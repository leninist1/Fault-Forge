"""E3b RCA quality rubric experiment for ASE NIER.

This companion experiment keeps the E3 CSV-only leakage contract, but evaluates
RCA answer quality on a 1-5 rubric instead of only root-service Top-1.

Predictor inputs:
  - technical_only: metrics.csv, logs.csv, traces.csv
  - technical_plus_business: metrics.csv, logs.csv, traces.csv, business.csv

Labels, metadata, PRISM verdicts, business JSON, service catalogs, and
fault-dimension catalogs are never exposed to the RCA predictor. Gold labels are
used only by the offline rubric evaluator.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import statistics
import time
from pathlib import Path
from typing import Any

import requests

try:
    from .e3_business_modality_value import (
        DEEPSEEK_BASE,
        DEEPSEEK_MODEL,
        TOOLS,
        CsvToolContext,
        _normalize_service,
        _read_json,
        _write_json,
    )
except ImportError:
    from e3_business_modality_value import (
        DEEPSEEK_BASE,
        DEEPSEEK_MODEL,
        TOOLS,
        CsvToolContext,
        _normalize_service,
        _read_json,
        _write_json,
    )


QUALITY_SYSTEM_PROMPT = """You are a careful Root Cause Analysis (RCA) engineer for Train-Ticket fault injection experiments.

You are given only CSV-derived tools. Do not assume access to fault labels,
metadata, business JSON files, service catalogs, architecture files, source
code, or a predefined fault taxonomy.

Investigate the CSVs and produce an RCA report. Use concrete evidence from the
available CSV tools. If business.csv is available, use it as additional
user-visible business evidence, but do not treat metric names as labels.

Output ONLY this JSON object:
{
  "root_cause_service": "<single most likely service>",
  "suspected_services": ["<up to 5 services>"],
  "fault_hypothesis": "<specific root-cause hypothesis>",
  "evidence": ["<evidence item 1>", "<evidence item 2>", "<evidence item 3>"],
  "causal_analysis": "<how the fault explains observed technical/business symptoms>",
  "business_impact": "<user-visible business impact, or 'not observed'>",
  "mitigation": "<concrete validation or repair action>",
  "confidence": "<low|medium|high>"
}

Complete the investigation in 3-5 tool turns. On the final turn, output JSON only."""


RUBRIC_SYSTEM_PROMPT = """You are an impartial RCA quality evaluator.

Score the candidate RCA answer against the gold fault description. The answer
may have been produced with or without business CSVs; do not infer or reward
the condition. Use only the supplied gold facts, evidence summary, and candidate
answer.

Use integer scores from 1 to 5:
1 = wrong or unsupported, 2 = weak/mostly wrong, 3 = partially correct,
4 = mostly correct with minor gaps, 5 = correct, specific, and well supported.

Dimensions:
- hypothesis: whether the root-cause hypothesis matches the gold fault mode.
- evidence: whether cited evidence is concrete and aligned with observed signals.
- causal_analysis: whether the answer explains how the fault produces symptoms.
- business_impact: whether the answer identifies the user-visible business impact and ties it to telemetry.
- mitigation: whether the proposed validation/repair is specific and actionable.

Output ONLY JSON:
{
  "hypothesis": 1-5,
  "evidence": 1-5,
  "causal_analysis": 1-5,
  "business_impact": 1-5,
  "mitigation": 1-5,
  "rationale": "<one short sentence>"
}"""


def _parse_json_response(content: str) -> dict[str, Any]:
    if not content:
        return {"error": "empty response", "raw": None}
    content = content.strip()
    if content.startswith("```"):
        parts = content.split("\n")
        content = "\n".join(parts[1:])
        if content.endswith("```"):
            content = content[:-3]
    content = content.strip()
    if content.startswith("json"):
        content = content[4:].strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        import re

        match = re.search(r"\{.*\}", content, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        return {"error": "json parse failed", "raw": content[:500]}


def _post_chat(
    api_key: str,
    messages: list[dict[str, Any]],
    base_url: str,
    model: str,
    *,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | None = None,
    max_tokens: int = 1000,
    temperature: float = 0.1,
    timeout: int = 120,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if tools is not None:
        payload["tools"] = tools
    if tool_choice is not None:
        payload["tool_choice"] = tool_choice
    resp = requests.post(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=payload,
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()


def _run_quality_rca_with_tools(
    ctx: CsvToolContext,
    api_key: str,
    base_url: str,
    model: str,
    max_turns: int = 6,
) -> dict[str, Any]:
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": QUALITY_SYSTEM_PROMPT},
        {"role": "user", "content": "Analyze this fault injection sample and produce the RCA quality-report JSON."},
    ]
    available_files = ctx._csv_files()
    tools = json.loads(json.dumps(TOOLS))
    for tool in tools:
        if tool["function"]["name"] == "read_csv":
            tool["function"]["parameters"]["properties"]["filename"]["enum"] = available_files
            tool["function"]["description"] = f"Read raw CSV rows. Available files: {available_files}"

    tool_trace: list[dict[str, str]] = []
    for turn in range(max_turns):
        if turn == max_turns - 1:
            messages.append({
                "role": "user",
                "content": "Final turn. Output the RCA JSON object now. Do not call more tools.",
            })
            tool_choice = "none"
        else:
            tool_choice = None

        try:
            body = _post_chat(
                api_key,
                messages,
                base_url,
                model,
                tools=tools,
                tool_choice=tool_choice,
                max_tokens=1200,
            )
        except Exception as exc:
            return {"error": str(exc), "raw": None, "tool_trace": tool_trace}

        choice = body["choices"][0]
        msg = choice["message"]
        finish = choice.get("finish_reason", "unknown")
        if msg.get("content") and finish == "stop" and not msg.get("tool_calls"):
            parsed = _parse_json_response(msg["content"])
            parsed["tool_trace"] = tool_trace
            return parsed

        if msg.get("tool_calls"):
            messages.append({"role": "assistant", "tool_calls": msg["tool_calls"], "content": msg.get("content") or ""})
            for tc in msg["tool_calls"]:
                fn_name = tc["function"]["name"]
                try:
                    fn_args = json.loads(tc["function"]["arguments"])
                except json.JSONDecodeError:
                    fn_args = {}

                if fn_name == "get_error_summary":
                    result = ctx.get_error_summary()
                elif fn_name == "get_service_errors":
                    result = ctx.get_service_errors(fn_args.get("service_name", ""), fn_args.get("limit", 10))
                elif fn_name == "get_metric_summary":
                    result = ctx.get_metric_summary()
                elif fn_name == "get_trace_summary":
                    result = ctx.get_trace_summary()
                elif fn_name == "get_business_anomalies":
                    result = ctx.get_business_anomalies()
                elif fn_name == "read_csv":
                    result = ctx.read_csv(fn_args.get("filename", ""), fn_args.get("offset", 0), fn_args.get("limit", 30))
                else:
                    result = f"Unknown tool: {fn_name}"

                if len(result) > 6000:
                    result = result[:6000] + f"\n... (truncated, original {len(result)} chars)"
                tool_trace.append({"tool": fn_name, "args": json.dumps(fn_args, ensure_ascii=False), "result_head": result[:1000]})
                messages.append({"role": "tool", "tool_call_id": tc["id"], "content": result})
        elif msg.get("content"):
            parsed = _parse_json_response(msg["content"])
            parsed["tool_trace"] = tool_trace
            return parsed
        else:
            return {"error": "no content or tool calls", "raw": None, "tool_trace": tool_trace}

    return {"error": "max turns exceeded", "raw": None, "tool_trace": tool_trace}


def _gold_reference(sample_dir: Path) -> dict[str, Any]:
    sample_id = sample_dir.name
    dataset_root = sample_dir.parent.parent if sample_dir.parent.name == "rca_inputs" else sample_dir.parent
    label_file = dataset_root / "labels" / f"{sample_id}.json"
    audit_dir = dataset_root / "audit" / sample_id
    label = _read_json(label_file) if label_file.exists() else {}
    audit_spec_file = audit_dir / "fault_spec.json"
    inline_spec_file = sample_dir / "fault_spec.json"
    audit_spec = (
        _read_json(audit_spec_file)
        if audit_spec_file.exists()
        else _read_json(inline_spec_file)
        if inline_spec_file.exists()
        else {}
    )
    spec = {**audit_spec, **label}
    metadata = _read_json(audit_dir / "metadata.json") if (audit_dir / "metadata.json").exists() else _read_json(sample_dir / "metadata.json") if (sample_dir / "metadata.json").exists() else {}
    evidence = metadata.get("quality_decision", {}).get("evidence", {}).get("strong_evidence", {})
    invs = metadata.get("invariant_violations") or metadata.get("quality_decision", {}).get("evidence", {}).get("new_invariant_violations", [])
    return {
        "sample_id": sample_id,
        "root_cause_service": spec.get("root_cause_service") or (spec.get("fault_point", {}) or {}).get("owner_service", "unknown"),
        "fault_dimension": spec.get("fault_dimension") or spec.get("dimension", "unknown"),
        "target_invariant": spec.get("target_invariant", "unknown"),
        "business_journey": spec.get("business_journey", "unknown"),
        "business_entity": spec.get("business_entity", "unknown"),
        "failure_mode": spec.get("fse_metadata", {}).get("failure_mode", "unknown"),
        "affected_services_count": evidence.get("affected_services_count", len(metadata.get("affected_services", []) or [])),
        "new_invariant_violations": evidence.get("new_invariant_violations", len(invs or [])),
        "invariant_ids": [row.get("invariant_id") for row in (invs or []) if isinstance(row, dict)][:5],
    }


def _score_localization(predicted_service: str, gold_service: str) -> int:
    pred = _normalize_service(predicted_service or "")
    gold = _normalize_service(gold_service or "")
    if pred == gold:
        return 5
    if pred and pred != "unknown" and pred.startswith("ts-"):
        return 2
    return 1


def _score_with_rubric(
    answer: dict[str, Any],
    gold: dict[str, Any],
    api_key: str,
    base_url: str,
    model: str,
) -> dict[str, Any]:
    if answer.get("error"):
        return {
            "overall": 1.0,
            "localization": 1,
            "hypothesis": 1,
            "evidence": 1,
            "causal_analysis": 1,
            "business_impact": 1,
            "mitigation": 1,
            "rationale": f"Parse or API error: {answer.get('error')}",
        }
    prompt = {
        "gold_fault": gold,
        "candidate_answer": {k: v for k, v in answer.items() if k != "tool_trace"},
    }
    messages = [
        {"role": "system", "content": RUBRIC_SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(prompt, ensure_ascii=False, indent=2)},
    ]
    try:
        body = _post_chat(api_key, messages, base_url, model, tools=None, max_tokens=500, temperature=0.0)
        content = body["choices"][0]["message"].get("content", "")
        judged = _parse_json_response(content)
    except Exception as exc:
        judged = {"error": str(exc)}

    localization = _score_localization(str(answer.get("root_cause_service", "")), gold["root_cause_service"])
    scores: dict[str, Any] = {"localization": localization}
    for key in ("hypothesis", "evidence", "causal_analysis", "business_impact", "mitigation"):
        value = judged.get(key, 1)
        try:
            value_int = int(value)
        except (TypeError, ValueError):
            value_int = 1
        scores[key] = max(1, min(5, value_int))
    weights = {
        "localization": 0.20,
        "hypothesis": 0.25,
        "evidence": 0.20,
        "causal_analysis": 0.20,
        "business_impact": 0.05,
        "mitigation": 0.10,
    }
    scores["overall"] = round(sum(scores[k] * w for k, w in weights.items()), 2)
    scores["rationale"] = judged.get("rationale", judged.get("error", ""))
    return scores


def _score_dry_run(gold: dict[str, Any]) -> dict[str, Any]:
    return {
        "overall": 5.0,
        "localization": 5,
        "hypothesis": 5,
        "evidence": 5,
        "causal_analysis": 5,
        "business_impact": 5,
        "mitigation": 5,
        "rationale": "dry_run_oracle_prediction",
    }


def _bootstrap_ci95(values: list[float], reps: int = 3000) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return 0.0
    rng = random.Random(20260501)
    means = []
    for _ in range(reps):
        sample = [values[rng.randrange(len(values))] for _ in values]
        means.append(sum(sample) / len(sample))
    means.sort()
    lo = means[int(0.025 * reps)]
    hi = means[int(0.975 * reps)]
    return round((hi - lo) / 2, 3)


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    metrics = ["overall", "localization", "hypothesis", "evidence", "causal_analysis", "business_impact", "mitigation"]
    for condition in ("technical_only", "technical_plus_business"):
        block = [r for r in rows if r["condition"] == condition]
        for metric in metrics:
            values = [float(r["scores"][metric]) for r in block]
            out.append({
                "group": "all",
                "condition": condition,
                "metric": metric,
                "samples": len(values),
                "mean_score": round(sum(values) / len(values), 3) if values else 0.0,
                "ci95": _bootstrap_ci95(values),
            })
    return out


def _plot_quality(aggregate_rows: list[dict[str, Any]], output_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metric_order = ["overall", "localization", "hypothesis", "evidence", "causal_analysis", "business_impact", "mitigation"]
    metric_labels = ["Overall", "Localization", "Hypothesis", "Evidence", "Analysis", "Business", "Mitigation"]
    fig, ax = plt.subplots(figsize=(8.8, 4.2))
    colors = {"technical_only": "#69778c", "technical_plus_business": "#287c76"}
    labels = {"technical_only": "w/o business CSV", "technical_plus_business": "w/ business CSV"}
    width = 0.36
    x = list(range(len(metric_order)))
    lookup = {(r["condition"], r["metric"]): r for r in aggregate_rows}
    for offset, condition in [(-width / 2, "technical_only"), (width / 2, "technical_plus_business")]:
        means = [lookup[(condition, metric)]["mean_score"] for metric in metric_order]
        errs = [lookup[(condition, metric)]["ci95"] for metric in metric_order]
        ax.bar([i + offset for i in x], means, width, yerr=errs, capsize=3, color=colors[condition], label=labels[condition])
    ax.axhline(3.0, color="#6b6f76", linestyle="--", linewidth=1.0)
    ax.set_title("E3: Pooled RCA Quality Across All Fault Modes", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, rotation=18, ha="right")
    ax.set_ylim(0, 5.4)
    ax.grid(axis="y", color="#e0e4e8", linewidth=0.8)
    ax.text(0.02, 0.94, f"n={lookup[('technical_only', 'overall')]['samples']}", transform=ax.transAxes, fontsize=9)
    ax.set_ylabel("Rubric score (1-5)")
    ax.legend(loc="upper right", frameon=True)
    fig.tight_layout()
    fig.savefig(output_dir / "fig_e3_quality_scores.png", dpi=220)
    plt.close(fig)


def run_quality_experiment(
    dataset_dir: Path,
    label: str,
    output_dir: Path,
    api_key: str,
    base_url: str = DEEPSEEK_BASE,
    model: str = DEEPSEEK_MODEL,
    judge_model: str = DEEPSEEK_MODEL,
    max_samples: int | None = None,
    reuse_rca: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rca_inputs = dataset_dir / "rca_inputs"
    sample_root = rca_inputs if rca_inputs.is_dir() else dataset_dir
    samples = sorted([p for p in sample_root.iterdir() if p.is_dir()])
    if max_samples:
        samples = samples[:max_samples]

    existing: dict[tuple[str, str], dict[str, Any]] = {}
    result_path = output_dir / "e3_quality_results.json"
    if reuse_rca and result_path.exists():
        old = _read_json(result_path)
        for row in old.get("per_sample", []):
            answer = row.get("answer") or {}
            if answer.get("error"):
                continue
            existing[(row["sample_id"], row["condition"])] = row

    rows: list[dict[str, Any]] = []
    print(f"Loaded {len(samples)} samples for E3b RCA quality experiment")
    for idx, sample_dir in enumerate(samples, start=1):
        gold = _gold_reference(sample_dir)
        print(f"[{idx}/{len(samples)}] {sample_dir.name} ...", flush=True)
        for condition, include_business in [("technical_only", False), ("technical_plus_business", True)]:
            if (sample_dir.name, condition) in existing:
                rows.append(existing[(sample_dir.name, condition)])
                print(f"  {condition}: reused")
                continue
            ctx = CsvToolContext(sample_dir, include_business=include_business)
            if dry_run:
                answer = {
                    "root_cause_service": gold["root_cause_service"],
                    "suspected_services": [gold["root_cause_service"]],
                    "fault_hypothesis": gold["failure_mode"],
                    "evidence": ["dry_run_oracle"],
                    "causal_analysis": "dry_run_oracle",
                    "business_impact": gold["target_invariant"],
                    "mitigation": "dry_run_oracle",
                    "confidence": "high",
                    "tool_trace": [],
                }
                scores = _score_dry_run(gold)
            else:
                answer = _run_quality_rca_with_tools(ctx, api_key, base_url, model)
                time.sleep(0.8)
                scores = _score_with_rubric(answer, gold, api_key, base_url, judge_model)
                time.sleep(0.8)
            row = {
                "sample_id": sample_dir.name,
                "subset": label,
                "condition": condition,
                "failure_mode": gold["failure_mode"],
                "gold": gold,
                "answer": answer,
                "scores": scores,
            }
            rows.append(row)
            print(f"  {condition}: overall={scores['overall']} loc={scores['localization']} hyp={scores['hypothesis']} ev={scores['evidence']}")

    aggregate = _aggregate(rows)
    result = {
        "experiment": "E3b RCA Quality Rubric",
        "method": "csv_only_rca_quality_rubric",
        "predictor_model": model,
        "judge_model": judge_model,
        "dataset": label,
        "total_samples": len(samples),
        "input_contract": {
            "technical_only": ["metrics.csv", "logs.csv", "traces.csv"],
            "technical_plus_business": ["metrics.csv", "logs.csv", "traces.csv", "business.csv"],
            "excluded_from_predictor": [
                "fault_spec.json",
                "metadata.json",
                "prism_verdict.json",
                "business_journey.json",
                "business_invariants.json",
                "business_state_snapshot.json",
                "service catalog",
                "fault-dimension catalog",
            ],
        },
        "score_scale": {
            "range": "1-5",
            "positive_threshold": 3,
            "weights": {
                "localization": 0.20,
                "hypothesis": 0.25,
                "evidence": 0.20,
                "causal_analysis": 0.20,
                "business_impact": 0.05,
                "mitigation": 0.10,
            },
        },
        "rubric": ["overall", "localization", "hypothesis", "evidence", "causal_analysis", "business_impact", "mitigation"],
        "aggregate": aggregate,
        "per_sample": rows,
    }
    _write_json(result_path, result)

    with (output_dir / "e3_quality_summary.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["group", "condition", "metric", "samples", "mean_score", "ci95"])
        writer.writeheader()
        writer.writerows(aggregate)

    _plot_quality(aggregate, output_dir)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="E3b RCA quality rubric experiment")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--label", default="all_family_60")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--api-key", default=os.environ.get("DEEPSEEK_API_KEY", ""))
    parser.add_argument("--base-url", default=DEEPSEEK_BASE)
    parser.add_argument("--model", default=DEEPSEEK_MODEL)
    parser.add_argument("--judge-model", default=DEEPSEEK_MODEL)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--reuse-rca", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not args.api_key and not args.dry_run:
        parser.error("--api-key or DEEPSEEK_API_KEY is required")
    run_quality_experiment(
        dataset_dir=args.dataset_dir,
        label=args.label,
        output_dir=args.output_dir,
        api_key=args.api_key,
        base_url=args.base_url,
        model=args.model,
        judge_model=args.judge_model,
        max_samples=args.max_samples,
        reuse_rca=args.reuse_rca,
        dry_run=args.dry_run,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
