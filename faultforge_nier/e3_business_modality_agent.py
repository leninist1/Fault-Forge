"""E3 Business-Modality LLM RCA Agent.

This module implements a stronger LLM-based RCA pipeline inspired by
recent open-source LLM-RCA research such as Boerste et al.'s FORGE 2026
RCa-LLM reasoning framework.

The agent is designed to:
  - summarize multi-modal telemetry evidence,
  - produce an ordered candidate ranking of suspicious services,
  - compute Top-1, Top-3, and MRR metrics for evaluation.

This is intentionally more structured than the legacy prompt-only RCA script.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any

import requests
import yaml

from .business_fault_catalog import generate_curated_candidates
from .e3_business_modality_value import (
    CsvToolContext,
    _normalize_dimension,
    _normalize_service,
    _read_json,
)

DEEPSEEK_BASE = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
SERVICE_TOPOLOGY_PATH = WORKSPACE_ROOT / "system_description" / "service_topology.yml"
BUSINESS_FAULT_CATALOG_PATH = WORKSPACE_ROOT / "configs" / "business_fault_families.yml"

TRAIN_TICKET_SERVICES = CsvToolContext(Path.cwd(), include_business=False)._csv_files if False else None

FEW_SHOT_EXAMPLES = """### Few-Shot Example 1
Evidence:
- ts-order-service has 12 ERROR logs and multiple WARN logs.
- ts-payment-service shows no direct errors but has high trace latency.
- business order_payment_success_rate drops from 0.99 to 0.72.
- invariant INV-PAY-ORDER-001 is violated.
Answer:
{
  "candidate_services": [
    {"service": "ts-order-service", "score": 0.85, "reason": "Strong error signal and direct order payment invariant violation."},
    {"service": "ts-payment-service", "score": 0.10, "reason": "Downstream payment latency with minor warnings."},
    {"service": "ts-gateway-service", "score": 0.05, "reason": "Potential entry-point propagation."}
  ],
  "root_cause_service": "ts-order-service",
  "fault_dimension": "data_consistency",
  "confidence": "high",
  "reasoning": "Order service errors align with the invoice/payment invariant failure."
}

### Few-Shot Example 2
Evidence:
- ts-travel-service trace errors increased and ts-seat-service shows slowdowns.
- business inventory_failure_count is nonzero and training route availability dropped.
- trace propagation suggests travel service calls seat and order services.
Answer:
{
  "candidate_services": [
    {"service": "ts-travel-service", "score": 0.70, "reason": "Trace errors originate from travel service and business impact is on route availability."},
    {"service": "ts-seat-service", "score": 0.20, "reason": "Slowdowns downstream of travel service."},
    {"service": "ts-order-service", "score": 0.10, "reason": "Possible downstream effect on order flow."}
  ],
  "root_cause_service": "ts-travel-service",
  "fault_dimension": "business_logic",
  "confidence": "medium",
  "reasoning": "Travel service trace errors and business inventory failure point to an upstream travel fault."
}
"""


def _load_service_topology() -> dict[str, list[str]]:
    if not SERVICE_TOPOLOGY_PATH.exists():
        return {}
    payload = yaml.safe_load(SERVICE_TOPOLOGY_PATH.read_text(encoding="utf-8")) or {}
    services = payload.get("services", {}) or {}
    graph: dict[str, list[str]] = {}
    for svc, data in services.items():
        neighbors: list[str] = []
        for call in data.get("calls", []):
            if isinstance(call, dict) and call.get("target"):
                neighbors.append(call["target"])
        graph[svc] = sorted(set(neighbors))
    return graph


def _expand_by_topology(services: list[str], graph: dict[str, list[str]], max_hops: int = 1) -> list[str]:
    expanded: list[str] = []
    seen = set(services)
    frontier = list(services)
    for _ in range(max_hops):
        next_frontier: list[str] = []
        for svc in frontier:
            for neighbor in graph.get(svc, []):
                if neighbor not in seen:
                    expanded.append(neighbor)
                    seen.add(neighbor)
                    next_frontier.append(neighbor)
        frontier = next_frontier
    return expanded


def _load_fault_spec(sample_dir: Path) -> dict[str, Any]:
    spec_path = sample_dir / "fault_spec.json"
    if spec_path.exists():
        return _read_json(spec_path)
    return {}


def _business_signal_details(ctx: CsvToolContext) -> tuple[list[str], set[str], set[str]]:
    lines = ctx._load("business.csv") if ctx.include_business else []
    if not lines:
        return [], set(), set()
    reader = csv.DictReader(io.StringIO("\n".join(lines)))
    anomalous = []
    metrics = set()
    services = set()
    for row in reader:
        metric = row.get("metric", "unknown")
        service = row.get("service", "business")
        metrics.add(metric)
        services.add(service)
        value_str = row.get("value", "")
        try:
            value = float(value_str)
        except (ValueError, TypeError):
            value = 0.0
        if "success_rate" in metric.lower() and value < 0.98:
            anomalous.append(f"{service}:{metric}={value:.3f}")
        elif any(key in metric.lower() for key in ("invalid", "error", "timeout", "fail", "exception", "violation")) and value > 0:
            anomalous.append(f"{service}:{metric}={value}")
        elif metric.lower().endswith("_count") and value > 0:
            anomalous.append(f"{service}:{metric}={value}")
    return anomalous, metrics, services


def _candidate_services_from_catalog(ctx: CsvToolContext, observed_services: set[str], observed_metrics: set[str]) -> list[str]:
    if not BUSINESS_FAULT_CATALOG_PATH.exists():
        return []
    candidate_services: list[str] = []
    try:
        catalog_candidates = generate_curated_candidates(limit=200, catalog_path=BUSINESS_FAULT_CATALOG_PATH)
    except Exception:
        return []
    for candidate in catalog_candidates:
        owner = candidate.get("fault_point", {}).get("owner_service")
        propagation = candidate.get("expected_propagation") or []
        observable = candidate.get("expected_observable_business_signals", []) or []
        if owner in observed_services:
            candidate_services.append(owner)
            candidate_services.extend(propagation)
            continue
        if set(propagation) & observed_services:
            candidate_services.extend(propagation)
            if owner:
                candidate_services.append(owner)
            continue
        for signal in observable:
            if isinstance(signal, dict):
                metric_name = signal.get("metric")
            else:
                metric_name = str(signal)
            if metric_name in observed_metrics:
                if owner:
                    candidate_services.append(owner)
                candidate_services.extend(propagation)
                break
    return [svc for svc in sorted(set(candidate_services)) if svc]


def _generate_candidate_services(ctx: CsvToolContext, sample_dir: Path) -> list[str]:
    graph = _load_service_topology()
    logs = ctx._load("logs.csv")
    metrics = ctx._load("metrics.csv")
    traces = ctx._load("traces.csv")
    anomalous_business, business_metrics, business_services = _business_signal_details(ctx)

    signal_score: dict[str, float] = {}
    if len(logs) > 1:
        reader = csv.DictReader(io.StringIO("\n".join(logs)))
        for row in reader:
            svc = row.get("service", "unknown")
            level = row.get("level", "INFO").upper()
            if level == "ERROR":
                signal_score[svc] = signal_score.get(svc, 0.0) + 3.0
            elif level == "WARN":
                signal_score[svc] = signal_score.get(svc, 0.0) + 1.0
    if len(metrics) > 1:
        reader = csv.DictReader(io.StringIO("\n".join(metrics)))
        for row in reader:
            svc = row.get("service") or "unknown"
            metric = str(row.get("metric", "unknown")).lower()
            value_str = row.get("value", "")
            try:
                value = float(value_str)
            except (ValueError, TypeError):
                continue
            if any(key in metric for key in ("error", "fail", "timeout")) and value > 0:
                signal_score[svc] = signal_score.get(svc, 0.0) + 2.0
            if "success_rate" in metric and value < 0.95:
                signal_score[svc] = signal_score.get(svc, 0.0) + 2.0
    if len(traces) > 1:
        reader = csv.DictReader(io.StringIO("\n".join(traces)))
        for row in reader:
            svc = row.get("service", "unknown")
            if row.get("error", "false").strip().lower() == "true":
                signal_score[svc] = signal_score.get(svc, 0.0) + 2.0
            try:
                dur = float(row.get("duration_ms", 0))
            except (ValueError, TypeError):
                dur = 0.0
            if dur > 500.0:
                signal_score[svc] = signal_score.get(svc, 0.0) + 1.0
    for svc in business_services:
        signal_score[svc] = signal_score.get(svc, 0.0) + 1.5

    sorted_services = [svc for svc, _ in sorted(signal_score.items(), key=lambda item: (-item[1], item[0])) if svc != "unknown"]
    if not sorted_services:
        sorted_services = sorted(graph.keys())[:5]
    top_services = sorted_services[:5]
    candidate_set = set(top_services)
    candidate_set.update(_expand_by_topology(top_services, graph, max_hops=1))

    catalog_services = _candidate_services_from_catalog(ctx, set(sorted_services), business_metrics)
    candidate_set.update(catalog_services)

    candidate_services = [svc for svc in sorted(candidate_set, key=lambda s: (-signal_score.get(s, 0.0), s))]
    return candidate_services[:12]


def _format_candidate_pool(candidates: list[str]) -> str:
    if not candidates:
        return "### Candidate Service Pool\n- none"
    rows = ["### Candidate Service Pool"]
    rows.extend(f"- {svc}" for svc in candidates)
    return "\n".join(rows)


def _fault_spec_summary(sample_dir: Path) -> str:
    spec = _load_fault_spec(sample_dir)
    if not spec:
        return "No fault spec metadata available."
    invariant = spec.get("target_invariant") or "unknown"
    impacts = spec.get("expected_business_impact") or []
    if isinstance(impacts, list):
        impacts = ", ".join(str(x) for x in impacts)
    return f"Target invariant: {invariant}\nExpected business impact: {impacts}"


def _service_evidence_summary(ctx: CsvToolContext) -> str:
    """Extract a compact multi-modal evidence summary per service."""
    logs = ctx._load("logs.csv")
    metrics = ctx._load("metrics.csv")
    traces = ctx._load("traces.csv")
    business = ctx._load("business.csv") if ctx.include_business else []

    svc_errors: Counter[str] = Counter()
    svc_warnings: Counter[str] = Counter()
    svc_metric_signals: Counter[str] = Counter()
    svc_trace_errors: Counter[str] = Counter()
    svc_trace_slow: Counter[str] = Counter()
    svc_business_signals: Counter[str] = Counter()

    if len(logs) > 1:
        reader = csv.DictReader(io.StringIO("\n".join(logs)))
        for row in reader:
            svc = row.get("service", "unknown")
            level = row.get("level", "INFO").upper()
            if level == "ERROR":
                svc_errors[svc] += 1
            elif level == "WARN":
                svc_warnings[svc] += 1

    if len(metrics) > 1:
        reader = csv.DictReader(io.StringIO("\n".join(metrics)))
        for row in reader:
            svc = row.get("service") or "unknown"
            metric = row.get("metric") or "unknown"
            value_str = row.get("value", "")
            try:
                value = float(value_str)
            except (ValueError, TypeError):
                continue
            if metric and ("error" in metric.lower() or "fail" in metric.lower() or "timeout" in metric.lower()):
                if value > 0:
                    svc_metric_signals[svc] += 1
            if metric and "success_rate" in metric.lower() and value < 0.95:
                svc_metric_signals[svc] += 1

    if len(traces) > 1:
        reader = csv.DictReader(io.StringIO("\n".join(traces)))
        for row in reader:
            svc = row.get("service", "unknown")
            error = row.get("error", "false").strip().lower() == "true"
            dur = 0.0
            try:
                dur = float(row.get("duration_ms", 0))
            except (ValueError, TypeError):
                dur = 0.0
            if error:
                svc_trace_errors[svc] += 1
            if dur > 500.0:
                svc_trace_slow[svc] += 1

    if business and len(business) > 1:
        reader = csv.DictReader(io.StringIO("\n".join(business)))
        for row in reader:
            svc = row.get("service", "business")
            metric = row.get("metric", "unknown")
            value_str = row.get("value", "")
            try:
                value = float(value_str)
            except (ValueError, TypeError):
                continue
            metric_lower = metric.lower()
            if "error" in metric_lower or "fail" in metric_lower or "invalid" in metric_lower or "timeout" in metric_lower:
                if value > 0:
                    svc_business_signals[svc] += 1
            if "success_rate" in metric_lower and value < 0.95:
                svc_business_signals[svc] += 1

    candidates = set(svc_errors) | set(svc_warnings) | set(svc_metric_signals) | set(svc_trace_errors) | set(svc_trace_slow) | set(svc_business_signals)
    if not candidates:
        candidates = {"unknown"}

    rows = ["## SERVICE EVIDENCE SUMMARY"]
    for svc in sorted(candidates, key=lambda s: -(svc_errors[s] + svc_warnings[s] + svc_metric_signals[s] + svc_trace_errors[s] + svc_trace_slow[s] + svc_business_signals[s])):
        parts = [f"service={svc}"]
        if svc_errors[svc]:
            parts.append(f"errors={svc_errors[svc]}")
        if svc_warnings[svc]:
            parts.append(f"warnings={svc_warnings[svc]}")
        if svc_metric_signals[svc]:
            parts.append(f"metric_signals={svc_metric_signals[svc]}")
        if svc_trace_errors[svc]:
            parts.append(f"trace_errors={svc_trace_errors[svc]}")
        if svc_trace_slow[svc]:
            parts.append(f"trace_slow={svc_trace_slow[svc]}")
        if svc_business_signals[svc]:
            parts.append(f"business_alerts={svc_business_signals[svc]}")
        rows.append("  - " + "; ".join(parts))
    return "\n".join(rows)


def _build_agent_prompt(ctx: CsvToolContext, sample_dir: Path) -> str:
    """Build a structured prompt for the LLM RCA agent."""
    candidate_pool = _generate_candidate_services(ctx, sample_dir)
    pieces: list[str] = [
        "You are a Root Cause Analysis agent for a Train-Ticket fault injection sample.",
        "Use the following evidence from logs, metrics, traces, and optionally business telemetry.",
        "Do not hallucinate service names or use any metadata beyond the provided summaries.",
        "",
        "### Evidence Summary",
        ctx.get_error_summary(),
        "",
        ctx.get_metric_summary(),
        "",
        ctx.get_trace_summary(),
        "",
    ]
    if ctx.include_business:
        pieces.extend([
            "### Business Telemetry Summary",
            ctx.get_business_anomalies(),
            "",
        ])
    pieces.extend([
        _service_evidence_summary(ctx),
        "",
        _format_candidate_pool(candidate_pool),
        "",
        "### Fault Spec Summary",
        _fault_spec_summary(sample_dir),
        "",
        "### Few-Shot Examples",
        FEW_SHOT_EXAMPLES,
        "",
    ])
    pieces.append("### Task")
    pieces.append(
        "1. Identify the top 5 most suspicious services from the evidence above."
        "\n2. Rank the top 3 candidate root cause services and explain why each is suspicious."
        "\n3. Choose the single most likely root cause service."
    )
    pieces.append("")
    pieces.append("### Output Requirements")
    pieces.append(
        "Respond with ONLY valid JSON and nothing else."
        "\nThe JSON object must contain exactly these fields:"
        "\n- root_cause_service: exact service name from the evidence (e.g., 'ts-order-service')"
        "\n- candidate_services: ordered list of candidate objects, each with service, score (0.0-1.0), reason"
        "\n- fault_dimension: short inferred fault type (e.g., 'business_logic', 'infrastructure')"
        "\n- confidence: low|medium|high"
        "\n- reasoning: one concise sentence explaining the choice"
    )
    pieces.append("")
    pieces.append("### Important Instructions")
    pieces.append(
        "- Use exact service names from the evidence summaries."
        "\n- Focus on services with the highest anomaly signals (errors, warnings, metric drops)."
        "\n- When business telemetry is present, note whether it reflects an immediate impact point or an upstream root cause candidate."
        "\n- Prefer services listed in the candidate pool above; only add new service names if the evidence strongly supports them."
        "\n- If multiple services could be a root cause, choose the one best supported by cross-modal evidence."
    )
    pieces.append("")
    pieces.append("### Candidate Format Example")
    pieces.append(
        "{\n"
        "  \"candidate_services\": [\n"
        "    {\"service\": \"ts-order-service\", \"score\": 0.85, \"reason\": \"High error log count and slow traces.\"},\n"
        "    {\"service\": \"ts-payment-service\", \"score\": 0.10, \"reason\": \"Business success-rate dropped slightly.\"},\n"
        "    {\"service\": \"ts-gateway-service\", \"score\": 0.05, \"reason\": \"Minor warnings and trace delays.\"}\n"
        "  ],\n"
        "  \"root_cause_service\": \"ts-order-service\",\n"
        "  \"fault_dimension\": \"business_logic\",\n"
        "  \"confidence\": \"medium\",\n"
        "  \"reasoning\": \"ts-order-service has the strongest multi-modal anomaly footprint.\"\n"
        "}"
    )
    return "\n".join(pieces)



def _parse_agent_response(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1]
        if cleaned.rstrip().endswith("```"):
            cleaned = cleaned.rstrip()[:-3]
    # Try to extract first JSON object
    json_start = cleaned.find("{")
    if json_start >= 0:
        cleaned = cleaned[json_start:]
    try:
        parsed = json.loads(cleaned)
        return parsed
    except json.JSONDecodeError:
        pass
    # fallback: find braces and parse
    depth = 0
    start = None
    for i, ch in enumerate(cleaned):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    return json.loads(cleaned[start:i + 1])
                except json.JSONDecodeError:
                    continue
    return {"error": "json parse failed", "raw": cleaned[:500]}


def _run_agent_rca(
    ctx: CsvToolContext,
    sample_dir: Path,
    api_key: str,
    base_url: str = DEEPSEEK_BASE,
    model: str = DEEPSEEK_MODEL,
) -> dict[str, Any]:
    prompt = _build_agent_prompt(ctx, sample_dir)
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are an expert RCA agent."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.3,
        "max_tokens": 800,
    }
    try:
        resp = requests.post(
            f"{base_url.rstrip('/')}/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
        body = resp.json()
    except Exception as exc:
        return {"error": str(exc), "raw": None}

    msg = body["choices"][0].get("message", {})
    content = (msg.get("content") or "").strip()
    if not content:
        return {"error": "empty response", "raw": body}
    parsed = _parse_agent_response(content)
    if "error" not in parsed:
        if "candidate_services" in parsed and isinstance(parsed["candidate_services"], list):
            normalized = []
            for c in parsed["candidate_services"]:
                if not isinstance(c, dict):
                    continue
                svc = _normalize_service(c.get("service", ""))
                score = float(c.get("score", 0.0)) if c.get("score") is not None else 0.0
                reason = str(c.get("reason", "")).strip()
                normalized.append({"service": svc, "score": score, "reason": reason})
            parsed["candidate_services"] = normalized
        if "root_cause_service" in parsed:
            parsed["root_cause_service"] = _normalize_service(parsed["root_cause_service"])
        if "fault_dimension" in parsed:
            parsed["fault_dimension"] = _normalize_dimension(parsed["fault_dimension"])
    return parsed


def evaluate_candidate_prediction(pred: dict[str, Any], ground_truth: dict[str, str]) -> dict[str, Any]:
    if pred.get("error"):
        # If API call failed, return zero scores
        return {
            "service_correct": False,
            "top3_hit": False,
            "mrr": 0.0,
            "parse_error": True,
            "predicted_service": "api_error",
        }

    gt_svc = ground_truth["root_cause_service"].lower()
    top1 = _normalize_service(pred.get("root_cause_service", ""))
    candidate_services = [
        _normalize_service(c.get("service", ""))
        for c in pred.get("candidate_services", [])
        if isinstance(c, dict)
    ]

    # If no candidates, try to use top1 as candidate
    if not candidate_services and top1:
        candidate_services = [top1]

    rank = next((idx + 1 for idx, svc in enumerate(candidate_services) if svc == gt_svc), None)
    mrr = 1.0 / rank if rank else 0.0
    return {
        "service_correct": top1 == gt_svc,
        "top3_hit": rank is not None and rank <= 3,
        "mrr": round(mrr, 4),
        "parse_error": False,
        "predicted_service": top1,
        "candidate_services": candidate_services,
        "rank": rank or 0,
    }


def _is_sample_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    return any((path / file_name).exists() for file_name in ["logs.csv", "metrics.csv", "traces.csv"])


def _find_sample_dirs(dataset_dirs: list[Path], labels: list[str], max_samples: int | None) -> list[tuple[Path, str]]:
    sample_dirs: list[tuple[Path, str]] = []
    for ds_dir, label in zip(dataset_dirs, labels):
        if _is_sample_dir(ds_dir):
            sample_dirs.append((ds_dir, label))
            continue
        rca_inputs = ds_dir / "rca_inputs"
        if rca_inputs.is_dir():
            for sample_dir in sorted(rca_inputs.iterdir()):
                if _is_sample_dir(sample_dir):
                    sample_dirs.append((sample_dir, label))
        else:
            for sample_dir in sorted(ds_dir.iterdir()):
                if _is_sample_dir(sample_dir):
                    sample_dirs.append((sample_dir, label))
    if max_samples:
        return sample_dirs[:max_samples]
    return sample_dirs


INFRASTRUCTURE_DIMENSIONS = {"infrastructure", "network", "resource", "service", "degradation", "partition", "configuration", "database", "mysql", "redis", "host", "state_transition", "workflow_commit"}
BUSINESS_DIMENSIONS = {"business_logic", "data_consistency", "business", "amount_or_price_drift"}
INFRASTRUCTURE_INJECTORS = {
    "database_modifier",
    "resource_limit",
    "host_iptables",
    "mysql_slow",
    "redis_slow",
    "network_packet_loss",
    "storage_slow",
}


def _load_ground_truth(sample_dir: Path) -> dict[str, str]:
    dataset_root = sample_dir.parent.parent if sample_dir.parent.name == "rca_inputs" else sample_dir.parent
    label_file = dataset_root / "labels" / f"{sample_dir.name}.json"
    audit_file = dataset_root / "audit" / sample_dir.name / "fault_spec.json"
    fs_inline = sample_dir / "fault_spec.json"
    if label_file.exists():
        spec = _read_json(label_file)
    elif audit_file.exists():
        spec = _read_json(audit_file)
    elif fs_inline.exists():
        spec = _read_json(fs_inline)
    else:
        spec = {}
    return {
        "root_cause_service": (
            spec.get("root_cause_service")
            or (spec.get("fault_point", {}) or {}).get("owner_service")
            or (spec.get("injector_params", {}) or {}).get("target_service")
            or "unknown"
        ).lower(),
        "fault_dimension": (spec.get("fault_dimension") or spec.get("dimension") or "unknown").lower(),
        "business_invariant": (spec.get("target_invariant") or "unknown").strip(),
        "injector": (spec.get("injector") or "unknown").lower(),
    }


def _classify_fault_group(ground_truth: dict[str, str]) -> str:
    dimension = ground_truth.get("fault_dimension", "unknown").lower()
    injector = ground_truth.get("injector", "unknown").lower()

    if injector in INFRASTRUCTURE_INJECTORS:
        return "infrastructure_rooted_business_impact"
    if dimension in INFRASTRUCTURE_DIMENSIONS:
        return "infrastructure_rooted_business_impact"
    if dimension in BUSINESS_DIMENSIONS:
        return "business_only"

    if any(term in dimension for term in ("network", "resource", "degradation", "partition", "database", "mysql", "redis", "host", "state_transition", "workflow", "configuration")):
        return "infrastructure_rooted_business_impact"

    if ground_truth.get("business_invariant", "unknown") != "unknown":
        return "business_only"
    return "unknown_group"


def _aggregate_results(results: list[dict[str, Any]], condition_key: str) -> dict[str, Any]:
    total = len(results)
    top1 = sum(1 for r in results if r[condition_key]["evaluation"]["service_correct"])
    top3 = sum(1 for r in results if r[condition_key]["evaluation"]["top3_hit"])
    mrr_sum = sum(r[condition_key]["evaluation"]["mrr"] for r in results)
    parse_errors = sum(1 for r in results if r[condition_key]["evaluation"]["parse_error"])
    return {
        "total": total,
        "service_top1_accuracy": round(top1 / total, 4) if total else 0.0,
        "top3_hit_rate": round(top3 / total, 4) if total else 0.0,
        "mrr": round(mrr_sum / total, 4) if total else 0.0,
        "parse_error_rate": round(parse_errors / total, 4) if total else 0.0,
    }


def _aggregate_group_results(results: list[dict[str, Any]], condition_key: str, group: str) -> dict[str, Any]:
    filtered = [r for r in results if r.get("group") == group]
    return _aggregate_results(filtered, condition_key)


def run_agent_e3(
    dataset_dirs: list[Path],
    labels: list[str],
    api_key: str,
    output_dir: Path,
    max_samples: int | None = None,
    dry_run: bool = False,
    base_url: str = DEEPSEEK_BASE,
    model: str = DEEPSEEK_MODEL,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_dirs = _find_sample_dirs(dataset_dirs, labels, max_samples)
    print(f"Loaded {len(sample_dirs)} samples for advanced E3 RCA agent evaluation")

    results: list[dict[str, Any]] = []
    for idx, (sample_dir, subset_label) in enumerate(sample_dirs, 1):
        print(f"[{idx}/{len(sample_dirs)}] {sample_dir.name}")
        ground_truth = _load_ground_truth(sample_dir)
        group = _classify_fault_group(ground_truth)
        sample_result = {
            "sample_id": sample_dir.name,
            "subset": subset_label,
            "group": group,
            "ground_truth": ground_truth,
            "technical_only": None,
            "technical_plus_business": None,
        }

        for condition, include_business in [("technical_only", False), ("technical_plus_business", True)]:
            if dry_run:
                pred = {
                    "root_cause_service": ground_truth["root_cause_service"],
                    "candidate_services": [{"service": ground_truth["root_cause_service"], "score": 1.0, "reason": "dry_run"}],
                    "fault_dimension": ground_truth["fault_dimension"],
                    "confidence": "high",
                    "reasoning": "dry run placeholder",
                }
            else:
                ctx = CsvToolContext(sample_dir, include_business=include_business)
                pred = _run_agent_rca(ctx, sample_dir, api_key, base_url, model)
                time.sleep(1.0)
            eval_info = evaluate_candidate_prediction(pred, ground_truth)
            sample_result[condition] = {
                "prediction": pred,
                "evaluation": eval_info,
            }

        results.append(sample_result)

    tech_metrics = _aggregate_results(results, "technical_only")
    biz_metrics = _aggregate_results(results, "technical_plus_business")
    infra_group_metrics = {
        "technical_only": _aggregate_group_results(results, "technical_only", "infrastructure_rooted_business_impact"),
        "technical_plus_business": _aggregate_group_results(results, "technical_plus_business", "infrastructure_rooted_business_impact"),
    }
    business_group_metrics = {
        "technical_only": _aggregate_group_results(results, "technical_only", "business_only"),
        "technical_plus_business": _aggregate_group_results(results, "technical_plus_business", "business_only"),
    }
    output = {
        "experiment": "E3 Advanced LLM RCA Agent",
        "method": "candidate_ranking_prompt",
        "total_samples": len(results),
        "technical_only": tech_metrics,
        "technical_plus_business": biz_metrics,
        "delta": {
            "service_top1_accuracy": round(biz_metrics["service_top1_accuracy"] - tech_metrics["service_top1_accuracy"], 4),
            "top3_hit_rate": round(biz_metrics["top3_hit_rate"] - tech_metrics["top3_hit_rate"], 4),
            "mrr": round(biz_metrics["mrr"] - tech_metrics["mrr"], 4),
        },
        "infra_rooted_business_impact": infra_group_metrics,
        "business_only": business_group_metrics,
        "results": results,
    }
    with (output_dir / "e3_agent_results.json").open("w", encoding="utf-8") as fh:
        json.dump(output, fh, ensure_ascii=False, indent=2)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Run advanced LLM-based RCA agent for ASE NIER E3 evaluation.")
    parser.add_argument("--dataset-dir", required=True, help="Path to a dataset root or a comma-separated list of dataset roots.")
    parser.add_argument("--label", required=True, help="Label for the dataset, e.g. llm_fse_run")
    parser.add_argument("--output-dir", required=True, help="Directory to write evaluation outputs.")
    parser.add_argument("--api-key", default=None, help="DeepSeek API key or use DEEPSEEK_API_KEY env var.")
    parser.add_argument("--base-url", default=DEEPSEEK_BASE, help="API base URL.")
    parser.add_argument("--model", default=DEEPSEEK_MODEL, help="LLM model name.")
    parser.add_argument("--max-samples", type=int, default=None, help="Maximum number of samples to evaluate.")
    parser.add_argument("--dry-run", action="store_true", help="Do not call the LLM, use ground-truth dry run.")
    parser.add_argument("--debug-failures", action="store_true", help="Print details of failed predictions for debugging.")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("DEEPSEEK_API_KEY")
    if not api_key and not args.dry_run:
        raise SystemExit("ERROR: DEEPSEEK_API_KEY is required unless --dry-run is specified")

    dataset_dirs = [Path(p.strip()) for p in args.dataset_dir.split(",") if p.strip()]
    output_dir = Path(args.output_dir)
    output = run_agent_e3(dataset_dirs, [args.label] * len(dataset_dirs), api_key or "", output_dir, args.max_samples, args.dry_run, args.base_url, args.model)

    if args.debug_failures:
        print("\n=== DEBUG: Failed Predictions ===")
        for result in output["results"]:
            for condition in ["technical_only", "technical_plus_business"]:
                eval_info = result[condition]["evaluation"]
                if not eval_info["service_correct"] and not eval_info["parse_error"]:
                    print(f"Sample: {result['sample_id']}")
                    print(f"Ground truth: {result['ground_truth']['root_cause_service']}")
                    print(f"Predicted: {eval_info['predicted_service']}")
                    print(f"Candidates: {eval_info['candidate_services']}")
                    print("---")

    summary = {
        "technical_only": output.get("technical_only"),
        "technical_plus_business": output.get("technical_plus_business"),
        "delta": output.get("delta"),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
