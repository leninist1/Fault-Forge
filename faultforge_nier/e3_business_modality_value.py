"""E3 Business-Modality RCA Value — ASE NIER 2026 paper evidence.

Tests whether adding the business CSV modality improves RCA top-1 accuracy
beyond conventional metrics, logs, and traces.

Uses DeepSeek function-calling API: the LLM explores CSV data through tools.

Two conditions per sample:
  - technical_only: metrics.csv + logs.csv + traces.csv tools
  - technical_plus_business: technical tools + business.csv

Leakage rule: fault_spec.json, metadata.json, prism_verdict.json, business JSON
files, explicit root-cause labels, service catalogs, and fault-dimension
catalogs are never exposed to the RCA predictor. Labels are read only for
offline evaluation.
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

DEEPSEEK_BASE = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"

# Evaluation-only normalization vocabularies. These are not included in prompts.
FAULT_DIMENSIONS = [
    "data_consistency", "state_transition", "amount_or_price_drift",
    "partial_workflow_commit", "stale_configuration", "business_logic",
]

TRAIN_TICKET_SERVICES = {
    "ts-admin-basic-info-service", "ts-admin-order-service", "ts-admin-travel-service",
    "ts-admin-user-service", "ts-assurance-service", "ts-auth-service", "ts-avatar-service",
    "ts-basic-service", "ts-cancel-service", "ts-config-service", "ts-consign-price-service",
    "ts-consign-service", "ts-contacts-service", "ts-delivery-service", "ts-execute-service",
    "ts-food-delivery-service", "ts-food-service", "ts-gateway-service", "ts-inside-payment-service",
    "ts-login-service", "ts-news-service", "ts-notification-service", "ts-order-other-service",
    "ts-order-service", "ts-payment-service", "ts-preserve-other-service", "ts-preserve-service",
    "ts-price-service", "ts-rebook-service", "ts-route-plan-service", "ts-route-service",
    "ts-seat-service", "ts-security-service", "ts-station-food-service", "ts-station-service",
    "ts-ticket-office-service", "ts-train-food-service", "ts-train-service", "ts-travel-plan-service",
    "ts-travel-service", "ts-travel2-service", "ts-user-service", "ts-verification-code-service",
    "ts-voucher-service", "ts-wait-order-service",
}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


# ── CSV tool implementations ─────────────────────────────────────────────────

class CsvToolContext:
    """Holds pre-loaded CSV data for one sample and implements tool functions."""

    def __init__(self, sample_dir: Path, include_business: bool):
        self.sample_dir = sample_dir
        self.include_business = include_business
        self._cache: dict[str, list[str]] = {}

    def _load(self, filename: str) -> list[str]:
        if filename not in self._cache:
            path = self.sample_dir / filename
            self._cache[filename] = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
        return self._cache[filename]

    def _csv_files(self) -> list[str]:
        files = ["metrics.csv", "logs.csv", "traces.csv"]
        if self.include_business:
            files.append("business.csv")
        return files

    def read_csv(self, filename: str, offset: int = 0, limit: int = 50) -> str:
        """Read a chunk of a CSV file. offset=0 is the header row."""
        lines = self._load(filename)
        if not lines:
            return f"({filename}: empty)"
        chunk = lines[offset:offset + limit]
        result = "\n".join(chunk)
        omitted = max(0, len(lines) - offset - limit)
        header = f"=== {filename} lines {offset}-{offset+len(chunk)-1} of {len(lines)} ===\n"
        trailer = f"\n... {omitted} more lines" if omitted > 0 else ""
        return header + result + trailer

    def get_error_summary(self) -> str:
        """Count ERROR/WARN/INFO per service from logs.csv."""
        lines = self._load("logs.csv")
        if len(lines) < 2:
            return "logs.csv is empty"
        reader = csv.DictReader(io.StringIO("\n".join(lines)))
        counts: dict[str, dict[str, int]] = {}
        total = 0
        for row in reader:
            total += 1
            svc = row.get("service", "unknown")
            level = row.get("level", "INFO")
            if svc not in counts:
                counts[svc] = {"ERROR": 0, "WARN": 0, "INFO": 0}
            if level in counts[svc]:
                counts[svc][level] += 1
        sorted_items = sorted(counts.items(), key=lambda x: -x[1]["ERROR"])
        rows = [f"{'service':<35} {'ERROR':>7} {'WARN':>7} {'INFO':>7}"]
        rows.append("-" * 56)
        for svc, c in sorted_items:
            rows.append(f"{svc:<35} {c['ERROR']:>7} {c['WARN']:>7} {c['INFO']:>7}")
        rows.append(f"\nTotal lines: {total}, Services with errors: {sum(1 for _,c in sorted_items if c['ERROR']>0)}")
        return "\n".join(rows)

    def get_service_errors(self, service_name: str, limit: int = 10) -> str:
        """Get actual error log lines for a specific service."""
        lines = self._load("logs.csv")
        reader = csv.DictReader(io.StringIO("\n".join(lines)))
        found = []
        for row in reader:
            if row.get("service") == service_name and row.get("level") in ("ERROR", "WARN"):
                msg = row.get("message", "")[:300]
                found.append(f"[{row['level']}] {msg}")
                if len(found) >= limit:
                    break
        if not found:
            return f"No ERROR/WARN log lines found for {service_name}"
        return f"=== First {len(found)} ERROR/WARN lines for {service_name} ===\n" + "\n".join(found)

    def get_metric_summary(self) -> str:
        """Extract per-service metric statistics from metrics.csv."""
        lines = self._load("metrics.csv")
        reader = csv.DictReader(io.StringIO("\n".join(lines)))
        svc_metrics: dict[str, list[float]] = {}
        svc_metric_names: dict[str, set[str]] = {}
        for row in reader:
            time_val = row.get("time", "")
            value_str = row.get("value", "")
            try:
                value = float(value_str)
            except (ValueError, TypeError):
                continue
            if row.get("service") and row.get("metric"):
                svc = row.get("service", "unknown")
                metric_name = row.get("metric", "unknown")
            else:
                # Legacy "timestamp:baseline:service:metric" format.
                parts = time_val.split(":")
                if len(parts) < 4:
                    continue
                svc = parts[2]
                metric_name = parts[3]
            if svc not in svc_metrics:
                svc_metrics[svc] = []
                svc_metric_names[svc] = set()
            svc_metrics[svc].append(value)
            svc_metric_names[svc].add(metric_name)

        rows = [f"{'service':<35} {'metrics':>6} {'mean':>8} {'max':>8} {'min':>8}"]
        rows.append("-" * 70)
        for svc in sorted(svc_metrics.keys()):
            vals = svc_metrics[svc]
            rows.append(f"{svc:<35} {len(svc_metric_names[svc]):>6} {sum(vals)/len(vals):>8.2f} {max(vals):>8.2f} {min(vals):>8.2f}")
        return "\n".join(rows)

    def get_trace_summary(self) -> str:
        """Per-service trace statistics."""
        lines = self._load("traces.csv")
        reader = csv.DictReader(io.StringIO("\n".join(lines)))
        svc_stats: dict[str, dict[str, Any]] = {}
        for row in reader:
            svc = row.get("service", "unknown")
            try:
                dur = float(row.get("duration_ms", 0))
            except (ValueError, TypeError):
                dur = 0.0
            is_error = row.get("error", "False").lower() == "true"
            if svc not in svc_stats:
                svc_stats[svc] = {"count": 0, "total_dur": 0.0, "max_dur": 0.0, "errors": 0}
            svc_stats[svc]["count"] += 1
            svc_stats[svc]["total_dur"] += dur
            svc_stats[svc]["max_dur"] = max(svc_stats[svc]["max_dur"], dur)
            if is_error:
                svc_stats[svc]["errors"] += 1

        rows = [f"{'service':<35} {'spans':>6} {'avg_ms':>8} {'max_ms':>8} {'errs':>5}"]
        rows.append("-" * 70)
        for svc in sorted(svc_stats.keys()):
            s = svc_stats[svc]
            avg = s["total_dur"] / s["count"] if s["count"] else 0
            rows.append(f"{svc:<35} {s['count']:>6} {avg:>8.1f} {s['max_dur']:>8.1f} {s['errors']:>5}")
        return "\n".join(rows)

    def get_business_anomalies(self) -> str:
        """Summarize business.csv and highlight genuinely anomalous business metrics."""
        if not self.include_business:
            return "Business data is not available in this condition."
        lines = self._load("business.csv")
        reader = csv.DictReader(io.StringIO("\n".join(lines)))
        anomalous_rows = []
        normal_rows = []
        for row in reader:
            metric = row.get("metric", "?")
            value_str = row.get("value", "0")
            try:
                value = float(value_str)
            except (ValueError, TypeError):
                value = 0.0

            metric_lower = metric.lower()
            is_anomalous = False
            if any(key in metric_lower for key in ("invalid", "error", "timeout", "exception", "4xx", "5xx")):
                is_anomalous = value > 0.0
            elif "success_rate" in metric_lower:
                is_anomalous = abs(value - 1.0) > 1e-9
            elif metric_lower.endswith("_count"):
                is_anomalous = value > 0.0
            elif metric_lower.endswith("_distribution_jsd"):
                is_anomalous = value >= 0.10

            row_text = f"  {metric}: {value}"
            if is_anomalous:
                anomalous_rows.append(row_text)
            else:
                normal_rows.append(row_text)

        rows = [f"=== business.csv ({len(anomalous_rows)} anomalous metrics) ==="]
        if anomalous_rows:
            rows.append("Anomalous metrics:")
            rows.extend(anomalous_rows[:40])
            if len(anomalous_rows) > 40:
                rows.append(f"... {len(anomalous_rows) - 40} more anomalous metrics")
        else:
            rows.append("No direct business failure signal found in business.csv.")
        rows.append(f"Normal/reference business metrics omitted: {len(normal_rows)}")
        return "\n".join(rows)


# ── Tool definitions (OpenAI-compatible schema) ─────────────────────────────

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_error_summary",
            "description": "Get per-service count of ERROR, WARN, INFO log lines from logs.csv. Start here to identify problematic services.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_service_errors",
            "description": "Get actual error/warning log messages for a specific service. Use after get_error_summary to investigate top suspects.",
            "parameters": {
                "type": "object",
                "properties": {
                    "service_name": {"type": "string", "description": "Exact service name from error summary"},
                    "limit": {"type": "integer", "description": "Max error lines to return (default 10)"},
                },
                "required": ["service_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_metric_summary",
            "description": "Get per-service metric statistics (mean, max, min) from simple_metrics.csv.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_trace_summary",
            "description": "Get per-service trace span counts, avg/max duration, and error counts from traces.csv.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_business_anomalies",
            "description": "Get business.csv metrics showing which business rules have nonzero/anomalous values.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_csv",
            "description": "Read raw CSV rows. Use sparingly to inspect specific data after using summary tools.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {"type": "string", "description": "CSV filename"},
                    "offset": {"type": "integer", "description": "Starting row (0=header)", "default": 0},
                    "limit": {"type": "integer", "description": "Max rows", "default": 30},
                },
                "required": ["filename", "offset"],
            },
        },
    },
]

SYSTEM_PROMPT = """You are a precise Root Cause Analysis (RCA) engineer. You analyze CSV data from fault injection experiments on Train-Ticket.

You are given only CSV-derived tools for this task. Do not assume access to
fault labels, metadata, business JSON files, service catalogs, architecture
files, source code, or a predefined fault-dimension taxonomy.

## Investigation Protocol
1. Start with **get_error_summary** — note ALL services with errors, not just the top one.
2. CRITICAL: Use **get_service_errors** on the top 3-5 suspect services. Many "errors" are normal:
   - "User already exists" is normal idempotent creation — NOT a fault signal.
   - JWT "Token expired" is a cascading effect from service restarts — NOT the root cause.
   - Look for errors that indicate DATA problems (missing fields, wrong values, consistency violations).
3. Use **get_metric_summary** — look for services with unusually high max values or anomalous patterns.
4. Use **get_trace_summary** — look for traces with errors (errs>0) or unusually high duration.
5. If **get_business_anomalies** is available, use business.csv as additional signal for user-visible business failures.
6. Synthesize: cross-reference ALL tool outputs. The root cause service should have evidence from MULTIPLE CSV sources where possible.

## Output Format
When you are ready, respond with ONLY this JSON (no markdown, no explanation):
{"root_cause_service": "<single most likely service name>", "fault_dimension": "<short inferred fault type>", "business_invariant": "<anomalous metric name or 'none'>", "confidence": "<low|medium|high>", "reasoning": "<one sentence>"}

IMPORTANT: Complete your investigation in 3-5 turns. After 5 turns, you MUST produce the JSON."""


def _run_rca_with_tools(
    ctx: CsvToolContext,
    api_key: str,
    base_url: str = DEEPSEEK_BASE,
    model: str = DEEPSEEK_MODEL,
    max_turns: int = 6,
) -> dict[str, Any]:
    """Run multi-turn tool-calling RCA for one sample/condition."""
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "Analyze this fault injection sample. Use tools to investigate. After 3-5 turns, produce the final JSON with root_cause_service, fault_dimension, business_invariant, confidence, and reasoning."},
    ]

    available_files = ctx._csv_files()
    # Deep copy base tools and inject available filenames
    tools = json.loads(json.dumps(TOOLS))
    for tool in tools:
        if tool["function"]["name"] == "read_csv":
            tool["function"]["parameters"]["properties"]["filename"]["enum"] = available_files
            tool["function"]["description"] = f"Read raw CSV rows. Available files: {available_files}"
    for turn in range(max_turns):
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "tools": tools,
            "temperature": 0.1,
            "max_tokens": 800,
        }
        # Force final answer on last turn
        if turn == max_turns - 1:
            payload["tool_choice"] = "none"
            payload["messages"].append({
                "role": "user",
                "content": "This is your final turn. You MUST produce the JSON answer NOW. Do not call any more tools. Output ONLY the JSON object.",
            })

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

        msg = body["choices"][0]["message"]
        finish = body["choices"][0].get("finish_reason", "unknown")

        # Check for final answer
        if msg.get("content") and finish == "stop" and not msg.get("tool_calls"):
            content = msg["content"].strip()
            return _parse_json_response(content)

        # Process tool calls
        if msg.get("tool_calls"):
            messages.append({"role": "assistant", "tool_calls": msg["tool_calls"], "content": msg.get("content") or ""})

            for tc in msg["tool_calls"]:
                fn_name = tc["function"]["name"]
                try:
                    fn_args = json.loads(tc["function"]["arguments"])
                except json.JSONDecodeError:
                    fn_args = {}

                # Execute tool
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

                # Truncate long results to avoid overwhelming the LLM
                max_result_chars = 6000
                if len(result) > max_result_chars:
                    result = result[:max_result_chars] + f"\n... (truncated, original {len(result)} chars)"

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": result,
                })
        else:
            # No tool calls, but may have content
            if msg.get("content"):
                return _parse_json_response(msg["content"].strip())
            return {"error": "no content or tool calls", "raw": None}

    return {"error": "max turns exceeded", "raw": None}


def _parse_json_response(content: str) -> dict[str, Any]:
    """Extract JSON from LLM response text."""
    if not content:
        return {"error": "empty response", "raw": None}
    # Remove markdown fences
    cleaned = content.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        cleaned = "\n".join(lines[1:])
        if cleaned.rstrip().endswith("```"):
            cleaned = cleaned.rstrip()[:-3]
    cleaned = cleaned.strip()
    if cleaned.lower().startswith("json"):
        cleaned = cleaned[4:].strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    # Try to find JSON object with balanced braces
    import re as _re
    for m in _re.finditer(r'\{', cleaned):
        start = m.start()
        depth = 0
        end = start
        for i, ch in enumerate(cleaned[start:], start):
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        candidate = cleaned[start:end]
        if '"root_cause_service"' in candidate:
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue
    return {"error": "json parse failed", "raw": content[:500]}


# ── prompt-mode RCA (single-turn, no tool-calling) ───────────────────────────

PROMPT_SYSTEM = """You are a precise Root Cause Analysis (RCA) engineer analyzing fault injection data from the Train-Ticket microservice benchmark.

You are given pre-computed summaries from four telemetry CSV files:
- metrics.csv: per-service performance metrics
- logs.csv: per-service log severity counts
- traces.csv: per-service distributed-trace span statistics
- business.csv: business-level probe results and data-quality checks

## Investigation Protocol
1. Start with the ERROR LOG SUMMARY — note ALL services with errors.
2. CRITICAL: Many errors are NORMAL in distributed systems:
   - "User already exists" = idempotent creation — NOT a fault signal
   - JWT "Token expired" = cascading from service restarts — NOT root cause
   - Look for errors indicating DATA problems (missing fields, wrong values, consistency violations)
3. Cross-reference with METRICS — services with unusually high max values or anomalous patterns
4. Cross-reference with TRACES — services with error spans or unusually high duration
5. If BUSINESS data is provided, check for user-visible business failures (success-rate drops, timeout/error increases, invalid counts, distribution shifts)
6. SYNTHESIZE: The root cause service must have evidence from MULTIPLE CSV sources

## Output Format
Respond with ONLY this JSON (no markdown, no explanation):
{"root_cause_service": "<exact service name from the data>", "fault_dimension": "<short inferred fault type>", "business_invariant": "<anomalous metric name or 'none'>", "confidence": "<low|medium|high>", "reasoning": "<one sentence>"}"""


def _build_analysis_prompt(ctx: CsvToolContext) -> str:
    """Pre-compute all CSV summaries and build a single comprehensive prompt."""
    parts: list[str] = []

    # 1. Error summary from logs
    parts.append("## ERROR LOG SUMMARY (logs.csv)")
    parts.append(ctx.get_error_summary())
    parts.append("")

    # 2. Top service errors for the most suspicious services
    lines = ctx._load("logs.csv")
    # Find top services by error count
    if len(lines) >= 2:
        reader = csv.DictReader(io.StringIO("\n".join(lines)))
        svc_errors: dict[str, int] = {}
        for row in reader:
            if row.get("level") in ("ERROR", "WARN"):
                svc = row.get("service", "unknown")
                svc_errors[svc] = svc_errors.get(svc, 0) + 1
        top_svcs = sorted(svc_errors, key=lambda k: -svc_errors[k])[:5]
        if top_svcs:
            parts.append("## TOP SUSPECT SERVICE ERROR DETAILS")
            for svc in top_svcs:
                parts.append(ctx.get_service_errors(svc, limit=5))
                parts.append("")
    parts.append("")

    # 3. Metric summary
    parts.append("## METRICS SUMMARY (metrics.csv)")
    parts.append(ctx.get_metric_summary())
    parts.append("")

    # 4. Trace summary
    parts.append("## TRACE SUMMARY (traces.csv)")
    parts.append(ctx.get_trace_summary())
    parts.append("")

    # 5. Business anomalies (if available)
    if ctx.include_business:
        parts.append("## BUSINESS TELEMETRY (business.csv)")
        parts.append(ctx.get_business_anomalies())
        parts.append("")

    # 6. Raw CSV samples (first few rows)
    parts.append("## RAW CSV SAMPLES (first 8 rows of each file)")
    for fname in ctx._csv_files():
        parts.append(ctx.read_csv(fname, offset=0, limit=8))
        parts.append("")

    return "\n".join(parts)


def _run_rca_with_prompt(
    ctx: CsvToolContext,
    api_key: str,
    base_url: str = DEEPSEEK_BASE,
    model: str = DEEPSEEK_MODEL,
) -> dict[str, Any]:
    """Run single-turn prompt-based RCA for one sample/condition."""
    prompt = _build_analysis_prompt(ctx)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": PROMPT_SYSTEM},
        {"role": "user", "content": f"Analyze this fault injection sample. Output ONLY the JSON object.\n\n{prompt}"},
    ]

    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": 0.1,
        "max_tokens": 500,
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
        return {"error": str(exc), "raw": None, "model_served": None}

    choice = body["choices"][0]
    msg = choice.get("message", {})
    content = (msg.get("content") or "").strip()
    model_served = body.get("model", "unknown")
    finish = choice.get("finish_reason", "unknown")

    if not content:
        return {"error": f"empty content, finish={finish}", "raw": None, "model_served": model_served}

    parsed = _parse_json_response(content)
    if "error" not in parsed:
        parsed["_model_served"] = model_served
        parsed["_finish_reason"] = finish
    else:
        parsed["model_served"] = model_served
    return parsed


# ── evaluation ─────────────────────────────────────────────────────────────

def _normalize_service(name: str) -> str:
    """Normalize a predicted service name to match the Train-Ticket set."""
    name = (name or "").strip().lower()
    if name in TRAIN_TICKET_SERVICES:
        return name
    if not name.startswith("ts-"):
        name = f"ts-{name}"
    if name in TRAIN_TICKET_SERVICES:
        return name
    for svc in sorted(TRAIN_TICKET_SERVICES, key=len, reverse=True):
        if name in svc or svc in name:
            return svc
    return name


def _normalize_dimension(dim: str) -> str:
    dim = (dim or "").strip().lower().replace(" ", "_").replace("-", "_")
    for d in FAULT_DIMENSIONS:
        if d in dim or dim in d:
            return d
    return dim


def evaluate_prediction(pred: dict[str, Any], ground_truth: dict[str, str]) -> dict[str, Any]:
    if "error" in pred and pred.get("error"):
        raw = pred.get("raw")
        if raw is None:
            raw = str(pred)
        return {"service_correct": False, "dimension_correct": False, "invariant_correct": False,
                "parse_error": True, "predicted_service": str(raw)[:80],
                "predicted_dimension": "", "predicted_invariant": "", "confidence": "unknown"}

    pred_svc = _normalize_service(pred.get("root_cause_service", ""))
    pred_dim = _normalize_dimension(pred.get("fault_dimension", ""))
    pred_inv = (pred.get("business_invariant") or "").strip()

    gt_svc = ground_truth["root_cause_service"].lower()
    gt_dim = ground_truth["fault_dimension"].lower()
    gt_inv = ground_truth["business_invariant"].strip()

    return {
        "service_correct": pred_svc == gt_svc,
        "dimension_correct": pred_dim == gt_dim,
        "invariant_correct": pred_inv == gt_inv,
        "parse_error": False,
        "predicted_service": pred_svc,
        "predicted_dimension": pred_dim,
        "predicted_invariant": pred_inv,
        "confidence": pred.get("confidence", "unknown"),
    }


# ── main runner ─────────────────────────────────────────────────────────────

def run_e3(
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

    # 1. Collect all sample directories (support both old and new layouts)
    sample_dirs: list[tuple[Path, str]] = []
    for ds_dir, label in zip(dataset_dirs, labels):
        rca_inputs = ds_dir / "rca_inputs"
        if rca_inputs.is_dir():
            # New layout: rca_inputs/<sample_id>/  with labels/ and audit/ siblings
            for sample_dir in sorted(rca_inputs.iterdir()):
                if sample_dir.is_dir():
                    sample_dirs.append((sample_dir, label))
        else:
            # Old layout: <sample_id>/ directly in dataset dir
            for sample_dir in sorted(ds_dir.iterdir()):
                if sample_dir.is_dir():
                    sample_dirs.append((sample_dir, label))

    if max_samples:
        sample_dirs = sample_dirs[:max_samples]

    print(f"Loaded {len(sample_dirs)} samples for E3 evaluation (tool-calling mode)")

    # 2. Evaluate each sample under both conditions
    results: list[dict[str, Any]] = []
    for i, (sample_dir, label) in enumerate(sample_dirs):
        sample_id = sample_dir.name
        print(f"[{i+1}/{len(sample_dirs)}] {sample_id} ...", end=" ", flush=True)

        # Ground truth labels are evaluator-only and never exposed to tools.
        # Support both new layout (labels/ at dataset root, audit/ for fault_spec)
        # and old layout (fault_spec.json in sample dir, labels in grandparent)
        dataset_root = sample_dir.parent.parent if sample_dir.parent.name == "rca_inputs" else sample_dir.parent
        label_file = dataset_root / "labels" / f"{sample_id}.json"
        audit_file = dataset_root / "audit" / sample_id / "fault_spec.json"
        fs_inline = sample_dir / "fault_spec.json"
        if label_file.exists():
            spec = _read_json(label_file)
        elif audit_file.exists():
            spec = _read_json(audit_file)
        elif fs_inline.exists():
            spec = _read_json(fs_inline)
        else:
            spec = {}
        ground_truth = {
            "root_cause_service": (
                spec.get("root_cause_service")
                or (spec.get("fault_point", {}) or {}).get("owner_service")
                or (spec.get("injector_params", {}) or {}).get("target_service")
                or "unknown"
            ),
            "fault_dimension": spec.get("fault_dimension") or spec.get("dimension", "unknown"),
            "business_invariant": spec.get("target_invariant", "unknown"),
        }

        if dry_run:
            dry_pred = {"root_cause_service": ground_truth["root_cause_service"],
                        "fault_dimension": ground_truth["fault_dimension"],
                        "business_invariant": ground_truth["business_invariant"],
                        "confidence": "high", "reasoning": "dry_run"}
            tech_pred = dry_pred
            biz_pred = dry_pred
        else:
            # Technical only
            tech_ctx = CsvToolContext(sample_dir, include_business=False)
            tech_pred = _run_rca_with_tools(tech_ctx, api_key, base_url, model)
            time.sleep(1.0)  # rate limit

            # Technical + business
            biz_ctx = CsvToolContext(sample_dir, include_business=True)
            biz_pred = _run_rca_with_tools(biz_ctx, api_key, base_url, model)
            time.sleep(1.0)

        tech_eval = evaluate_prediction(tech_pred, ground_truth)
        biz_eval = evaluate_prediction(biz_pred, ground_truth)

        results.append({
            "sample_id": sample_id,
            "subset": label,
            "ground_truth": ground_truth,
            "technical_only": {"prediction": tech_pred, "evaluation": tech_eval},
            "technical_plus_business": {"prediction": biz_pred, "evaluation": biz_eval},
        })
        print(f"tech: svc={'Y' if tech_eval['service_correct'] else 'N'} "
              f"dim={'Y' if tech_eval['dimension_correct'] else 'N'} "
              f"inv={'Y' if tech_eval['invariant_correct'] else 'N'} | "
              f"tech+biz: svc={'Y' if biz_eval['service_correct'] else 'N'} "
              f"dim={'Y' if biz_eval['dimension_correct'] else 'N'} "
              f"inv={'Y' if biz_eval['invariant_correct'] else 'N'}")

    # 3. Compute aggregate metrics
    def _compute_metrics(entries: list[dict[str, Any]], key: str) -> dict[str, Any]:
        total = len(entries)
        svc_correct = sum(1 for e in entries if e[key]["evaluation"]["service_correct"])
        dim_correct = sum(1 for e in entries if e[key]["evaluation"]["dimension_correct"])
        inv_correct = sum(1 for e in entries if e[key]["evaluation"]["invariant_correct"])
        parse_errors = sum(1 for e in entries if e[key]["evaluation"].get("parse_error"))
        return {
            "total": total,
            "root_service_top1_accuracy": round(svc_correct / total, 4) if total else 0,
            "fault_dimension_accuracy": round(dim_correct / total, 4) if total else 0,
            "business_invariant_accuracy": round(inv_correct / total, 4) if total else 0,
            "parse_error_rate": round(parse_errors / total, 4) if total else 0,
            "raw_counts": {
                "service_correct": svc_correct,
                "dimension_correct": dim_correct,
                "invariant_correct": inv_correct,
                "parse_errors": parse_errors,
            },
        }

    tech_metrics = _compute_metrics(results, "technical_only")
    biz_metrics = _compute_metrics(results, "technical_plus_business")

    # 4. Per-dimension breakdowns
    def _per_breakdown(results, key, field_accessor, values):
        breakdown = {}
        for v in values:
            subset = [e for e in results if field_accessor(e) == v]
            if subset:
                breakdown[v] = _compute_metrics(subset, key)
        return breakdown

    gt_dims = sorted(set(e["ground_truth"]["fault_dimension"] for e in results))
    tech_by_dim = _per_breakdown(results, "technical_only", lambda e: e["ground_truth"]["fault_dimension"], gt_dims)
    biz_by_dim = _per_breakdown(results, "technical_plus_business", lambda e: e["ground_truth"]["fault_dimension"], gt_dims)

    # 5. Write outputs
    e3_results = {
        "experiment": "E3 Business-Modality RCA Value (CSV-only tool-calling mode)",
        "method": "csv_only_deepseek_tool_calling",
        "input_contract": {
        "technical_only": ["metrics.csv", "logs.csv", "traces.csv"],
        "technical_plus_business": ["metrics.csv", "logs.csv", "traces.csv", "business.csv"],
            "excluded_from_prompt": [
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
        "total_samples": len(results),
        "technical_only": tech_metrics,
        "technical_plus_business": biz_metrics,
        "delta": {
            "service_accuracy_improvement": round(biz_metrics["root_service_top1_accuracy"] - tech_metrics["root_service_top1_accuracy"], 4),
            "dimension_accuracy_improvement": round(biz_metrics["fault_dimension_accuracy"] - tech_metrics["fault_dimension_accuracy"], 4),
            "invariant_accuracy_improvement": round(biz_metrics["business_invariant_accuracy"] - tech_metrics["business_invariant_accuracy"], 4),
        },
        "by_dimension": {
            "technical_only": tech_by_dim,
            "technical_plus_business": biz_by_dim,
        },
        "per_sample": [
            {
                "sample_id": r["sample_id"],
                "subset": r["subset"],
                "ground_truth": r["ground_truth"],
                "technical_only": r["technical_only"]["evaluation"],
                "technical_plus_business": r["technical_plus_business"]["evaluation"],
            }
            for r in results
        ],
    }
    _write_json(output_dir / "e3_results.json", e3_results)

    # CSV table
    csv_path = output_dir / "e3_summary_table.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["condition", "samples", "service_top1_acc", "dimension_acc", "invariant_acc", "parse_errors"])
        for label, m in [("technical_only", tech_metrics), ("technical_plus_business", biz_metrics)]:
            writer.writerow([
                label, m["total"],
                m["root_service_top1_accuracy"], m["fault_dimension_accuracy"],
                m["business_invariant_accuracy"], m["parse_error_rate"],
            ])

    return e3_results


def run_e3_prompt(
    dataset_dirs: list[Path],
    labels: list[str],
    api_key: str,
    output_dir: Path,
    max_samples: int | None = None,
    dry_run: bool = False,
    base_url: str = DEEPSEEK_BASE,
    model: str = DEEPSEEK_MODEL,
    min_score: float | None = None,
) -> dict[str, Any]:
    """Prompt-mode E3: pre-computed summaries in a single prompt, no tool-calling."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Collect samples, optionally filtering by score
    sample_dirs: list[tuple[Path, str]] = []
    for ds_dir, label in zip(dataset_dirs, labels):
        rca_inputs = ds_dir / "rca_inputs"
        labels_root = ds_dir / "labels"
        if rca_inputs.is_dir():
            for sample_dir in sorted(rca_inputs.iterdir()):
                if not sample_dir.is_dir():
                    continue
                sid = sample_dir.name
                label_file = labels_root / f"{sid}.json"
                if min_score is not None and min_score > 0 and label_file.exists():
                    try:
                        lab = _read_json(label_file)
                        score = lab.get("prism_score") or lab.get("score", 0)
                        if score < min_score:
                            continue
                    except Exception:
                        pass
                sample_dirs.append((sample_dir, label))
        else:
            for sample_dir in sorted(ds_dir.iterdir()):
                if sample_dir.is_dir():
                    sample_dirs.append((sample_dir, label))

    if max_samples:
        sample_dirs = sample_dirs[:max_samples]

    mode_desc = "prompt (single-turn, no tools)" if not dry_run else "dry-run"
    filter_desc = f" score>={min_score}" if min_score else ""
    print(f"Loaded {len(sample_dirs)} samples for E3 evaluation ({mode_desc}){filter_desc}")

    # 2. Evaluate each sample under both conditions
    results: list[dict[str, Any]] = []
    for i, (sample_dir, label) in enumerate(sample_dirs):
        sample_id = sample_dir.name
        print(f"[{i+1}/{len(sample_dirs)}] {sample_id} ...", end=" ", flush=True)

        # Ground truth
        dataset_root = sample_dir.parent.parent if sample_dir.parent.name == "rca_inputs" else sample_dir.parent
        label_file = dataset_root / "labels" / f"{sample_id}.json"
        audit_file = dataset_root / "audit" / sample_id / "fault_spec.json"
        fs_inline = sample_dir / "fault_spec.json"
        if label_file.exists():
            spec = _read_json(label_file)
        elif audit_file.exists():
            spec = _read_json(audit_file)
        elif fs_inline.exists():
            spec = _read_json(fs_inline)
        else:
            spec = {}
        ground_truth = {
            "root_cause_service": (
                spec.get("root_cause_service")
                or (spec.get("fault_point", {}) or {}).get("owner_service")
                or (spec.get("injector_params", {}) or {}).get("target_service")
                or "unknown"
            ),
            "fault_dimension": spec.get("fault_dimension") or spec.get("dimension", "unknown"),
            "business_invariant": spec.get("target_invariant", "unknown"),
        }

        if dry_run:
            dry_pred = {"root_cause_service": ground_truth["root_cause_service"],
                        "fault_dimension": ground_truth["fault_dimension"],
                        "business_invariant": ground_truth["business_invariant"],
                        "confidence": "high", "reasoning": "dry_run"}
            tech_pred = dict(dry_pred)
            biz_pred = dict(dry_pred)
            tech_model = "dry_run"
            biz_model = "dry_run"
        else:
            tech_ctx = CsvToolContext(sample_dir, include_business=False)
            tech_pred = _run_rca_with_prompt(tech_ctx, api_key, base_url, model)
            tech_model = tech_pred.pop("_model_served", "unknown")
            time.sleep(0.5)

            biz_ctx = CsvToolContext(sample_dir, include_business=True)
            biz_pred = _run_rca_with_prompt(biz_ctx, api_key, base_url, model)
            biz_model = biz_pred.pop("_model_served", "unknown")
            time.sleep(0.5)

        tech_eval = evaluate_prediction(tech_pred, ground_truth)
        biz_eval = evaluate_prediction(biz_pred, ground_truth)

        te = tech_eval
        be = biz_eval
        results.append({
            "sample_id": sample_id,
            "subset": label,
            "ground_truth": ground_truth,
            "technical_only": {"prediction": tech_pred, "evaluation": tech_eval, "model_served": tech_model},
            "technical_plus_business": {"prediction": biz_pred, "evaluation": biz_eval, "model_served": biz_model},
        })
        print(f"tech: svc={'Y' if te['service_correct'] else 'N'} "
              f"dim={'Y' if te['dimension_correct'] else 'N'} "
              f"inv={'Y' if te['invariant_correct'] else 'N'} "
              f"parse={'ERR' if te.get('parse_error') else 'OK'} | "
              f"+biz: svc={'Y' if be['service_correct'] else 'N'} "
              f"dim={'Y' if be['dimension_correct'] else 'N'} "
              f"inv={'Y' if be['invariant_correct'] else 'N'} "
              f"parse={'ERR' if be.get('parse_error') else 'OK'} "
              f"[{tech_model}]")

    # 3. Compute aggregate metrics
    def _compute_metrics(entries, key):
        total = len(entries)
        svc_correct = sum(1 for e in entries if e[key]["evaluation"]["service_correct"])
        dim_correct = sum(1 for e in entries if e[key]["evaluation"]["dimension_correct"])
        inv_correct = sum(1 for e in entries if e[key]["evaluation"]["invariant_correct"])
        parse_errors = sum(1 for e in entries if e[key]["evaluation"].get("parse_error"))
        return {
            "total": total,
            "root_service_top1_accuracy": round(svc_correct / total, 4) if total else 0,
            "fault_dimension_accuracy": round(dim_correct / total, 4) if total else 0,
            "business_invariant_accuracy": round(inv_correct / total, 4) if total else 0,
            "parse_error_rate": round(parse_errors / total, 4) if total else 0,
            "raw_counts": {"service_correct": svc_correct, "dimension_correct": dim_correct,
                           "invariant_correct": inv_correct, "parse_errors": parse_errors},
        }

    tech_metrics = _compute_metrics(results, "technical_only")
    biz_metrics = _compute_metrics(results, "technical_plus_business")

    gt_dims = sorted(set(e["ground_truth"]["fault_dimension"] for e in results))
    def _per_breakdown(results, key, field_accessor, values):
        breakdown = {}
        for v in values:
            subset = [e for e in results if field_accessor(e) == v]
            if subset:
                breakdown[v] = _compute_metrics(subset, key)
        return breakdown
    tech_by_dim = _per_breakdown(results, "technical_only", lambda e: e["ground_truth"]["fault_dimension"], gt_dims)
    biz_by_dim = _per_breakdown(results, "technical_plus_business", lambda e: e["ground_truth"]["fault_dimension"], gt_dims)

    # Per-score breakdown
    def _score_bucket(entry):
        label_file = entry.get("_label_file")
        # use the prism score from label if available
        return "unknown"
    score_buckets = sorted(set(
        str(e.get("ground_truth", {}).get("_score_bucket", "unknown")) for e in results
    ))

    e3_results = {
        "experiment": "E3 Business-Modality RCA Value (prompt mode — single-turn, no tool-calling)",
        "method": "csv_summary_prompt_deepseek",
        "input_contract": {
            "technical_only": ["metrics.csv", "logs.csv", "traces.csv"],
            "technical_plus_business": ["metrics.csv", "logs.csv", "traces.csv", "business.csv"],
            "excluded_from_prompt": [
                "fault_spec.json", "metadata.json", "prism_verdict.json",
                "business_journey.json", "business_invariants.json",
                "business_state_snapshot.json", "service catalog", "fault-dimension catalog",
            ],
        },
        "total_samples": len(results),
        "technical_only": tech_metrics,
        "technical_plus_business": biz_metrics,
        "delta": {
            "service_accuracy_improvement": round(biz_metrics["root_service_top1_accuracy"] - tech_metrics["root_service_top1_accuracy"], 4),
            "dimension_accuracy_improvement": round(biz_metrics["fault_dimension_accuracy"] - tech_metrics["fault_dimension_accuracy"], 4),
            "invariant_accuracy_improvement": round(biz_metrics["business_invariant_accuracy"] - tech_metrics["business_invariant_accuracy"], 4),
        },
        "by_dimension": {"technical_only": tech_by_dim, "technical_plus_business": biz_by_dim},
        "per_sample": [
            {"sample_id": r["sample_id"], "subset": r["subset"],
             "ground_truth": r["ground_truth"],
             "technical_only": r["technical_only"]["evaluation"],
             "technical_plus_business": r["technical_plus_business"]["evaluation"],
             "model_served_tech": r["technical_only"].get("model_served", "unknown"),
             "model_served_biz": r["technical_plus_business"].get("model_served", "unknown"),
             }
            for r in results
        ],
    }
    _write_json(output_dir / "e3_results.json", e3_results)

    csv_path = output_dir / "e3_summary_table.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["condition", "samples", "service_top1_acc", "dimension_acc", "invariant_acc", "parse_errors"])
        for label, m in [("technical_only", tech_metrics), ("technical_plus_business", biz_metrics)]:
            writer.writerow([label, m["total"], m["root_service_top1_accuracy"],
                             m["fault_dimension_accuracy"], m["business_invariant_accuracy"],
                             m["parse_error_rate"]])

    return e3_results


def main() -> int:
    parser = argparse.ArgumentParser(description="E3 Business-Modality RCA Value")
    parser.add_argument("--dataset-dir", action="append", default=[], help="RCAEval dataset dir(s)")
    parser.add_argument("--label", action="append", default=[], help="Label for each dataset")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--api-key", default=None, help="DeepSeek API key (or DEEPSEEK_API_KEY env)")
    parser.add_argument("--base-url", default=DEEPSEEK_BASE)
    parser.add_argument("--model", default=DEEPSEEK_MODEL)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--min-score", type=float, default=None, help="Only include samples with prism_score >= this value")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--prompt-mode", action="store_true", default=True,
                        help="Use single-turn prompt mode (default). Use --no-prompt-mode for tool-calling.")
    parser.add_argument("--no-prompt-mode", action="store_false", dest="prompt_mode",
                        help="Use multi-turn tool-calling mode instead of prompt mode")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("DEEPSEEK_API_KEY")
    if not api_key and not args.dry_run:
        print("ERROR: No API key. Set DEEPSEEK_API_KEY or pass --api-key.")
        return 1

    if not args.dataset_dir:
        parser.error("At least one --dataset-dir is required")

    labels = args.label if args.label else [f"ds_{i}" for i in range(len(args.dataset_dir))]

    if args.prompt_mode:
        results = run_e3_prompt(
            dataset_dirs=[Path(d) for d in args.dataset_dir],
            labels=labels,
            api_key=api_key or "",
            output_dir=Path(args.output_dir),
            max_samples=args.max_samples,
            dry_run=args.dry_run,
            base_url=args.base_url,
            model=args.model,
            min_score=args.min_score,
        )
    else:
        results = run_e3(
            dataset_dirs=[Path(d) for d in args.dataset_dir],
            labels=labels,
            api_key=api_key or "",
            output_dir=Path(args.output_dir),
            max_samples=args.max_samples,
            dry_run=args.dry_run,
            base_url=args.base_url,
            model=args.model,
        )

    print(f"\n=== E3 Results ===")
    tm = results["technical_only"]
    bm = results["technical_plus_business"]
    d = results["delta"]
    print(f"{'Metric':<35} {'TechOnly':>10} {'Tech+Business':>14} {'Delta':>10}")
    print(f"{'---':<35} {'---':>10} {'---':>14} {'---':>10}")
    for name, t_key, b_key, d_key in [
        ("Root-service top-1 accuracy", "root_service_top1_accuracy", "root_service_top1_accuracy", "service_accuracy_improvement"),
        ("Fault-dimension accuracy", "fault_dimension_accuracy", "fault_dimension_accuracy", "dimension_accuracy_improvement"),
        ("Business-invariant accuracy", "business_invariant_accuracy", "business_invariant_accuracy", "invariant_accuracy_improvement"),
    ]:
        print(f"{name:<35} {tm[t_key]:>10.2%} {bm[b_key]:>14.2%} {d[d_key]:>+10.2%}")

    print(f"\nOutputs: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
