"""Telemetry-only PRISM dynamic admission for FaultForge RCA samples."""

from __future__ import annotations

import csv
import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from .admission_profiles import infer_admission_profile
from .telemetry_contract import (
    ALLOWED_BUSINESS_METRICS,
    BUSINESS_CSV_HEADER,
    BUSINESS_SOURCES,
    BUSINESS_UNITS,
    BUSINESS_WINDOWS,
    FORBIDDEN_VISIBLE_TOKENS,
    LATENCY_BUSINESS_SUFFIXES,
)
from .profiles import load_profile

TELEMETRY_PRISM_WEIGHTS = {
    "A_observability_completeness": 0.20,
    "B_business_telemetry_shift": 0.30,
    "C_technical_telemetry_shift": 0.20,
    "D_cross_modal_alignment": 0.20,
    "E_non_leakage_and_data_quality": 0.10,
}

DEFAULT_THRESHOLDS = {
    "min_business_rows_per_window": 10,
    "min_technical_rows_per_window": 10,
    "min_success_rate_drop": 0.05,
    "min_invalid_rate_increase": 0.05,
    "min_error_rate_increase": 0.03,
    "min_timeout_rate_increase": 0.03,
    "min_request_exception_rate_increase": 0.03,
    "min_http_4xx_rate_increase": 0.03,
    "min_json_decode_error_rate_increase": 0.02,
    "min_semantic_count_increase": 1.0,
    "min_count_relative_increase": 0.50,
    "min_distribution_js_divergence": 0.10,
    "min_latency_p95_increase_ms": 100.0,
    "min_latency_p99_increase_ms": 150.0,
    "max_positive_success_rate_jump": 0.10,
    "max_dirty_baseline_error_rate": 0.05,
    "max_dirty_baseline_invalid_rate": 0.02,
    "cross_modal_max_lag_seconds": 60.0,
    "latency_only_max_verdict": "BORDERLINE",
    "require_primary_business_anomaly_for_gold": True,
    "require_technical_anomaly_for_gold": True,
    "require_cross_modal_alignment_for_gold": True,
    "reject_visible_csv_leakage": True,
}


def load_thresholds(config_path: str | None = None) -> dict[str, object]:
    if config_path is None:
        config_path = str(
            Path(__file__).resolve().parent.parent / "configs" / "telemetry_prism.yml"
        )
    p = Path(config_path)
    if p.exists():
        raw = load_profile(p)
        overrides = raw.get("telemetry_prism", {})
        merged = dict(DEFAULT_THRESHOLDS)
        for key in DEFAULT_THRESHOLDS:
            if key in overrides:
                merged[key] = overrides[key]
        for extra_key in ("latency_only_max_verdict",):
            if extra_key in overrides and extra_key not in DEFAULT_THRESHOLDS:
                merged[extra_key] = overrides[extra_key]
        return merged
    return dict(DEFAULT_THRESHOLDS)


@dataclass
class TelemetryRows:
    metrics: list[dict[str, str]]
    logs: list[dict[str, str]]
    traces: list[dict[str, str]]
    business: list[dict[str, str]]
    files: dict[str, Path]


def dynamic_validate_telemetry(
    record: Mapping[str, Any],
    thresholds: dict[str, object] | None = None,
) -> dict[str, Any]:
    if thresholds is None:
        thresholds = load_thresholds()
    t = thresholds
    telemetry = _load_telemetry(record)
    profile = infer_admission_profile(record)
    a_score, a_rules = _score_observability(telemetry, t)
    b_score, b_summary = _score_business_shift(telemetry.business, t)
    c_score, c_summary = _score_technical_shift(telemetry, t)
    d_score, d_summary = _score_alignment(b_summary, c_summary, t)
    e_score, e_summary = _score_non_leakage(telemetry, t)

    axes = {
        "A_observability_completeness": a_score,
        "B_business_telemetry_shift": b_score,
        "C_technical_telemetry_shift": c_score,
        "D_cross_modal_alignment": d_score,
        "E_non_leakage_and_data_quality": e_score,
    }
    aggregate = sum(
        TELEMETRY_PRISM_WEIGHTS[name] * score for name, score in axes.items()
    )
    has_primary_business = bool(b_summary["primary_anomalies"])
    has_business_anomaly = has_primary_business or bool(b_summary["latency_anomalies"])
    has_technical_anomaly = bool(c_summary["technical_anomalies"])
    has_any_anomaly = has_business_anomaly or has_technical_anomaly
    latency_only = bool(b_summary["latency_anomalies"]) and not has_primary_business
    blocking = []
    if e_score < 1.0:
        blocking.extend(e_summary["violations"])

    require_primary = (
        t.get("require_primary_business_anomaly_for_gold", True)
        and profile.require_primary_business_for_realistic
    )
    require_technical = t.get("require_technical_anomaly_for_gold", True)
    require_alignment = t.get("require_cross_modal_alignment_for_gold", True)
    meets_business_requirement = (
        has_primary_business if require_primary else has_business_anomaly
    )
    if (
        profile.allow_infra_weak_business_realistic
        and has_technical_anomaly
        and (has_business_anomaly or b_score >= profile.min_business_score_for_realistic)
    ):
        meets_business_requirement = True
    latency_blocks_realistic = latency_only and not profile.allow_latency_only_realistic

    if (
        a_score >= 0.80
        and meets_business_requirement
        and b_score >= profile.min_business_score_for_realistic
        and (has_technical_anomaly if require_technical else True)
        and c_score >= 0.60
        and (d_score >= 0.60 if require_alignment else True)
        and e_score == 1.0
        and aggregate >= 0.75
        and not latency_blocks_realistic
    ):
        decision = "REALISTIC"
    elif e_score == 1.0 and aggregate >= 0.55 and has_any_anomaly:
        max_verdict = str(t.get("latency_only_max_verdict", "BORDERLINE")).upper()
        if latency_only and max_verdict not in (
            "BORDERLINE",
            "REALISTIC",
            "UNREALISTIC",
        ):
            decision = "REJECT_DYNAMIC"
        else:
            decision = "BORDERLINE"
    else:
        decision = "REJECT_DYNAMIC"

    return {
        "prism_version": "telemetry_first_v1",
        "stage": "dynamic",
        "axis_scores": {name: round(score, 3) for name, score in axes.items()},
        "rule_results": {
            **a_rules,
            "admission_profile": profile.name,
            "B_primary_business_anomaly": has_primary_business,
            "B_latency_only_business_anomaly": latency_only,
            "B_profile_business_requirement": meets_business_requirement,
            "C_technical_anomaly": has_technical_anomaly,
            "D_cross_modal_alignment": d_score >= 0.60,
            "E_visible_csv_non_leaking": e_score == 1.0,
        },
        "aggregate_score": round(aggregate, 3),
        "decision": decision,
        "blocking_errors": blocking,
        "warnings": e_summary["warnings"],
        "evidence_refs": {
            "visible_csv_files": sorted(str(p) for p in telemetry.files.values())
        },
        "runtime_evidence_summary": {
            "primary_business_anomalies": b_summary["primary_anomalies"],
            "latency_anomalies": b_summary["latency_anomalies"],
            "technical_anomalies": c_summary["technical_anomalies"],
            "alignment_lag_seconds": d_summary["lag_seconds"],
        },
        "llm_rationale": {"text": "", "authoritative_for_score": False},
    }


def _load_telemetry(record: Mapping[str, Any]) -> TelemetryRows:
    files = _resolve_files(record)
    if files:
        return TelemetryRows(
            metrics=_read_csv(
                files.get("metrics.csv") or files.get("simple_metrics.csv")
            ),
            logs=_read_csv(files.get("logs.csv")),
            traces=_read_csv(files.get("traces.csv")),
            business=_read_csv(files.get("business.csv")),
            files={name: path for name, path in files.items() if path},
        )
    return _telemetry_from_record(record)


def _resolve_files(record: Mapping[str, Any]) -> dict[str, Path]:
    raw = {}
    for key in ("visible_csv_dir", "rca_input_dir", "sample_dir"):
        if record.get(key):
            root = Path(str(record[key]))
            raw.update(
                {
                    name: root / name
                    for name in (
                        "metrics.csv",
                        "simple_metrics.csv",
                        "logs.csv",
                        "traces.csv",
                        "business.csv",
                    )
                    if (root / name).exists()
                }
            )
    for source_key in ("files", "artifact_paths"):
        for name, value in (record.get(source_key) or {}).items():
            if (
                name
                in {
                    "metrics.csv",
                    "simple_metrics.csv",
                    "logs.csv",
                    "traces.csv",
                    "business.csv",
                }
                and value
            ):
                path = Path(str(value))
                if path.exists():
                    raw[name] = path
    return raw


def _telemetry_from_record(record: Mapping[str, Any]) -> TelemetryRows:
    baseline = (
        record.get("baseline_slis")
        or (record.get("bifi_result") or {}).get("baseline_slis")
        or {}
    )
    fault = (
        record.get("fault_slis")
        or (record.get("bifi_result") or {}).get("fault_slis")
        or {}
    )
    bifi = record.get("bifi_result") or {}
    business = []
    now = _iso(record.get("inject_time") or time.time())
    for window, slis in (("baseline", baseline), ("fault", fault)):
        for metric, value in sorted((slis or {}).items()):
            if metric in ALLOWED_BUSINESS_METRICS:
                business.append(
                    {
                        "timestamp": now,
                        "window": window,
                        "metric": str(metric),
                        "value": str(value),
                        "unit": _unit_for_metric(str(metric)),
                        "source": "business_probe",
                    }
                )
    metrics = []
    logs = []
    traces = []
    for window_name in ("baseline_metrics", "fault_metrics"):
        window = record.get(window_name) or bifi.get(window_name) or {}
        phase = "baseline" if window_name.startswith("baseline") else "fault"
        timestamp = _iso(
            window.get("timestamp") or record.get("inject_time") or time.time()
        )
        for service, stats in (window.get("stats") or {}).items():
            for key, value in (stats or {}).items():
                metrics.append(
                    {
                        "timestamp": timestamp,
                        "window": phase,
                        "service": str(service),
                        "metric": str(key),
                        "value": str(value),
                    }
                )
        for service, count in (window.get("error_log_counts") or {}).items():
            metrics.append(
                {
                    "timestamp": timestamp,
                    "window": phase,
                    "service": str(service),
                    "metric": "error_log_count",
                    "value": str(count),
                }
            )
        for service, lines in (window.get("raw_logs") or {}).items():
            for line in lines or []:
                logs.append(
                    {
                        "timestamp": timestamp,
                        "window": phase,
                        "service": str(service),
                        "level": _level(str(line)),
                        "message": str(line),
                    }
                )
        for trace in window.get("traces") or []:
            processes = trace.get("processes") or {}
            for span in trace.get("spans") or []:
                process = processes.get(span.get("processID"), {})
                traces.append(
                    {
                        "timestamp": timestamp,
                        "window": phase,
                        "trace_id": str(
                            trace.get("traceID") or trace.get("trace_id") or ""
                        ),
                        "span_id": str(span.get("spanID") or span.get("span_id") or ""),
                        "service": str(
                            process.get("serviceName") or span.get("serviceName") or ""
                        ),
                        "operation": str(
                            span.get("operationName") or span.get("operation") or ""
                        ),
                        "duration_ms": str(
                            float(span.get("duration", 0) or 0) / 1000.0
                        ),
                        "error": str(_span_error(span)),
                    }
                )
    return TelemetryRows(
        metrics=metrics, logs=logs, traces=traces, business=business, files={}
    )


def _read_csv(path: Path | None) -> list[dict[str, str]]:
    if not path or not path.exists():
        return []
    with path.open(newline="", encoding="utf-8", errors="ignore") as fh:
        return [dict(row) for row in csv.DictReader(fh)]


def _score_observability(
    rows: TelemetryRows, t: dict[str, object]
) -> tuple[float, dict[str, bool]]:
    rules = {
        "A_metrics_present": bool(rows.metrics),
        "A_logs_present": bool(rows.logs),
        "A_traces_present": bool(rows.traces),
        "A_business_present": bool(rows.business),
        "A_business_baseline_and_fault": _has_windows(rows.business),
        "A_technical_fault_present": bool(
            _fault_rows(rows.metrics)
            or _fault_rows(rows.logs)
            or _fault_rows(rows.traces)
        ),
    }
    return sum(1.0 for ok in rules.values() if ok) / len(rules), rules


def _score_business_shift(
    rows: list[dict[str, str]], t: dict[str, object]
) -> tuple[float, dict[str, Any]]:
    grouped: dict[str, dict[str, float]] = {}
    warnings = []
    for row in rows:
        metric = row.get("metric", "")
        if metric not in ALLOWED_BUSINESS_METRICS:
            warnings.append(f"unknown_business_metric:{metric}")
            continue
        grouped.setdefault(metric, {})[row.get("window", "")] = _float(row.get("value"))
    primary = []
    latency = []
    for metric, values in grouped.items():
        base = values.get("baseline")
        fault = values.get("fault")
        if base is None or fault is None:
            continue
        delta = fault - base
        if metric.endswith("_success_rate") and -delta >= float(
            t.get("min_success_rate_drop", 0.05)
        ):
            primary.append(metric)
        elif metric.endswith("_business_invalid_rate") and delta >= float(
            t.get("min_invalid_rate_increase", 0.05)
        ):
            primary.append(metric)
        elif metric.endswith("_timeout_rate") and delta >= float(
            t.get("min_timeout_rate_increase", 0.03)
        ):
            primary.append(metric)
        elif metric.endswith(
            ("_http_5xx_rate", "_request_exception_rate")
        ) and delta >= float(t.get("min_error_rate_increase", 0.03)):
            primary.append(metric)
        elif metric.endswith("_http_4xx_rate") and delta >= float(
            t.get("min_http_4xx_rate_increase", 0.03)
        ):
            primary.append(metric)
        elif metric.endswith("_json_decode_error_rate") and delta >= float(
            t.get("min_json_decode_error_rate_increase", 0.02)
        ):
            primary.append(metric)
        elif metric.endswith("_count") and delta >= float(
            t.get("min_semantic_count_increase", 1.0)
        ):
            primary.append(metric)
        elif metric.endswith("_distribution_jsd") and fault >= float(
            t.get("min_distribution_js_divergence", 0.10)
        ):
            primary.append(metric)
        elif metric.endswith(LATENCY_BUSINESS_SUFFIXES):
            threshold = (
                float(t.get("min_latency_p99_increase_ms", 150.0))
                if metric.endswith("_p99")
                else float(t.get("min_latency_p95_increase_ms", 100.0))
            )
            if delta >= threshold:
                latency.append(metric)
    score = 0.0
    if primary:
        score = min(1.0, 0.70 + 0.10 * min(len(primary), 3))
    elif latency:
        score = 0.50
    return score, {
        "primary_anomalies": primary,
        "latency_anomalies": latency,
        "warnings": warnings,
    }


def _score_technical_shift(
    rows: TelemetryRows, t: dict[str, object]
) -> tuple[float, dict[str, Any]]:
    anomalies = []
    for row in rows.metrics:
        metric = row.get("metric") or row.get("time", "")
        value = _float(row.get("value"))
        if (
            row.get("window") == "fault"
            and ("error" in metric.lower() or "exception" in metric.lower())
            and value > 0
        ):
            anomalies.append(str(metric))
        elif (
            row.get("window") == "fault"
            and any(key in metric.lower() for key in ("cpu", "mem", "latency"))
            and value > 0
        ):
            anomalies.append(str(metric))
        elif "fault" in str(row.get("time", "")).lower() and value > 0:
            anomalies.append(str(metric))
    for row in rows.logs:
        level = str(row.get("level", "")).upper()
        message = str(row.get("message", "")).lower()
        if row.get("window") == "fault" and (
            level in {"ERROR", "WARN"} or "error" in message or "exception" in message
        ):
            anomalies.append("log_error")
    for row in rows.traces:
        if row.get("window") == "fault" and (
            _float(row.get("duration_ms")) > 0
            or str(row.get("error", "")).lower() == "true"
        ):
            anomalies.append("trace_fault")
    score = (
        0.0 if not anomalies else min(1.0, 0.60 + 0.10 * min(len(set(anomalies)), 4))
    )
    return score, {"technical_anomalies": sorted(set(anomalies))}


def _score_alignment(
    business: Mapping[str, Any], technical: Mapping[str, Any], t: dict[str, object]
) -> tuple[float, dict[str, Any]]:
    if business["primary_anomalies"] and technical["technical_anomalies"]:
        return 0.80, {"lag_seconds": 0.0}
    if (business["primary_anomalies"] or business["latency_anomalies"]) and technical[
        "technical_anomalies"
    ]:
        return 0.60, {"lag_seconds": 0.0}
    return 0.0, {"lag_seconds": None}


def _score_non_leakage(
    rows: TelemetryRows, t: dict[str, object]
) -> tuple[float, dict[str, Any]]:
    violations = []
    warnings = []
    for name, path in rows.files.items():
        text = path.read_text(encoding="utf-8", errors="ignore")
        lower = text.lower()
        for token in FORBIDDEN_VISIBLE_TOKENS:
            if token.lower() in lower:
                violations.append(f"visible_leakage:{name}:{token}")
    for row in rows.business:
        if set(row) and list(row.keys()) != BUSINESS_CSV_HEADER:
            warnings.append("business_csv_schema_not_canonical")
        if row.get("window") and row["window"] not in BUSINESS_WINDOWS:
            violations.append(f"business_bad_window:{row['window']}")
        if row.get("unit") and row["unit"] not in BUSINESS_UNITS:
            violations.append(f"business_bad_unit:{row['unit']}")
        if row.get("source") and row["source"] not in BUSINESS_SOURCES:
            violations.append(f"business_bad_source:{row['source']}")
    for metric, base, fault in _business_pairs(rows.business):
        if metric.endswith("_success_rate") and fault - base > float(
            t.get("max_positive_success_rate_jump", 0.10)
        ):
            violations.append(f"positive_success_rate_jump:{metric}")
        if metric.endswith(
            ("_http_5xx_rate", "_request_exception_rate")
        ) and base > float(t.get("max_dirty_baseline_error_rate", 0.05)):
            violations.append(f"dirty_baseline_error_rate:{metric}")
        if metric.endswith("_business_invalid_rate") and base > float(
            t.get("max_dirty_baseline_invalid_rate", 0.02)
        ):
            violations.append(f"dirty_baseline_invalid_rate:{metric}")
    return (0.0 if violations else 1.0), {
        "violations": violations,
        "warnings": warnings,
    }


def _business_pairs(
    rows: Iterable[Mapping[str, str]],
) -> Iterable[tuple[str, float, float]]:
    values: dict[str, dict[str, float]] = {}
    for row in rows:
        values.setdefault(row.get("metric", ""), {})[row.get("window", "")] = _float(
            row.get("value")
        )
    for metric, pair in values.items():
        if "baseline" in pair and "fault" in pair:
            yield metric, pair["baseline"], pair["fault"]


def _has_windows(rows: Iterable[Mapping[str, str]]) -> bool:
    windows = {row.get("window") for row in rows}
    return {"baseline", "fault"}.issubset(windows)


def _fault_rows(rows: Iterable[Mapping[str, str]]) -> list[Mapping[str, str]]:
    return [
        row
        for row in rows
        if row.get("window") == "fault" or "fault" in str(row.get("time", "")).lower()
    ]


def _float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        number = float(value)
        if math.isnan(number):
            return default
        return number
    except (TypeError, ValueError):
        return default


def _unit_for_metric(metric: str) -> str:
    if metric.endswith("_count"):
        return "count"
    if metric.endswith(("_latency_ms_p95", "_latency_ms_p99")):
        return "ms"
    if metric.endswith("_distribution_jsd"):
        return "score"
    return "ratio"


def _iso(value: Any) -> str:
    if isinstance(value, str) and value:
        return value
    try:
        return datetime.fromtimestamp(float(value), tz=timezone.utc).isoformat()
    except (TypeError, ValueError, OSError):
        return datetime.now(timezone.utc).isoformat()


def _level(line: str) -> str:
    low = line.lower()
    if "error" in low or "exception" in low or "fatal" in low:
        return "ERROR"
    if "warn" in low:
        return "WARN"
    return "INFO"


def _span_error(span: Mapping[str, Any]) -> bool:
    for tag in span.get("tags") or []:
        if isinstance(tag, Mapping) and str(tag.get("key", "")).lower() == "error":
            return str(tag.get("value", "")).lower() == "true"
    return False
