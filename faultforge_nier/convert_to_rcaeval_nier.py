"""RCAEval-style exporter for ASE NIER tiered records.

The visible sample remains telemetry-only. Export policy decides which tiered
records are materialized:

* ``gold``: gold tier only;
* ``scored``: any tier with a dynamic PRISM decision and aggregate score;
* ``all``: every tiered runtime record that has enough modalities to export.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .telemetry_contract import ALLOWED_BUSINESS_METRICS, BUSINESS_CSV_HEADER


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_empty_csv(path: Path, headers: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerow(headers)


def _csv_data_rows(path: Path) -> int:
    try:
        with path.open(encoding="utf-8", errors="ignore") as fh:
            return max(0, sum(1 for _ in fh) - 1)
    except OSError:
        return 0


def _fault_id(record: dict[str, Any], fallback: str) -> str:
    spec = record.get("fault_spec", {})
    return str(spec.get("fault_id") or record.get("fault_id") or fallback)


def _target_service(record: dict[str, Any]) -> str:
    spec = record.get("fault_spec", {})
    return (
        spec.get("fault_point", {}).get("owner_service")
        or spec.get("injector_params", {}).get("target_service")
        or record.get("target_service")
        or "unknown"
    )


def _build_business_journey(record: dict[str, Any]) -> dict[str, Any]:
    """Build business journey data from fault_spec when explicit journey trace is absent."""
    spec = record.get("fault_spec", {})
    return {
        "journey_name": spec.get("business_journey", "unknown"),
        "entity": spec.get("business_entity", "unknown"),
        "target_invariant": spec.get("target_invariant", "unknown"),
        "fault_point": spec.get("fault_point", {}),
        "expected_propagation": spec.get("expected_propagation", []),
        "expected_business_impact": spec.get("expected_business_impact", []),
        "semantic_violation_type": spec.get("semantic_violation_type", "unknown"),
    }


def _build_business_state_snapshot(record: dict[str, Any]) -> dict[str, Any]:
    """Build business state snapshot from runtime evidence when not explicitly captured."""
    bifi = record.get("bifi_result", {})
    decision = record.get("quality_decision", {})
    return {
        "target_service": _target_service(record),
        "fault_injected": bifi.get("fault_injected", False),
        "baseline_healthy": decision.get("gate_results", {}).get("baseline_health", {}).get("passed", False),
        "business_slis": _business_slis(record),
        "business_sli_deltas": _business_sli_deltas(record),
        "affected_services": record.get("affected_services") or bifi.get("affected_services", []),
        "invariant_violations_observed": record.get("new_invariant_violations") or bifi.get("new_invariant_violations", []),
    }


def _build_invariant_violations(record: dict[str, Any]) -> list[dict[str, Any]]:
    """Build invariant violations list. When no runtime violations are observed,
    include the target invariant from fault_spec as an expected-but-unobserved entry."""
    spec = record.get("fault_spec", {})
    target_inv = spec.get("target_invariant")
    if not target_inv:
        return []
    return [{
        "invariant_id": target_inv,
        "observed": False,
        "status": "expected_not_observed",
        "note": "Fault targeted this invariant but runtime checker did not observe a violation",
    }]


def _business_slis(record: dict[str, Any]) -> dict[str, Any]:
    return record.get("fault_slis") or record.get("bifi_result", {}).get("fault_slis") or {}


def _business_sli_deltas(record: dict[str, Any]) -> dict[str, Any]:
    return record.get("business_sli_deltas") or record.get("sli_deltas") or record.get("bifi_result", {}).get("sli_deltas") or {}


def _prism_axis(record: dict[str, Any], axis: str) -> dict[str, Any]:
    prism = record.get("prism_verdict") or record.get("prism_dynamic") or {}
    return (prism.get("axes") or {}).get(axis, {}) or {}


def _observed_invariant_evidence(record: dict[str, Any]) -> list[dict[str, Any]]:
    """Return runtime-observed invariant evidence without using fault_spec labels."""
    evidence: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add(item: dict[str, Any]) -> None:
        invariant_id = str(item.get("invariant_id") or item.get("id") or "").strip()
        if not invariant_id or invariant_id in seen:
            return
        seen.add(invariant_id)
        evidence.append(item)

    for item in record.get("new_invariant_violations") or []:
        if isinstance(item, dict):
            add(item)
    for item in (record.get("bifi_result", {}) or {}).get("new_invariant_violations", []) or []:
        if isinstance(item, dict):
            add(item)

    for block in _prism_axis(record, "B").get("evidence") or []:
        for raw in block.get("raw") or []:
            if isinstance(raw, dict) and raw.get("violated"):
                add(raw)
        for invariant_id in block.get("observed") or []:
            add({"invariant_id": invariant_id, "violated": True, "source": "prism_axis_B"})
    return evidence


def _business_sli_delta_evidence(record: dict[str, Any]) -> dict[str, Any]:
    """Return business SLI deltas observed at runtime and by PRISM dynamic scoring."""
    deltas = dict(_business_sli_deltas(record))
    for block in _prism_axis(record, "D").get("evidence") or []:
        for key, value in (block.get("all_deltas") or {}).items():
            deltas.setdefault(key, value)
    return deltas


def _impactful_business_slis(record: dict[str, Any]) -> dict[str, Any]:
    impactful: dict[str, Any] = {}
    for block in _prism_axis(record, "D").get("evidence") or []:
        impactful.update(block.get("impactful_slis") or {})
    return impactful


def _write_business_csv(record: dict[str, Any], path: Path) -> int:
    """Write the fixed telemetry-only long-table business modality."""
    rows: list[list[Any]] = []
    timestamp = _iso_timestamp(record.get("inject_time") or record.get("bifi_result", {}).get("fault_start") or time.time())
    baseline = record.get("baseline_slis") or record.get("bifi_result", {}).get("baseline_slis") or {}
    fault = _business_slis(record)
    for window, slis in (("baseline", baseline), ("fault", fault)):
        for metric, value in sorted((slis or {}).items()):
            if metric not in ALLOWED_BUSINESS_METRICS:
                continue
            rows.append([timestamp, window, metric, value, _unit_for_business_metric(metric), _source_for_business_metric(metric)])

    # Add explicit semantic/data-quality metrics only when they are already
    # runtime-observed telemetry using allowlisted metric names.
    for metric, delta in sorted(_business_sli_delta_evidence(record).items()):
        if metric not in ALLOWED_BUSINESS_METRICS or metric in baseline or metric in fault:
            continue
        base = 0.0
        rows.append([timestamp, "baseline", metric, base, _unit_for_business_metric(metric), _source_for_business_metric(metric)])
        rows.append([timestamp, "fault", metric, float(delta), _unit_for_business_metric(metric), _source_for_business_metric(metric)])

    _write_empty_csv(path, BUSINESS_CSV_HEADER)
    with path.open("a", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows(rows)
    return len(rows)


def _iso_timestamp(value: Any) -> str:
    if isinstance(value, str) and value:
        return value
    try:
        return datetime.fromtimestamp(float(value), tz=timezone.utc).isoformat()
    except (TypeError, ValueError, OSError):
        return datetime.now(timezone.utc).isoformat()


def _unit_for_business_metric(metric: str) -> str:
    if metric.endswith("_count"):
        return "count"
    if metric.endswith(("_latency_ms_p95", "_latency_ms_p99")):
        return "ms"
    if metric.endswith("_distribution_jsd"):
        return "score"
    return "ratio"


def _source_for_business_metric(metric: str) -> str:
    if metric.endswith("_distribution_jsd"):
        return "business_entity_distribution"
    if metric.endswith("_count"):
        return "business_data_quality"
    return "business_probe"


def _copy_existing(record: dict[str, Any], sample_dir: Path, name: str) -> str | None:
    source = record.get("files", {}).get(name) or record.get("artifact_paths", {}).get(name)
    if name == "metrics.csv" and not source:
        source = record.get("files", {}).get("simple_metrics.csv") or record.get("artifact_paths", {}).get("simple_metrics.csv")
    target = sample_dir / name
    if source and Path(source).exists():
        shutil.copy2(source, target)
        if _csv_data_rows(target) > 0:
            return "real"
        target.unlink(missing_ok=True)
    return None


def _metric_windows(record: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    bifi = record.get("bifi_result", {})
    return [
        ("baseline", record.get("baseline_metrics") or bifi.get("baseline_metrics") or {}),
        ("fault", record.get("fault_metrics") or bifi.get("fault_metrics") or {}),
    ]


def _write_real_metrics(record: dict[str, Any], path: Path) -> int:
    rows: list[list[Any]] = []
    for phase, window in _metric_windows(record):
        timestamp = window.get("timestamp") or record.get("inject_time") or time.time()
        for service, stats in (window.get("stats") or {}).items():
            if not isinstance(stats, dict):
                continue
            for key, value in sorted(stats.items()):
                rows.append([f"{timestamp}:{phase}:{service}:{key}", value])
        for service, count in (window.get("error_log_counts") or {}).items():
            rows.append([f"{timestamp}:{phase}:{service}:error_log_count", count])
    _write_empty_csv(path, ["time", "value"])
    with path.open("a", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows(rows)
    return len(rows)


def _level_for_log(line: str) -> str:
    low = line.lower()
    if "error" in low or "exception" in low or "fatal" in low:
        return "ERROR"
    if "warn" in low:
        return "WARN"
    return "INFO"


def _write_real_logs(record: dict[str, Any], path: Path) -> int:
    rows: list[list[Any]] = []
    for phase, window in _metric_windows(record):
        timestamp = window.get("timestamp") or record.get("inject_time") or time.time()
        for service, lines in (window.get("raw_logs") or {}).items():
            for line in lines or []:
                text = str(line).strip()
                if text:
                    rows.append([timestamp, service, _level_for_log(text), text])
    _write_empty_csv(path, ["timestamp", "service", "level", "message"])
    with path.open("a", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows(rows)
    return len(rows)


def _write_real_traces(record: dict[str, Any], path: Path) -> int:
    rows: list[list[Any]] = []
    for _phase, window in _metric_windows(record):
        for trace in window.get("traces") or []:
            trace_id = trace.get("traceID") or trace.get("trace_id") or ""
            processes = trace.get("processes") or {}
            for span in trace.get("spans") or []:
                process = processes.get(span.get("processID"), {})
                service = process.get("serviceName") or span.get("serviceName") or ""
                operation = span.get("operationName") or span.get("operation") or ""
                duration_ms = float(span.get("duration", 0) or 0) / 1000.0
                error = any(
                    str(tag.get("key", "")).lower() == "error" and str(tag.get("value", "")).lower() == "true"
                    for tag in span.get("tags") or []
                    if isinstance(tag, dict)
                )
                rows.append([trace_id, span.get("spanID") or span.get("span_id") or "", service, operation, duration_ms, error])
    _write_empty_csv(path, ["trace_id", "span_id", "service", "operation", "duration_ms", "error"])
    with path.open("a", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows(rows)
    return len(rows)


def _materialize_required_modalities(record: dict[str, Any], sample_dir: Path) -> dict[str, str]:
    states: dict[str, str] = {}
    generators = {
        "metrics.csv": _write_real_metrics,
        "logs.csv": _write_real_logs,
        "traces.csv": _write_real_traces,
    }
    names = {"metrics.csv": "metrics", "logs.csv": "logs", "traces.csv": "traces"}
    for filename, generator in generators.items():
        target = sample_dir / filename
        state = _copy_existing(record, sample_dir, filename)
        if state is None:
            rows = generator(record, target)
            state = "real" if rows > 0 else "missing"
        if _csv_data_rows(target) <= 0:
            raise ValueError(f"{_fault_id(record, 'unknown')} has empty required {names[filename]} modality")
        states[names[filename]] = state
    return states


def _has_prism_score(record: dict[str, Any]) -> bool:
    prism = record.get("prism_verdict") or record.get("prism_dynamic") or {}
    if not (prism.get("decision") or prism.get("verdict")):
        return False
    try:
        float(prism.get("aggregate_score", prism.get("score")))
    except (TypeError, ValueError):
        return False
    return True


def convert_record(record_path: Path, output_root: Path, run_id: str = "ase_nier", *, require_gold: bool = True, require_prism_score: bool = False) -> Path:
    record = _read_json(record_path)
    decision = record.get("quality_decision", {})
    if require_gold and decision.get("tier") != "gold":
        raise ValueError(f"{record_path} is not a gold record")
    if require_prism_score and not _has_prism_score(record):
        raise ValueError(f"{record_path} does not contain a dynamic PRISM score")

    fid = _fault_id(record, record_path.stem)
    sample_dir = output_root / "rca_inputs" / fid
    labels_dir = output_root / "labels"
    audit_dir = output_root / "audit" / fid
    sample_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    audit_dir.mkdir(parents=True, exist_ok=True)

    modality_states = _materialize_required_modalities(record, sample_dir)
    _write_business_csv(record, sample_dir / "business.csv")

    spec = record.get("fault_spec", {})
    label = {
        "sample_id": fid,
        "root_cause_service": _target_service(record),
        "fault_dimension": spec.get("dimension", ""),
        "fault_name": spec.get("name", ""),
        "target_invariant": spec.get("target_invariant", ""),
        "business_journey": spec.get("business_journey", ""),
        "business_entity": spec.get("business_entity", ""),
        "injector": spec.get("injector", ""),
        "quality_tier": decision.get("tier", ""),
        "prism_decision": (record.get("prism_verdict") or record.get("prism_dynamic") or {}).get("decision", ""),
        "prism_score": (record.get("prism_verdict") or record.get("prism_dynamic") or {}).get("aggregate_score", ""),
    }
    _write_json(labels_dir / f"{fid}.json", label)

    metadata = {
        "fault_id": fid,
        "source_fault_id": spec.get("fault_id", fid),
        "fault_name": spec.get("name", ""),
        "dimension": spec.get("dimension", ""),
        "target_service": _target_service(record),
        "injector": spec.get("injector", ""),
        "fault_spec": spec,
        "quality_decision": decision,
        "dataset_export_policy": "gold" if require_gold else "scored" if require_prism_score else "all",
        "selection": record.get("selection", {}),
        "dynamic_evidence": record.get("dynamic_evidence", {}),
        "baseline_health": decision.get("gate_results", {}).get("baseline_health", {}),
        "business_slis": _business_slis(record),
        "business_sli_deltas": _business_sli_deltas(record),
        "invariant_violations": _observed_invariant_evidence(record),
        "affected_services": record.get("affected_services") or record.get("bifi_result", {}).get("affected_services", []),
        "propagation_depth": record.get("propagation_depth") or record.get("bifi_result", {}).get("propagation_depth", 0),
        "modality_manifest": {
            "metrics": modality_states["metrics"],
            "logs": modality_states["logs"],
            "traces": modality_states["traces"],
            "business": "real",
        },
        "provenance": {
            "generator": "FaultForge-ASE-NIER",
            "run_id": run_id,
            "converter": "convert_to_rcaeval_nier.py",
            "real_only_mode": True,
            "dataset_tier": decision.get("tier", ""),
        },
    }
    _write_json(audit_dir / "metadata.json", metadata)
    _write_json(audit_dir / "fault_spec.json", record.get("fault_spec", {}))
    _write_json(audit_dir / "prism_verdict.json", record.get("prism_verdict") or record.get("prism_dynamic") or {})
    _write_json(audit_dir / "bifi_result.json", record.get("bifi_result", {}))
    return sample_dir


def _tier_paths(run_root: Path, tiers: tuple[str, ...]) -> list[Path]:
    paths: list[Path] = []
    for tier in tiers:
        paths.extend(sorted((run_root / "tiered_records" / tier / "per_fault").glob("*.json")))
    return paths


def convert_tiered_run(run_root: Path, output_root: Path, run_id: str = "ase_nier", *, export_policy: str = "gold") -> list[Path]:
    export_policy = str(export_policy or "gold").lower()
    if export_policy not in {"gold", "scored", "all"}:
        raise ValueError(f"unsupported export_policy={export_policy}")
    tiers = ("gold",) if export_policy == "gold" else ("gold", "candidate", "rejected")
    output_root.mkdir(parents=True, exist_ok=True)
    converted = []
    for record_path in _tier_paths(run_root, tiers):
        try:
            converted.append(
                convert_record(
                    record_path,
                    output_root,
                    run_id,
                    require_gold=export_policy == "gold",
                    require_prism_score=export_policy == "scored",
                )
            )
        except ValueError:
            fid = record_path.stem
            shutil.rmtree(output_root / "rca_inputs" / fid, ignore_errors=True)
            shutil.rmtree(output_root / "audit" / fid, ignore_errors=True)
            (output_root / "labels" / f"{fid}.json").unlink(missing_ok=True)
            if export_policy == "gold":
                raise
    return converted


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-id", default="ase_nier")
    parser.add_argument("--export-policy", choices=["gold", "scored", "all"], default="gold")
    args = parser.parse_args()
    converted = convert_tiered_run(Path(args.run_root), Path(args.output_dir), args.run_id, export_policy=args.export_policy)
    print(json.dumps({"converted_gold_total": len(converted), "output_dir": args.output_dir}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
