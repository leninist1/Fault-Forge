"""Injector family calibration planning for ASE NIER gold production."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .business_fault_catalog import DEFAULT_CATALOG_PATH, generate_curated_candidates, load_fault_families


REQUIRED_FAMILY_FIELDS = {
    "id",
    "journey",
    "entity",
    "invariant",
    "dimension",
    "semantic_violation_type",
    "injector_family",
    "expected_business_slis",
    "expected_invariants",
    "expected_services",
    "cleanup_strategy",
    "variants",
}
REQUIRED_VARIANT_FIELDS = {
    "id",
    "mode",
    "target_service",
    "target_db",
    "target_table",
    "target_field",
    "condition",
    "modify_value",
    "expected_propagation",
}
DEFAULT_RESULTS_ROOT = DEFAULT_CATALOG_PATH.parents[1] / "experiments" / "production_gold" / "run"


def _missing_fields(item: dict[str, Any], required: set[str]) -> list[str]:
    missing = []
    for field in sorted(required):
        value = item.get(field)
        if value is None or value == "" or value == []:
            missing.append(field)
    return missing


def _norm(value: Any) -> str:
    return str(value if value is not None else "").strip().lower()


def _variant_signature(variant: dict[str, Any]) -> tuple[str, str, str, str, str, str]:
    return (
        _norm(variant.get("target_service")),
        _norm(variant.get("target_db")),
        _norm(variant.get("target_table")),
        _norm(variant.get("target_field")),
        _norm(variant.get("condition")),
        _norm(variant.get("modify_value")),
    )


def _record_signature(spec: dict[str, Any]) -> tuple[str, str, str, str, str, str]:
    params = spec.get("injector_params") or {}
    fault_point = spec.get("fault_point") or {}
    return (
        _norm(params.get("target_service") or fault_point.get("owner_service")),
        _norm(params.get("target_db") or fault_point.get("database")),
        _norm(params.get("target_table") or fault_point.get("table")),
        _norm(params.get("target_field") or fault_point.get("field")),
        _norm(params.get("condition") or params.get("where_clause")),
        _norm(params.get("modify_value")),
    )


def _family_signature_index(families: list[dict[str, Any]]) -> dict[tuple[str, str, str, str, str, str], str]:
    index: dict[tuple[str, str, str, str, str, str], str] = {}
    for family in families:
        family_id = str(family.get("id", ""))
        for variant in family.get("variants") or []:
            index[_variant_signature(variant)] = family_id
    return index


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _iter_result_records(results_root: Path | None) -> list[dict[str, Any]]:
    if not results_root or not results_root.exists():
        return []

    records: list[dict[str, Any]] = []
    seen_record_keys: set[str] = set()
    for path in sorted(results_root.glob("iterations/*/evaluation/per_fault_full/*.json")):
        try:
            record = _load_json(path)
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(record, dict):
            # A targeted canary may repeat the same deterministic fault id across
            # separate runtime attempts. Count each per-fault artifact as an
            # observed attempt, and use this key only to avoid double-counting
            # the aggregate fallback for the same iteration.
            seen_record_keys.add(f"{path.parent.parent}:{(record.get('fault_spec') or {}).get('fault_id')}")
            records.append(record)

    # Older or interrupted runs may only have the aggregate evaluation list.
    for path in sorted(results_root.glob("iterations/*/evaluation/injection_and_prism_results.json")):
        try:
            payload = _load_json(path)
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, list):
            continue
        for record in payload:
            if not isinstance(record, dict):
                continue
            fault_id = str((record.get("fault_spec") or {}).get("fault_id") or path)
            record_key = f"{path.parent}:{fault_id}"
            if record_key in seen_record_keys:
                continue
            seen_record_keys.add(record_key)
            records.append(record)
    return records


def _quality_tier(record: dict[str, Any]) -> str:
    decision = record.get("quality_decision") or {}
    tier = str(decision.get("tier") or "").lower()
    if tier:
        return tier
    if (record.get("selection") or {}).get("accepted_for_dataset"):
        return "gold"
    return "rejected"


def _gate_ok(record: dict[str, Any], gate_name: str) -> bool:
    decision = record.get("quality_decision") or {}
    gate = (decision.get("gate_results") or {}).get(gate_name) or {}
    if "ok" in gate:
        return bool(gate.get("ok"))
    return gate_name in set(decision.get("passed_gates") or [])


def _family_observations(
    families: list[dict[str, Any]],
    results_root: Path | None,
) -> dict[str, dict[str, Any]]:
    family_ids = {str(family.get("id")) for family in families}
    signature_index = _family_signature_index(families)
    observations: dict[str, dict[str, Any]] = {
        family_id: {
            "attempts": 0,
            "bifi_successes": 0,
            "gold": 0,
            "candidate": 0,
            "rejected": 0,
            "cleanup_healthy": 0,
            "strong_evidence": 0,
            "rejection_reasons": Counter(),
        }
        for family_id in family_ids
        if family_id
    }

    for record in _iter_result_records(results_root):
        spec = record.get("fault_spec") or {}
        metadata = spec.get("fse_metadata") or {}
        family_id = str(metadata.get("family_id") or "")
        if not family_id:
            family_id = signature_index.get(_record_signature(spec), "")
        if family_id not in observations:
            continue

        row = observations[family_id]
        row["attempts"] += 1
        if bool((record.get("bifi_result") or {}).get("succeeded")):
            row["bifi_successes"] += 1
        tier = _quality_tier(record)
        if tier in {"gold", "candidate", "rejected"}:
            row[tier] += 1
        else:
            row["rejected"] += 1
        if bool(record.get("post_cleanup_healthy")):
            row["cleanup_healthy"] += 1
        if _gate_ok(record, "strong_evidence"):
            row["strong_evidence"] += 1
        for reason in (record.get("quality_decision") or {}).get("reasons") or []:
            row["rejection_reasons"][str(reason)] += 1

    for row in observations.values():
        attempts = int(row["attempts"])
        row["bifi_success_rate"] = round(row["bifi_successes"] / attempts, 4) if attempts else 0.0
        row["gold_rate"] = round(row["gold"] / attempts, 4) if attempts else 0.0
        row["cleanup_healthy_rate"] = round(row["cleanup_healthy"] / attempts, 4) if attempts else 0.0
        row["strong_evidence_rate"] = round(row["strong_evidence"] / attempts, 4) if attempts else 0.0
        row["top_rejection_reasons"] = [
            {"reason": reason, "count": count}
            for reason, count in row["rejection_reasons"].most_common(5)
        ]
        row.pop("rejection_reasons", None)
    return observations


def build_calibration_report(
    catalog_path: Path | None = None,
    results_root: Path | None = DEFAULT_RESULTS_ROOT,
) -> dict[str, Any]:
    catalog = load_fault_families(catalog_path)
    gold_admission = catalog.get("gold_admission") or {}
    families = catalog.get("families") or []
    candidates = generate_curated_candidates(catalog_path=catalog_path)
    observations = _family_observations(families, results_root)
    rows: list[dict[str, Any]] = []

    for family in families:
        family_id = str(family.get("id"))
        family_missing = _missing_fields(family, REQUIRED_FAMILY_FIELDS)
        variants = family.get("variants") or []
        mode_counts = Counter(str(v.get("mode", "unknown")) for v in variants)
        variant_missing: dict[str, list[str]] = {}
        for variant in variants:
            missing = _missing_fields(variant, REQUIRED_VARIANT_FIELDS)
            if missing:
                variant_missing[str(variant.get("id", "<unknown>"))] = missing

        has_single = mode_counts["single_point"] > 0
        has_cascade = mode_counts["cascading"] > 0
        metadata_ok = not family_missing and not variant_missing and has_single and has_cascade
        canary_status = str(family.get("canary_status", "pending_runtime_validation"))
        observed = observations.get(family_id, {})
        attempts = int(observed.get("attempts", 0))
        min_attempts = int(gold_admission.get("canary_attempts", 3) or 3)
        min_gold_rate = float(gold_admission.get("min_gold_rate", 0.5) or 0.5)
        cleanup_ok = (
            not gold_admission.get("require_cleanup_healthy", True)
            or attempts == 0
            or float(observed.get("cleanup_healthy_rate", 0.0)) >= 1.0
        )
        strong_evidence_ok = (
            not gold_admission.get("require_strong_evidence", True)
            or attempts == 0
            or int(observed.get("strong_evidence", 0)) >= int(observed.get("gold", 0))
        )
        passed_canary = (
            attempts >= min_attempts
            and float(observed.get("gold_rate", 0.0)) >= min_gold_rate
            and cleanup_ok
            and strong_evidence_ok
        )
        admitted = metadata_ok and (canary_status == "ready_known_gold_path" or passed_canary)
        if admitted:
            action = "admit_to_production_catalog"
        elif attempts >= min_attempts:
            action = "quarantine_or_redesign"
        elif attempts > 0:
            action = "continue_runtime_canary"
        elif metadata_ok:
            action = "run_runtime_canary"
        else:
            action = "fix_catalog_metadata"

        rows.append(
            {
                "family_id": family_id,
                "journey": family.get("journey"),
                "invariant": family.get("invariant"),
                "injector_family": family.get("injector_family"),
                "canary_status": canary_status,
                "variant_total": len(variants),
                "single_point_variants": mode_counts["single_point"],
                "cascading_variants": mode_counts["cascading"],
                "metadata_ok": metadata_ok,
                "admitted": admitted,
                "next_action": action,
                "missing_family_fields": family_missing,
                "missing_variant_fields": variant_missing,
                "observed_attempts": attempts,
                "bifi_successes": observed.get("bifi_successes", 0),
                "gold_count": observed.get("gold", 0),
                "candidate_count": observed.get("candidate", 0),
                "rejected_count": observed.get("rejected", 0),
                "bifi_success_rate": observed.get("bifi_success_rate", 0.0),
                "gold_rate": observed.get("gold_rate", 0.0),
                "cleanup_healthy_rate": observed.get("cleanup_healthy_rate", 0.0),
                "strong_evidence_rate": observed.get("strong_evidence_rate", 0.0),
                "top_rejection_reasons": observed.get("top_rejection_reasons", []),
            }
        )

    summary = {
        "catalog_path": str(catalog_path or DEFAULT_CATALOG_PATH),
        "results_root": str(results_root) if results_root else None,
        "family_total": len(families),
        "candidate_total": len(candidates),
        "admitted_family_total": sum(1 for row in rows if row["admitted"]),
        "pending_canary_family_total": sum(1 for row in rows if row["next_action"] == "run_runtime_canary"),
        "continuing_canary_family_total": sum(1 for row in rows if row["next_action"] == "continue_runtime_canary"),
        "quarantined_family_total": sum(1 for row in rows if row["next_action"] == "quarantine_or_redesign"),
        "metadata_fix_family_total": sum(1 for row in rows if row["next_action"] == "fix_catalog_metadata"),
        "gold_admission": gold_admission,
        "families": rows,
    }
    return summary


def write_calibration_report(
    output_dir: Path,
    catalog_path: Path | None = None,
    results_root: Path | None = DEFAULT_RESULTS_ROOT,
) -> dict[str, Any]:
    report = build_calibration_report(catalog_path, results_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "injector_calibration_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    with (output_dir / "injector_calibration_table.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "family_id",
                "journey",
                "invariant",
                "injector_family",
                "canary_status",
                "variant_total",
                "single_point_variants",
                "cascading_variants",
                "metadata_ok",
                "admitted",
                "next_action",
                "observed_attempts",
                "bifi_successes",
                "gold_count",
                "gold_rate",
                "cleanup_healthy_rate",
                "strong_evidence_rate",
            ],
        )
        writer.writeheader()
        for row in report["families"]:
            writer.writerow({key: row[key] for key in writer.fieldnames})

    lines = [
        "# Injector Calibration Report",
        "",
        f"- family_total: {report['family_total']}",
        f"- candidate_total: {report['candidate_total']}",
        f"- admitted_family_total: {report['admitted_family_total']}",
        f"- pending_canary_family_total: {report['pending_canary_family_total']}",
        f"- continuing_canary_family_total: {report['continuing_canary_family_total']}",
        f"- quarantined_family_total: {report['quarantined_family_total']}",
        f"- metadata_fix_family_total: {report['metadata_fix_family_total']}",
        "",
        "| family | journey | variants | attempts | gold_rate | cleanup_rate | status | next_action |",
        "|---|---|---:|---:|---:|---:|---|---|",
    ]
    for row in report["families"]:
        lines.append(
            "| {family_id} | {journey} | {variant_total} | {observed_attempts} | {gold_rate:.2f} | {cleanup_healthy_rate:.2f} | {canary_status} | {next_action} |".format(
                **row
            )
        )
    (output_dir / "injector_calibration_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", default=str(DEFAULT_CATALOG_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_CATALOG_PATH.parents[1] / "reports"))
    parser.add_argument("--results-root", default=str(DEFAULT_RESULTS_ROOT))
    args = parser.parse_args()
    report = write_calibration_report(Path(args.output_dir), Path(args.catalog), Path(args.results_root))
    print(json.dumps({k: report[k] for k in report if k != "families"}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
