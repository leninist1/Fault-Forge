"""Report builders for ASE NIER FSE/PRISM smoke and paper tables."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


def _write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _priority_score(candidate: dict[str, Any]) -> float:
    signals = candidate.get("expected_observable_signals", {})
    propagation = candidate.get("expected_propagation", [])
    business_value = 1.0 if candidate.get("target_invariant") else 0.0
    observer_signal_coverage = min(1.0, len(signals.get("business_slis", [])) / 2.0)
    bifi_feasibility = 1.0 if candidate.get("injector") in {"database_modifier", "resource_limit", "host_iptables"} else 0.4
    propagation_value = min(1.0, len(propagation) / 3.0)
    return round(
        0.20 * 1.0
        + 0.15 * 1.0
        + 0.20 * 1.0
        + 0.10 * propagation_value
        + 0.15 * bifi_feasibility
        + 0.15 * observer_signal_coverage
        + 0.05 * business_value,
        3,
    )


def write_fse_reports(catalog: list[dict[str, Any]], report_dir: Path) -> dict[str, Any]:
    report_dir = Path(report_dir)
    journeys = Counter(c.get("business_journey", "unknown") for c in catalog)
    entities = Counter(c.get("business_entity", "unknown") for c in catalog)
    invariants = Counter(c.get("target_invariant", "unknown") for c in catalog)
    services = Counter(
        (c.get("fault_point", {}) or {}).get("owner_service")
        or (c.get("injector_params", {}) or {}).get("target_service")
        or "unknown"
        for c in catalog
    )
    dimensions = Counter(c.get("dimension", "unknown") for c in catalog)
    injectors = Counter(c.get("injector", "unknown") for c in catalog)
    signatures = Counter(_semantic_signature(c) for c in catalog)
    duplicate_total = sum(count - 1 for count in signatures.values() if count > 1)

    matrix_rows = []
    status_by_pair: dict[tuple[str, str], str] = defaultdict(lambda: "not_generated")
    for c in catalog:
        status_by_pair[(c.get("business_journey", "unknown"), c.get("target_invariant", "unknown"))] = "generated"
    for (journey, invariant), status in sorted(status_by_pair.items()):
        matrix_rows.append({"dimension": "journey_x_invariant", "left": journey, "right": invariant, "status": status})
    _write_csv(report_dir / "fault_space_matrix.csv", matrix_rows, ["dimension", "left", "right", "status"])

    ranking_rows = []
    for c in catalog:
        ranking_rows.append(
            {
                "fault_id": c.get("fault_id"),
                "business_journey": c.get("business_journey"),
                "business_entity": c.get("business_entity"),
                "target_invariant": c.get("target_invariant"),
                "target_service": (c.get("fault_point", {}) or {}).get("owner_service", ""),
                "injector": c.get("injector"),
                "priority_score": _priority_score(c),
            }
        )
    ranking_rows.sort(key=lambda row: row["priority_score"], reverse=True)
    _write_csv(
        report_dir / "fse_priority_ranking.csv",
        ranking_rows,
        ["fault_id", "business_journey", "business_entity", "target_invariant", "target_service", "injector", "priority_score"],
    )

    coverage = {
        "catalog_total": len(catalog),
        "journey_coverage": len(journeys),
        "entity_coverage": len(entities),
        "invariant_coverage": len(invariants),
        "service_coverage": len(services),
        "fault_dimension_coverage": len(dimensions),
        "injector_coverage": len(injectors),
        "semantic_duplicate_total": duplicate_total,
        "semantic_duplicate_rate": round(duplicate_total / len(catalog), 3) if catalog else 0.0,
        "journeys": dict(journeys),
        "entities": dict(entities),
        "invariants": dict(invariants),
        "services": dict(services),
        "dimensions": dict(dimensions),
        "injectors": dict(injectors),
    }
    lines = [
        "# FSE Coverage Report",
        "",
        f"- catalog_total: {coverage['catalog_total']}",
        f"- journeys: {coverage['journey_coverage']}",
        f"- entities: {coverage['entity_coverage']}",
        f"- invariants: {coverage['invariant_coverage']}",
        f"- services: {coverage['service_coverage']}",
        f"- fault_dimensions: {coverage['fault_dimension_coverage']}",
        f"- injectors: {coverage['injector_coverage']}",
        f"- semantic_duplicate_rate: {coverage['semantic_duplicate_rate']}",
        "",
        "## Top Undercovered Journeys",
    ]
    if journeys:
        min_count = min(journeys.values())
        lines.extend(f"- {name}: {count}" for name, count in sorted(journeys.items()) if count == min_count)
    else:
        lines.append("- none")
    (report_dir / "fse_coverage_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (report_dir / "fse_coverage_report.json").write_text(json.dumps(coverage, indent=2, ensure_ascii=False), encoding="utf-8")
    return coverage


def write_prism_reports(records: list[dict[str, Any]], report_dir: Path) -> dict[str, Any]:
    report_dir = Path(report_dir)
    rows = []
    axis_rows = []
    decisions = Counter()
    for record in records:
        spec = record.get("fault_spec", {})
        prism = record.get("prism_static") or record.get("prism") or {}
        decisions[prism.get("decision", "UNKNOWN")] += 1
        rows.append(
            {
                "fault_id": spec.get("fault_id", ""),
                "decision": prism.get("decision", "UNKNOWN"),
                "aggregate_score": prism.get("aggregate_score", 0.0),
                "blocking_errors": ";".join(prism.get("blocking_errors", [])),
            }
        )
        axes = prism.get("axis_scores", {})
        axis_rows.append(
            {
                "fault_id": spec.get("fault_id", ""),
                "A": axes.get("A_local_business_grounding", ""),
                "B": axes.get("B_invariant_oracle_strength", ""),
                "C": axes.get("C_propagation_realism", ""),
                "D": axes.get("D_business_user_impact", ""),
                "E": axes.get("E_triggerability_executability", ""),
            }
        )
    _write_csv(report_dir / "prism_static_filtering.csv", rows, ["fault_id", "decision", "aggregate_score", "blocking_errors"])
    _write_csv(report_dir / "prism_axis_distribution.csv", axis_rows, ["fault_id", "A", "B", "C", "D", "E"])
    lines = ["# PRISM Quality Report", "", "## Static Decisions"]
    lines.extend(f"- {key}: {value}" for key, value in sorted(decisions.items()))
    (report_dir / "prism_quality_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"decisions": dict(decisions), "records": len(records)}


def write_business_modality_report(run_root: Path, report_dir: Path) -> dict[str, Any]:
    run_root = Path(run_root)
    report_dir = Path(report_dir)
    records = []
    for path in sorted((run_root / "tiered_records" / "gold" / "per_fault").glob("*.json")):
        records.append(json.loads(path.read_text(encoding="utf-8")))
    counts = Counter()
    for record in records:
        manifest = record.get("modality_manifest") or record.get("dynamic_evidence", {}).get("modality_manifest", {})
        for key in ("business", "business_slis", "business_journey", "business_invariants"):
            state = manifest.get(key)
            if state:
                counts[f"{key}:{state}"] += 1
    payload = {"gold_records": len(records), "modality_counts": dict(counts)}
    lines = ["# Business Modality Report", "", f"- gold_records: {len(records)}"]
    lines.extend(f"- {key}: {value}" for key, value in sorted(counts.items()))
    (report_dir / "business_modality_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def _semantic_signature(candidate: dict[str, Any]) -> str:
    fault_point = candidate.get("fault_point", {}) or {}
    params = candidate.get("injector_params", {}) or {}
    return "|".join(
        [
            str(candidate.get("business_journey", "")),
            str(candidate.get("business_entity", "")),
            str(candidate.get("target_invariant", "")),
            str(candidate.get("semantic_violation_type", "")),
            str(candidate.get("dimension", "")),
            str(fault_point.get("owner_service") or params.get("target_service") or ""),
            str(fault_point.get("table") or params.get("target_table") or ""),
            str(fault_point.get("field") or params.get("target_field") or ""),
            str(candidate.get("injector", "")),
        ]
    )
