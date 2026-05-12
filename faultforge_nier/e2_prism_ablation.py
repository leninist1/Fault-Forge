"""E2 PRISM ablation study — ASE NIER 2026 paper evidence.

Runs a permissive injection pipeline to produce records across all quality tiers,
then applies four offline admission policies to the same records:
  - full_faultforge: all 8 gates (standard paper thresholds)
  - no_prism: all gates except PRISM
  - static_only_prism: static PRISM + BIFI execution + cleanup
  - no_business_evidence: all gates except business_signal and strong_evidence
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .prism_driver import static_validate
from .quality_gate import TierClassifier, QualityThresholds

ALL_GATES = [
    "baseline_health",
    "bifi_execution",
    "dynamic_observation",
    "technical_modalities",
    "business_signal",
    "strong_evidence",
    "prism",
    "cleanup",
]

POLICY_GATES: dict[str, set[str]] = {
    "full_faultforge": set(ALL_GATES),
    "no_prism": set(ALL_GATES) - {"prism"},
    "static_only_prism": {"bifi_execution", "cleanup"},
    "no_business_evidence": set(ALL_GATES) - {"business_signal", "strong_evidence"},
}

# Standard paper thresholds (strict) used for policy admission.
PAPER_THRESHOLDS = QualityThresholds(
    baseline_min_success_rate=0.90,
    baseline_max_5xx_rate=0.05,
    business_min_probe_samples=10,
    strong_evidence_min_sli_drop=0.05,
    strong_evidence_min_new_invariants=1,
    strong_evidence_min_affected_services=2,
    strong_evidence_min_propagation_depth=2,
    final_score_threshold=0.70,
    final_allowed_verdicts=("REALISTIC",),
    max_positive_sli_delta=0.20,
    reject_dirty_baseline=True,
)

PAPER_CLASSIFIER = TierClassifier(PAPER_THRESHOLDS)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _find_all_tiered_records(run_root: Path) -> list[Path]:
    """Find ALL per-fault JSON records across all tiers (gold, candidate, rejected)."""
    records = []
    for per_fault_dir in run_root.glob("**/tiered_records/*/per_fault"):
        if per_fault_dir.is_dir():
            records.extend(sorted(per_fault_dir.glob("*.json")))
    return records


def _reclassify(record: dict[str, Any]) -> dict[str, Any]:
    """Re-run TierClassifier with standard paper thresholds on a raw per-fault record."""
    classification = PAPER_CLASSIFIER.classify(record)
    return classification.to_dict()


def build_manifest(run_root: Path, subset_label: str, rcaeval_root: Path | None = None) -> list[dict[str, Any]]:
    """Build manifest from ALL tiered records under a run root."""
    manifest = []
    for record_path in _find_all_tiered_records(run_root):
        record = _read_json(record_path)
        spec = record.get("fault_spec", {})
        original_decision = record.get("quality_decision", {})

        # Re-classify with paper-standard thresholds
        reclassified = _reclassify(record)
        gate_results = reclassified.get("gate_results", {})

        fid = spec.get("fault_id", record_path.stem)
        sample_dir = None
        if rcaeval_root:
            candidate_dir = rcaeval_root / fid
            if candidate_dir.exists():
                sample_dir = str(candidate_dir)

        entry = {
            "subset": subset_label,
            "fault_id": fid,
            "record_path": str(record_path),
            "sample_dir": sample_dir,
            "family_id": spec.get("name", spec.get("fault_id", "")),
            "business_journey": spec.get("business_journey", ""),
            "target_service": (
                (spec.get("fault_point", {}) or {}).get("owner_service")
                or (spec.get("injector_params", {}) or {}).get("target_service")
                or ""
            ),
            "dimension": spec.get("dimension", ""),
            "target_invariant": spec.get("target_invariant", ""),
            "injector": spec.get("injector", ""),
            "prism_verdict": (record.get("prism_verdict", {}) or {}).get("verdict", ""),
            "prism_aggregate_score": (record.get("prism_verdict", {}) or {}).get("aggregate_score", 0),
            "original_tier": original_decision.get("tier", ""),
            "reclassified_tier": reclassified.get("tier", ""),
            "gate_results": gate_results,
            "failed_gates": reclassified.get("failed_gates", []),
            "passed_gates": reclassified.get("passed_gates", []),
            "new_invariant_violations": len(
                record.get("new_invariant_violations")
                or (record.get("bifi_result", {}) or {}).get("new_invariant_violations", [])
                or []
            ),
            "affected_services_count": len(
                record.get("affected_services")
                or (record.get("bifi_result", {}) or {}).get("affected_services", [])
                or []
            ),
            "post_cleanup_healthy": record.get("post_cleanup_healthy", True),
            # Keep raw data for re-classification
            "_raw": {
                "bifi_succeeded": record.get("bifi_succeeded", False),
                "baseline_slis": record.get("baseline_slis", {}) or (record.get("bifi_result", {}) or {}).get("baseline_slis", {}),
                "fault_slis": record.get("fault_slis", {}) or (record.get("bifi_result", {}) or {}).get("fault_slis", {}),
                "new_invariant_violations": record.get("new_invariant_violations", []) or (record.get("bifi_result", {}) or {}).get("new_invariant_violations", []),
                "affected_services": record.get("affected_services", []) or (record.get("bifi_result", {}) or {}).get("affected_services", []),
                "propagation_depth": record.get("propagation_depth", 0),
                "business_sli_deltas": record.get("business_sli_deltas", {}),
            },
        }
        manifest.append(entry)

    return manifest


def _gates_pass(entry: dict[str, Any], gate_set: set[str]) -> bool:
    gate_results = entry.get("gate_results", {})
    for name in gate_set:
        gate = gate_results.get(name, {})
        if not gate.get("ok", False):
            return False
    return True


def _static_prism_pass(entry: dict[str, Any]) -> bool:
    record_path = entry.get("record_path", "")
    if not record_path:
        return False
    record = _read_json(Path(record_path))
    spec = record.get("fault_spec", {})
    if not spec:
        return False
    try:
        result = static_validate(spec, model=None)
        return result.get("decision", "") == "EXECUTE"
    except Exception:
        return False


def _business_modality_present(entry: dict[str, Any]) -> bool:
    sample_dir = entry.get("sample_dir")
    if not sample_dir:
        return True  # No RCAEval dir to check → not applicable
    sd = Path(sample_dir)
    for fname in ("business.csv", "business_journey.json", "business_invariants.json", "business_state_snapshot.json"):
        fp = sd / fname
        if not fp.exists() or fp.stat().st_size == 0:
            return False
    return True


def _dirty_baseline(entry: dict[str, Any]) -> bool:
    gate = entry.get("gate_results", {}).get("baseline_health", {})
    return not gate.get("ok", False)


def compute_policy_metrics(
    manifest: list[dict[str, Any]],
    policy_name: str,
    gate_set: set[str],
) -> dict[str, Any]:
    admitted = []
    rejected = []
    for entry in manifest:
        passes = _gates_pass(entry, gate_set)
        if policy_name == "static_only_prism":
            passes = _gates_pass(entry, gate_set) and _static_prism_pass(entry)

        if passes:
            admitted.append(entry)
        else:
            rejected.append(entry)

    families = Counter(e.get("family_id", "") for e in admitted if e.get("family_id"))
    journeys = Counter(e.get("business_journey", "") for e in admitted if e.get("business_journey"))
    services = set()
    for e in admitted:
        svcs = e.get("target_service", "")
        if svcs:
            services.add(svcs)
    dimensions = Counter(e.get("dimension", "") for e in admitted if e.get("dimension"))
    invariants = Counter(e.get("target_invariant", "") for e in admitted if e.get("target_invariant"))

    avg_inv = (sum(e.get("new_invariant_violations", 0) for e in admitted) / len(admitted)) if admitted else 0.0
    avg_svc = (sum(e.get("affected_services_count", 0) for e in admitted) / len(admitted)) if admitted else 0.0
    dirty = sum(1 for e in admitted if _dirty_baseline(e))
    missing_biz = sum(1 for e in admitted if not _business_modality_present(e))
    non_real = sum(1 for e in admitted if e.get("prism_verdict", "").upper() not in ("REALISTIC",))

    return {
        "policy": policy_name,
        "gates_included": sorted(gate_set),
        "total_records": len(manifest),
        "admitted_count": len(admitted),
        "rejected_count": len(rejected),
        "admission_rate": round(len(admitted) / len(manifest), 4) if manifest else 0,
        "accepted_family_coverage": len(families),
        "accepted_family_ids": sorted(families.keys()),
        "unique_journeys": len(journeys),
        "journey_distribution": dict(journeys.most_common()),
        "unique_services": len(services),
        "unique_dimensions": len(dimensions),
        "dimension_distribution": dict(dimensions.most_common()),
        "unique_invariants": len(invariants),
        "avg_invariant_violations_per_sample": round(avg_inv, 3),
        "avg_affected_services_per_sample": round(avg_svc, 2),
        "dirty_baseline_admitted": dirty,
        "samples_missing_business_modality": missing_biz,
        "non_realistic_admitted": non_real,
        "admitted_fault_ids": [e["fault_id"] for e in admitted],
        "rejected_fault_ids": [e["fault_id"] for e in rejected],
        "admitted_tiers": dict(Counter(e.get("reclassified_tier", "") for e in admitted)),
        "rejected_tiers": dict(Counter(e.get("reclassified_tier", "") for e in rejected)),
    }


def _per_subset_breakdown(manifest: list[dict[str, Any]], policies: list[dict[str, Any]]) -> dict[str, Any]:
    subsets: dict[str, Any] = {}
    for label in sorted(set(e.get("subset", "") for e in manifest)):
        subset = [e for e in manifest if e.get("subset") == label]
        if not subset:
            continue
        sp: dict[str, Any] = {}
        for pr in policies:
            admitted_ids = set(pr["admitted_fault_ids"])
            subset_admitted = [e for e in subset if e["fault_id"] in admitted_ids]
            sp[pr["policy"]] = {
                "total": len(subset),
                "admitted": len(subset_admitted),
                "admission_rate": round(len(subset_admitted) / len(subset), 4),
            }
        subsets[label] = {"total": len(subset), "policies": sp}
    return subsets


def run_e2_ablation(
    run_roots: list[tuple[str, Path, Path | None]],
    output_dir: Path,
) -> dict[str, Any]:
    """Run E2 ablation over one or more run directories.

    Args:
        run_roots: list of (label, run_root, optional_rcaeval_root) tuples.
        output_dir: where to write results.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    for label, run_root, rcaeval_root in run_roots:
        entries = build_manifest(run_root, label, rcaeval_root)
        manifest.extend(entries)

    _write_json(output_dir / "e2_manifest.json", manifest)

    policies = []
    for policy_name in ("full_faultforge", "no_prism", "static_only_prism", "no_business_evidence"):
        result = compute_policy_metrics(manifest, policy_name, POLICY_GATES[policy_name])
        policies.append(result)

    subset_breakdown = _per_subset_breakdown(manifest, policies)

    results = {
        "experiment": "E2 PRISM Ablation",
        "description": "Records re-classified with standard paper thresholds; 4 policies applied.",
        "total_records": len(manifest),
        "original_tier_distribution": dict(Counter(e.get("original_tier", "") for e in manifest)),
        "reclassified_tier_distribution": dict(Counter(e.get("reclassified_tier", "") for e in manifest)),
        "subset_breakdown": subset_breakdown,
        "policies": policies,
    }
    _write_json(output_dir / "prism_ablation_results.json", results)

    # CSV table
    csv_path = output_dir / "prism_ablation_table.csv"
    headers = [
        "policy", "admitted", "admission_rate", "families", "journeys",
        "services", "dimensions", "invariants",
        "avg_invariant_violations", "avg_affected_services",
        "dirty_baseline", "missing_business_modality", "non_realistic",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()
        for p in policies:
            writer.writerow({
                "policy": p["policy"],
                "admitted": p["admitted_count"],
                "admission_rate": p["admission_rate"],
                "families": p["accepted_family_coverage"],
                "journeys": p["unique_journeys"],
                "services": p["unique_services"],
                "dimensions": p["unique_dimensions"],
                "invariants": p["unique_invariants"],
                "avg_invariant_violations": p["avg_invariant_violations_per_sample"],
                "avg_affected_services": p["avg_affected_services_per_sample"],
                "dirty_baseline": p["dirty_baseline_admitted"],
                "missing_business_modality": p["samples_missing_business_modality"],
                "non_realistic": p["non_realistic_admitted"],
            })

    # Per-gate pass/fail across the full manifest (using reclassified gate_results)
    gate_disc_path = output_dir / "gate_discrimination.csv"
    with gate_disc_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["policy", "gate", "passed_count", "failed_count", "pass_rate"])
        for policy_name, gate_set in POLICY_GATES.items():
            for gate_name in ALL_GATES:
                passed = sum(
                    1 for e in manifest
                    if e.get("gate_results", {}).get(gate_name, {}).get("ok", False)
                )
                failed = len(manifest) - passed
                writer.writerow([
                    policy_name, gate_name, passed, failed,
                    round(passed / max(1, len(manifest)), 4),
                ])

    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="E2 PRISM Ablation Study")
    parser.add_argument("--run-root", action="append", default=[], help="Run root dir(s); repeat for each")
    parser.add_argument("--rcaeval-root", action="append", default=[], help="Optional RCAEval dir(s); repeat for each")
    parser.add_argument("--label", action="append", default=[], help="Label for each run root")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    if not args.run_root:
        parser.error("At least one --run-root is required")

    labels = args.label if args.label else [f"run_{i}" for i in range(len(args.run_root))]
    rcaevals = args.rcaeval_root if args.rcaeval_root else [None] * len(args.run_root)

    roots = []
    for i, run_root in enumerate(args.run_root):
        label = labels[i] if i < len(labels) else f"run_{i}"
        rcaeval = Path(rcaevals[i]) if i < len(rcaevals) and rcaevals[i] else None
        roots.append((label, Path(run_root), rcaeval))

    results = run_e2_ablation(roots, Path(args.output_dir))
    for policy in results["policies"]:
        print(
            f"{policy['policy']:30s}  admitted={policy['admitted_count']:3d}  "
            f"rate={policy['admission_rate']:.2%}  families={policy['accepted_family_coverage']:2d}  "
            f"journeys={policy['unique_journeys']:2d}  dims={policy['unique_dimensions']}  "
            f"dirty={policy['dirty_baseline_admitted']}  biz_miss={policy['samples_missing_business_modality']}  "
            f"non_real={policy['non_realistic_admitted']}"
        )
    print(f"\nOriginal tiers: {results['original_tier_distribution']}")
    print(f"Reclassified tiers: {results['reclassified_tier_distribution']}")
    print(f"\nOutputs: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
