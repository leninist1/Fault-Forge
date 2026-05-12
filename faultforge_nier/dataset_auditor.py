"""Dataset audit utilities for the ASE NIER FaultForge workspace."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .telemetry_contract import (
    BUSINESS_SOURCES,
    FORBIDDEN_VISIBLE_TOKENS,
    LATENCY_BUSINESS_SUFFIXES,
    VISIBLE_CSV_FILES,
)


@dataclass
class AuditReport:
    status: str
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)
    distributions: dict[str, dict[str, int]] = field(default_factory=dict)
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _record_files(run_root: Path, tier: str) -> list[Path]:
    direct = sorted((run_root / "tiered_records" / tier / "per_fault").glob("*.json"))
    if direct:
        return direct
    return sorted((run_root / "iterations").glob(f"*/tiered_records/{tier}/per_fault/*.json"))


def _unique_record_files(files: list[Path]) -> list[Path]:
    by_label: dict[str, Path] = {}
    for path in sorted(files):
        by_label[path.stem] = path
    return [by_label[label] for label in sorted(by_label)]


def _has_prism_score(record: dict[str, Any]) -> bool:
    prism = record.get("prism_verdict") or record.get("prism_dynamic") or {}
    if not (prism.get("decision") or prism.get("verdict")):
        return False
    try:
        float(prism.get("aggregate_score", prism.get("score")))
    except (TypeError, ValueError):
        return False
    return True


def _rcaeval_metadata_files(path: Path) -> list[Path]:
    if not path.exists():
        return []
    audit_files = sorted((path / "audit").glob("*/metadata.json"))
    if audit_files:
        return audit_files
    return sorted(path.glob("*/metadata.json"))


def _rca_input_dir(rcaeval_dir: Path, metadata_path: Path) -> Path:
    if metadata_path.parent.parent.name == "audit":
        return rcaeval_dir / "rca_inputs" / metadata_path.parent.name
    return metadata_path.parent


def _csv_data_rows(path: Path) -> int:
    try:
        with path.open(encoding="utf-8", errors="ignore") as fh:
            return max(0, sum(1 for _ in fh) - 1)
    except OSError:
        return 0


def _csv_contains_forbidden_tokens(path: Path) -> list[str]:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return []
    lowered = text.lower()
    for src in BUSINESS_SOURCES:
        lowered = lowered.replace(src, "")
    return [token for token in FORBIDDEN_VISIBLE_TOKENS if token.lower() in lowered]


def _business_csv_summary(path: Path) -> dict[str, Any]:
    by_metric: dict[str, dict[str, float]] = {}
    try:
        with path.open(newline="", encoding="utf-8", errors="ignore") as fh:
            for row in csv.DictReader(fh):
                metric = str(row.get("metric") or "")
                window = str(row.get("window") or "")
                if not metric or window not in {"baseline", "fault"}:
                    continue
                try:
                    value = float(row.get("value") or 0.0)
                except (TypeError, ValueError):
                    continue
                by_metric.setdefault(metric, {})[window] = value
    except OSError:
        pass

    primary: list[str] = []
    latency: list[str] = []
    for metric, windows in by_metric.items():
        if "baseline" not in windows or "fault" not in windows:
            continue
        delta = windows["fault"] - windows["baseline"]
        if metric.endswith("_success_rate") and -delta >= 0.05:
            primary.append("success_rate_drop")
        elif metric.endswith("_business_invalid_rate") and delta >= 0.05:
            primary.append("invalid_rate_increase")
        elif metric.endswith("_timeout_rate") and delta >= 0.03:
            primary.append("timeout_rate_increase")
        elif metric.endswith(("_http_5xx_rate", "_request_exception_rate")) and delta >= 0.03:
            primary.append("error_rate_increase")
        elif metric.endswith("_count") and delta >= 1.0:
            primary.append("semantic_count_increase")
        elif metric.endswith("_distribution_jsd") and windows["fault"] >= 0.10:
            primary.append("distribution_shift")
        elif metric.endswith(LATENCY_BUSINESS_SUFFIXES) and delta > 0:
            latency.append(metric)

    return {
        "primary_types": sorted(set(primary)),
        "latency_anomalies": sorted(latency),
        "latency_only": bool(latency and not primary),
    }


def _target_service(record: dict[str, Any]) -> str:
    spec = record.get("fault_spec", {})
    return (
        spec.get("fault_point", {}).get("owner_service")
        or spec.get("injector_params", {}).get("target_service")
        or record.get("target_service")
        or "unknown"
    )


def _signature(record: dict[str, Any]) -> str:
    spec = record.get("fault_spec", {})
    payload = {
        "dimension": spec.get("dimension"),
        "injector": spec.get("injector"),
        "fault_point": spec.get("fault_point", {}),
        "injector_params": spec.get("injector_params", {}),
        "target_invariant": spec.get("target_invariant") or spec.get("expected_invariant_violations", []),
    }
    return json.dumps(payload, sort_keys=True, ensure_ascii=False)


class DatasetAuditor:
    def __init__(self, run_root: Path, rcaeval_dir: Path | None = None, report_dir: Path | None = None):
        self.run_root = Path(run_root)
        self.rcaeval_dir = Path(rcaeval_dir) if rcaeval_dir else self.run_root / "rcaeval_gold"
        self.report_dir = Path(report_dir) if report_dir else self.run_root / "reports"

    def audit(self) -> AuditReport:
        gold_artifact_files = _record_files(self.run_root, "gold")
        gold_files = _unique_record_files(gold_artifact_files)
        candidate_files = _record_files(self.run_root, "candidate")
        rejected_files = _record_files(self.run_root, "rejected")
        all_files = _unique_record_files(gold_artifact_files + candidate_files + rejected_files)
        rcaeval_files = _rcaeval_metadata_files(self.rcaeval_dir)
        rcaeval_by_id = {path.parent.name: path for path in rcaeval_files}
        records = [_read_json(path) for path in gold_files]
        all_records_by_id = {path.stem: _read_json(path) for path in all_files}
        exported_metadata = []
        for path in rcaeval_files:
            try:
                exported_metadata.append(_read_json(path))
            except Exception:  # pylint: disable=broad-except
                exported_metadata.append({})
        export_policy = "gold"
        if any(meta.get("dataset_export_policy") in {"scored", "all"} for meta in exported_metadata):
            export_policy = "scored"
        expected_export_files = gold_files
        if export_policy == "scored":
            expected_export_files = [
                path for path in all_files
                if _has_prism_score(all_records_by_id.get(path.stem, {}))
            ]
        errors: list[str] = []
        warnings: list[str] = []

        if len(expected_export_files) != len(rcaeval_files):
            errors.append(
                f"{export_policy}_rcaeval_count_mismatch:raw={len(expected_export_files)},rcaeval={len(rcaeval_files)}"
            )

        service_counts: Counter[str] = Counter()
        dimension_counts: Counter[str] = Counter()
        injector_counts: Counter[str] = Counter()
        signature_counts: Counter[str] = Counter()
        evidence_counts: Counter[str] = Counter()
        visible_leakage_count = 0
        primary_business_count = 0
        latency_only_count = 0
        primary_type_counts: Counter[str] = Counter()

        for path, record in zip(gold_files, records):
            label = path.stem
            spec = record.get("fault_spec")
            decision = record.get("quality_decision")
            if not spec:
                errors.append(f"{label}:missing_fault_spec")
                continue
            if not decision:
                errors.append(f"{label}:missing_quality_decision")
                continue
            if decision.get("tier") != "gold":
                errors.append(f"{label}:quality_decision_not_gold")
            gate_results = decision.get("gate_results", {})
            if not gate_results.get("baseline_health", {}).get("ok", False):
                errors.append(f"{label}:dirty_baseline")
            if not gate_results.get("strong_evidence", {}).get("ok", False):
                errors.append(f"{label}:missing_strong_evidence")
            prism = gate_results.get("prism", {}).get("metrics", {})
            if prism.get("verdict") != "REALISTIC":
                errors.append(f"{label}:non_realistic_prism")
            manifest = record.get("modality_manifest") or record.get("dynamic_evidence", {}).get("modality_manifest", {})
            if any(value == "synthetic" for value in manifest.values()):
                errors.append(f"{label}:synthetic_core_evidence")
            modality_gate = gate_results.get("technical_modalities", {})
            if not modality_gate.get("ok", False):
                errors.append(f"{label}:missing_required_technical_modalities")

            service_counts[_target_service(record)] += 1
            dimension_counts[str(spec.get("dimension", "unknown"))] += 1
            injector_counts[str(spec.get("injector", "unknown"))] += 1
            signature_counts[_signature(record)] += 1
            strong = decision.get("evidence", {}).get("strong_evidence", {})
            for evidence_type in strong.get("evidence_types", []):
                evidence_counts[str(evidence_type)] += 1

        for path in expected_export_files:
            label = path.stem
            metadata_path = rcaeval_by_id.get(label)
            if not metadata_path:
                errors.append(f"{label}:missing_rcaeval_export")
                continue
            sample_dir = _rca_input_dir(self.rcaeval_dir, metadata_path)
            for modality, filename in (
                ("metrics", "metrics.csv"),
                ("logs", "logs.csv"),
                ("traces", "traces.csv"),
                ("business", "business.csv"),
            ):
                rows = _csv_data_rows(sample_dir / filename)
                if rows <= 0:
                    errors.append(f"{label}:empty_rcaeval_{modality}")
            leaked = False
            for filename in VISIBLE_CSV_FILES:
                leaks = _csv_contains_forbidden_tokens(sample_dir / filename)
                if leaks:
                    leaked = True
                    errors.append(f"{label}:visible_csv_leakage:{filename}:{','.join(leaks)}")
            if leaked:
                visible_leakage_count += 1
            business_summary = _business_csv_summary(sample_dir / "business.csv")
            if business_summary["primary_types"]:
                primary_business_count += 1
                primary_type_counts.update(business_summary["primary_types"])
            if business_summary["latency_only"]:
                latency_only_count += 1

        duplicate_total = sum(count - 1 for count in signature_counts.values() if count > 1)
        duplicate_ratio = duplicate_total / len(gold_files) if gold_files else 0.0
        if duplicate_ratio > 0.10:
            errors.append(f"duplicate_signature_ratio_too_high:{duplicate_ratio:.2f}")
        elif duplicate_total:
            warnings.append(f"duplicate_signatures:{duplicate_total}")

        if gold_files:
            top_service_ratio = max(service_counts.values(), default=0) / len(gold_files)
            top_injector_ratio = max(injector_counts.values(), default=0) / len(gold_files)
            invariant_ratio = (
                evidence_counts.get("new_invariant_violation", 0)
                + evidence_counts.get("hidden_new_invariant_violation", 0)
            ) / len(gold_files)
            if top_service_ratio > 0.50:
                warnings.append(f"top_service_ratio_high:{top_service_ratio:.2f}")
            if top_injector_ratio > 0.60:
                warnings.append(f"top_injector_ratio_high:{top_injector_ratio:.2f}")
            if invariant_ratio < 0.20:
                warnings.append(f"invariant_violation_evidence_ratio_low:{invariant_ratio:.2f}")

        counts = {
            "gold_raw": len(gold_files),
            "gold_raw_artifacts": len(gold_artifact_files),
            "candidate_raw": len(candidate_files),
            "rejected_raw": len(rejected_files),
            "rcaeval_gold_metadata": len(rcaeval_files),
            "rcaeval_export_metadata": len(rcaeval_files),
            "rcaeval_export_expected": len(expected_export_files),
            "duplicate_signatures": duplicate_total,
            "visible_csv_leakage_samples": visible_leakage_count,
            "business_primary_anomaly_samples": primary_business_count,
            "business_latency_only_samples": latency_only_count,
        }
        status = "fail" if errors else "pass"
        return AuditReport(
            status=status,
            errors=errors,
            warnings=warnings,
            counts=counts,
            distributions={
                "services": dict(service_counts),
                "dimensions": dict(dimension_counts),
                "injectors": dict(injector_counts),
            },
            evidence={
                "evidence_type_counts": dict(evidence_counts),
                "business_primary_type_counts": dict(primary_type_counts),
                "visible_csv_leakage_samples": visible_leakage_count,
                "business_primary_anomaly_samples": primary_business_count,
                "business_latency_only_samples": latency_only_count,
                "duplicate_signature_ratio": duplicate_ratio,
            },
        )

    def write_reports(self, report: AuditReport) -> None:
        self.report_dir.mkdir(parents=True, exist_ok=True)
        (self.report_dir / "dataset_quality_report.json").write_text(
            json.dumps(report.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
        )
        lines = [
            "# Dataset Quality Report",
            "",
            f"Status: {report.status}",
            "",
            "## Counts",
        ]
        lines.extend(f"- {key}: {value}" for key, value in report.counts.items())
        lines.extend(["", "## Errors"])
        lines.extend(f"- {item}" for item in report.errors) or lines.append("- none")
        lines.extend(["", "## Warnings"])
        lines.extend(f"- {item}" for item in report.warnings) or lines.append("- none")
        (self.report_dir / "dataset_quality_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        self._write_distribution_csv("service_distribution.csv", report.distributions.get("services", {}))
        self._write_distribution_csv("injector_yield_table.csv", report.distributions.get("injectors", {}))
        self._write_funnel_csv(report)
        (self.report_dir / "case_study_candidates.json").write_text("[]\n", encoding="utf-8")

    def _write_distribution_csv(self, name: str, distribution: dict[str, int]) -> None:
        with (self.report_dir / name).open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["name", "count"])
            for key, value in sorted(distribution.items()):
                writer.writerow([key, value])

    def _write_funnel_csv(self, report: AuditReport) -> None:
        with (self.report_dir / "funnel_table.csv").open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["stage", "count"])
            for key in ("gold_raw", "candidate_raw", "rejected_raw", "rcaeval_gold_metadata"):
                writer.writerow([key, report.counts.get(key, 0)])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--rcaeval-dir")
    parser.add_argument("--report-dir")
    args = parser.parse_args()
    auditor = DatasetAuditor(Path(args.run_root), Path(args.rcaeval_dir) if args.rcaeval_dir else None, Path(args.report_dir) if args.report_dir else None)
    report = auditor.audit()
    auditor.write_reports(report)
    print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
    return 0 if report.status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
