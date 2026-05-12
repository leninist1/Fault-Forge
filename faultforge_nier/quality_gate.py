"""Evidence-first tier classification for FaultForge ASE NIER records."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

from .admission_profiles import infer_admission_profile
from .telemetry_contract import LATENCY_BUSINESS_SUFFIXES, PRIMARY_BUSINESS_SUFFIXES

CORE_PROBES = (
    "login",
    "trip_search",
    "contacts_fetch",
    "order_read",
    "booking_precheck",
)

ALL_PROBES = (
    "login",
    "trip_search",
    "contacts_fetch",
    "order_read",
    "booking_precheck",
    "payment_submit",
    "cancel_submit",
    "addon_query",
)


@dataclass
class QualityThresholds:
    baseline_min_success_rate: float = 0.90
    baseline_max_5xx_rate: float = 0.05
    business_min_probe_samples: int = 10
    strong_evidence_min_sli_drop: float = 0.05
    strong_evidence_min_new_invariants: int = 1
    strong_evidence_min_affected_services: int = 2
    strong_evidence_min_propagation_depth: int = 2
    strong_evidence_min_invalid_rate_increase: float = 0.05
    strong_evidence_min_error_rate_increase: float = 0.03
    strong_evidence_min_timeout_rate_increase: float = 0.03
    strong_evidence_min_semantic_count_increase: float = 1.0
    strong_evidence_min_distribution_jsd: float = 0.10
    final_score_threshold: float = 0.70
    final_allowed_verdicts: tuple[str, ...] = ("REALISTIC",)
    max_positive_sli_delta: float = 0.20
    reject_dirty_baseline: bool = True

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None) -> "QualityThresholds":
        if not raw:
            return cls()
        values = {}
        for key in cls.__dataclass_fields__:  # type: ignore[attr-defined]
            if key in raw:
                values[key] = raw[key]
        if "final_allowed_verdicts" in values and isinstance(
            values["final_allowed_verdicts"], str
        ):
            values["final_allowed_verdicts"] = tuple(
                x.strip().upper()
                for x in values["final_allowed_verdicts"].split(",")
                if x.strip()
            )
        return cls(**values)


@dataclass
class GateResult:
    name: str
    ok: bool
    reasons: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class QualityDecision:
    tier: str
    accepted_for_gold: bool
    accepted_for_candidate: bool
    failed_gates: list[str]
    passed_gates: list[str]
    reasons: list[str]
    evidence: dict[str, Any]
    gate_results: dict[str, GateResult]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["gate_results"] = {
            name: result.to_dict() for name, result in self.gate_results.items()
        }
        return payload


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _get(record: Mapping[str, Any], *paths: str, default: Any = None) -> Any:
    for path in paths:
        cur: Any = record
        ok = True
        for part in path.split("."):
            if isinstance(cur, Mapping) and part in cur:
                cur = cur[part]
            else:
                ok = False
                break
        if ok:
            return cur
    return default


def _csv_data_rows(path: Any) -> int:
    try:
        p = Path(str(path))
        if not p.exists():
            return 0
        with p.open(encoding="utf-8", errors="ignore") as fh:
            return max(0, sum(1 for _ in fh) - 1)
    except (OSError, TypeError, ValueError):
        return 0


def _nonempty_sequence_or_mapping(value: Any) -> int:
    if isinstance(value, Mapping):
        return sum(1 for item in value.values() if item)
    if isinstance(value, (list, tuple, set)):
        return len(value)
    return 0


class TierClassifier:
    """Classify a runtime record into gold, candidate, or rejected."""

    def __init__(self, thresholds: QualityThresholds | Mapping[str, Any] | None = None):
        if isinstance(thresholds, QualityThresholds):
            self.thresholds = thresholds
        else:
            self.thresholds = QualityThresholds.from_mapping(thresholds)

    def classify(self, record_context: Mapping[str, Any]) -> QualityDecision:
        gates = {
            "bifi_execution": self._bifi_gate(record_context),
            "baseline_health": self._baseline_gate(record_context),
            "dynamic_observation": self._dynamic_observation_gate(record_context),
            "technical_modalities": self._technical_modalities_gate(record_context),
            "business_signal": self._business_signal_gate(record_context),
            "strong_evidence": self._strong_evidence_gate(record_context),
            "prism": self._prism_gate(record_context),
            "cleanup": self._cleanup_gate(record_context),
        }
        failed = [name for name, gate in gates.items() if not gate.ok]
        passed = [name for name, gate in gates.items() if gate.ok]
        reasons = [reason for gate in gates.values() for reason in gate.reasons]
        evidence = self._evidence_summary(record_context)

        hard_reject = (
            not gates["bifi_execution"].ok
            or not gates["dynamic_observation"].ok
            or not gates["technical_modalities"].ok
            or not gates["cleanup"].ok
            or gates["prism"].metrics.get("rejected", False)
            or (
                self.thresholds.reject_dirty_baseline
                and not gates["baseline_health"].ok
            )
        )
        gold_ok = all(gates[name].ok for name in gates)

        if gold_ok:
            tier = "gold"
        elif hard_reject:
            tier = "rejected"
        else:
            tier = "candidate"

        return QualityDecision(
            tier=tier,
            accepted_for_gold=tier == "gold",
            accepted_for_candidate=tier in {"gold", "candidate"},
            failed_gates=failed,
            passed_gates=passed,
            reasons=reasons,
            evidence=evidence,
            gate_results=gates,
        )

    def _bifi_gate(self, record: Mapping[str, Any]) -> GateResult:
        succeeded = bool(
            _get(
                record,
                "bifi_succeeded",
                "bifi_result.succeeded",
                "bifi_result.success",
                default=False,
            )
        )
        reasons = [] if succeeded else ["bifi_failed"]
        return GateResult(
            "bifi_execution", succeeded, reasons, {"bifi_succeeded": succeeded}
        )

    def _baseline_gate(self, record: Mapping[str, Any]) -> GateResult:
        baseline_slis = (
            _get(record, "baseline_slis", "bifi_result.baseline_slis", default={}) or {}
        )
        service_health = (
            _get(record, "service_health", "baseline_health.service_health", default={})
            or {}
        )
        reasons: list[str] = []
        metrics: dict[str, Any] = {}
        for probe in CORE_PROBES:
            success_key = f"{probe}_success_rate"
            count_key = f"{probe}_sample_count"
            rate = _as_float(baseline_slis.get(success_key), -1.0)
            count = _as_int(baseline_slis.get(count_key), 0)
            rate_5xx = _as_float(baseline_slis.get(f"{probe}_http_5xx_rate"), 0.0)
            metrics[success_key] = rate
            metrics[count_key] = count
            metrics[f"{probe}_http_5xx_rate"] = rate_5xx
            if rate < 0:
                reasons.append(f"missing_baseline_sli:{success_key}")
                continue
            if rate < self.thresholds.baseline_min_success_rate:
                reasons.append(f"low_baseline_success:{probe}={rate:.2f}")
            if rate_5xx > self.thresholds.baseline_max_5xx_rate:
                reasons.append(f"high_5xx:{probe}={rate_5xx:.2f}")
            if count and count < self.thresholds.business_min_probe_samples:
                reasons.append(f"low_baseline_probe_count:{probe}={count}")
        unhealthy = [
            name
            for name, state in service_health.items()
            if isinstance(state, Mapping)
            and str(state.get("status", "")).lower() not in {"", "running", "healthy"}
        ]
        if unhealthy:
            reasons.append("unhealthy_services:" + ",".join(sorted(unhealthy)))
        metrics["unhealthy_service_count"] = len(unhealthy)
        return GateResult("baseline_health", not reasons, reasons, metrics)

    def _dynamic_observation_gate(self, record: Mapping[str, Any]) -> GateResult:
        observation = _get(
            record,
            "dynamic_observation",
            "observer_result",
            "bifi_result.observation",
            default=None,
        )
        fault_slis = (
            _get(record, "fault_slis", "bifi_result.fault_slis", default={}) or {}
        )
        has_observation = bool(observation) or bool(fault_slis)
        reasons = [] if has_observation else ["no_real_dynamic_observation"]
        return GateResult(
            "dynamic_observation",
            has_observation,
            reasons,
            {"has_observation": has_observation},
        )

    def _technical_modalities_gate(self, record: Mapping[str, Any]) -> GateResult:
        files = _get(record, "files", "artifact_paths", default={}) or {}
        baseline_metrics = (
            _get(record, "baseline_metrics", "bifi_result.baseline_metrics", default={})
            or {}
        )
        fault_metrics = (
            _get(record, "fault_metrics", "bifi_result.fault_metrics", default={}) or {}
        )

        metrics_count = _csv_data_rows(files.get("simple_metrics.csv"))
        logs_count = _csv_data_rows(files.get("logs.csv"))
        traces_count = _csv_data_rows(files.get("traces.csv"))

        metrics_count += _nonempty_sequence_or_mapping(baseline_metrics.get("stats"))
        metrics_count += _nonempty_sequence_or_mapping(fault_metrics.get("stats"))
        metrics_count += _nonempty_sequence_or_mapping(
            baseline_metrics.get("error_log_counts")
        )
        metrics_count += _nonempty_sequence_or_mapping(
            fault_metrics.get("error_log_counts")
        )

        logs_count += _nonempty_sequence_or_mapping(baseline_metrics.get("raw_logs"))
        logs_count += _nonempty_sequence_or_mapping(fault_metrics.get("raw_logs"))
        traces_count += _nonempty_sequence_or_mapping(baseline_metrics.get("traces"))
        traces_count += _nonempty_sequence_or_mapping(fault_metrics.get("traces"))

        counts = {
            "metrics_rows": metrics_count,
            "logs_rows": logs_count,
            "traces_rows": traces_count,
        }
        reasons = [
            name
            for name, count in (
                ("empty_metrics_modality", metrics_count),
                ("empty_logs_modality", logs_count),
                ("empty_traces_modality", traces_count),
            )
            if count <= 0
        ]
        return GateResult("technical_modalities", not reasons, reasons, counts)

    def _business_signal_gate(self, record: Mapping[str, Any]) -> GateResult:
        baseline = (
            _get(record, "baseline_slis", "bifi_result.baseline_slis", default={}) or {}
        )
        fault = _get(record, "fault_slis", "bifi_result.fault_slis", default={}) or {}
        deltas = (
            _get(
                record,
                "business_sli_deltas",
                "sli_deltas",
                "bifi_result.sli_deltas",
                default={},
            )
            or {}
        )
        reasons: list[str] = []
        metrics: dict[str, Any] = {}
        for probe in CORE_PROBES:
            count = _as_int(fault.get(f"{probe}_sample_count"), 0)
            success = _as_float(fault.get(f"{probe}_success_rate"), -1.0)
            base_success = _as_float(baseline.get(f"{probe}_success_rate"), -1.0)
            delta = _as_float(
                deltas.get(f"{probe}_success_rate"),
                success - base_success if success >= 0 and base_success >= 0 else 0.0,
            )
            metrics[f"{probe}_fault_sample_count"] = count
            metrics[f"{probe}_fault_success_rate"] = success
            metrics[f"{probe}_success_delta"] = delta
            if count and count < self.thresholds.business_min_probe_samples:
                reasons.append(f"low_fault_probe_count:{probe}={count}")
            if success == 0.0 and base_success == 0.0:
                reasons.append(f"all_zero_telemetry:{probe}")
            if delta > self.thresholds.max_positive_sli_delta:
                reasons.append(f"suspicious_positive_sli_jump:{probe}={delta:.2f}")
        return GateResult("business_signal", not reasons, reasons, metrics)

    def _strong_evidence_gate(self, record: Mapping[str, Any]) -> GateResult:
        profile = infer_admission_profile(record)
        deltas = (
            _get(
                record,
                "business_sli_deltas",
                "sli_deltas",
                "bifi_result.sli_deltas",
                default={},
            )
            or {}
        )
        invariant_violations = (
            _get(
                record,
                "new_invariant_violations",
                "bifi_result.new_invariant_violations",
                default=[],
            )
            or []
        )
        affected_services = (
            _get(
                record, "affected_services", "bifi_result.affected_services", default=[]
            )
            or []
        )
        depth = _as_int(
            _get(
                record, "propagation_depth", "bifi_result.propagation_depth", default=0
            ),
            0,
        )
        max_drop = 0.0
        primary_business = []
        secondary_business = []
        for key, value in deltas.items():
            delta = _as_float(value, 0.0)
            if key.endswith("_success_rate"):
                max_drop = max(max_drop, -delta)
                if -delta >= self.thresholds.strong_evidence_min_sli_drop:
                    primary_business.append("business_success_rate_drop")
            elif (
                key.endswith("_business_invalid_rate")
                and delta >= self.thresholds.strong_evidence_min_invalid_rate_increase
            ):
                primary_business.append("business_invalid_rate_increase")
            elif (
                key.endswith("_timeout_rate")
                and delta >= self.thresholds.strong_evidence_min_timeout_rate_increase
            ):
                primary_business.append("business_timeout_rate_increase")
            elif (
                key.endswith(("_http_5xx_rate", "_request_exception_rate"))
                and delta >= self.thresholds.strong_evidence_min_error_rate_increase
            ):
                primary_business.append("business_error_rate_increase")
            elif (
                key.endswith("_http_4xx_rate")
                and delta >= self.thresholds.strong_evidence_min_error_rate_increase
            ):
                primary_business.append("business_client_error_rate_increase")
            elif (
                key.endswith("_json_decode_error_rate")
                and delta >= self.thresholds.strong_evidence_min_error_rate_increase
            ):
                primary_business.append("business_decode_error_rate_increase")
            elif (
                key.endswith("_count")
                and delta >= self.thresholds.strong_evidence_min_semantic_count_increase
            ):
                primary_business.append("semantic_count_increase")
            elif (
                key.endswith("_distribution_jsd")
                and _as_float(value, 0.0)
                >= self.thresholds.strong_evidence_min_distribution_jsd
            ):
                primary_business.append("entity_distribution_shift")
            elif key.endswith(LATENCY_BUSINESS_SUFFIXES) and delta > 0:
                secondary_business.append("latency_shift")
        evidence_types = []
        if primary_business:
            evidence_types.extend(sorted(set(primary_business)))
        if profile.allow_infra_weak_business_realistic and secondary_business:
            evidence_types.append("infra_business_latency_signal")
        if (
            len(invariant_violations)
            >= self.thresholds.strong_evidence_min_new_invariants
        ):
            evidence_types.append("hidden_new_invariant_violation")
        if (
            len(affected_services)
            >= self.thresholds.strong_evidence_min_affected_services
        ):
            evidence_types.append("hidden_service_propagation")
        if depth >= self.thresholds.strong_evidence_min_propagation_depth:
            evidence_types.append("hidden_propagation_depth")
        ok = bool(primary_business) or bool(
            profile.allow_infra_weak_business_realistic and secondary_business
        )
        reasons = [] if ok else ["no_strong_evidence"]
        return GateResult(
            "strong_evidence",
            ok,
            reasons,
            {
                "max_core_sli_drop": max_drop,
                "primary_business_anomaly": bool(primary_business),
                "secondary_latency_only": bool(
                    secondary_business and not primary_business
                ),
                "admission_profile": profile.name,
                "new_invariant_violations": len(invariant_violations),
                "affected_services_count": len(affected_services),
                "propagation_depth": depth,
                "evidence_types": evidence_types,
            },
        )

    def _prism_gate(self, record: Mapping[str, Any]) -> GateResult:
        prism = (
            _get(record, "prism_verdict", "prism_dynamic", "prism", default={}) or {}
        )
        verdict = str(prism.get("decision") or prism.get("verdict") or "").upper()
        score = _as_float(prism.get("aggregate_score", prism.get("score")), 0.0)
        precheck_ok = bool(prism.get("precheck_ok", True))
        rejected = verdict in {
            "REJECT_DYNAMIC",
            "REJECT_STATIC",
            "UNREALISTIC",
            "REJECTED",
        }
        reasons: list[str] = []
        if not verdict:
            reasons.append("missing_prism_verdict")
        elif verdict not in set(self.thresholds.final_allowed_verdicts):
            reasons.append(f"prism_verdict_not_gold:{verdict}")
        if score < self.thresholds.final_score_threshold:
            reasons.append(f"low_prism_score:{score:.2f}")
        if not precheck_ok:
            reasons.append("prism_precheck_failed")
        ok = (
            bool(verdict)
            and verdict in set(self.thresholds.final_allowed_verdicts)
            and score >= self.thresholds.final_score_threshold
            and precheck_ok
        )
        return GateResult(
            "prism",
            ok,
            reasons,
            {
                "verdict": verdict,
                "score": score,
                "precheck_ok": precheck_ok,
                "rejected": rejected,
            },
        )

    def _cleanup_gate(self, record: Mapping[str, Any]) -> GateResult:
        cleanup_ok = bool(
            _get(
                record,
                "post_cleanup_healthy",
                "cleanup_result.healthy",
                "bifi_result.cleanup_ok",
                default=True,
            )
        )
        reasons = [] if cleanup_ok else ["cleanup_or_restoration_failed"]
        return GateResult(
            "cleanup", cleanup_ok, reasons, {"post_cleanup_healthy": cleanup_ok}
        )

    def _evidence_summary(self, record: Mapping[str, Any]) -> dict[str, Any]:
        strong = self._strong_evidence_gate(record)
        return {
            "bifi_succeeded": bool(
                _get(
                    record,
                    "bifi_succeeded",
                    "bifi_result.succeeded",
                    "bifi_result.success",
                    default=False,
                )
            ),
            "has_real_dynamic_observation": self._dynamic_observation_gate(record).ok,
            "technical_modalities": self._technical_modalities_gate(record).metrics,
            "strong_evidence": strong.metrics,
            "business_sli_deltas": _get(
                record,
                "business_sli_deltas",
                "sli_deltas",
                "bifi_result.sli_deltas",
                default={},
            )
            or {},
            "new_invariant_violations": _get(
                record,
                "new_invariant_violations",
                "bifi_result.new_invariant_violations",
                default=[],
            )
            or [],
            "affected_services": _get(
                record, "affected_services", "bifi_result.affected_services", default=[]
            )
            or [],
        }
