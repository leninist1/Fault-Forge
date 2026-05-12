"""Admission profiles for telemetry-first dataset quality gates.

Profiles are used only by PRISM and offline quality gates. They must not be
exported into RCA-visible CSV inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


INFRA_DIMENSIONS = {
    "resource_degradation",
    "network_partition",
    "network_degradation",
    "infra_business_impact",
}
INFRA_LAYERS = {"resource", "network", "database", "cache"}
INFRA_INJECTORS = {
    "resource_limit",
    "host_iptables",
    "host_tc",
    "mysql_slow",
    "redis_slow",
}

SEMANTIC_DIMENSIONS = {
    "data_consistency",
    "state_transition",
    "amount_or_price_drift",
    "partial_workflow_commit",
}

CONFIG_DIMENSIONS = {"stale_configuration", "business_logic"}
CONFIG_INJECTORS = {"config_modifier"}


@dataclass(frozen=True)
class AdmissionProfile:
    name: str
    min_business_score_for_realistic: float
    require_primary_business_for_realistic: bool
    allow_latency_only_realistic: bool
    allow_infra_weak_business_realistic: bool = False


DEFAULT_PROFILE = AdmissionProfile(
    name="default",
    min_business_score_for_realistic=0.70,
    require_primary_business_for_realistic=True,
    allow_latency_only_realistic=False,
)

INFRA_PROFILE = AdmissionProfile(
    name="infra",
    min_business_score_for_realistic=0.35,
    require_primary_business_for_realistic=False,
    allow_latency_only_realistic=True,
    allow_infra_weak_business_realistic=True,
)

SEMANTIC_PROFILE = AdmissionProfile(
    name="semantic_business",
    min_business_score_for_realistic=0.70,
    require_primary_business_for_realistic=True,
    allow_latency_only_realistic=False,
)

CONFIG_PROFILE = AdmissionProfile(
    name="config_business",
    min_business_score_for_realistic=0.60,
    require_primary_business_for_realistic=True,
    allow_latency_only_realistic=False,
)


def infer_admission_profile(record: Mapping[str, Any]) -> AdmissionProfile:
    """Infer the offline admission profile from hidden fault metadata."""
    spec = record.get("fault_spec") or {}
    dimension = _norm(spec.get("dimension") or record.get("fault_dimension"))
    layer = _norm(spec.get("root_injection_layer") or record.get("root_injection_layer"))
    injector = _norm(
        spec.get("injector")
        or spec.get("injector_family")
        or (spec.get("injector_params") or {}).get("injector")
        or record.get("injector")
    )

    if (
        dimension in INFRA_DIMENSIONS
        or layer in INFRA_LAYERS
        or injector in INFRA_INJECTORS
    ):
        return INFRA_PROFILE
    if dimension in SEMANTIC_DIMENSIONS:
        return SEMANTIC_PROFILE
    if dimension in CONFIG_DIMENSIONS or injector in CONFIG_INJECTORS:
        return CONFIG_PROFILE
    return DEFAULT_PROFILE


def _norm(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_")
