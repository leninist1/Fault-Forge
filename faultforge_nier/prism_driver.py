"""Rule-computed PRISM static scoring for ASE NIER candidates."""

from __future__ import annotations

from typing import Any

from .telemetry_prism import dynamic_validate_telemetry
from .train_ticket_business_model import BusinessModel, load_default


SUPPORTED_INJECTORS = {
    "database_modifier",
    "resource_limit",
    "host_iptables",
    "host_tc",
    "mysql_slow",
    "redis_slow",
    "config_modifier",
}


def static_validate(
    candidate: dict[str, Any], model: BusinessModel | None = None
) -> dict[str, Any]:
    model = model or load_default()
    journeys = model.data["journeys"].get("journeys", {})
    entities = model.data["entities"].get("entities", {})
    invariants = model.data["invariants"].get("invariants", {})
    ownership = model.data["ownership"].get("services", {})

    service = candidate.get("fault_point", {}).get("owner_service") or candidate.get(
        "injector_params", {}
    ).get("target_service")
    invariant = invariants.get(candidate.get("target_invariant"), {})
    rules = {
        "A1_business_journey_exists": candidate.get("business_journey") in journeys,
        "A2_business_entity_exists": candidate.get("business_entity") in entities,
        "A3_target_invariant_exists": candidate.get("target_invariant") in invariants,
        "A4_service_owns_or_participates_in_entity": service in ownership,
        "A6_expected_business_impact_defined": bool(
            candidate.get("expected_business_impact")
            or candidate.get("expected_observable_signals")
        ),
        "E1_target_service_exists": service in ownership,
        "E4_injector_supported": candidate.get("injector") in SUPPORTED_INJECTORS,
        "E5_injector_params_complete": bool(candidate.get("injector_params")),
        "B1_target_invariant_declared": bool(candidate.get("target_invariant")),
        "B3_invariant_has_executable_or_observable_check": bool(
            invariant.get("observation_sources")
        ),
        "D1_expected_business_impact_defined": bool(
            candidate.get("expected_business_impact")
        ),
    }
    a_score = sum(rules[k] for k in rules if k.startswith("A")) / 5
    b_score = (
        rules["B1_target_invariant_declared"]
        + rules["B3_invariant_has_executable_or_observable_check"]
    ) / 2
    c_score = (
        1.0
        if service in ownership and candidate.get("expected_propagation")
        else 0.5
        if service in ownership
        else 0.0
    )
    d_score = 1.0 if rules["D1_expected_business_impact_defined"] else 0.0
    e_score = sum(rules[k] for k in rules if k.startswith("E")) / 3
    aggregate = (
        0.25 * a_score
        + 0.20 * b_score
        + 0.15 * c_score
        + 0.15 * d_score
        + 0.25 * e_score
    )
    blocking = []
    if candidate.get("target_invariant") not in invariants:
        blocking.append("unknown_target_invariant")
    if service not in ownership:
        blocking.append("unknown_target_service")
    if candidate.get("injector") not in SUPPORTED_INJECTORS:
        blocking.append("unsupported_injector")
    decision = (
        "EXECUTE"
        if aggregate >= 0.75 and a_score >= 0.70 and e_score >= 0.80 and not blocking
        else "REPAIR"
        if aggregate >= 0.50
        else "REJECT_STATIC"
    )
    return {
        "prism_version": "ase_nier_evidence_first_v1",
        "stage": "static",
        "axis_scores": {
            "A_local_business_grounding": round(a_score, 3),
            "B_invariant_oracle_strength": round(b_score, 3),
            "C_propagation_realism": round(c_score, 3),
            "D_business_user_impact": round(d_score, 3),
            "E_triggerability_executability": round(e_score, 3),
        },
        "rule_results": rules,
        "aggregate_score": round(aggregate, 3),
        "decision": decision,
        "blocking_errors": blocking,
        "warnings": [],
        "evidence_refs": {"business_model_files": []},
        "llm_rationale": {"text": "", "authoritative_for_score": False},
    }


def dynamic_validate(
    record: dict[str, Any], model: BusinessModel | None = None
) -> dict[str, Any]:
    """Compute dynamic PRISM verdict from LLM-visible telemetry only."""
    return dynamic_validate_telemetry(record)


def _mean_bool(values: list[Any]) -> float:
    if not values:
        return 0.0
    return sum(1.0 if value else 0.0 for value in values) / len(values)


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default
