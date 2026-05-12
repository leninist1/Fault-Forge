"""Deterministic business fault-space explorer seed implementation."""

from __future__ import annotations

import json
import copy
from pathlib import Path
from typing import Any

from .business_fault_catalog import generate_curated_candidates
from .models import FaultCandidate
from .train_ticket_business_model import BusinessModel, load_default


TEMPLATES = {
    "T1": "state_transition_violation",
    "T2": "cross_service_state_mismatch",
    "T3": "amount_or_price_drift",
    "T4": "inventory_reservation_mismatch",
    "T5": "ownership_violation",
    "T6": "stale_business_configuration",
    "T7": "partial_workflow_commit",
    "T8": "infra_to_business_impact",
}


def generate_candidates(
    model: BusinessModel | None = None,
    limit: int = 50,
    family_ids: list[str] | set[str] | tuple[str, ...] | None = None,
    exploration_round: int = 0,
) -> list[dict[str, Any]]:
    model = model or load_default()
    curated = generate_curated_candidates(limit=limit, family_ids=family_ids)
    if family_ids:
        if curated and len(curated) < limit:
            base = list(curated)
            repeat_index = max(1, exploration_round * max(1, len(base)))
            while len(curated) < limit:
                source = base[(len(curated) - len(base)) % len(base)]
                expanded = _mutate_candidate(
                    source,
                    exploration_round=max(1, exploration_round),
                    variant_index=repeat_index,
                )
                expanded["fault_id"] = (
                    f"{source['fault_id']}-R{repeat_index:03d}"
                    f"-X{max(1, exploration_round):03d}"
                )
                expanded["name"] = (
                    f"{source.get('name', source['fault_id'])}"
                    f" / repeat {repeat_index:03d}"
                )
                expanded.setdefault("injector_params", {})["_faultforge_repeat_index"] = repeat_index
                expanded.setdefault("fse_metadata", {})["scaleup_repeat_index"] = repeat_index
                expanded["fse_metadata"]["scaleup_source_fault_id"] = source["fault_id"]
                curated.append(expanded)
                repeat_index += 1
        return curated[:limit]

    invariants = model.data["invariants"].get("invariants", {})
    ownership = model.data["ownership"].get("services", {})
    dimensions = model.data["fault_dimensions"].get("dimensions", {})
    candidates: list[FaultCandidate] = []
    output: list[dict[str, Any]] = list(curated)
    counter = len(output) + 1
    template_ids = list(TEMPLATES)
    dimension_ids = list(dimensions) or ["data_consistency"]
    pairs = []
    for inv_id, inv in invariants.items():
        for service in inv.get("services", []):
            pairs.append((inv_id, inv, service))
    while len(candidates) < limit and pairs:
        for inv_id, inv, service in pairs:
            owner = ownership.get(service, {})
            entity = inv.get("entity", "order")
            table = (owner.get("tables") or [entity])[0]
            template_id = template_ids[(counter - 1) % len(template_ids)]
            dimension = dimension_ids[(counter - 1) % len(dimension_ids)]
            candidate = FaultCandidate(
                fault_id=f"FSE-{counter:03d}",
                name=f"{inv.get('name', inv_id)} via {TEMPLATES[template_id]}",
                dimension=dimension,
                business_journey=inv.get("journey", "ticket_booking"),
                business_entity=entity,
                target_invariant=inv_id,
                semantic_violation_type=TEMPLATES[template_id],
                fault_point={
                    "owner_service": service,
                    "database": owner.get("database", ""),
                    "table": table,
                    "field": inv.get("default_field", "status"),
                    "downstream_services": inv.get("downstream_services", []),
                },
                injector="database_modifier" if template_id != "T8" else "resource_limit",
                injector_params={
                    "target_service": service,
                    "target_db": owner.get("database", ""),
                    "target_table": table,
                    "target_field": inv.get("default_field", "status"),
                    "condition": inv.get("example_condition", "id IS NOT NULL"),
                    "modify_value": inv.get("example_bad_value", "INVALID"),
                },
                expected_business_impact=inv.get("expected_impacts", []),
                expected_observable_signals={
                    "business_slis": inv.get("business_slis", []),
                    "invariants": [inv_id],
                    "logs": [service],
                    "traces": inv.get("trace_patterns", []),
                },
                expected_propagation=[service] + inv.get("downstream_services", []),
                production_scenario=inv.get("production_scenario", ""),
                fse_metadata={"generation_source": "template", "template_id": template_id, "llm_used": False},
            )
            candidates.append(candidate)
            counter += 1
            output.append(candidate.to_dict())
            if len(output) >= limit:
                return _with_exploration_variants(
                    output,
                    limit=limit,
                    exploration_round=exploration_round,
                )
    return _with_exploration_variants(
        output,
        limit=limit,
        exploration_round=exploration_round,
    )


def _with_exploration_variants(
    base: list[dict[str, Any]],
    *,
    limit: int,
    exploration_round: int,
) -> list[dict[str, Any]]:
    """Return unique base candidates plus round-specific fault-intensity variants."""
    if len(base) >= limit and exploration_round <= 1:
        return base[:limit]
    seed_count = limit if exploration_round <= 1 else max(1, limit // 2)
    output = list(base[:seed_count])
    if not base:
        return output

    variant_index = 1
    cursor = max(0, exploration_round - 1) * max(1, limit)
    while len(output) < limit:
        source = base[(cursor + variant_index - 1) % len(base)]
        variant = _mutate_candidate(source, exploration_round, variant_index)
        output.append(variant)
        variant_index += 1
    return output


def _mutate_candidate(
    source: dict[str, Any],
    exploration_round: int,
    variant_index: int,
) -> dict[str, Any]:
    candidate = copy.deepcopy(source)
    params = candidate.setdefault("injector_params", {})
    metadata = candidate.setdefault("fse_metadata", {})
    injector = str(candidate.get("injector") or "").lower()
    field = str(
        params.get("target_field")
        or (candidate.get("fault_point") or {}).get("field")
        or ""
    ).lower()

    severity = ((exploration_round + variant_index - 1) % 4) + 1
    candidate["fault_id"] = (
        f"{source.get('fault_id', 'FSE')}-X{exploration_round:03d}V{variant_index:03d}"
    )
    candidate["name"] = (
        f"{source.get('name', source.get('fault_id', 'fault'))}"
        f" / exploration r{exploration_round} v{variant_index}"
    )
    metadata.update(
        {
            "generation_source": "feedback_exploration",
            "exploration_round": exploration_round,
            "exploration_variant_index": variant_index,
            "exploration_source_fault_id": source.get("fault_id"),
            "exploration_severity": severity,
        }
    )
    params["_faultforge_exploration_round"] = exploration_round
    params["_faultforge_exploration_variant_index"] = variant_index

    if injector == "resource_limit":
        if "mem" in field or "memory" in field:
            params["modify_value"] = ["memory_limit_192m", "memory_limit_128m", "memory_limit_96m", "memory_limit_64m"][severity - 1]
        else:
            params["modify_value"] = ["cpu_quota_25_percent", "cpu_quota_15_percent", "cpu_quota_10_percent", "cpu_quota_5_percent"][severity - 1]
    elif injector in {"mysql_slow", "redis_slow"}:
        params["delay_ms"] = [300, 600, 1000, 1500][severity - 1]
        params["affected_ratio"] = [0.5, 0.75, 1.0, 1.0][severity - 1]
        params["modify_value"] = f"latency_{params['delay_ms']}ms"
    elif injector == "host_iptables":
        params["action"] = ["DROP", "DROP", "REJECT", "DROP"][severity - 1]
        params["direction"] = ["to", "both", "both", "both"][severity - 1]
        if field in {"latency_ms", "packet_loss", "loss_rate"}:
            params["modify_value"] = ["latency_300ms", "latency_600ms", "loss_30_percent", "loss_50_percent"][severity - 1]
    elif injector == "database_modifier":
        params["condition"] = _condition_variant(str(params.get("condition") or ""), severity)
        params["where_clause"] = params["condition"]
    else:
        params["exploration_severity"] = severity

    return candidate


def _condition_variant(condition: str, severity: int) -> str:
    condition = condition.strip() or "id IS NOT NULL"
    if severity <= 1:
        return condition
    if " limit " in condition.lower():
        return condition
    if severity == 2:
        return f"({condition})"
    if severity == 3:
        return f"({condition}) AND id IS NOT NULL"
    return "id IS NOT NULL"


def write_catalog(output_path: Path, limit: int = 50) -> None:
    candidates = generate_candidates(limit=limit)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(candidates, indent=2, ensure_ascii=False), encoding="utf-8")
