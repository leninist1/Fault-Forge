"""Curated Train-Ticket business fault-family catalog support."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from .models import FaultCandidate


WORKSPACE = Path(__file__).resolve().parents[1]
DEFAULT_CATALOG_PATH = WORKSPACE / "configs" / "business_fault_families.yml"


def load_fault_families(path: Path | None = None) -> dict[str, Any]:
    """Load the curated business fault-family catalog."""
    catalog_path = path or DEFAULT_CATALOG_PATH
    payload = yaml.safe_load(catalog_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid fault-family catalog: {catalog_path}")
    payload.setdefault("families", [])
    payload.setdefault("gold_admission", {})
    return payload


def _variant_fault_id(counter: int, family_id: str, variant_id: str) -> str:
    family_part = "".join(ch for ch in family_id.upper() if ch.isalnum())[:10]
    variant_part = "".join(ch for ch in variant_id.upper() if ch.isalnum())[:8]
    return f"CFF-{counter:03d}-{family_part}-{variant_part}"


def _candidate_from_variant(counter: int, family: dict[str, Any], variant: dict[str, Any]) -> dict[str, Any]:
    family_id = str(family["id"])
    variant_id = str(variant["id"])
    target_service = str(variant["target_service"])
    target_db = str(variant.get("target_db", ""))
    target_table = str(variant["target_table"])
    target_field = str(variant["target_field"])
    mode = str(variant.get("mode", "single_point"))
    expected_propagation = list(variant.get("expected_propagation") or family.get("expected_services") or [])

    candidate = FaultCandidate(
        fault_id=_variant_fault_id(counter, family_id, variant_id),
        name=f"{family.get('name', family_id)} / {variant_id}",
        dimension=str(family["dimension"]),
        business_journey=str(family["journey"]),
        business_entity=str(family["entity"]),
        target_invariant=str(family["invariant"]),
        semantic_violation_type=str(family["semantic_violation_type"]),
        fault_point={
            "owner_service": target_service,
            "database": target_db,
            "table": target_table,
            "field": target_field,
            "downstream_services": expected_propagation[1:] if len(expected_propagation) > 1 else [],
        },
        injector=str(family.get("injector_family", "database_modifier")),
        injector_params={
            "target_service": target_service,
            "target_db": target_db,
            "target_table": target_table,
            "target_field": target_field,
            "condition": variant.get("condition", "id IS NOT NULL"),
            "where_clause": variant.get("condition", "id IS NOT NULL"),
            "modify_type": variant.get("modify_type", "set"),
            "modify_value": variant.get("modify_value"),
        },
        expected_business_impact=list(family.get("expected_impacts") or []),
        expected_observable_signals={
            "business_slis": list(family.get("expected_business_slis") or []),
            "invariants": list(family.get("expected_invariants") or [family["invariant"]]),
            "logs": list(family.get("expected_services") or [target_service]),
            "traces": expected_propagation,
        },
        expected_propagation=expected_propagation,
        production_scenario=str(family.get("name", family_id)),
        fse_metadata={
            "generation_source": "curated_fault_family",
            "family_id": family_id,
            "variant_id": variant_id,
            "failure_mode": mode,
            "canary_status": family.get("canary_status", "pending_runtime_validation"),
            "cleanup_strategy": family.get("cleanup_strategy", ""),
            "requires_canary": family.get("canary_status") != "ready_known_gold_path",
        },
    )
    payload = candidate.to_dict()
    payload["root_injection_layer"] = variant.get("root_injection_layer") or family.get("root_injection_layer", "database")
    payload["business_effect_type"] = family.get("business_effect_type", "cross_entity_consistency")
    payload["expected_observable_business_signals"] = list(
        variant.get("expected_observable_business_signals")
        or family.get("expected_observable_business_signals")
        or [{"metric": metric, "direction": "down" if metric.endswith("_success_rate") else "up"} for metric in family.get("expected_business_slis", [])]
    )
    return payload


def generate_curated_candidates(
    *,
    limit: int | None = None,
    catalog_path: Path | None = None,
    include_pending: bool = True,
    family_ids: list[str] | set[str] | tuple[str, ...] | None = None,
) -> list[dict[str, Any]]:
    """Generate deterministic candidates from curated fault families."""
    catalog = load_fault_families(catalog_path)
    family_filter = {str(item).strip() for item in (family_ids or []) if str(item).strip()}
    candidates: list[dict[str, Any]] = []
    counter = 1
    for family in catalog.get("families", []):
        if family_filter and str(family.get("id", "")) not in family_filter:
            continue
        if not include_pending and family.get("canary_status") != "ready_known_gold_path":
            continue
        for variant in family.get("variants", []):
            candidates.append(_candidate_from_variant(counter, family, variant))
            counter += 1
            if limit is not None and len(candidates) >= limit:
                return candidates
    return candidates


def write_curated_catalog(
    output_path: Path,
    *,
    limit: int | None = None,
    family_ids: list[str] | set[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Write curated candidates in the auto-loop seed-catalog format."""
    faults = generate_curated_candidates(limit=limit, family_ids=family_ids)
    payload = {
        "faults": faults,
        "stats": {
            "generated": len(faults),
            "accepted": len(faults),
            "generation_source": "curated_fault_family",
            "family_ids": sorted({str(item).strip() for item in (family_ids or []) if str(item).strip()}),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(WORKSPACE / "datasets" / "generated_fault_catalogs" / "curated_fault_catalog.json"))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--family-ids",
        default="",
        help="Optional comma-separated fault family ids to include",
    )
    args = parser.parse_args()
    limit = args.limit if args.limit > 0 else None
    family_ids = [item.strip() for item in args.family_ids.split(",") if item.strip()]
    payload = write_curated_catalog(Path(args.output), limit=limit, family_ids=family_ids)
    print(json.dumps(payload["stats"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
