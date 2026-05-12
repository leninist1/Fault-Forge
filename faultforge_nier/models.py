"""Canonical ASE NIER data models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class FaultCandidate:
    fault_id: str
    name: str
    dimension: str
    business_journey: str
    business_entity: str
    target_invariant: str
    semantic_violation_type: str
    fault_point: dict[str, Any]
    injector: str
    injector_params: dict[str, Any]
    expected_business_impact: list[str] = field(default_factory=list)
    expected_observable_signals: dict[str, Any] = field(default_factory=dict)
    expected_propagation: list[str] = field(default_factory=list)
    production_scenario: str = ""
    fse_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "fault_id": self.fault_id,
            "name": self.name,
            "dimension": self.dimension,
            "business_journey": self.business_journey,
            "business_entity": self.business_entity,
            "target_invariant": self.target_invariant,
            "semantic_violation_type": self.semantic_violation_type,
            "fault_point": self.fault_point,
            "injector": self.injector,
            "injector_params": self.injector_params,
            "expected_business_impact": self.expected_business_impact,
            "expected_observable_signals": self.expected_observable_signals,
            "expected_propagation": self.expected_propagation,
            "production_scenario": self.production_scenario,
            "fse_metadata": self.fse_metadata,
        }

    def semantic_signature(self) -> str:
        table = self.fault_point.get("table") or self.injector_params.get("target_table") or ""
        field = self.fault_point.get("field") or self.injector_params.get("target_field") or ""
        service = self.fault_point.get("owner_service") or self.injector_params.get("target_service") or ""
        return "|".join(
            [
                self.business_journey,
                self.business_entity,
                self.target_invariant,
                self.semantic_violation_type,
                self.dimension,
                service,
                table,
                field,
                self.injector,
            ]
        )
