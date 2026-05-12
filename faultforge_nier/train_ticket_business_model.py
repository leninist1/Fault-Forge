"""Loader for the seed Train-Ticket business model."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


MODEL_FILES = {
    "entities": "business_entities.yml",
    "journeys": "business_journeys.yml",
    "invariants": "business_invariants.yml",
    "state_machines": "business_state_machines.yml",
    "ownership": "service_business_ownership.yml",
    "fault_dimensions": "fault_space_dimensions.yml",
}


class BusinessModel:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.data = {key: self._load(self.root / filename) for key, filename in MODEL_FILES.items()}

    def _load(self, path: Path) -> dict[str, Any]:
        if not path.exists():
            raise FileNotFoundError(path)
        text = path.read_text(encoding="utf-8")
        if yaml is not None:
            return yaml.safe_load(text) or {}
        return json.loads(text)

    def validate(self) -> list[str]:
        errors: list[str] = []
        entities = self.data["entities"].get("entities", {})
        journeys = self.data["journeys"].get("journeys", {})
        invariants = self.data["invariants"].get("invariants", {})
        state_machines = self.data["state_machines"].get("state_machines", {})
        if len(entities) < 10:
            errors.append("business_entities.yml must define at least 10 entities")
        if len(journeys) < 6:
            errors.append("business_journeys.yml must define at least 6 journeys")
        if len(invariants) < 10:
            errors.append("business_invariants.yml must define at least 10 invariants")
        if len(state_machines) < 3:
            errors.append("business_state_machines.yml must define at least 3 state machines")
        for inv_id, inv in invariants.items():
            if not inv.get("services"):
                errors.append(f"{inv_id}: missing services")
            if not inv.get("observation_sources"):
                errors.append(f"{inv_id}: missing observation_sources")
        return errors


def load_default() -> BusinessModel:
    return BusinessModel(Path(__file__).resolve().parents[1] / "business_model")
