"""LLM-guided, schema-grounded FSE adapter for ASE NIER.

The LLM proposes weakness intents, not executable FaultSpecs. This adapter
grounds those intents against local system facts and the curated executable
catalog, then emits only canonical candidate specs.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

from .business_fault_catalog import generate_curated_candidates
from .fse_driver import _mutate_candidate


SYSTEM_PROMPT = """You are FaultForge FSE's weakness analyst.

Use only the supplied system facts. Do not invent services, injectors, business
journeys, metrics, tables, or fields. Your task is to identify high-value
weakness intents that are likely to produce observable RCA telemetry.

Actively use runtime_feedback guidance:
- avoid repeatedly selecting overrepresented injectors/services/dimensions;
- prefer underexplored combinations that remain executable and observable;
- if prior runs show export/audit issues, steer away from repeatedly similar
  weak intents unless required for coverage.

Return JSON with this exact shape:
{"weakness_intents": [
  {
    "id": "short-id",
    "rationale": "one sentence grounded in facts",
    "target_service": "one allowed service",
    "business_journey": "one allowed journey",
    "preferred_injector_family": "one allowed injector",
    "root_layer": "resource|network|database|cache|configuration|application",
    "expected_business_signals": ["allowed metric names"],
    "severity_goal": "medium|high|high_but_recoverable"
  }
]}
"""


def build_system_fact_pack(
    *,
    system_description_dir: Path,
    workspace: Path,
    previous_run_root: Path | None = None,
    feedback_run_roots: list[Path] | None = None,
) -> dict[str, Any]:
    """Build a compact, allowlisted fact pack for LLM FSE."""
    sysdir = Path(system_description_dir)
    families_path = Path(workspace) / "configs" / "business_fault_families.yml"
    families_doc = _load_yaml(families_path)
    journeys_doc = _load_yaml(sysdir / "business_journeys.yml")
    topology_doc = _load_yaml(sysdir / "service_topology.yml")
    infra_doc = _load_yaml(sysdir / "infrastructure.yml")
    injectors_doc = _load_yaml(sysdir / "available_injectors.yml")

    curated = generate_curated_candidates(limit=500)
    services = sorted(
        {
            _target_service(candidate)
            for candidate in curated
            if _target_service(candidate) != "unknown"
        }
    )
    journeys = sorted(
        {
            str(candidate.get("business_journey") or "")
            for candidate in curated
            if candidate.get("business_journey")
        }
    )
    injectors = sorted(
        {
            str(candidate.get("injector") or "")
            for candidate in curated
            if candidate.get("injector")
        }
    )
    metrics = sorted(
        {
            metric
            for candidate in curated
            for metric in _expected_business_slis(candidate)
        }
    )

    feedback = _load_feedback(feedback_run_roots, previous_run_root)
    return {
        "allowed": {
            "services": services,
            "business_journeys": journeys,
            "injector_families": injectors,
            "business_metrics": metrics,
        },
        "business_fault_families": _family_summaries(families_doc),
        "topology_summary": _topology_summary(topology_doc),
        "infrastructure_summary": infra_doc,
        "injectors_summary": injectors_doc,
        "business_journeys_summary": _journey_summary(journeys_doc),
        "runtime_feedback": feedback,
    }


def run_llm_fse(
    *,
    llm: Any,
    system_description_dir: Path,
    workspace: Path,
    output_dir: Path,
    limit: int,
    exploration_round: int,
    previous_run_root: Path | None = None,
    feedback_run_roots: list[Path] | None = None,
) -> dict[str, Any]:
    """Run LLM weakness analysis and compile intents into executable specs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fact_pack = build_system_fact_pack(
        system_description_dir=system_description_dir,
        workspace=workspace,
        previous_run_root=previous_run_root,
        feedback_run_roots=feedback_run_roots,
    )
    (output_dir / "system_fact_pack.json").write_text(
        json.dumps(fact_pack, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    response = llm.chat_json(
        system=SYSTEM_PROMPT,
        user=json.dumps(fact_pack, indent=2, ensure_ascii=False),
        temperature=0.2,
        max_tokens=5000,
        schema_hint="All values must be selected from the allowed lists.",
    )
    intents = _sanitize_intents(response.get("weakness_intents", []), fact_pack)
    compiled = compile_intents_to_faults(
        intents=intents,
        limit=limit,
        exploration_round=exploration_round,
        feedback_guidance=(fact_pack.get("runtime_feedback") or {}).get("guidance") or {},
    )
    payload = {
        "faults": compiled,
        "stats": {
            "generated": len(compiled),
            "accepted": len(compiled),
            "generation_source": "llm_guided_schema_grounded",
            "weakness_intents": len(intents),
            "exploration_round": exploration_round,
        },
        "weakness_intents": intents,
    }
    (output_dir / "weakness_intents.json").write_text(
        json.dumps({"weakness_intents": intents}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "fault_catalog.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "vulnerability_surface.json").write_text(
        json.dumps(
            {
                "generation_source": "llm_guided_schema_grounded",
                "weakness_intents": intents,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    _write_report(output_dir, fact_pack, intents, compiled)
    return payload


def compile_intents_to_faults(
    *,
    intents: list[dict[str, Any]],
    limit: int,
    exploration_round: int,
    feedback_guidance: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Ground LLM intents into known executable candidates."""
    curated = generate_curated_candidates(limit=500)
    scored: list[tuple[float, int, dict[str, Any], dict[str, Any]]] = []
    for intent_index, intent in enumerate(intents):
        for candidate in curated:
            score = _match_score(intent, candidate)
            if score <= 0:
                continue
            score *= _feedback_multiplier(candidate, feedback_guidance or {})
            scored.append((score, intent_index, candidate, intent))

    if not scored:
        scored = [
            (1.0, 0, candidate, {})
            for candidate in curated
            if candidate.get("injector") in {"resource_limit", "host_iptables", "mysql_slow", "redis_slow", "database_modifier"}
        ]

    scored.sort(key=lambda item: item[0], reverse=True)
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    variant_index = 1
    for _score, intent_index, candidate, intent in scored:
        if len(output) >= limit:
            break
        grounded = _mutate_candidate(
            candidate,
            exploration_round=max(1, exploration_round),
            variant_index=variant_index,
        )
        metadata = grounded.setdefault("fse_metadata", {})
        metadata.update(
            {
                "generation_source": "llm_guided_schema_grounded",
                "weakness_intent_id": intent.get("id", f"intent-{intent_index}"),
                "weakness_intent_rank": intent_index,
                "llm_selected_target_service": intent.get("target_service", ""),
                "llm_selected_business_journey": intent.get("business_journey", ""),
                "llm_selected_injector_family": intent.get("preferred_injector_family", ""),
            }
        )
        sig = json.dumps(
            {
                "dimension": grounded.get("dimension"),
                "injector": grounded.get("injector"),
                "fault_point": grounded.get("fault_point", {}),
                "injector_params": grounded.get("injector_params", {}),
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        if sig in seen:
            continue
        seen.add(sig)
        output.append(grounded)
        variant_index += 1
    return output


def _sanitize_intents(
    raw_intents: list[dict[str, Any]],
    fact_pack: Mapping[str, Any],
) -> list[dict[str, Any]]:
    allowed = fact_pack.get("allowed", {})
    services = set(allowed.get("services", []))
    journeys = set(allowed.get("business_journeys", []))
    injectors = set(allowed.get("injector_families", []))
    metrics = set(allowed.get("business_metrics", []))
    clean: list[dict[str, Any]] = []
    for idx, raw in enumerate(raw_intents):
        service = str(raw.get("target_service") or "")
        journey = str(raw.get("business_journey") or "")
        injector = str(raw.get("preferred_injector_family") or "")
        if service not in services or journey not in journeys or injector not in injectors:
            continue
        signals = [
            str(metric)
            for metric in raw.get("expected_business_signals", [])
            if str(metric) in metrics
        ]
        clean.append(
            {
                "id": str(raw.get("id") or f"intent-{idx:03d}"),
                "rationale": str(raw.get("rationale") or ""),
                "target_service": service,
                "business_journey": journey,
                "preferred_injector_family": injector,
                "root_layer": str(raw.get("root_layer") or ""),
                "expected_business_signals": signals,
                "severity_goal": str(raw.get("severity_goal") or "high_but_recoverable"),
            }
        )
    return clean


def _match_score(intent: Mapping[str, Any], candidate: Mapping[str, Any]) -> float:
    score = 0.0
    preferred_injector = intent.get("preferred_injector_family")
    if preferred_injector and preferred_injector != candidate.get("injector"):
        return 0.0
    if intent.get("target_service") == _target_service(candidate):
        score += 4.0
    if intent.get("business_journey") == candidate.get("business_journey"):
        score += 2.0
    if intent.get("preferred_injector_family") == candidate.get("injector"):
        score += 3.0
    expected = set(intent.get("expected_business_signals") or [])
    candidate_metrics = set(_expected_business_slis(candidate))
    score += min(2.0, len(expected & candidate_metrics))
    return score


def _feedback_multiplier(candidate: Mapping[str, Any], guidance: Mapping[str, Any]) -> float:
    """Softly downweight overrepresented choices from prior run feedback."""
    if not guidance:
        return 1.0
    injector = str(candidate.get("injector") or "")
    service = _target_service(candidate)
    dimension = str(candidate.get("dimension") or "")
    multiplier = 1.0
    if injector and injector in set(guidance.get("avoid_injectors", [])):
        multiplier *= 0.35
    if service and service in set(guidance.get("avoid_services", [])):
        multiplier *= 0.55
    if dimension and dimension in set(guidance.get("avoid_dimensions", [])):
        multiplier *= 0.65
    return max(0.05, multiplier)


def _target_service(candidate: Mapping[str, Any]) -> str:
    fp = candidate.get("fault_point") or {}
    params = candidate.get("injector_params") or {}
    return str(params.get("target_service") or fp.get("owner_service") or "unknown")


def _expected_business_slis(candidate: Mapping[str, Any]) -> list[str]:
    signals = candidate.get("expected_observable_signals") or {}
    return [
        str(item)
        for item in (
            signals.get("business_slis")
            or candidate.get("expected_business_slis")
            or []
        )
        if item
    ]


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _family_summaries(doc: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for family in doc.get("families", []) or []:
        rows.append(
            {
                "id": family.get("id"),
                "journey": family.get("journey"),
                "dimension": family.get("dimension"),
                "injector_family": family.get("injector_family"),
                "expected_business_slis": family.get("expected_business_slis", []),
                "expected_services": family.get("expected_services", []),
                "variant_count": len(family.get("variants", []) or []),
            }
        )
    return rows


def _journey_summary(doc: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "journeys": [j.get("name") for j in doc.get("business_journeys", []) or []],
        "invariants": [
            {
                "id": inv.get("id"),
                "journey": inv.get("journey"),
                "entity": inv.get("entity"),
                "services": inv.get("services", []),
                "business_slis": inv.get("business_slis", []),
            }
            for inv in doc.get("business_invariants", []) or []
        ],
    }


def _topology_summary(doc: Mapping[str, Any]) -> dict[str, Any]:
    if "services" not in doc:
        return doc
    services = doc.get("services", {})
    if isinstance(services, dict):
        return {
            "service_count": len(services),
            "services": sorted(services)[:80],
        }
    return doc


def _load_feedback(
    feedback_run_roots: Iterable[Path] | None,
    previous_run_root: Path | None,
) -> dict[str, Any]:
    roots: list[Path] = []
    seen: set[str] = set()
    for raw in list(feedback_run_roots or []) + ([previous_run_root] if previous_run_root else []):
        if not raw:
            continue
        root = Path(raw)
        key = str(root.resolve())
        if key in seen:
            continue
        seen.add(key)
        roots.append(root)
    snapshots = [_feedback_snapshot(root) for root in roots]
    snapshots = [item for item in snapshots if item]
    return {
        "sources": snapshots,
        "guidance": _feedback_guidance(snapshots),
    }


def _feedback_snapshot(run_root: Path) -> dict[str, Any]:
    report_path = run_root / "reports" / "dataset_quality_report.json"
    final_path = run_root / "final_summary.json"
    payload: dict[str, Any] = {"run_root": str(run_root)}
    if report_path.exists():
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
            payload["dataset_quality"] = {
                "status": report.get("status", "unknown"),
                "warnings": list(report.get("warnings") or []),
                "errors": list(report.get("errors") or []),
                "distributions": {
                    "injectors": dict((report.get("distributions") or {}).get("injectors") or {}),
                    "services": dict((report.get("distributions") or {}).get("services") or {}),
                    "dimensions": dict((report.get("distributions") or {}).get("dimensions") or {}),
                },
                "counts": dict(report.get("counts") or {}),
            }
        except json.JSONDecodeError:
            pass
    if final_path.exists():
        try:
            final = json.loads(final_path.read_text(encoding="utf-8"))
            payload["final_summary"] = {
                "final_dataset_count": final.get("final_dataset_count"),
                "reached_target": final.get("reached_target"),
                "dataset_audit": dict(final.get("dataset_audit") or {}),
            }
        except json.JSONDecodeError:
            pass
    return payload if "dataset_quality" in payload or "final_summary" in payload else {}


def _feedback_guidance(snapshots: list[dict[str, Any]]) -> dict[str, Any]:
    injector_totals: Counter[str] = Counter()
    service_totals: Counter[str] = Counter()
    dimension_totals: Counter[str] = Counter()
    warning_counts: Counter[str] = Counter()
    error_counts: Counter[str] = Counter()
    for snapshot in snapshots:
        quality = snapshot.get("dataset_quality") or {}
        dist = quality.get("distributions") or {}
        injector_totals.update({str(k): int(v) for k, v in dict(dist.get("injectors") or {}).items()})
        service_totals.update({str(k): int(v) for k, v in dict(dist.get("services") or {}).items()})
        dimension_totals.update({str(k): int(v) for k, v in dict(dist.get("dimensions") or {}).items()})
        warning_counts.update(str(item) for item in quality.get("warnings") or [])
        error_counts.update(str(item) for item in quality.get("errors") or [])

    def dominant(counter: Counter[str], threshold: float) -> list[str]:
        total = float(sum(counter.values()))
        if total <= 0:
            return []
        return sorted([name for name, count in counter.items() if (count / total) >= threshold])

    return {
        "runs_considered": len(snapshots),
        "avoid_injectors": dominant(injector_totals, threshold=0.60),
        "avoid_services": dominant(service_totals, threshold=0.50),
        "avoid_dimensions": dominant(dimension_totals, threshold=0.50),
        "warning_summary": dict(warning_counts),
        "error_summary": dict(error_counts),
    }


def _write_report(
    output_dir: Path,
    fact_pack: Mapping[str, Any],
    intents: list[dict[str, Any]],
    faults: list[dict[str, Any]],
) -> None:
    injectors = Counter(str(f.get("injector") or "unknown") for f in faults)
    lines = [
        "# LLM-Guided FSE Exploration Report",
        "",
        f"- allowed_services: {len(fact_pack.get('allowed', {}).get('services', []))}",
        f"- weakness_intents: {len(intents)}",
        f"- compiled_faults: {len(faults)}",
        "",
        "## Injector Distribution",
    ]
    lines.extend(f"- {name}: {count}" for name, count in sorted(injectors.items()))
    (output_dir / "exploration_report.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
