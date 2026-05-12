"""Build and optionally execute ASE NIER production-gold auto-loop commands."""

from __future__ import annotations

import argparse
import os
import json
import shlex
import shutil
import subprocess
from pathlib import Path

import yaml

from .injector_calibration import build_calibration_report


DISABLED_PRODUCTION_INJECTORS = (
    "docker_exec",
    "docker_kill",
    "docker_pause",
    "docker_stop",
    "hybrid",
    "logic_fault",
    "mysql_down",
    "network_latency",
    "packet_loss",
    "network_partition",
    "cpu_high",
    "memory_leak",
    "selective_database_modifier",
)
DISABLED_SCORED_DATASET_INJECTORS = (
    "docker_exec",
    "docker_kill",
    "docker_pause",
    "docker_stop",
    "mysql_down",
    "selective_database_modifier",
)
LOW_YIELD_SCORED_INJECTORS = (
    "config_modifier",  # blocked: upstream ConfigModifierInjector MySQL issue
    "host_tc",  # blocked: Docker NET_ADMIN capability required
)
# mysql_slow and redis_slow promoted: validated as scored data producers (Session 6)
DEFAULT_SCALE_UP_GOLD_THRESHOLD = 30
DEFAULT_MIN_ADMITTED_FAMILIES_FOR_SCALE_UP = 2


class CalibrationGateError(RuntimeError):
    """Raised when a production scale-up is attempted before calibration is ready."""


def load_env_file(path: Path) -> dict[str, str]:
    """Parse a simple KEY=VALUE env file without overriding shell-provided env."""
    if not path.exists():
        return {}
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            values[key] = value
    return values


def _ensure_runtime_injector(
    injectors_path: Path, definition: dict[str, Any]
) -> None:
    """Add a missing injector definition to available_injectors.yml at runtime."""
    if not injectors_path.exists():
        return
    data = yaml.safe_load(injectors_path.read_text(encoding="utf-8")) or {}
    injectors = data.get("injectors")
    if not isinstance(injectors, list):
        return
    existing = {i["name"] for i in injectors if isinstance(i, dict) and "name" in i}
    name = definition.get("name")
    if name and name not in existing:
        injectors.append(definition)
        injectors_path.write_text(yaml.dump(data, default_flow_style=False), encoding="utf-8")


def prepare_runtime_system_description(
    workspace: Path, output_root: Path | None = None
) -> Path:
    source_sysdir = workspace / "system_description"
    if output_root is None:
        runtime_sysdir = source_sysdir
    else:
        runtime_sysdir = output_root / "runtime_system_description"
        if runtime_sysdir.exists():
            shutil.rmtree(runtime_sysdir)
        shutil.copytree(source_sysdir, runtime_sysdir)

    target = runtime_sysdir / "business_journeys.yml"
    source = runtime_sysdir / "business_journeys_reference.yml"
    if not target.exists() and source.exists():
        shutil.copyfile(source, target)
    injectors_path = runtime_sysdir / "available_injectors.yml"
    legacy_injectors_path = (
        workspace.parent
        / "fault-injection"
        / "system_description"
        / "available_injectors.yml"
    )
    if injectors_path.exists() and legacy_injectors_path.exists():
        injectors = yaml.safe_load(injectors_path.read_text(encoding="utf-8")) or {}
        if isinstance(injectors.get("injectors"), dict):
            shutil.copyfile(legacy_injectors_path, injectors_path)
    _ensure_runtime_injector(
        injectors_path,
        {
            "name": "redis_slow",
            "class": "fault_injection.fault_injector.RedisSlowInjector",
            "target_type": "cache",
            "reversible": True,
            "capabilities": ["cache_slow_query"],
            "params_schema": {
                "target_db": {"type": "string", "required": False},
                "delay_ms": {
                    "type": "int",
                    "range": [100, 10000],
                },
                "affected_ratio": {
                    "type": "float",
                    "default": 1.0,
                    "range": [0, 1],
                },
            },
        },
    )
    return runtime_sysdir


def build_pythonpath(workspace: Path, current_pythonpath: str | None = None) -> str:
    """Return the runtime PYTHONPATH needed by the isolated ASE NIER wrapper."""
    entries = [str(workspace)]
    fault_injection_root = workspace.parent / "fault-injection"
    if fault_injection_root.exists():
        entries.append(str(fault_injection_root))
    if current_pythonpath:
        entries.append(current_pythonpath)
    return os.pathsep.join(entries)


def candidate_pool_args(
    target_gold_count: int, max_iterations: int, max_injections_per_iteration: int
) -> list[str]:
    """Scale FSE candidate-pool work to the requested production-gold run size."""
    injection_budget = max(1, max_iterations) * max(1, max_injections_per_iteration)
    max_candidates_per_iteration = min(
        60, max(6, target_gold_count, max(1, max_injections_per_iteration) * 3)
    )
    prefetch = min(300, max(20, target_gold_count * 8, injection_budget * 4))
    pool_min = min(prefetch, max_candidates_per_iteration)
    refill = min(100, max(20, target_gold_count * 2, injection_budget * 2))
    return [
        "--max-candidates-per-iteration",
        str(max_candidates_per_iteration),
        "--candidate-pool-prefetch",
        str(prefetch),
        "--candidate-pool-min",
        str(pool_min),
        "--candidate-pool-refill",
        str(refill),
        "--candidate-pool-refill-rounds",
        "1",
    ]


def enforce_calibration_gate(
    *,
    workspace: Path,
    target_gold_count: int,
    scale_up_gold_threshold: int = DEFAULT_SCALE_UP_GOLD_THRESHOLD,
    min_admitted_families: int = DEFAULT_MIN_ADMITTED_FAMILIES_FOR_SCALE_UP,
) -> dict[str, object]:
    """Block broad production runs until enough injector families have passed calibration."""
    report = build_calibration_report(
        catalog_path=workspace / "configs" / "business_fault_families.yml",
        results_root=workspace / "experiments" / "production_gold" / "run",
    )
    admitted = [
        row["family_id"] for row in report.get("families", []) if row.get("admitted")
    ]
    pending = [
        row["family_id"]
        for row in report.get("families", [])
        if row.get("next_action") in {"run_runtime_canary", "continue_runtime_canary"}
    ]

    gate = {
        "enabled": True,
        "target_gold_count": target_gold_count,
        "scale_up_gold_threshold": scale_up_gold_threshold,
        "min_admitted_families": min_admitted_families,
        "admitted_family_total": len(admitted),
        "admitted_families": admitted,
        "pending_or_continuing_families": pending,
        "passed": target_gold_count < scale_up_gold_threshold
        or len(admitted) >= min_admitted_families,
    }
    if not gate["passed"]:
        raise CalibrationGateError(
            "Calibration gate blocked production scale-up: "
            f"target_gold_count={target_gold_count}, "
            f"admitted_family_total={len(admitted)}, "
            f"required_admitted_families={min_admitted_families}. "
            "Run injector canaries first with target_gold_count below the scale-up threshold, "
            "or pass --skip-calibration-gate for an explicitly ungated experiment."
        )
    return gate


def production_family_ids(
    *,
    calibration_gate: dict[str, object],
    canary_families: list[str] | tuple[str, ...] | None = None,
) -> list[str]:
    """Return curated families that should be included in the production command."""
    canary = [
        str(item).strip() for item in (canary_families or []) if str(item).strip()
    ]
    if canary:
        return canary
    if not calibration_gate.get("enabled", True):
        return []
    return [
        str(item)
        for item in calibration_gate.get("admitted_families", [])
        if str(item).strip()
    ]


def family_injectors(
    workspace: Path, family_ids: list[str] | tuple[str, ...]
) -> set[str]:
    wanted = {str(item).strip() for item in family_ids if str(item).strip()}
    if not wanted:
        return set()
    catalog_path = workspace / "configs" / "business_fault_families.yml"
    if not catalog_path.exists():
        return set()
    catalog = yaml.safe_load(catalog_path.read_text(encoding="utf-8")) or {}
    injectors = set()
    for family in catalog.get("families", []):
        if str(family.get("id", "")) in wanted:
            injectors.add(str(family.get("injector_family", "")).strip())
    return {item for item in injectors if item}


def disabled_injectors_for_run(
    workspace: Path,
    family_ids: list[str] | tuple[str, ...],
    export_policy: str = "gold",
) -> list[str]:
    base = (
        DISABLED_SCORED_DATASET_INJECTORS
        if str(export_policy).lower() == "scored"
        else DISABLED_PRODUCTION_INJECTORS
    )
    disabled = set(base)
    # Targeted canaries are allowed to exercise the specific injector declared
    # by the requested family; ordinary scale-up keeps the conservative list.
    disabled.difference_update(family_injectors(workspace, family_ids))
    return sorted(disabled)


def low_yield_injectors_for_run(
    workspace: Path,
    family_ids: list[str] | tuple[str, ...],
    export_policy: str = "gold",
) -> list[str]:
    base = LOW_YIELD_SCORED_INJECTORS if str(export_policy).lower() == "scored" else ()
    low_yield = set(base)
    fi = family_injectors(workspace, family_ids)
    low_yield -= fi
    return sorted(low_yield)


def build_command(
    workspace: Path,
    output_root: Path,
    dataset_dir: Path,
    target_gold_count: int,
    max_iterations: int,
    max_injections_per_iteration: int,
    system_description_dir: Path | None = None,
    curated_family_ids: list[str] | tuple[str, ...] | None = None,
    final_allowed_verdicts: str = "REALISTIC,BORDERLINE,UNREALISTIC,INCONCLUSIVE",
    final_score_threshold: float = 0.0,
    rcaeval_export_policy: str = "scored",
    llm_fse: bool = False,
) -> list[str]:
    system_description_dir = system_description_dir or workspace / "system_description"
    command = [
        "python",
        "-m",
        "faultforge_nier.auto_loop_nier",
        "--production-gold",
        "--target-count",
        str(target_gold_count),
        "--target-gold-count",
        str(target_gold_count),
        "--max-iterations",
        str(max_iterations),
        "--max-injections-per-iteration",
        str(max_injections_per_iteration),
        "--output-root",
        str(output_root),
        "--dataset-dir",
        str(dataset_dir),
        "--system-description",
        str(system_description_dir),
        "--traffic-probes",
        "10",
        "--baseline-seconds",
        "30",
        "--fault-seconds",
        "60",
        "--cooldown-seconds",
        "15",
        "--fault-seconds-min",
        "30",
        "--cooldown-seconds-min",
        "10",
        "--business-min-probe-samples",
        "10",
        "--baseline-min-success-rate",
        "0.90",
        "--baseline-max-5xx-rate",
        "0.05",
        "--strong-evidence-min-sli-drop",
        "0.05",
        "--final-allowed-verdicts",
        str(final_allowed_verdicts),
        "--final-score-threshold",
        str(final_score_threshold),
        "--rcaeval-export-policy",
        str(rcaeval_export_policy),
        "--disabled-injectors",
        ",".join(DISABLED_PRODUCTION_INJECTORS),
        "--low-yield-injectors",
        ",".join(DISABLED_PRODUCTION_INJECTORS),
        "--write-per-fault-full",
        "--gold-only-rcaeval",
        "--reject-dirty-baseline",
    ]
    if llm_fse:
        command.append("--llm-fse")
    else:
        command.insert(3, "--deterministic-fse")
    family_ids = [
        str(item).strip() for item in (curated_family_ids or []) if str(item).strip()
    ]
    disabled_injectors = disabled_injectors_for_run(
        workspace, family_ids, rcaeval_export_policy
    )
    low_yield = low_yield_injectors_for_run(
        workspace, family_ids, rcaeval_export_policy
    )
    if family_ids:
        command.extend(["--curated-family-ids", ",".join(family_ids)])
    disabled_csv = ",".join(disabled_injectors)
    disabled_index = command.index("--disabled-injectors") + 1
    low_yield_index = command.index("--low-yield-injectors") + 1
    command[disabled_index] = disabled_csv
    command[low_yield_index] = ",".join(low_yield)
    command.extend(
        candidate_pool_args(
            target_gold_count, max_iterations, max_injections_per_iteration
        )
    )
    return command


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--output-root")
    parser.add_argument("--dataset-dir")
    parser.add_argument("--target-gold-count", type=int, default=30)
    parser.add_argument("--max-iterations", type=int, default=10)
    parser.add_argument("--max-injections-per-iteration", type=int, default=3)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--skip-calibration-gate", action="store_true")
    parser.add_argument(
        "--scale-up-gold-threshold", type=int, default=DEFAULT_SCALE_UP_GOLD_THRESHOLD
    )
    parser.add_argument(
        "--min-admitted-families",
        type=int,
        default=DEFAULT_MIN_ADMITTED_FAMILIES_FOR_SCALE_UP,
    )
    parser.add_argument(
        "--canary-family",
        action="append",
        default=[],
        help="Curated fault family id to target during a runtime canary; may be repeated",
    )
    parser.add_argument(
        "--final-allowed-verdicts",
        default="REALISTIC,BORDERLINE,UNREALISTIC,INCONCLUSIVE",
        help="Comma-separated PRISM verdicts allowed through the final gate",
    )
    parser.add_argument(
        "--final-score-threshold",
        type=float,
        default=0.0,
        help="Minimum PRISM aggregate score for the final gate",
    )
    parser.add_argument(
        "--rcaeval-export-policy",
        choices=["gold", "scored", "all"],
        default="scored",
        help="Which tiered records to export to RCAEval layout",
    )
    parser.add_argument(
        "--llm-fse",
        action="store_true",
        help="Use LLM-guided schema-grounded FSE instead of deterministic enumeration",
    )
    args = parser.parse_args()
    workspace = Path(args.workspace).resolve()
    output_root = (
        Path(args.output_root).resolve()
        if args.output_root
        else workspace / "experiments" / "production_gold" / "run"
    )
    dataset_dir = (
        Path(args.dataset_dir).resolve()
        if args.dataset_dir
        else workspace / "datasets" / "rcaeval_export_gold"
    )
    runtime_sysdir = (
        prepare_runtime_system_description(workspace, output_root)
        if args.execute
        else workspace / "system_description"
    )
    if args.skip_calibration_gate:
        calibration_gate = {"enabled": False, "passed": True}
    else:
        try:
            calibration_gate = enforce_calibration_gate(
                workspace=workspace,
                target_gold_count=args.target_gold_count,
                scale_up_gold_threshold=args.scale_up_gold_threshold,
                min_admitted_families=args.min_admitted_families,
            )
        except CalibrationGateError as exc:
            calibration_gate = {"enabled": True, "passed": False, "error": str(exc)}
    family_ids = production_family_ids(
        calibration_gate=calibration_gate,
        canary_families=args.canary_family,
    )
    command = build_command(
        workspace=workspace,
        output_root=output_root,
        dataset_dir=dataset_dir,
        target_gold_count=args.target_gold_count,
        max_iterations=args.max_iterations,
        max_injections_per_iteration=args.max_injections_per_iteration,
        system_description_dir=runtime_sysdir,
        curated_family_ids=family_ids,
        final_allowed_verdicts=args.final_allowed_verdicts,
        final_score_threshold=args.final_score_threshold,
        rcaeval_export_policy=args.rcaeval_export_policy,
        llm_fse=args.llm_fse,
    )
    payload = {
        "execute": args.execute,
        "workspace": str(workspace),
        "output_root": str(output_root),
        "dataset_dir": str(dataset_dir),
        "canary_families": args.canary_family,
        "production_family_ids": family_ids,
        "command": command,
        "shell_command": " ".join(shlex.quote(part) for part in command),
        "calibration_gate": calibration_gate,
    }
    if not calibration_gate.get("passed"):
        print(json.dumps(payload, indent=2))
        return 2
    print(json.dumps(payload, indent=2))
    if not args.execute:
        return 0
    env = os.environ.copy()
    for key, value in load_env_file(workspace / ".env").items():
        env.setdefault(key, value)
    env["PYTHONPATH"] = build_pythonpath(workspace, env.get("PYTHONPATH"))
    proc = subprocess.run(
        command, cwd=workspace, env=env, check=False, stdin=subprocess.DEVNULL
    )
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
