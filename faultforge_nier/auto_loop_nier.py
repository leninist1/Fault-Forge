#!/usr/bin/env python3
"""FaultForge end-to-end auto-loop runner.

Workflow per iteration:
1) FSE explores fault space and generates a fresh fault catalog.
2) PRISM static precheck filters non-executable specs (no simulated observation).
3) Remaining faults are injected into Train-Ticket via BIFI in real time.
4) Injection observations are judged by PRISM using live observation only.
5) Accepted records are converted to RCAEval format.
6) Check dataset size against target; continue until target reached.

Example:
    python -m faultforge.auto_loop \
      --target-count 50 \
      --max-iterations 10 \
      --output-root faultforge_dataset/auto_loop_run
"""

from __future__ import annotations

import argparse
import copy
import glob
import json
import logging
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import subprocess
import sys
import time
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import requests

# Allow running as python -m faultforge.auto_loop from fault-injection/
HERE = Path(__file__).resolve().parent
FI_ROOT = HERE.parent
LEGACY_FI_ROOT = FI_ROOT.parent / "fault-injection"
if str(FI_ROOT) not in sys.path:
    sys.path.insert(0, str(FI_ROOT))
if LEGACY_FI_ROOT.exists() and str(LEGACY_FI_ROOT) not in sys.path:
    sys.path.insert(0, str(LEGACY_FI_ROOT))

from bifi import BIFIDispatcher
from convert_to_rcaeval import convert_faultforge_dataset
from fault_space_explorer.fse import FaultSpaceExplorer
from fault_space_explorer.llm_client import LLMClient, LLMConfig
from prism import PRISM
from prism.corpus.rag_retriever import BM25Retriever
from prism.invariants import InvariantRunner
from prism.invariants.invariant_runner import DBConfig
from prism.types import Verdict

logger = logging.getLogger("faultforge.auto_loop")

try:
    from faultforge_nier.admission_profiles import infer_admission_profile
    from faultforge_nier import http_client
    from faultforge_nier.convert_to_rcaeval_nier import convert_tiered_run
    from faultforge_nier.dataset_auditor import DatasetAuditor
    from faultforge_nier.fse_driver import (
        generate_candidates as generate_nier_candidates,
    )
    from faultforge_nier.llm_fse_adapter import run_llm_fse
    from faultforge_nier.observer import Observer, ObserverConfig
    from faultforge_nier.report_builder import write_business_modality_report
    from faultforge_nier.quality_gate import QualityThresholds, TierClassifier
    from faultforge_nier.telemetry_contract import (
        LATENCY_BUSINESS_SUFFIXES,
        PRIMARY_BUSINESS_SUFFIXES,
    )
    from faultforge_nier.telemetry_prism import dynamic_validate_telemetry
    from faultforge_nier.runtime_invariant_patch import apply_invariant_runner_patch
    from faultforge_nier.runtime_schema_patch import (
        apply_database_modifier_schema_patch,
    )
except (
    Exception
):  # pragma: no cover - keeps the copied reference runnable from unusual paths
    from faultforge.observer import Observer, ObserverConfig

    http_client = None
    convert_tiered_run = None
    DatasetAuditor = None
    generate_nier_candidates = None
    run_llm_fse = None
    write_business_modality_report = None
    QualityThresholds = None
    infer_admission_profile = None
    TierClassifier = None
    LATENCY_BUSINESS_SUFFIXES = ("_latency_ms_p95", "_latency_ms_p99")
    PRIMARY_BUSINESS_SUFFIXES = (
        "_success_rate",
        "_business_invalid_rate",
        "_timeout_rate",
        "_http_5xx_rate",
        "_http_4xx_rate",
        "_request_exception_rate",
        "_json_decode_error_rate",
        "_count",
        "_distribution_jsd",
    )
    dynamic_validate_telemetry = None
    apply_invariant_runner_patch = None
    apply_database_modifier_schema_patch = None

if apply_invariant_runner_patch is not None:
    apply_invariant_runner_patch()

if apply_database_modifier_schema_patch is not None:
    apply_database_modifier_schema_patch()

# BIFIDispatcher._run_business currently supports only these business injectors.
SUPPORTED_BUSINESS_INJECTORS = {
    "database_modifier",
    "config_modifier",
    "hybrid",
    "logic_fault",
}

# BIFIDispatcher._run_infra method_map keys.
SUPPORTED_INFRA_INJECTORS = {
    "docker_pause",
    "docker_stop",
    "docker_kill",
    "host_tc",
    "host_iptables",
    "resource_limit",
    "mysql_down",
    "mysql_slow",
    "redis_slow",
    "network_latency",
    "packet_loss",
    "network_partition",
    "cpu_high",
    "memory_leak",
    "docker_exec",
}


def _now_str() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )


def _llm_api_key(args: argparse.Namespace) -> Optional[str]:
    return (
        args.api_key
        or os.environ.get("DEEPSEEK_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
    )


def _canonicalize_nier_fault_spec(spec: Dict[str, Any]) -> Dict[str, Any]:
    """Adapt ASE NIER deterministic candidates to the original auto-loop schema."""
    out = copy.deepcopy(spec)
    params = out.setdefault("injector_params", {})
    fault_point = out.get("fault_point") or {}
    target_service = params.get("target_service") or fault_point.get("owner_service")
    target_table = params.get("target_table") or fault_point.get("table")
    target_field = params.get("target_field") or fault_point.get("field")

    if out.get("injector") == "database_modifier":
        params.setdefault("modify_type", "set")
        if params.get("condition") and not params.get("where_clause"):
            params["where_clause"] = params["condition"]
        if target_service == "ts-order-service" and target_table == "orders":
            out["target_invariant"] = "BK-3" if target_field == "status" else "BK-1"
            if target_field == "status":
                params["modify_value"] = 99
                params["condition"] = "status = 1"
                params["where_clause"] = "status = 1"
            elif target_field == "price":
                params["modify_value"] = -1
        elif target_service == "ts-price-service":
            out["target_invariant"] = "BK-7"
            if target_field in {"price", "basicPriceRate", "basic_price_rate"}:
                params["target_field"] = "basic_price_rate"
                params["modify_value"] = -1

    if out.get("injector") == "resource_limit":
        params.setdefault("cpu_quota", 20000)
        params.pop("target_db", None)

    if out.get("injector") == "host_iptables":
        params.pop("target_db", None)
        params.setdefault("direction", "both")
        params.setdefault("action", "DROP")

    if out.get("injector") == "host_tc":
        params.pop("target_db", None)
        params.setdefault("direction", "both")
        if not params.get("latency_ms") and not params.get("loss_percent"):
            params.setdefault("latency_ms", 500)

    if out.get("injector") == "mysql_slow":
        params.setdefault("modify_type", "set")
        if not params.get("latency_ms"):
            params.setdefault("latency_ms", 500)

    if out.get("injector") == "redis_slow":
        params.pop("target_db", None)
        params.setdefault(
            "target_service", params.get("target_service", "ts-order-service")
        )
        if not params.get("latency_ms"):
            params.setdefault("latency_ms", 300)

    if out.get("injector") == "config_modifier":
        params.setdefault("modify_type", "set")
        if params.get("condition") and not params.get("where_clause"):
            params["where_clause"] = params["condition"]
        config_key = params.get("target_field") or params.get("config_key") or ""
        if config_key and not params.get("config_key"):
            params["config_key"] = config_key
        if config_key and not params.get("target_config"):
            params["target_config"] = config_key
        new_val = params.get("modify_value")
        if new_val is not None:
            if not params.get("new_value"):
                params["new_value"] = str(new_val)
            if not params.get("injected_value"):
                params["injected_value"] = str(new_val)
            if not params.get("value"):
                params["value"] = str(new_val)

    target_invariant = out.get("target_invariant")
    if target_invariant and not out.get("expected_invariant_violations"):
        out["expected_invariant_violations"] = [target_invariant]
    if out.get("expected_observable_signals") and not out.get("observable_signals"):
        out["observable_signals"] = out["expected_observable_signals"]
    if isinstance(out.get("expected_propagation"), list):
        out["expected_propagation"] = {
            "services": out["expected_propagation"],
            "propagation_type": "III",
        }
    return out


def _fault_signature(spec: Dict[str, Any]) -> str:
    """Return a stable signature for deduping semantically same faults."""
    payload = {
        "dimension": spec.get("dimension"),
        "injector": spec.get("injector"),
        "fault_point": spec.get("fault_point", {}),
        "injector_params": spec.get("injector_params", {}),
        "expected_invariant_violations": spec.get("expected_invariant_violations", []),
    }
    return json.dumps(payload, sort_keys=True, ensure_ascii=False)


def _supports_bifi(
    spec: Dict[str, Any], disabled_injectors: Optional[Set[str]] = None
) -> bool:
    injector = str(spec.get("injector", "")).lower()
    if disabled_injectors and injector in disabled_injectors:
        return False
    return (
        injector in SUPPORTED_BUSINESS_INJECTORS
        or injector in SUPPORTED_INFRA_INJECTORS
    )


def _count_rcaeval_faults(dataset_dir: Path) -> int:
    if not dataset_dir.exists():
        return 0
    audit_dir = dataset_dir / "audit"
    if audit_dir.exists():
        return sum(1 for path in audit_dir.glob("*/metadata.json"))
    rca_inputs = dataset_dir / "rca_inputs"
    if rca_inputs.exists():
        return sum(1 for child in rca_inputs.iterdir() if child.is_dir())
    return sum(
        1
        for child in dataset_dir.iterdir()
        if child.is_dir() and (child / "metadata.json").exists()
    )


def _parse_verdicts(raw: str) -> Set[str]:
    return {x.strip().upper() for x in raw.split(",") if x.strip()}


def _parse_csv_lower(raw: str) -> Set[str]:
    return {x.strip().lower() for x in raw.split(",") if x.strip()}


def _parse_csv(raw: str) -> Set[str]:
    return {x.strip() for x in raw.split(",") if x.strip()}


@dataclass
class IterationStats:
    iteration: int
    catalog_total: int
    deduped_total: int
    supported_total: int
    static_selected: int
    injected_total: int
    accepted_total: int
    dataset_total: int
    elapsed_sec: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "iteration": self.iteration,
            "catalog_total": self.catalog_total,
            "deduped_total": self.deduped_total,
            "supported_total": self.supported_total,
            "static_selected": self.static_selected,
            "injected_total": self.injected_total,
            "accepted_total": self.accepted_total,
            "dataset_total": self.dataset_total,
            "elapsed_sec": round(self.elapsed_sec, 1),
        }


@dataclass
class InjectorHealthStats:
    attempts: int = 0
    bifi_successes: int = 0
    accepted: int = 0
    consecutive_failures: int = 0
    consecutive_rejections: int = 0
    disabled_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "attempts": self.attempts,
            "bifi_successes": self.bifi_successes,
            "accepted": self.accepted,
            "consecutive_failures": self.consecutive_failures,
            "consecutive_rejections": self.consecutive_rejections,
            "disabled_reason": self.disabled_reason,
        }


class AutoLoopRunner:
    def __init__(self, args: argparse.Namespace):
        self._apply_runtime_profile(args)
        self.args = args

        self.output_root = Path(args.output_root).resolve()
        self.output_root.mkdir(parents=True, exist_ok=True)

        self.runs_dir = self.output_root / "iterations"
        self.runs_dir.mkdir(parents=True, exist_ok=True)

        if args.dataset_dir:
            self.dataset_dir = Path(args.dataset_dir).resolve()
        else:
            self.dataset_dir = (self.output_root / "rcaeval_dataset").resolve()
        self.dataset_dir.mkdir(parents=True, exist_ok=True)

        self.system_description_dir = Path(args.system_description)
        if not self.system_description_dir.is_absolute():
            self.system_description_dir = (
                FI_ROOT / self.system_description_dir
            ).resolve()

        self.corpus_path = Path(args.corpus)
        if not self.corpus_path.is_absolute():
            self.corpus_path = (FI_ROOT / self.corpus_path).resolve()
            if not self.corpus_path.exists() and LEGACY_FI_ROOT.exists():
                legacy_corpus_path = (LEGACY_FI_ROOT / args.corpus).resolve()
                if legacy_corpus_path.exists():
                    self.corpus_path = legacy_corpus_path

        self.final_allow = _parse_verdicts(args.final_allowed_verdicts)
        self.disabled_injectors = _parse_csv_lower(args.disabled_injectors)
        self.low_yield_injectors = _parse_csv_lower(args.low_yield_injectors)
        self.precheck_reject_warning_prefixes = _parse_csv(
            args.precheck_reject_warning_prefixes
        )
        self.logic_fault_penalty = max(0.0, min(1.0, float(args.logic_fault_penalty)))
        self.max_logic_fault_ratio = max(
            0.0, min(1.0, float(args.max_logic_fault_ratio))
        )
        self._accepted_service_counter: Counter[str] = Counter()
        self._accepted_dimension_counter: Counter[str] = Counter()
        self._accepted_injector_counter: Counter[str] = Counter()
        self._accepted_family_counter: Counter[str] = Counter()
        self._load_existing_distribution()

        api_key = _llm_api_key(args)
        if api_key:
            llm_cfg = LLMConfig(
                api_key=api_key,
                base_url=args.base_url,
                model=args.model,
                temperature=args.temperature,
            )
            self.llm = LLMClient(llm_cfg)
        else:
            logger.info(
                "No LLM API key configured; using deterministic ASE NIER FSE and PRISM heuristics."
            )
            self.llm = None

        self.invariant_runner = InvariantRunner(
            self.system_description_dir,
            db_config=DBConfig(),
        )
        self.retriever = BM25Retriever(self.corpus_path)
        self.observer = Observer(
            ObserverConfig(
                gateway_url=args.gateway_url,
                traffic_probes_per_window=args.traffic_probes,
            ),
            system_description_dir=self.system_description_dir,
        )

        self.dispatcher = BIFIDispatcher(
            baseline_seconds=args.baseline_seconds,
            fault_seconds=args.fault_seconds,
            cooldown_seconds=args.cooldown_seconds,
            invariant_runner=self.invariant_runner,
            observer=self.observer,
            collect_raw_fault_data=args.collect_raw_fault_data,
            fault_injector_factory=self._fault_injector_factory,
        )

        self.prism = PRISM(
            system_description_dir=str(self.system_description_dir),
            llm=self.llm,
            retriever=self.retriever,
            invariant_runner=self.invariant_runner,
        )

        self.seen_signatures: Set[str] = set()
        self.candidate_pool: List[Dict[str, Any]] = []
        self.fse_pool_dir = self.output_root / "fse_pool"
        self.fse_pool_dir.mkdir(parents=True, exist_ok=True)
        self._fse_batch_index = 0
        self.iteration_summaries: List[Dict[str, Any]] = []
        self._recovery_fault_injector = None
        self._gateway_last_recover_ts = 0.0
        self._gateway_grace_until_ts = 0.0
        self.injector_runtime_stats: Dict[str, InjectorHealthStats] = {}
        self._load_existing_iteration_summaries()

    def _resolve_feedback_run_roots(self) -> List[Path]:
        """Collect prior run roots whose summary/audit can guide LLM-FSE."""
        if not getattr(self.args, "feedback_loop", True):
            return []
        roots: List[Path] = []
        seen: Set[str] = set()

        def add(root: Path) -> None:
            root = Path(root).resolve()
            key = str(root)
            if key in seen:
                return
            # Only keep roots with at least one usable feedback artifact.
            has_feedback = (
                (root / "final_summary.json").exists()
                or (root / "reports" / "dataset_quality_report.json").exists()
            )
            if not has_feedback:
                return
            seen.add(key)
            roots.append(root)

        # Resume of the same run can reuse its previous summary/report.
        add(self.output_root)

        # Auto-discover sibling runs under experiments/e123_llm_fse/*/run.
        try:
            run_group_root = self.output_root.parents[1]
        except IndexError:
            run_group_root = self.output_root.parent
        sibling_runs = sorted(
            [p for p in run_group_root.glob("*/run") if p.is_dir() and p.resolve() != self.output_root],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for root in sibling_runs:
            add(root)
            if len(roots) >= int(self.args.feedback_max_runs):
                break

        # Optional explicit feedback run glob (absolute or relative patterns).
        pattern = str(getattr(self.args, "feedback_run_glob", "") or "").strip()
        if pattern:
            for root in sorted(HERE.parent.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True):
                if root.is_dir():
                    add(root)
                if len(roots) >= int(self.args.feedback_max_runs):
                    break
        return roots[: int(self.args.feedback_max_runs)]

    def _fault_injector_factory(self) -> Any:
        from fault_injector import FaultInjector

        return FaultInjector(
            config_path=str((LEGACY_FI_ROOT / "fault-config-enhanced.yml").resolve())
        )

    @staticmethod
    def _apply_runtime_profile(args: argparse.Namespace) -> None:
        profile = str(
            getattr(args, "runtime_profile", "standard") or "standard"
        ).lower()
        if profile != "throughput":
            return

        if getattr(args, "production_gold", False):
            args.traffic_probes = max(
                int(args.traffic_probes),
                int(getattr(args, "business_min_probe_samples", 10)),
            )
        else:
            args.traffic_probes = min(int(args.traffic_probes), 4)
        args.baseline_seconds = min(int(args.baseline_seconds), 10)
        args.fault_seconds = min(int(args.fault_seconds), 20)
        args.cooldown_seconds = min(int(args.cooldown_seconds), 5)
        args.fault_seconds_min = min(int(args.fault_seconds_min), 8)
        args.cooldown_seconds_min = min(int(args.cooldown_seconds_min), 2)
        args.health_check_every_n_injections = max(
            int(args.health_check_every_n_injections),
            10,
        )
        args.collect_raw_fault_data = bool(getattr(args, "production_gold", False))
        args.skip_prism_on_bifi_failure = True
        args.skip_prism_on_no_dynamic_observation = True
        args.static_precheck_workers = max(int(args.static_precheck_workers), 16)
        args.precheck_max_warnings = 0
        if not getattr(args, "precheck_reject_warning_prefixes", ""):
            args.precheck_reject_warning_prefixes = (
                "missing_expected_invariants,target_table_unusual_for_service"
            )

    def run(self) -> int:
        start = time.time()
        current_total = _count_rcaeval_faults(self.dataset_dir)

        logger.info("=" * 72)
        logger.info("FaultForge auto loop start")
        logger.info("Output root: %s", self.output_root)
        logger.info("RCAEval dataset dir: %s", self.dataset_dir)
        logger.info(
            "Runtime profile: %s (baseline=%ss fault=%ss cooldown=%ss probes=%s raw_fault_data=%s)",
            self.args.runtime_profile,
            self.args.baseline_seconds,
            self.args.fault_seconds,
            self.args.cooldown_seconds,
            self.args.traffic_probes,
            self.args.collect_raw_fault_data,
        )
        logger.info("Initial dataset count: %d", current_total)
        logger.info("Target dataset count: %d", self.args.target_count)
        logger.info("=" * 72)

        if current_total >= self.args.target_count:
            final = {
                "target_count": self.args.target_count,
                "final_dataset_count": current_total,
                "reached_target": True,
                "max_iterations": self.args.max_iterations,
                "elapsed_sec": round(time.time() - start, 1),
                "iterations": self.iteration_summaries,
                "injector_health": self._injector_health_report(),
            }
            _write_json(self.output_root / "final_summary.json", final)
            logger.info(
                "Initial dataset count already reached target; skip FSE/injection."
            )
            logger.info("Summary: %s", self.output_root / "final_summary.json")
            return 0

        if self.args.candidate_pool_seed_glob:
            seed_stats = self._load_seed_catalogs(self.args.candidate_pool_seed_glob)
            _write_json(self.output_root / "seed_pool_stats.json", seed_stats)
            logger.info(
                "Seeded candidate pool: files=%d catalog=%d deduped=%d supported=%d added=%d pool_size=%d",
                seed_stats["files_scanned"],
                seed_stats["catalog_total"],
                seed_stats["deduped_total"],
                seed_stats["supported_total"],
                seed_stats["added_to_pool"],
                len(self.candidate_pool),
            )

        if self.args.candidate_pool_prefetch > 0:
            prefetch_need = max(
                0, int(self.args.candidate_pool_prefetch) - len(self.candidate_pool)
            )
        else:
            prefetch_need = 0
        if prefetch_need > 0:
            self._fill_candidate_pool(
                min_new=prefetch_need,
                reason="initial_prefetch",
            )

        for iteration in range(1, self.args.max_iterations + 1):
            if current_total >= self.args.target_count:
                break

            iter_start = time.time()
            remaining = self.args.target_count - current_total
            iter_dir = self.runs_dir / f"iter_{iteration:03d}_{_now_str()}"
            eval_out = iter_dir / "evaluation"
            eval_out.mkdir(parents=True, exist_ok=True)
            accepted_dir = iter_dir / "accepted_records"
            (accepted_dir / "per_fault").mkdir(parents=True, exist_ok=True)
            tiered_dir = iter_dir / "tiered_records"
            for tier in ("gold", "candidate", "rejected"):
                (tiered_dir / tier / "per_fault").mkdir(parents=True, exist_ok=True)

            logger.info("[Iter %d] Remaining target=%d", iteration, remaining)

            if not self._ensure_system_healthy(f"iter_{iteration:03d}_preflight"):
                logger.error(
                    "[Iter %d] System unhealthy before FSE/injection, aborting auto-loop.",
                    iteration,
                )
                break

            refill_needed = max(
                self.args.candidate_pool_min,
                self.args.max_candidates_per_iteration
                if self.args.max_candidates_per_iteration > 0
                else 1,
            )
            fse_stats = {
                "catalog_total": 0,
                "deduped_total": 0,
                "supported_total": 0,
                "added_to_pool": 0,
                "pool_before_refill": len(self.candidate_pool),
            }
            if len(self.candidate_pool) < refill_needed:
                refill_stats = self._fill_candidate_pool(
                    min_new=self.args.candidate_pool_refill,
                    reason=f"iter_{iteration:03d}_refill",
                )
                for k in (
                    "catalog_total",
                    "deduped_total",
                    "supported_total",
                    "added_to_pool",
                ):
                    fse_stats[k] += refill_stats.get(k, 0)
            fse_stats["pool_after_refill"] = len(self.candidate_pool)

            if not self.candidate_pool:
                logger.error(
                    "[Iter %d] candidate pool is empty after refill; aborting auto-loop.",
                    iteration,
                )
                break

            if self.args.max_candidates_per_iteration > 0:
                take_n = min(
                    self.args.max_candidates_per_iteration, len(self.candidate_pool)
                )
            else:
                take_n = len(self.candidate_pool)
            supported = self._select_candidates(take_n)
            supported_total = len(supported)
            catalog_total = fse_stats["catalog_total"]
            deduped_total = fse_stats["deduped_total"]

            _write_json(iter_dir / "pool_stats.json", fse_stats)

            prechecked = self._static_precheck(supported, eval_out)
            static_selected = len(prechecked)

            if self.args.max_injections_per_iteration <= 0:
                inject_limit = static_selected
            elif self.args.rcaeval_export_policy == "scored":
                inject_limit = min(
                    self.args.max_injections_per_iteration,
                    static_selected,
                )
            else:
                inject_limit = min(
                    remaining,
                    self.args.max_injections_per_iteration,
                    static_selected,
                )
            selected = prechecked[:inject_limit]

            accepted_count = self._inject_and_collect(
                iteration=iteration,
                selected_specs=selected,
                accepted_dir=accepted_dir,
                tiered_dir=tiered_dir,
                eval_dir=eval_out,
            )

            if len(selected) > 0:
                if convert_tiered_run is not None:
                    convert_tiered_run(
                        iter_dir,
                        self.dataset_dir,
                        run_id=iter_dir.name,
                        export_policy=self.args.rcaeval_export_policy,
                    )
                else:
                    if accepted_count > 0:
                        convert_faultforge_dataset(
                            accepted_dir, self.dataset_dir, format_type=self.args.format
                        )

            current_total = _count_rcaeval_faults(self.dataset_dir)

            stats = IterationStats(
                iteration=iteration,
                catalog_total=catalog_total,
                deduped_total=deduped_total,
                supported_total=supported_total,
                static_selected=static_selected,
                injected_total=len(selected),
                accepted_total=accepted_count,
                dataset_total=current_total,
                elapsed_sec=time.time() - iter_start,
            )
            self.iteration_summaries.append(stats.to_dict())
            _write_json(iter_dir / "iteration_summary.json", stats.to_dict())

            logger.info(
                "[Iter %d] pool_before=%d pool_after_refill=%d pool_after_consume=%d catalog=%d deduped=%d supported=%d static_selected=%d injected=%d accepted=%d dataset_total=%d",
                iteration,
                fse_stats["pool_before_refill"],
                fse_stats.get("pool_after_refill", 0),
                len(self.candidate_pool),
                catalog_total,
                deduped_total,
                supported_total,
                static_selected,
                len(selected),
                accepted_count,
                current_total,
            )

        final = {
            "target_count": self.args.target_count,
            "final_dataset_count": current_total,
            "reached_target": current_total >= self.args.target_count,
            "max_iterations": self.args.max_iterations,
            "elapsed_sec": round(time.time() - start, 1),
            "iterations": self.iteration_summaries,
            "injector_health": self._injector_health_report(),
        }
        if DatasetAuditor is not None:
            report_dir = (
                Path(self.args.quality_report_dir).resolve()
                if self.args.quality_report_dir
                else self.output_root / "reports"
            )
            try:
                auditor = DatasetAuditor(self.output_root, self.dataset_dir, report_dir)
                audit_report = auditor.audit()
                auditor.write_reports(audit_report)
                if write_business_modality_report is not None:
                    write_business_modality_report(self.output_root, report_dir)
                final["dataset_audit"] = {
                    "status": audit_report.status,
                    "report_path": str(report_dir / "dataset_quality_report.json"),
                }
            except Exception as exc:  # pylint: disable=broad-except
                final["dataset_audit"] = {"status": "error", "error": str(exc)}

        _write_json(self.output_root / "final_summary.json", final)

        logger.info("=" * 72)
        logger.info(
            "Auto loop finished. reached_target=%s final=%d target=%d elapsed=%.1fs",
            final["reached_target"],
            current_total,
            self.args.target_count,
            final["elapsed_sec"],
        )
        logger.info("Summary: %s", self.output_root / "final_summary.json")
        logger.info("=" * 72)

        return 0 if final["reached_target"] else 2

    def _load_existing_distribution(self) -> None:
        if not self.dataset_dir.exists():
            return
        metadata_paths = sorted((self.dataset_dir / "audit").glob("*/metadata.json"))
        if not metadata_paths:
            metadata_paths = [
                child / "metadata.json"
                for child in self.dataset_dir.iterdir()
                if child.is_dir() and (child / "metadata.json").exists()
            ]
        for meta in metadata_paths:
            try:
                payload = json.loads(meta.read_text(encoding="utf-8"))
            except Exception:  # pylint: disable=broad-except
                continue
            dim = str(payload.get("dimension") or "")
            svc = str(payload.get("target_service") or "")
            inj = str(payload.get("injector") or "").lower()
            fam = str((payload.get("fse_metadata") or {}).get("family_id") or "")
            if dim:
                self._accepted_dimension_counter[dim] += 1
            if svc:
                self._accepted_service_counter[svc] += 1
            if inj:
                self._accepted_injector_counter[inj] += 1
            if fam:
                self._accepted_family_counter[fam] += 1

    def _load_existing_iteration_summaries(self) -> None:
        summaries: List[Dict[str, Any]] = []
        for summary_path in sorted(self.runs_dir.glob("*/iteration_summary.json")):
            try:
                payload = json.loads(summary_path.read_text(encoding="utf-8"))
            except Exception:  # pylint: disable=broad-except
                continue
            if isinstance(payload, dict):
                summaries.append(payload)
        self.iteration_summaries = summaries

    def _injector_health_report(self) -> Dict[str, Any]:
        return {
            injector: stats.to_dict()
            for injector, stats in sorted(self.injector_runtime_stats.items())
        }

    def _injector_is_runtime_disabled(self, injector: str) -> bool:
        stats = self.injector_runtime_stats.get(injector)
        return bool(stats and stats.disabled_reason)

    def _record_injector_outcome(
        self,
        *,
        injector: str,
        bifi_succeeded: bool,
        accepted: bool,
    ) -> None:
        if not injector:
            return
        stats = self.injector_runtime_stats.setdefault(injector, InjectorHealthStats())
        stats.attempts += 1
        if bifi_succeeded:
            stats.bifi_successes += 1
            stats.consecutive_failures = 0
        else:
            stats.consecutive_failures += 1

        if accepted:
            stats.accepted += 1
            stats.consecutive_rejections = 0
        elif bifi_succeeded:
            stats.consecutive_rejections += 1
        else:
            # Dataset rejection is only meaningful when BIFI actually executed
            # and produced an observation. Injection failures are tracked by the
            # failure breaker and should not poison the rejection streak.
            stats.consecutive_rejections = 0

        min_attempts = max(1, int(self.args.injector_disable_min_attempts))
        if stats.attempts < min_attempts or stats.disabled_reason:
            return

        if (
            self.args.injector_disable_after_consecutive_failures > 0
            and stats.consecutive_failures
            >= int(self.args.injector_disable_after_consecutive_failures)
        ):
            stats.disabled_reason = f"consecutive_bifi_failures>={self.args.injector_disable_after_consecutive_failures}"
            return

        if (
            injector in self.low_yield_injectors
            and stats.bifi_successes > 0
            and self.args.injector_disable_after_consecutive_rejections > 0
            and stats.consecutive_rejections
            >= int(self.args.injector_disable_after_consecutive_rejections)
            and stats.accepted == 0
        ):
            stats.disabled_reason = (
                "consecutive_dataset_rejections"
                f">={self.args.injector_disable_after_consecutive_rejections}"
            )
            return

        success_rate = stats.bifi_successes / float(max(1, stats.attempts))
        accept_rate = stats.accepted / float(max(1, stats.attempts))
        if success_rate < float(
            self.args.injector_min_bifi_success_rate
        ) and accept_rate <= float(self.args.injector_min_accept_rate):
            stats.disabled_reason = (
                "low_yield:"
                f"bifi_success_rate={success_rate:.3f},accept_rate={accept_rate:.3f}"
            )

    def _fill_candidate_pool(self, *, min_new: int, reason: str) -> Dict[str, int]:
        """Fill candidate pool with unique+BIFI-supported faults from one or more FSE batches."""
        min_new = max(0, int(min_new))
        stats = {
            "catalog_total": 0,
            "deduped_total": 0,
            "supported_total": 0,
            "added_to_pool": 0,
        }
        if min_new <= 0:
            return stats

        rounds = 0
        while (
            stats["added_to_pool"] < min_new
            and rounds < self.args.candidate_pool_refill_rounds
        ):
            rounds += 1
            self._fse_batch_index += 1
            batch_dir = (
                self.fse_pool_dir / f"batch_{self._fse_batch_index:04d}_{_now_str()}"
            )
            faults = self._run_fse(batch_dir)

            batch_catalog = len(faults)
            batch_deduped = 0
            batch_supported = 0
            batch_added = 0

            for spec in faults:
                sig = _fault_signature(spec)
                if sig in self.seen_signatures:
                    continue
                self.seen_signatures.add(sig)
                batch_deduped += 1
                injector = str(spec.get("injector") or "").lower()
                if not _supports_bifi(spec, self.disabled_injectors):
                    continue
                if self._injector_is_runtime_disabled(injector):
                    continue
                batch_supported += 1
                self.candidate_pool.append(spec)
                batch_added += 1

            batch_summary = {
                "reason": reason,
                "batch_index": self._fse_batch_index,
                "catalog_total": batch_catalog,
                "deduped_total": batch_deduped,
                "supported_total": batch_supported,
                "added_to_pool": batch_added,
                "pool_size_after_batch": len(self.candidate_pool),
            }
            _write_json(batch_dir / "batch_summary.json", batch_summary)

            stats["catalog_total"] += batch_catalog
            stats["deduped_total"] += batch_deduped
            stats["supported_total"] += batch_supported
            stats["added_to_pool"] += batch_added

            logger.info(
                "[PoolFill:%s] batch=%d catalog=%d deduped=%d supported=%d added=%d pool_size=%d",
                reason,
                self._fse_batch_index,
                batch_catalog,
                batch_deduped,
                batch_supported,
                batch_added,
                len(self.candidate_pool),
            )

            if batch_catalog == 0:
                break

        logger.info(
            "[PoolFill:%s] done rounds=%d added=%d target=%d pool_size=%d",
            reason,
            rounds,
            stats["added_to_pool"],
            min_new,
            len(self.candidate_pool),
        )
        return stats

    def _load_seed_catalogs(self, pattern: str) -> Dict[str, int]:
        stats = {
            "files_scanned": 0,
            "catalog_total": 0,
            "deduped_total": 0,
            "supported_total": 0,
            "added_to_pool": 0,
        }
        paths = sorted(glob.glob(pattern, recursive=True))
        stats["files_scanned"] = len(paths)
        for fp in paths:
            p = Path(fp)
            try:
                payload = json.loads(p.read_text(encoding="utf-8"))
            except Exception as exc:  # pylint: disable=broad-except
                logger.warning("Skip invalid seed catalog %s: %s", p, exc)
                continue
            faults = payload.get("faults", [])
            stats["catalog_total"] += len(faults)
            for spec in faults:
                sig = _fault_signature(spec)
                if sig in self.seen_signatures:
                    continue
                self.seen_signatures.add(sig)
                stats["deduped_total"] += 1
                injector = str(spec.get("injector") or "").lower()
                if not _supports_bifi(spec, self.disabled_injectors):
                    continue
                if self._injector_is_runtime_disabled(injector):
                    continue
                stats["supported_total"] += 1
                self.candidate_pool.append(spec)
                stats["added_to_pool"] += 1
        return stats

    def _service_of_spec(self, spec: Dict[str, Any]) -> str:
        fp = spec.get("fault_point") or {}
        ip = spec.get("injector_params") or {}
        return str(ip.get("target_service") or fp.get("owner_service") or "unknown")

    def _table_of_spec(self, spec: Dict[str, Any]) -> str:
        fp = spec.get("fault_point") or {}
        ip = spec.get("injector_params") or {}
        return str(ip.get("target_table") or fp.get("table") or "unknown")

    def _field_of_spec(self, spec: Dict[str, Any]) -> str:
        fp = spec.get("fault_point") or {}
        ip = spec.get("injector_params") or {}
        return str(ip.get("target_field") or fp.get("field") or "unknown")

    def _dimension_of_spec(self, spec: Dict[str, Any]) -> str:
        return str(spec.get("dimension") or "unknown")

    def _spec_priority(self, spec: Dict[str, Any]) -> float:
        injector = str(spec.get("injector") or "").lower()
        service = self._service_of_spec(spec)
        table = self._table_of_spec(spec)
        field = self._field_of_spec(spec)
        dimension = self._dimension_of_spec(spec)
        metadata = spec.get("fse_metadata") or {}
        family = str(metadata.get("family_id") or "")
        score = 1.0
        if infer_admission_profile is not None:
            profile = infer_admission_profile({"fault_spec": spec})
            if profile.name == "infra":
                score *= 1.35
            elif profile.name in {"semantic_business", "config_business"}:
                score *= 1.15
        expected_signals = spec.get("expected_observable_signals") or {}
        expected_slis = (
            expected_signals.get("business_slis")
            or spec.get("expected_business_slis")
            or []
        )
        if expected_slis:
            score *= min(2.0, 1.0 + 0.20 * len(expected_slis))
        if metadata.get("generation_source") == "feedback_exploration":
            severity = float(metadata.get("exploration_severity") or 1.0)
            score *= 1.0 + 0.10 * min(severity, 4.0)
        if metadata.get("scaleup_repeat_index") is not None:
            score *= 0.1
        if injector in self.low_yield_injectors:
            score *= 0.4
        health = getattr(self, "injector_runtime_stats", {}).get(injector)
        if health and health.attempts > 0:
            bifi_rate = health.bifi_successes / float(max(1, health.attempts))
            accept_rate = health.accepted / float(max(1, health.attempts))
            score *= 0.50 + bifi_rate
            score *= 0.75 + accept_rate
        if injector == "logic_fault":
            score *= self.logic_fault_penalty
        if injector == "database_modifier":
            if (
                service == "ts-order-service"
                and table == "orders"
                and field == "status"
            ):
                score *= 3.0
            if service == "ts-price-service" and table in {"price_config", "prices"}:
                score *= 0.25
        score *= 1.0 / (1.0 + float(self._accepted_service_counter.get(service, 0)))
        score *= 1.0 / (
            1.0 + 0.5 * float(self._accepted_dimension_counter.get(dimension, 0))
        )
        score *= 1.0 / (1.0 + float(self._accepted_injector_counter.get(injector, 0)))
        if family:
            score *= 1.0 / (1.0 + float(self._accepted_family_counter.get(family, 0)))
        return score

    def _select_candidates(self, take_n: int) -> List[Dict[str, Any]]:
        if take_n <= 0 or not self.candidate_pool:
            return []
        ranked = sorted(self.candidate_pool, key=self._spec_priority, reverse=True)
        selected: List[Dict[str, Any]] = []
        service_seen: Counter[str] = Counter()
        dim_seen: Counter[str] = Counter()
        injector_seen: Counter[str] = Counter()
        family_seen: Counter[str] = Counter()
        logic_count = 0
        max_per_service = max(0, int(self.args.max_per_service_per_iteration))
        max_per_dimension = max(0, int(self.args.max_per_dimension_per_iteration))
        max_per_injector = max(
            0, int(getattr(self.args, "max_per_injector_per_iteration", 0))
        )
        max_per_family = max(
            0, int(getattr(self.args, "max_per_family_per_iteration", 0))
        )
        for spec in ranked:
            if len(selected) >= take_n:
                break
            injector = str(spec.get("injector") or "").lower()
            if self._injector_is_runtime_disabled(injector):
                continue
            service = self._service_of_spec(spec)
            dimension = self._dimension_of_spec(spec)
            family = str((spec.get("fse_metadata") or {}).get("family_id") or "")
            if max_per_service > 0 and service_seen[service] >= max_per_service:
                continue
            if max_per_dimension > 0 and dim_seen[dimension] >= max_per_dimension:
                continue
            if max_per_injector > 0 and injector_seen[injector] >= max_per_injector:
                continue
            if max_per_family > 0 and family and family_seen[family] >= max_per_family:
                continue
            if injector == "logic_fault":
                next_ratio = float(logic_count + 1) / float(max(1, len(selected) + 1))
                if next_ratio > self.max_logic_fault_ratio:
                    continue
            selected.append(spec)
            service_seen[service] += 1
            dim_seen[dimension] += 1
            injector_seen[injector] += 1
            if family:
                family_seen[family] += 1
            if injector == "logic_fault":
                logic_count += 1
        if len(selected) < take_n:
            relaxed_caps = (
                max_per_service <= 0
                and max_per_dimension <= 0
                and max_per_injector <= 0
                and max_per_family <= 0
            )
            if not relaxed_caps:
                logger.warning(
                    "Diversity caps active but quota-aware selection only chose %d/%d; "
                    "filling remaining slots with soft diversity limits (2x caps)",
                    len(selected),
                    take_n,
                )
            soft_max_service = max_per_service * 2 if max_per_service > 0 else 0
            soft_max_dimension = max_per_dimension * 2 if max_per_dimension > 0 else 0
            soft_max_injector = max_per_injector * 2 if max_per_injector > 0 else 0
            soft_max_family = max_per_family * 2 if max_per_family > 0 else 0
            selected_ids = {id(s) for s in selected}
            for spec in ranked:
                if len(selected) >= take_n:
                    break
                if id(spec) in selected_ids:
                    continue
                s_injector = str(spec.get("injector") or "").lower()
                if self._injector_is_runtime_disabled(s_injector):
                    continue
                s_service = self._service_of_spec(spec)
                s_dimension = self._dimension_of_spec(spec)
                s_family = str((spec.get("fse_metadata") or {}).get("family_id") or "")
                if soft_max_service > 0 and service_seen[s_service] >= soft_max_service:
                    continue
                if (
                    soft_max_dimension > 0
                    and dim_seen[s_dimension] >= soft_max_dimension
                ):
                    continue
                if (
                    soft_max_injector > 0
                    and injector_seen[s_injector] >= soft_max_injector
                ):
                    continue
                if (
                    soft_max_family > 0
                    and s_family
                    and family_seen[s_family] >= soft_max_family
                ):
                    continue
                if s_injector == "logic_fault":
                    next_ratio = float(logic_count + 1) / float(
                        max(1, len(selected) + 1)
                    )
                    if next_ratio > self.max_logic_fault_ratio:
                        continue
                selected.append(spec)
                selected_ids.add(id(spec))
                service_seen[s_service] += 1
                dim_seen[s_dimension] += 1
                injector_seen[s_injector] += 1
                if s_family:
                    family_seen[s_family] += 1
                if s_injector == "logic_fault":
                    logic_count += 1
        selected_ids = {id(s) for s in selected}
        self.candidate_pool = [
            s for s in self.candidate_pool if id(s) not in selected_ids
        ]
        return selected

    def _business_signal_gate(
        self,
        *,
        baseline_slis: Dict[str, Any],
        fault_slis: Dict[str, Any],
    ) -> Tuple[bool, Dict[str, Any]]:
        # Gate1: minimum traffic volume for core probes
        core = ("login", "trip_search", "contacts_fetch", "booking_precheck")
        min_volume = max(1, int(self.args.business_min_probe_samples))
        details: Dict[str, Any] = {"reasons": []}
        volumes = {
            name: int(
                max(
                    float(baseline_slis.get(f"{name}_sample_count", 0.0) or 0.0),
                    float(fault_slis.get(f"{name}_sample_count", 0.0) or 0.0),
                )
            )
            for name in core
        }
        too_low = [name for name, c in volumes.items() if c < min_volume]
        if too_low:
            details["reasons"].append(f"min_volume_gate_failed:{','.join(too_low)}")

        # Gate2: suspicious all-zero core success rates.
        def _rate(d: Dict[str, Any], n: str) -> float:
            return float(d.get(f"{n}_success_rate", 0.0) or 0.0)

        all_zero_baseline = all(_rate(baseline_slis, n) <= 1e-6 for n in core)
        all_zero_fault = all(_rate(fault_slis, n) <= 1e-6 for n in core)
        if all_zero_baseline and all_zero_fault:
            details["reasons"].append("telemetry_suspect_all_core_zero")

        # Gate3: suspicious positive improvement during fault window.
        max_positive = float(self.args.max_positive_sli_delta)
        positive_keys = []
        for n in core:
            delta = _rate(fault_slis, n) - _rate(baseline_slis, n)
            if delta > max_positive:
                positive_keys.append(f"{n}:{delta:.3f}")
        if positive_keys:
            details["reasons"].append("positive_sli_delta:" + ",".join(positive_keys))

        ok = not details["reasons"]
        details["ok"] = ok
        details["volumes"] = volumes
        return ok, details

    def _strong_evidence_gate(
        self,
        *,
        baseline_slis: Dict[str, Any],
        fault_slis: Dict[str, Any],
        new_invariant_violations: List[Dict[str, Any]],
        affected_services: List[str],
        propagation_trace: Dict[str, Any],
    ) -> Tuple[bool, Dict[str, Any]]:
        deltas = self._sli_deltas(baseline_slis, fault_slis)
        degradations: Dict[str, float] = {}
        primary_business: List[str] = []
        latency_only: List[str] = []
        min_drop = float(self.args.strong_evidence_min_sli_drop)
        min_invalid = float(
            getattr(self.args, "strong_evidence_min_invalid_rate_increase", 0.05)
        )
        min_timeout = float(
            getattr(self.args, "strong_evidence_min_timeout_rate_increase", 0.03)
        )
        min_error = float(
            getattr(self.args, "strong_evidence_min_error_rate_increase", 0.03)
        )
        min_count = float(
            getattr(self.args, "strong_evidence_min_semantic_count_increase", 1.0)
        )
        min_jsd = float(
            getattr(self.args, "strong_evidence_min_distribution_jsd", 0.10)
        )
        for key, delta in deltas.items():
            if key.endswith("_success_rate"):
                drop = -float(delta)
                if drop > 0:
                    degradations[key] = round(drop, 4)
                if drop >= min_drop:
                    primary_business.append("business_success_rate_drop")
            elif key.endswith("_business_invalid_rate") and float(delta) >= min_invalid:
                primary_business.append("business_invalid_rate_increase")
            elif key.endswith("_timeout_rate") and float(delta) >= min_timeout:
                primary_business.append("business_timeout_rate_increase")
            elif (
                key.endswith(("_http_5xx_rate", "_request_exception_rate"))
                and float(delta) >= min_error
            ):
                primary_business.append("business_error_rate_increase")
            elif key.endswith("_http_4xx_rate") and float(delta) >= min_error:
                primary_business.append("business_client_error_rate_increase")
            elif key.endswith("_json_decode_error_rate") and float(delta) >= min_error:
                primary_business.append("business_decode_error_rate_increase")
            elif key.endswith("_count") and float(delta) >= min_count:
                primary_business.append("semantic_count_increase")
            elif key.endswith("_distribution_jsd") and float(delta) >= min_jsd:
                primary_business.append("entity_distribution_shift")
            elif key.endswith(LATENCY_BUSINESS_SUFFIXES) and float(delta) > 0:
                latency_only.append(key)

        max_degradation = max(degradations.values(), default=0.0)
        inv_count = len(new_invariant_violations or [])
        affected_count = len(set(affected_services or []))
        propagation_depth = int((propagation_trace or {}).get("max_depth", 0) or 0)

        reasons: List[str] = []
        if primary_business:
            reasons.extend(sorted(set(primary_business)))
        if inv_count >= int(self.args.strong_evidence_min_new_invariants):
            reasons.append("hidden_new_invariant_violation")
        if affected_count >= int(
            self.args.strong_evidence_min_affected_services
        ) or propagation_depth >= int(self.args.strong_evidence_min_propagation_depth):
            reasons.append("hidden_service_propagation")

        ok = bool(primary_business)
        return ok, {
            "ok": ok,
            "reasons": reasons,
            "primary_business_anomaly": ok,
            "secondary_latency_only": bool(latency_only and not primary_business),
            "max_core_sli_drop": round(max_degradation, 4),
            "sli_degradations": degradations,
            "new_invariant_violations": inv_count,
            "affected_services_count": affected_count,
            "propagation_depth": propagation_depth,
        }

    @staticmethod
    def _merge_effective_invariant_violations(
        *,
        spec: Dict[str, Any],
        observed: List[Dict[str, Any]],
        fault_invariants: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        expected = {
            str(item)
            for item in (spec.get("expected_invariant_violations") or [])
            if str(item)
        }
        merged = list(observed or [])
        if not expected:
            return merged
        seen = {
            str(item.get("invariant_id"))
            for item in merged
            if isinstance(item, dict) and item.get("invariant_id")
        }
        missing = expected - seen
        if not missing:
            return merged
        for item in fault_invariants or []:
            inv_id = str(item.get("invariant_id", ""))
            if inv_id in missing and item.get("violated"):
                merged.append(item)
                seen.add(inv_id)
        return merged

    def _should_run_full_prism(
        self,
        *,
        bifi_succeeded: bool,
        has_real_observation: bool,
    ) -> Tuple[bool, str]:
        if not bifi_succeeded and self.args.skip_prism_on_bifi_failure:
            return False, "bifi_failed"
        if not has_real_observation and self.args.skip_prism_on_no_dynamic_observation:
            return False, "no_dynamic_observation"
        return True, ""

    def _precheck_quality_gate(
        self,
        *,
        spec: Dict[str, Any],
        errors: List[str],
        warnings: List[str],
    ) -> Tuple[bool, List[str]]:
        reasons = list(errors)
        warning_count = len(warnings)
        max_warnings = int(self.args.precheck_max_warnings)
        if warning_count > max_warnings:
            reasons.append(f"too_many_warnings:{warning_count}>{max_warnings}")
        for warning in warnings:
            prefix = warning.split(":", 1)[0]
            if prefix in self.precheck_reject_warning_prefixes:
                reasons.append(f"reject_warning:{warning}")

        dimension = self._dimension_of_spec(spec)
        if dimension == "business_logic" and not (
            spec.get("expected_invariant_violations") or []
        ):
            reasons.append("business_logic_missing_expected_invariants")
        return (len(reasons) == 0), reasons

    def _apply_adaptive_windows(self, spec: Dict[str, Any]) -> Tuple[int, int, int]:
        base = int(self.args.baseline_seconds)
        fault = int(self.args.fault_seconds)
        cool = int(self.args.cooldown_seconds)
        if not self.args.adaptive_windows:
            return base, fault, cool
        inj = str(spec.get("injector") or "").lower()
        # Conservative defaults: only shorten known low-yield/low-signal modes.
        if inj in self.low_yield_injectors:
            fault = max(
                int(self.args.fault_seconds_min),
                int(fault * self.args.adaptive_fault_ratio),
            )
            cool = max(
                int(self.args.cooldown_seconds_min),
                int(cool * self.args.adaptive_cooldown_ratio),
            )
        if inj == "logic_fault":
            fault = max(
                int(self.args.fault_seconds_min),
                int(fault * self.args.logic_fault_window_ratio),
            )
        return base, fault, cool

    def _run_fse(self, output_dir: Path) -> List[Dict[str, Any]]:
        if (
            getattr(self.args, "llm_fse", False)
            and self.llm is not None
            and run_llm_fse is not None
        ):
            feedback_roots = self._resolve_feedback_run_roots()
            logger.info("Running LLM-guided schema-grounded ASE NIER FSE -> %s", output_dir)
            if feedback_roots:
                logger.info(
                    "LLM-FSE feedback loop sources=%d latest=%s",
                    len(feedback_roots),
                    feedback_roots[0],
                )
            result = run_llm_fse(
                llm=self.llm,
                system_description_dir=self.system_description_dir,
                workspace=HERE.parent,
                output_dir=output_dir,
                limit=max(50, int(self.args.candidate_pool_refill)),
                exploration_round=self._fse_batch_index,
                previous_run_root=self.output_root,
                feedback_run_roots=feedback_roots,
            )
            faults = [
                _canonicalize_nier_fault_spec(spec)
                for spec in result.get("faults", [])
            ]
            result["faults"] = faults
            _write_json(output_dir / "fault_catalog.json", result)
            logger.info("LLM-guided ASE NIER FSE produced %d grounded faults", len(faults))
            return faults

        if (
            self.llm is None
            or self.args.deterministic_fse
        ) and generate_nier_candidates is not None:
            logger.info("Running deterministic ASE NIER FSE -> %s", output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            family_ids = [
                item.strip()
                for item in str(
                    getattr(self.args, "curated_family_ids", "") or ""
                ).split(",")
                if item.strip()
            ]
            faults = [
                _canonicalize_nier_fault_spec(spec)
                for spec in generate_nier_candidates(
                    limit=max(50, int(self.args.candidate_pool_refill)),
                    family_ids=family_ids or None,
                    exploration_round=self._fse_batch_index,
                )
            ]
            catalog = {
                "faults": faults,
                "stats": {
                    "generated": len(faults),
                    "accepted": len(faults),
                    "generation_source": "ase_nier_deterministic_curated",
                    "exploration_mode": "round_intensity_variants",
                    "exploration_round": self._fse_batch_index,
                    "llm_bypassed": bool(
                        self.llm is not None and self.args.deterministic_fse
                    ),
                    "curated_family_ids": family_ids,
                },
            }
            _write_json(output_dir / "fault_catalog.json", catalog)
            _write_json(
                output_dir / "system_understanding.json",
                {"generation_source": "ase_nier_deterministic_curated"},
            )
            _write_json(
                output_dir / "vulnerability_surface.json",
                {
                    "generation_source": "ase_nier_deterministic_curated",
                    "exploration_mode": "round_intensity_variants",
                    "exploration_round": self._fse_batch_index,
                },
            )
            (output_dir / "exploration_report.md").write_text(
                "# FSE Exploration Report\n\n"
                "- generation_source: ase_nier_deterministic_curated\n"
                "- exploration_mode: round_intensity_variants\n"
                f"- exploration_round: {self._fse_batch_index}\n"
                f"- curated_family_ids: {family_ids}\n"
                f"- generated: {len(faults)}\n",
                encoding="utf-8",
            )
            logger.info("Deterministic ASE NIER FSE produced %d faults", len(faults))
            return faults

        retries = max(0, self.args.fse_max_retries)
        for attempt in range(retries + 1):
            logger.info(
                "Running FSE -> %s (attempt %d/%d)",
                output_dir,
                attempt + 1,
                retries + 1,
            )
            output_dir.mkdir(parents=True, exist_ok=True)
            fse = FaultSpaceExplorer(
                system_description_dir=self.system_description_dir,
                output_dir=output_dir,
                llm=self.llm,
            )
            try:
                result = fse.run(skip_if_exists=self.args.skip_fse_if_exists)
                faults = [
                    _canonicalize_nier_fault_spec(spec)
                    for spec in result.fault_catalog.get("faults", [])
                ]
                logger.info("FSE produced %d faults", len(faults))
                return faults
            except Exception as exc:  # pylint: disable=broad-except
                if attempt >= retries:
                    raise
                logger.warning(
                    "FSE attempt %d failed (%s). Retrying in %.1fs ...",
                    attempt + 1,
                    exc,
                    self.args.fse_retry_backoff_sec,
                )
                if output_dir.exists():
                    shutil.rmtree(output_dir, ignore_errors=True)
                time.sleep(self.args.fse_retry_backoff_sec)
        return []

    def _static_precheck(
        self,
        specs: List[Dict[str, Any]],
        eval_dir: Path,
    ) -> List[Dict[str, Any]]:
        def _precheck_one(spec: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
            t0 = time.time()
            precheck = self.prism.precheck.check(spec)
            quality_ok, quality_reasons = self._precheck_quality_gate(
                spec=spec,
                errors=precheck.errors,
                warnings=precheck.warnings,
            )
            duration = round(time.time() - t0, 2)
            ok = precheck.ok and quality_ok
            rec = {
                "fault_id": spec.get("fault_id"),
                "injector": spec.get("injector"),
                "dimension": spec.get("dimension"),
                "static_precheck": {
                    "ok": precheck.ok,
                    "errors": precheck.errors,
                    "warnings": precheck.warnings,
                },
                "quality_gate": {
                    "ok": quality_ok,
                    "reasons": quality_reasons,
                },
                "passed": ok,
                "elapsed_sec": duration,
            }
            return rec, ok

        out_records = []
        passed: List[Dict[str, Any]] = []

        workers = max(1, int(self.args.static_precheck_workers))
        if workers == 1 or len(specs) <= 1:
            for spec in specs:
                rec, ok = _precheck_one(spec)
                out_records.append(rec)
                if ok:
                    passed.append(spec)
        else:
            with ThreadPoolExecutor(max_workers=min(workers, len(specs))) as pool:
                for spec, (rec, ok) in zip(specs, pool.map(_precheck_one, specs)):
                    out_records.append(rec)
                    if ok:
                        passed.append(spec)

        _write_json(eval_dir / "static_precheck.json", out_records)
        return passed

    def _has_real_dynamic_observation(self, bifi_result) -> bool:
        """Dynamic evidence check: accept only real injection evidence.

        Some environments may not expose docker stats/state cleanly, so we treat
        live invariants/SLIs/windows/timestamps as acceptable dynamic evidence.
        """
        if not bifi_result.succeeded:
            return False

        if bifi_result.baseline_start <= 0 or bifi_result.fault_start <= 0:
            return False

        has_invariants = bool(
            bifi_result.baseline_invariants or bifi_result.fault_invariants
        )
        has_slis = bool(bifi_result.baseline_slis or bifi_result.fault_slis)
        has_windows = bool(bifi_result.baseline_metrics or bifi_result.fault_metrics)

        if not (has_invariants or has_slis or has_windows):
            return False
        return True

    def _inject_and_collect(
        self,
        *,
        iteration: int,
        selected_specs: List[Dict[str, Any]],
        accepted_dir: Path,
        tiered_dir: Path,
        eval_dir: Path,
    ) -> int:
        accepted = 0
        run_records = []

        for idx, spec in enumerate(selected_specs, start=1):
            health_every = max(1, int(self.args.health_check_every_n_injections))
            spec_run = copy.deepcopy(spec)
            base_fault_id = spec.get("fault_id", f"F-UNK-{idx:03d}")
            unique_fault_id = f"{base_fault_id}-IT{iteration:03d}-N{idx:03d}"
            spec_run["fault_id"] = unique_fault_id
            spec_run.setdefault("source_fault_id", base_fault_id)
            injector = str(spec_run.get("injector") or "").lower()

            logger.info(
                "[Inject %d/%d] %s (src=%s, injector=%s)",
                idx,
                len(selected_specs),
                unique_fault_id,
                base_fault_id,
                spec_run.get("injector"),
            )

            should_probe_health = (
                idx == 1
                or idx == len(selected_specs)
                or ((idx - 1) % health_every == 0)
            )
            if should_probe_health and not self._ensure_system_healthy(
                f"{unique_fault_id}_before_inject"
            ):
                logger.error(
                    "[Inject %s] system unhealthy before injection, skip this fault.",
                    unique_fault_id,
                )
                run_records.append(
                    {
                        "fault_spec": spec_run,
                        "bifi_result": {
                            "fault_id": unique_fault_id,
                            "injector": spec_run.get("injector"),
                            "target_service": spec_run.get("fault_point", {}).get(
                                "owner_service", "unknown"
                            ),
                            "succeeded": False,
                            "error": "system unhealthy before injection",
                        },
                        "prism_verdict": {},
                        "selection": {
                            "selected_by_static_precheck": True,
                            "has_real_dynamic_observation": False,
                            "accepted_for_dataset": False,
                        },
                        "timings": {
                            "bifi_sec": 0.0,
                            "prism_sec": 0.0,
                            "total_sec": 0.0,
                        },
                    }
                )
                continue

            if self._injector_is_runtime_disabled(injector):
                reason = self.injector_runtime_stats[injector].disabled_reason
                logger.warning(
                    "[Inject %s] injector=%s skipped by runtime circuit breaker: %s",
                    unique_fault_id,
                    injector,
                    reason,
                )
                run_records.append(
                    {
                        "fault_spec": spec_run,
                        "bifi_result": {
                            "fault_id": unique_fault_id,
                            "injector": injector,
                            "target_service": spec_run.get("fault_point", {}).get(
                                "owner_service", "unknown"
                            ),
                            "succeeded": False,
                            "error": f"runtime injector disabled: {reason}",
                        },
                        "prism_verdict": {},
                        "selection": {
                            "selected_by_static_precheck": True,
                            "has_real_dynamic_observation": False,
                            "accepted_for_dataset": False,
                            "runtime_injector_disabled": True,
                        },
                        "timings": {
                            "bifi_sec": 0.0,
                            "prism_sec": 0.0,
                            "total_sec": 0.0,
                        },
                    }
                )
                continue

            old_base = self.dispatcher.baseline_seconds
            old_fault = self.dispatcher.fault_seconds
            old_cool = self.dispatcher.cooldown_seconds
            new_base, new_fault, new_cool = self._apply_adaptive_windows(spec_run)
            self.dispatcher.baseline_seconds = new_base
            self.dispatcher.fault_seconds = new_fault
            self.dispatcher.cooldown_seconds = new_cool

            self.observer.reset_bqd_baseline()
            bifi_t0 = time.time()
            bifi_result = self.dispatcher.run(spec_run, dry_run=False)
            bifi_t1 = time.time()
            self.dispatcher.baseline_seconds = old_base
            self.dispatcher.fault_seconds = old_fault
            self.dispatcher.cooldown_seconds = old_cool

            self._force_cleanup(spec_run)
            post_healthy = True
            if should_probe_health:
                post_healthy = self._ensure_system_healthy(
                    f"{unique_fault_id}_post_cleanup"
                )
            if not post_healthy:
                logger.error(
                    "[Inject %s] post-cleanup health check failed; mark as rejected.",
                    unique_fault_id,
                )

            has_real_observation = self._has_real_dynamic_observation(bifi_result)
            should_run_prism, prism_skip_reason = self._should_run_full_prism(
                bifi_succeeded=bool(bifi_result.succeeded),
                has_real_observation=has_real_observation,
            )
            prism_t0 = time.time()
            if should_run_prism:
                prism_record = {
                    "fault_spec": spec_run,
                    "bifi_result": bifi_result.to_dict(),
                    "baseline_slis": bifi_result.baseline_slis or {},
                    "fault_slis": bifi_result.fault_slis or {},
                    "business_sli_deltas": bifi_result.sli_deltas or {},
                    "baseline_metrics": bifi_result.baseline_metrics or {},
                    "fault_metrics": bifi_result.fault_metrics or {},
                    "dynamic_observation": getattr(bifi_result, "observation", {})
                    or bifi_result.to_dict(),
                }
                final_verdict = dynamic_validate_telemetry(prism_record)
            else:
                final_verdict = None
            prism_t1 = time.time()

            business_gate_ok, business_gate = self._business_signal_gate(
                baseline_slis=bifi_result.baseline_slis or {},
                fault_slis=bifi_result.fault_slis or {},
            )
            evidence_gate_ok, evidence_gate = self._strong_evidence_gate(
                baseline_slis=bifi_result.baseline_slis or {},
                fault_slis=bifi_result.fault_slis or {},
                new_invariant_violations=bifi_result.new_invariant_violations or [],
                affected_services=bifi_result.affected_services or [],
                propagation_trace=bifi_result.propagation_trace or {},
            )

            final_ok = (
                bifi_result.succeeded
                and has_real_observation
                and final_verdict is not None
                and str(final_verdict.get("decision", "")).upper() in self.final_allow
                and float(final_verdict.get("aggregate_score", 0.0))
                >= self.args.final_score_threshold
                and business_gate_ok
                and evidence_gate_ok
                and post_healthy
            )

            record = {
                "fault_spec": spec_run,
                "bifi_result": bifi_result.to_dict(),
                "prism_verdict": final_verdict if final_verdict else {},
                "selection": {
                    "selected_by_static_precheck": True,
                    "has_real_dynamic_observation": has_real_observation,
                    "business_signal_gate": business_gate,
                    "strong_evidence_gate": evidence_gate,
                    "full_prism_evaluated": should_run_prism,
                    "prism_skip_reason": prism_skip_reason,
                    "accepted_for_dataset": final_ok,
                    "scored_for_dataset": final_verdict is not None,
                },
                "timings": {
                    "bifi_sec": round(bifi_t1 - bifi_t0, 1),
                    "prism_sec": round(prism_t1 - prism_t0, 1)
                    if should_run_prism
                    else 0.0,
                    "total_sec": round(prism_t1 - bifi_t0, 1),
                },
                "post_cleanup_healthy": post_healthy,
                "bifi_succeeded": bool(bifi_result.succeeded),
                "baseline_slis": bifi_result.baseline_slis or {},
                "fault_slis": bifi_result.fault_slis or {},
                "business_sli_deltas": self._sli_deltas(
                    bifi_result.baseline_slis or {}, bifi_result.fault_slis or {}
                ),
                "new_invariant_violations": bifi_result.new_invariant_violations or [],
                "affected_services": bifi_result.affected_services or [],
                "propagation_depth": len(
                    (bifi_result.propagation_trace or {}).get("path", [])
                ),
            }
            decision = self._classify_quality(record)
            record["quality_decision"] = decision
            record["selection"]["accepted_for_dataset"] = decision["tier"] == "gold"
            record["selection"]["scored_for_dataset"] = final_verdict is not None
            run_records.append(record)
            dataset_accepted_for_health = (
                final_verdict is not None
                if self.args.rcaeval_export_policy == "scored"
                else decision["tier"] == "gold"
            )
            self._record_injector_outcome(
                injector=injector,
                bifi_succeeded=bool(bifi_result.succeeded),
                accepted=dataset_accepted_for_health,
            )

            if self.args.write_per_fault_full:
                one_out = eval_dir / "per_fault_full" / f"{unique_fault_id}.json"
                _write_json(one_out, record)

            tier_out = (
                tiered_dir / decision["tier"] / "per_fault" / f"{unique_fault_id}.json"
            )
            _write_json(tier_out, record)

            if decision["tier"] == "gold":
                accepted += 1
                accepted_out = accepted_dir / "per_fault" / f"{unique_fault_id}.json"
                _write_json(accepted_out, record)

            if dataset_accepted_for_health:
                self._accepted_service_counter[self._service_of_spec(spec_run)] += 1
                self._accepted_dimension_counter[self._dimension_of_spec(spec_run)] += 1
                self._accepted_injector_counter[injector] += 1
                family_id = str(
                    (spec_run.get("fse_metadata") or {}).get("family_id") or ""
                )
                if family_id:
                    self._accepted_family_counter[family_id] += 1

            logger.info(
                "[Inject %s] bifi_ok=%s final_verdict=%s score=%.2f biz_gate_ok=%s evidence_ok=%s accepted=%s",
                unique_fault_id,
                bifi_result.succeeded,
                final_verdict.get("decision")
                if final_verdict
                else f"SKIPPED:{prism_skip_reason}",
                float(final_verdict.get("aggregate_score", 0.0))
                if final_verdict
                else 0.0,
                business_gate_ok,
                evidence_gate_ok,
                decision["tier"] == "gold",
            )

        _write_json(eval_dir / "injection_and_prism_results.json", run_records)
        return accepted

    def _classify_quality(self, record: Dict[str, Any]) -> Dict[str, Any]:
        if TierClassifier is None or QualityThresholds is None:
            return {
                "tier": "gold"
                if record.get("selection", {}).get("accepted_for_dataset")
                else "rejected",
                "accepted_for_gold": bool(
                    record.get("selection", {}).get("accepted_for_dataset")
                ),
                "accepted_for_candidate": bool(
                    record.get("selection", {}).get("accepted_for_dataset")
                ),
                "failed_gates": [],
                "passed_gates": [],
                "reasons": ["quality_gate_import_unavailable"],
                "evidence": {},
                "gate_results": {},
            }
        thresholds = QualityThresholds(
            baseline_min_success_rate=float(self.args.baseline_min_success_rate),
            baseline_max_5xx_rate=float(self.args.baseline_max_5xx_rate),
            business_min_probe_samples=int(self.args.business_min_probe_samples),
            strong_evidence_min_sli_drop=float(self.args.strong_evidence_min_sli_drop),
            final_score_threshold=float(self.args.final_score_threshold),
            final_allowed_verdicts=tuple(self.final_allow),
            reject_dirty_baseline=bool(
                getattr(self.args, "reject_dirty_baseline", False)
            ),
        )
        return TierClassifier(thresholds).classify(record).to_dict()

    def _sli_deltas(
        self, baseline_slis: Dict[str, Any], fault_slis: Dict[str, Any]
    ) -> Dict[str, float]:
        deltas: Dict[str, float] = {}
        for key, fault_value in fault_slis.items():
            if (
                not key.endswith(PRIMARY_BUSINESS_SUFFIXES + LATENCY_BUSINESS_SUFFIXES)
                or key not in baseline_slis
            ):
                continue
            try:
                deltas[key] = float(fault_value) - float(baseline_slis[key])
            except (TypeError, ValueError):
                continue
        return deltas

    def _run_cmd(self, cmd: List[str], timeout: int = 20) -> tuple[int, str, str]:
        try:
            p = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
            return p.returncode, p.stdout.strip(), p.stderr.strip()
        except Exception as exc:  # pylint: disable=broad-except
            return 1, "", str(exc)

    def _list_compose_containers(self) -> List[Dict[str, str]]:
        code, out, err = self._run_cmd(
            [
                "docker",
                "ps",
                "-a",
                "--filter",
                f"label=com.docker.compose.project={self.args.compose_project}",
                "--format",
                "{{.Names}}|{{.State}}|{{.Status}}",
            ],
            timeout=30,
        )
        if code != 0:
            logger.warning("docker ps failed during health check: %s", err)
            return []
        rows = []
        for line in out.splitlines():
            if not line.strip():
                continue
            parts = line.split("|", 2)
            rows.append(
                {
                    "name": parts[0],
                    "state": parts[1] if len(parts) > 1 else "unknown",
                    "status": parts[2] if len(parts) > 2 else "",
                }
            )
        return rows

    def _gateway_health_urls(self) -> List[str]:
        primary = self.args.gateway_health_url or (
            self.args.gateway_url.rstrip("/") + self.args.gateway_health_path
        )
        urls = [primary]
        for raw in (self.args.gateway_health_fallback_urls or "").split(","):
            u = raw.strip()
            if u:
                urls.append(u)
        # Keep order but remove duplicates.
        dedup = []
        seen = set()
        for u in urls:
            if u in seen:
                continue
            seen.add(u)
            dedup.append(u)
        return dedup

    def _gateway_ok_statuses(self) -> Set[int]:
        out: Set[int] = set()
        for raw in (self.args.gateway_health_ok_statuses or "200").split(","):
            raw = raw.strip()
            if not raw:
                continue
            try:
                out.add(int(raw))
            except ValueError:
                logger.warning("Ignore invalid gateway status code in config: %s", raw)
        return out or {200}

    def _check_gateway_health(self) -> bool:
        ok_codes = self._gateway_ok_statuses()
        for health_url in self._gateway_health_urls():
            try:
                if http_client is not None:
                    r = http_client.get(health_url, timeout=8)
                else:
                    r = requests.get(health_url, timeout=8)
                if r.status_code in ok_codes:
                    return True
            except Exception:  # pylint: disable=broad-except
                continue
        return False

    def _guess_gateway_container_name(self) -> str:
        # Prefer explicit override; fallback to compose naming convention.
        if self.args.gateway_container_name:
            return self.args.gateway_container_name
        return f"{self.args.compose_project}-ts-gateway-service-1"

    def _recover_gateway(self) -> None:
        now = time.time()
        cooldown = max(0, int(self.args.gateway_recover_cooldown_sec))
        if now - self._gateway_last_recover_ts < cooldown:
            return

        gateway_container = self._guess_gateway_container_name()
        logger.warning(
            "gateway unhealthy; trying self-recovery via docker restart: %s",
            gateway_container,
        )
        code, _, err = self._run_cmd(
            ["docker", "restart", gateway_container], timeout=120
        )
        if code != 0:
            logger.warning("gateway restart failed for %s: %s", gateway_container, err)
            return

        self._gateway_last_recover_ts = now
        self._gateway_grace_until_ts = now + max(
            0, int(self.args.gateway_startup_grace_sec)
        )

    def _recover_containers(self, unhealthy: List[Dict[str, str]]) -> None:
        for c in unhealthy:
            name = c["name"]
            state = c["state"].lower()
            if state == "paused":
                self._run_cmd(["docker", "unpause", name], timeout=30)
                continue
            if state in {"exited", "created", "dead"}:
                self._run_cmd(["docker", "start", name], timeout=60)
                continue
            if state in {"restarting", "removing"}:
                self._run_cmd(["docker", "restart", name], timeout=60)
                continue
            if state == "running" and "unhealthy" in c["status"].lower():
                self._run_cmd(["docker", "restart", name], timeout=60)

    def _ensure_system_healthy(self, stage: str) -> bool:
        if not self.args.require_system_healthy:
            return True

        deadline = time.time() + self.args.health_check_timeout_sec
        attempt = 0
        while time.time() < deadline:
            attempt += 1
            containers = self._list_compose_containers()
            unhealthy = []
            if not containers:
                unhealthy.append(
                    {"name": "compose_project", "state": "unknown", "status": "empty"}
                )
            else:
                for c in containers:
                    state = c["state"].lower()
                    status = c["status"].lower()
                    if (
                        state != "running"
                        or "unhealthy" in status
                        or "paused" in status
                    ):
                        unhealthy.append(c)

            gateway_ok = self._check_gateway_health()
            if not unhealthy and gateway_ok:
                logger.info(
                    "[HealthCheck:%s] healthy (attempt=%d, containers=%d)",
                    stage,
                    attempt,
                    len(containers),
                )
                return True

            logger.warning(
                "[HealthCheck:%s] unhealthy attempt=%d containers_bad=%d gateway_ok=%s",
                stage,
                attempt,
                len(unhealthy),
                gateway_ok,
            )
            if self.args.health_check_auto_recover and unhealthy:
                self._recover_containers(unhealthy)
            if self.args.gateway_auto_recover and not gateway_ok:
                self._recover_gateway()
                # If gateway has just been restarted, extend deadline for warmup.
                if self._gateway_grace_until_ts > time.time():
                    deadline = max(deadline, self._gateway_grace_until_ts)

            time.sleep(self.args.health_check_interval_sec)

        logger.error("[HealthCheck:%s] timeout, system still unhealthy", stage)
        return False

    def _force_cleanup(self, spec: Dict[str, Any]) -> None:
        target = (
            spec.get("injector_params", {}).get("target_service")
            or spec.get("fault_point", {}).get("owner_service")
            or "unknown"
        )
        injector = spec.get("injector", "")
        if target == "unknown":
            return
        try:
            if self._recovery_fault_injector is None:
                from fault_injector import FaultInjector

                self._recovery_fault_injector = FaultInjector(
                    config_path=str(
                        (LEGACY_FI_ROOT / "fault-config-enhanced.yml").resolve()
                    )
                )
            fi = self._recovery_fault_injector
            fi.cleanup_fault(target, fault_name=injector)
            container = fi._get_container_name(target)
            fi.cleanup_all(container)
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("force cleanup failed for %s/%s: %s", target, injector, exc)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="FaultForge auto loop: FSE -> PRISM筛选 -> 注入 -> 数据集 -> 目标计数循环"
    )

    # Loop control
    parser.add_argument(
        "--target-count", type=int, required=True, help="Target dataset size"
    )
    parser.add_argument(
        "--target-gold-count",
        type=int,
        default=None,
        help="Alias for --target-count in ASE NIER gold-tier runs",
    )
    parser.add_argument(
        "--target-candidate-count",
        type=int,
        default=0,
        help="Optional candidate-tier target for accounting/reporting",
    )
    parser.add_argument(
        "--dataset-tier-mode",
        choices=["strict", "compat"],
        default="strict",
        help="strict writes gold/candidate/rejected tiered records",
    )
    parser.add_argument(
        "--max-iterations", type=int, default=10, help="Maximum loop iterations"
    )
    parser.add_argument(
        "--max-candidates-per-iteration",
        type=int,
        default=60,
        help="Max candidate faults to static precheck each iteration",
    )
    parser.add_argument(
        "--max-injections-per-iteration",
        type=int,
        default=0,
        help="Max selected faults to inject each iteration; 0 means inject all prechecked faults",
    )
    parser.add_argument(
        "--candidate-pool-prefetch",
        type=int,
        default=300,
        help="Initial candidate pool size target filled by FSE before loop",
    )
    parser.add_argument(
        "--candidate-pool-min",
        type=int,
        default=60,
        help="Min pool size trigger for FSE refill before each iteration",
    )
    parser.add_argument(
        "--candidate-pool-refill",
        type=int,
        default=200,
        help="Target new candidates to add per refill",
    )
    parser.add_argument(
        "--candidate-pool-refill-rounds",
        type=int,
        default=3,
        help="Max FSE batches per refill cycle",
    )
    parser.add_argument(
        "--candidate-pool-seed-glob",
        default="",
        help="Optional glob pattern for existing fault_catalog.json files to seed candidate pool",
    )
    parser.add_argument(
        "--disabled-injectors",
        default="host_tc",
        help="Comma-separated injectors disabled from candidate pool and execution",
    )
    parser.add_argument(
        "--low-yield-injectors",
        default="config_modifier,docker_exec,host_tc",
        help="Comma-separated low-yield injectors used for scheduling down-weight and adaptive windows",
    )
    parser.add_argument(
        "--max-per-service-per-iteration",
        type=int,
        default=12,
        help="Max candidates per target service in one iteration (0 disables)",
    )
    parser.add_argument(
        "--max-per-dimension-per-iteration",
        type=int,
        default=30,
        help="Max candidates per dimension in one iteration (0 disables)",
    )
    parser.add_argument(
        "--max-per-injector-per-iteration",
        type=int,
        default=4,
        help="Max candidates per injector type in one iteration (0 disables)",
    )
    parser.add_argument(
        "--max-per-family-per-iteration",
        type=int,
        default=2,
        help="Max candidates per fault family in one iteration (0 disables)",
    )
    parser.add_argument(
        "--max-logic-fault-ratio",
        type=float,
        default=0.25,
        help="Max ratio of logic_fault specs among selected candidates per iteration",
    )
    parser.add_argument(
        "--logic-fault-penalty",
        type=float,
        default=0.35,
        help="Priority multiplier for logic_fault in scheduler (0..1)",
    )

    # Paths
    parser.add_argument(
        "--system-description",
        default="system_description",
        help="Path to system_description dir",
    )
    parser.add_argument(
        "--corpus",
        default="prism/corpus/incident_corpus/passages.jsonl",
        help="PRISM retrieval corpus",
    )
    parser.add_argument(
        "--output-root",
        default=f"faultforge_dataset/auto_loop_{_now_str()}",
        help="Directory for loop artifacts",
    )
    parser.add_argument(
        "--dataset-dir",
        default=None,
        help="Final RCAEval dataset output directory",
    )
    parser.add_argument(
        "--format",
        choices=["RE1", "RE2"],
        default="RE2",
        help="Output dataset format",
    )
    parser.add_argument(
        "--runtime-profile",
        choices=["standard", "throughput"],
        default="throughput",
        help="Pipeline profile: throughput favors fast sample generation for rapid iteration",
    )

    # FSE / PRISM / LLM
    parser.add_argument(
        "--api-key", default=None, help="LLM API key (or use DEEPSEEK_API_KEY)"
    )
    parser.add_argument("--base-url", default="https://api.deepseek.com")
    parser.add_argument("--model", default="deepseek-chat")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument(
        "--skip-fse-if-exists",
        action="store_true",
        help="Reuse stage outputs when fse_output already exists",
    )
    parser.add_argument(
        "--llm-fse",
        action="store_true",
        help="Use LLM-guided, schema-grounded ASE NIER FSE instead of deterministic enumeration when an LLM key is available",
    )
    parser.add_argument(
        "--feedback-loop",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable cross-run feedback loop: feed prior run summaries/audits back into LLM-FSE",
    )
    parser.add_argument(
        "--feedback-max-runs",
        type=int,
        default=5,
        help="Max prior run roots to include in each LLM-FSE feedback pack",
    )
    parser.add_argument(
        "--feedback-run-glob",
        default="",
        help="Optional glob (relative to workspace root) for additional run roots to include as feedback",
    )

    # PRISM dynamic judgement policy
    parser.add_argument(
        "--final-allowed-verdicts",
        default="REALISTIC",
        help="Comma-separated final PRISM verdicts allowed for dataset",
    )
    parser.add_argument(
        "--final-score-threshold",
        type=float,
        default=0.70,
        help="Minimum PRISM aggregate score after injection",
    )
    parser.add_argument(
        "--gold-only-rcaeval",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Deprecated compatibility flag; use --rcaeval-export-policy",
    )
    parser.add_argument(
        "--rcaeval-export-policy",
        choices=["gold", "scored", "all"],
        default="gold",
        help="Which tiered records to export: gold only, PRISM-scored records, or all records",
    )
    parser.add_argument(
        "--production-gold",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use strict ASE NIER production gold defaults",
    )
    parser.add_argument(
        "--deterministic-fse",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Bypass LLM FSE and enumerate ASE NIER curated deterministic candidates",
    )
    parser.add_argument(
        "--curated-family-ids",
        default="",
        help="Optional comma-separated curated fault family ids for deterministic FSE canaries",
    )
    parser.add_argument(
        "--reject-dirty-baseline",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reject records whose baseline health gate fails",
    )
    parser.add_argument(
        "--quality-report-dir",
        default="",
        help="Directory for dataset quality reports after a run",
    )

    # Injection observation windows
    parser.add_argument(
        "--gateway-url", default=os.environ.get("TT_GATEWAY", "http://localhost:18889")
    )
    parser.add_argument("--traffic-probes", type=int, default=10)
    parser.add_argument("--baseline-seconds", type=int, default=30)
    parser.add_argument("--fault-seconds", type=int, default=60)
    parser.add_argument("--cooldown-seconds", type=int, default=15)
    parser.add_argument(
        "--collect-raw-fault-data",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Collect raw logs/traces during fault window (higher cost, richer artifacts)",
    )
    parser.add_argument(
        "--skip-prism-on-bifi-failure",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip full PRISM evaluation when BIFI injection already failed",
    )
    parser.add_argument(
        "--skip-prism-on-no-dynamic-observation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip full PRISM evaluation when no real dynamic observation was captured",
    )
    parser.add_argument(
        "--adaptive-windows",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable adaptive fault/cooldown windows for low-yield injectors",
    )
    parser.add_argument(
        "--adaptive-fault-ratio",
        type=float,
        default=0.7,
        help="Fault window ratio for low-yield injectors",
    )
    parser.add_argument(
        "--adaptive-cooldown-ratio",
        type=float,
        default=0.7,
        help="Cooldown window ratio for low-yield injectors",
    )
    parser.add_argument(
        "--logic-fault-window-ratio",
        type=float,
        default=0.6,
        help="Fault window ratio override for logic_fault",
    )
    parser.add_argument("--fault-seconds-min", type=int, default=15)
    parser.add_argument("--cooldown-seconds-min", type=int, default=5)

    # FSE robustness
    parser.add_argument(
        "--fse-max-retries",
        type=int,
        default=2,
        help="Retry times when FSE fails (e.g., malformed LLM JSON)",
    )
    parser.add_argument(
        "--fse-retry-backoff-sec",
        type=float,
        default=2.0,
        help="Backoff seconds between FSE retries",
    )
    parser.add_argument(
        "--static-precheck-workers",
        type=int,
        default=8,
        help="Worker count for concurrent static precheck",
    )

    # System health checks
    parser.add_argument(
        "--compose-project",
        default=os.environ.get("TT_COMPOSE_PROJECT", "docker-compose-manifests"),
        help="Docker compose project name for train-ticket containers",
    )
    parser.add_argument(
        "--require-system-healthy",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require train-ticket system health before/after each injection",
    )
    parser.add_argument(
        "--health-check-auto-recover",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Try to auto recover unhealthy containers during health check",
    )
    parser.add_argument(
        "--health-check-timeout-sec",
        type=int,
        default=240,
        help="Timeout for a health-check round",
    )
    parser.add_argument(
        "--health-check-interval-sec",
        type=int,
        default=5,
        help="Polling interval for health checks",
    )
    parser.add_argument(
        "--gateway-health-path",
        default="/api/v1/stationservice/stations",
        help="Path appended to --gateway-url for gateway health check",
    )
    parser.add_argument(
        "--gateway-health-url",
        default=None,
        help="Optional full URL for gateway health check (overrides path)",
    )
    parser.add_argument(
        "--gateway-health-fallback-urls",
        default="",
        help="Comma-separated fallback health URLs, used if primary gateway check fails",
    )
    parser.add_argument(
        "--gateway-health-ok-statuses",
        default="200,401,403",
        help="Comma-separated HTTP status codes considered healthy for gateway probes",
    )
    parser.add_argument(
        "--gateway-container-name",
        default=os.environ.get("TT_GATEWAY_CONTAINER", ""),
        help="Gateway container name for auto-recovery (default: <compose_project>-ts-gateway-service-1)",
    )
    parser.add_argument(
        "--gateway-auto-recover",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Auto-restart gateway container when gateway health check keeps failing",
    )
    parser.add_argument(
        "--gateway-recover-cooldown-sec",
        type=int,
        default=120,
        help="Minimum seconds between two gateway auto-restart attempts",
    )
    parser.add_argument(
        "--gateway-startup-grace-sec",
        type=int,
        default=420,
        help="Additional health-check wait budget after gateway restart",
    )
    parser.add_argument(
        "--health-check-every-n-injections",
        type=int,
        default=5,
        help="Run pre/post health check once every N injections (plus first/last)",
    )
    parser.add_argument(
        "--write-per-fault-full",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Write per-fault full records to reduce I/O by default",
    )
    parser.add_argument(
        "--business-min-probe-samples",
        type=int,
        default=3,
        help="Minimum probe count per core business SLI before a sample can be accepted",
    )
    parser.add_argument(
        "--baseline-min-success-rate",
        type=float,
        default=0.90,
        help="Minimum baseline success rate for each core business probe",
    )
    parser.add_argument(
        "--baseline-max-5xx-rate",
        type=float,
        default=0.05,
        help="Maximum baseline HTTP 5xx rate for each core business probe",
    )
    parser.add_argument(
        "--max-positive-sli-delta",
        type=float,
        default=0.2,
        help="Reject samples where core business success_rate rises more than this in fault window",
    )
    parser.add_argument(
        "--precheck-max-warnings",
        type=int,
        default=2,
        help="Reject candidates whose static precheck emits more warnings than this threshold",
    )
    parser.add_argument(
        "--precheck-reject-warning-prefixes",
        default="",
        help="Comma-separated warning prefixes that should be treated as static-precheck rejection reasons",
    )
    parser.add_argument(
        "--strong-evidence-min-sli-drop",
        type=float,
        default=0.05,
        help="Minimum core business SLI drop required to count as strong evidence",
    )
    parser.add_argument(
        "--strong-evidence-min-new-invariants",
        type=int,
        default=1,
        help="Minimum number of new invariant violations needed to count as strong evidence",
    )
    parser.add_argument(
        "--strong-evidence-min-affected-services",
        type=int,
        default=2,
        help="Minimum affected services needed to count as strong evidence",
    )
    parser.add_argument(
        "--strong-evidence-min-propagation-depth",
        type=int,
        default=2,
        help="Minimum propagation depth needed to count as strong evidence",
    )
    parser.add_argument(
        "--injector-disable-min-attempts",
        type=int,
        default=6,
        help="Minimum attempts before runtime injector circuit-breaker can disable an injector",
    )
    parser.add_argument(
        "--injector-disable-after-consecutive-failures",
        type=int,
        default=4,
        help="Disable injector after this many consecutive BIFI failures",
    )
    parser.add_argument(
        "--injector-disable-after-consecutive-rejections",
        type=int,
        default=6,
        help="Disable injector after this many consecutive dataset rejections when it still has zero accepted samples",
    )
    parser.add_argument(
        "--injector-min-bifi-success-rate",
        type=float,
        default=0.15,
        help="Disable injector if its runtime BIFI success rate stays below this threshold",
    )
    parser.add_argument(
        "--injector-min-accept-rate",
        type=float,
        default=0.05,
        help="Disable injector if its runtime dataset acceptance rate stays at or below this threshold",
    )

    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    # Enforce non-synthetic dataset generation by default.
    os.environ.setdefault("FAULTFORGE_REAL_ONLY", "1")

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)-5s %(name)s | %(message)s",
    )

    if args.target_gold_count is not None:
        args.target_count = args.target_gold_count
    if args.production_gold:
        if args.rcaeval_export_policy == "gold":
            args.final_allowed_verdicts = "REALISTIC"
            args.final_score_threshold = max(float(args.final_score_threshold), 0.70)
        args.business_min_probe_samples = max(int(args.business_min_probe_samples), 10)
        args.baseline_min_success_rate = max(
            float(args.baseline_min_success_rate), 0.90
        )
        args.baseline_max_5xx_rate = min(float(args.baseline_max_5xx_rate), 0.05)
        args.reject_dirty_baseline = True
        args.gold_only_rcaeval = args.rcaeval_export_policy == "gold"
        args.deterministic_fse = True

    # Guardrails
    if args.target_count <= 0:
        raise SystemExit("--target-count must be > 0")
    if args.max_iterations <= 0:
        raise SystemExit("--max-iterations must be > 0")
    if args.max_injections_per_iteration < 0:
        raise SystemExit("--max-injections-per-iteration must be >= 0")
    if args.candidate_pool_prefetch < 0:
        raise SystemExit("--candidate-pool-prefetch must be >= 0")
    if args.candidate_pool_min < 0:
        raise SystemExit("--candidate-pool-min must be >= 0")
    if args.candidate_pool_refill <= 0:
        raise SystemExit("--candidate-pool-refill must be > 0")
    if args.candidate_pool_refill_rounds <= 0:
        raise SystemExit("--candidate-pool-refill-rounds must be > 0")
    if args.feedback_max_runs <= 0:
        raise SystemExit("--feedback-max-runs must be > 0")
    if args.static_precheck_workers <= 0:
        raise SystemExit("--static-precheck-workers must be > 0")
    if args.health_check_every_n_injections <= 0:
        raise SystemExit("--health-check-every-n-injections must be > 0")
    if args.gateway_recover_cooldown_sec < 0:
        raise SystemExit("--gateway-recover-cooldown-sec must be >= 0")
    if args.gateway_startup_grace_sec < 0:
        raise SystemExit("--gateway-startup-grace-sec must be >= 0")
    if args.max_per_service_per_iteration < 0:
        raise SystemExit("--max-per-service-per-iteration must be >= 0")
    if args.max_per_dimension_per_iteration < 0:
        raise SystemExit("--max-per-dimension-per-iteration must be >= 0")
    if args.max_per_injector_per_iteration < 0:
        raise SystemExit("--max-per-injector-per-iteration must be >= 0")
    if args.max_per_family_per_iteration < 0:
        raise SystemExit("--max-per-family-per-iteration must be >= 0")
    if not (0.0 <= args.max_logic_fault_ratio <= 1.0):
        raise SystemExit("--max-logic-fault-ratio must be in [0,1]")
    if not (0.0 <= args.logic_fault_penalty <= 1.0):
        raise SystemExit("--logic-fault-penalty must be in [0,1]")
    if args.business_min_probe_samples <= 0:
        raise SystemExit("--business-min-probe-samples must be > 0")
    if args.precheck_max_warnings < 0:
        raise SystemExit("--precheck-max-warnings must be >= 0")
    if args.strong_evidence_min_new_invariants < 0:
        raise SystemExit("--strong-evidence-min-new-invariants must be >= 0")
    if args.strong_evidence_min_affected_services < 0:
        raise SystemExit("--strong-evidence-min-affected-services must be >= 0")
    if args.strong_evidence_min_propagation_depth < 0:
        raise SystemExit("--strong-evidence-min-propagation-depth must be >= 0")
    if args.injector_disable_min_attempts <= 0:
        raise SystemExit("--injector-disable-min-attempts must be > 0")
    if args.injector_disable_after_consecutive_failures < 0:
        raise SystemExit("--injector-disable-after-consecutive-failures must be >= 0")
    if args.injector_disable_after_consecutive_rejections < 0:
        raise SystemExit("--injector-disable-after-consecutive-rejections must be >= 0")
    if not (0.0 <= args.injector_min_bifi_success_rate <= 1.0):
        raise SystemExit("--injector-min-bifi-success-rate must be in [0,1]")
    if not (0.0 <= args.injector_min_accept_rate <= 1.0):
        raise SystemExit("--injector-min-accept-rate must be in [0,1]")

    # Validate verdict names early
    known = {v.value for v in Verdict}
    for v in _parse_verdicts(args.final_allowed_verdicts):
        if v not in known:
            raise SystemExit(f"Unknown verdict '{v}', expected one of {sorted(known)}")

    runner = AutoLoopRunner(args)
    return runner.run()


if __name__ == "__main__":
    raise SystemExit(main())
