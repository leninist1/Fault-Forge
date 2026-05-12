"""FaultForge Observer — collects real observations during baseline and fault windows.

Three data sources, all live:

  1. Business SLIs via synthetic traffic
     - Login success rate          (gateway POST /api/v1/users/login)
     - Trip search success rate    (gateway GET  /api/v1/travelservice/trips/...)
     - Contacts fetch success rate (gateway GET  /api/v1/contactservice/contacts)
     - End-to-end booking-start success rate
     Each SLI is computed as successful_requests / attempted_requests during the window.

  2. Per-service health via docker (stats + recent log error counts)
     - CPU%, Memory%
     - Container status (running/restart)
     - Last-30s error log occurrences

  3. Propagation summary
     - Services whose error-log rate rose between baseline and fault windows
     - BFS on service topology from fault target to observed changes
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

import requests
import yaml

from . import http_client
from .business_probes import BusinessDataQualityCollector

logger = logging.getLogger(__name__)


@dataclass
class ObserverConfig:
    gateway_url: str = os.environ.get("TT_GATEWAY", "http://localhost:18889")
    username: str = "fdse_microservice"
    password: str = "111111"
    request_timeout: int = int(os.environ.get("OBSERVER_REQUEST_TIMEOUT_SEC", "8"))
    traffic_probes_per_window: int = int(
        os.environ.get("SLI_TRAFFIC_PROBES_PER_WINDOW", "10")
    )  # how many probes to issue
    sli_concurrency: int = int(os.environ.get("SLI_CONCURRENCY", "8"))
    sli_rate_limit_per_sec: float = float(os.environ.get("SLI_RATE_LIMIT_PER_SEC", "0"))
    compose_project: str = os.environ.get(
        "TT_COMPOSE_PROJECT_PREFIX", "docker-compose-manifests-"
    )
    window_seconds: int = int(os.environ.get("OBSERVER_WINDOW_SECONDS", "30"))
    trace_query_limit_per_service: int = int(
        os.environ.get("TRACE_QUERY_LIMIT_PER_SERVICE", "5000")
    )


class Observer:
    """Live observer against a running Train-Ticket deployment."""

    def __init__(
        self,
        config: Optional[ObserverConfig] = None,
        system_description_dir: Optional[Path] = None,
    ):
        self.cfg = config or ObserverConfig()
        self.token: Optional[str] = None
        self.account_id: Optional[str] = None
        self._bqd_collector: Optional[BusinessDataQualityCollector] = None
        self._baseline_bqd_distributions: dict[str, dict[str, float]] = {}
        if system_description_dir:
            topo_path = Path(system_description_dir) / "service_topology.yml"
            self.topology = yaml.safe_load(topo_path.read_text(encoding="utf-8"))
        else:
            self.topology = None

    # ---------------------------------------------------------------- business data quality
    def _get_bqd_collector(self) -> BusinessDataQualityCollector:
        if self._bqd_collector is None:
            self._bqd_collector = BusinessDataQualityCollector(
                gateway_url=self.cfg.gateway_url,
                token=self.token,
                timeout=self.cfg.request_timeout,
            )
        return self._bqd_collector

    def _collect_business_data_quality(self) -> dict[str, float]:
        """Collect business data-quality metrics, computing JSD against stored baseline."""
        try:
            collector = self._get_bqd_collector()
            collector.token = self.token
            if not self._baseline_bqd_distributions:
                order_counts = collector.collect_order_counts(self.account_id)
                payment_counts = collector.collect_payment_counts(self.account_id)
                order_dist = order_counts.get("_order_status_distribution", {})
                payment_dist = payment_counts.get("_payment_state_distribution", {})
                if isinstance(order_dist, dict) and order_dist:
                    self._baseline_bqd_distributions["order_status_distribution"] = order_dist
                if isinstance(payment_dist, dict) and payment_dist:
                    self._baseline_bqd_distributions["payment_state_distribution"] = payment_dist
            baseline = (
                self._baseline_bqd_distributions
                if self._baseline_bqd_distributions
                else None
            )
            results = collector.collect_all(
                account_id=self.account_id, baseline_distributions=baseline
            )
            return {k: v for k, v in results.items() if not isinstance(v, dict)}
        except Exception as exc:
            logger.warning("business data quality collection failed: %s", exc)
            return {}

    def reset_bqd_baseline(self) -> None:
        """Reset stored baseline distributions for a new fault injection cycle."""
        self._baseline_bqd_distributions.clear()

    # ------------------------------------------------------------------ auth
    def _ensure_login(self) -> bool:
        if self.token:
            return True
        try:
            r = http_client.post(
                f"{self.cfg.gateway_url}/api/v1/users/login",
                json={
                    "username": self.cfg.username,
                    "password": self.cfg.password,
                    "verificationCode": "1234",
                },
                timeout=self.cfg.request_timeout,
            )
            if r.ok and isinstance(r.json(), dict):
                data = r.json().get("data") or {}
                self.token = data.get("token")
                self.account_id = data.get("userId")
                return bool(self.token)
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("login failed: %s", exc)
        return False

    # ------------------------------------------------------------------ SLIs
    def collect_slis(self, spec: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """Return business SLIs with compatibility + richer latency/error metrics."""
        n = max(1, int(self.cfg.traffic_probes_per_window))
        _ = spec  # reserved for future per-fault adaptive probes

        self._ensure_login()
        auth_headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}
        account_id = self.account_id or ""

        probes: List[Dict[str, Any]] = [
            {
                "name": "login",
                "method": "POST",
                "path": "/api/v1/users/login",
                "attempts": n,
                "headers_builder": lambda: {},
                "payload_builder": lambda: {
                    "username": self.cfg.username,
                    "password": self.cfg.password,
                    "verificationCode": "1234",
                },
                "success_checker": lambda _, body: (
                    isinstance(body, dict)
                    and body.get("status") == 1
                    and isinstance(body.get("data"), dict)
                    and bool(body.get("data", {}).get("token"))
                ),
            },
            {
                "name": "trip_search",
                "method": "POST",
                "path": "/api/v1/travelservice/trips/left",
                "attempts": n,
                "headers_builder": lambda: auth_headers,
                "payload_builder": lambda: {
                    "startingPlace": "Shang Hai",
                    "endPlace": "Su Zhou",
                    "departureTime": "2026-06-30",
                },
                "success_checker": lambda _, body: (
                    isinstance(body, dict)
                    and body.get("status") == 1
                    and isinstance(body.get("data"), list)
                ),
            },
            {
                "name": "contacts_fetch",
                "method": "GET",
                "path": f"/api/v1/contactservice/contacts/account/{account_id}",
                "attempts": n,
                "headers_builder": lambda: auth_headers,
                "payload_builder": lambda: None,
                "success_checker": lambda _, body: (
                    isinstance(body, dict) and body.get("status") == 1
                ),
            },
            {
                "name": "order_read",
                "method": "POST",
                # Query is safer than refresh for low-state accounts: the live
                # environment may have zero orders, but the order service should
                # still answer successfully.
                "path": "/api/v1/orderservice/order/query",
                "attempts": n,
                "headers_builder": lambda: auth_headers,
                "payload_builder": lambda: {"loginId": account_id, "buyDate": None},
                "success_checker": lambda _, body: (
                    isinstance(body, dict)
                    and body.get("status") == 1
                    and isinstance(body.get("data"), list)
                ),
            },
            {
                "name": "booking_precheck",
                "method": "POST",
                "path": "/api/v1/travelservice/trips/left",
                "attempts": n,
                "headers_builder": lambda: auth_headers,
                "payload_builder": lambda: {
                    "startingPlace": "Shang Hai",
                    "endPlace": "Nan Jing",
                    "departureTime": "2026-06-30",
                },
                "success_checker": lambda _, body: (
                    isinstance(body, dict)
                    and body.get("status") == 1
                    and isinstance(body.get("data"), list)
                ),
            },
            {
                "name": "payment_submit",
                "method": "GET",
                "path": "/api/v1/paymentservice/payment",
                "attempts": n,
                "headers_builder": lambda: auth_headers,
                "payload_builder": lambda: None,
                "success_checker": lambda _, body: (
                    isinstance(body, dict) and body.get("status") == 1
                ),
            },
            {
                "name": "cancel_submit",
                "method": "GET",
                "path": "/api/v1/cancelservice/welcome",
                "attempts": n,
                "headers_builder": lambda: auth_headers,
                "payload_builder": lambda: None,
                "success_checker": lambda _, body: (
                    isinstance(body, str) and "Cancel Service" in body
                ),
            },
            {
                "name": "addon_query",
                "method": "GET",
                "path": "/api/v1/foodservice/orders",
                "attempts": n,
                "headers_builder": lambda: auth_headers,
                "payload_builder": lambda: None,
                "success_checker": lambda _, body: (
                    isinstance(body, dict) and body.get("status") == 1
                ),
            },
        ]

        results: Dict[str, float] = {}
        for probe in probes:
            metrics = self._run_sli_probe(
                name=probe["name"],
                method=probe["method"],
                path=probe["path"],
                attempts=int(probe["attempts"]),
                headers_builder=probe["headers_builder"],
                payload_builder=probe["payload_builder"],
                success_checker=probe["success_checker"],
            )
            results.update(metrics)
        bqd = self._collect_business_data_quality()
        results.update(bqd)
        return results

    def _run_sli_probe(
        self,
        *,
        name: str,
        method: str,
        path: str,
        attempts: int,
        headers_builder: Callable[[], Dict[str, str]],
        payload_builder: Callable[[], Optional[Dict[str, Any]]],
        success_checker: Callable[[requests.Response, Any], bool],
    ) -> Dict[str, float]:
        records: List[Dict[str, Any]] = []
        concurrency = max(1, int(self.cfg.sli_concurrency))
        attempts = max(1, int(attempts))
        per_req_delay = (
            1.0 / float(self.cfg.sli_rate_limit_per_sec)
            if self.cfg.sli_rate_limit_per_sec and self.cfg.sli_rate_limit_per_sec > 0
            else 0.0
        )
        submit_start = time.perf_counter()

        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = []
            for idx in range(attempts):
                if per_req_delay > 0:
                    target_submit = submit_start + (idx * per_req_delay)
                    sleep_for = target_submit - time.perf_counter()
                    if sleep_for > 0:
                        time.sleep(sleep_for)
                futures.append(
                    pool.submit(
                        self._execute_sli_probe_once,
                        method=method,
                        path=path,
                        headers=headers_builder(),
                        payload=payload_builder(),
                        success_checker=success_checker,
                    )
                )
            for fut in as_completed(futures):
                try:
                    records.append(fut.result())
                except Exception:  # pylint: disable=broad-except
                    records.append(
                        {
                            "success": False,
                            "latency_ms": float(self.cfg.request_timeout * 1000),
                            "error_type": "probe_internal_error",
                        }
                    )

        return self._aggregate_sli_records(name=name, records=records)

    def _execute_sli_probe_once(
        self,
        *,
        method: str,
        path: str,
        headers: Dict[str, str],
        payload: Optional[Dict[str, Any]],
        success_checker: Callable[[requests.Response, Any], bool],
    ) -> Dict[str, Any]:
        t0 = time.perf_counter()
        url = f"{self.cfg.gateway_url}{path}"
        try:
            resp = http_client.request(
                method=method,
                url=url,
                json=payload,
                headers=headers,
                timeout=self.cfg.request_timeout,
            )
            latency_ms = (time.perf_counter() - t0) * 1000.0

            if resp.status_code >= 500:
                return {
                    "success": False,
                    "latency_ms": latency_ms,
                    "error_type": "http_5xx",
                }
            if resp.status_code >= 400:
                return {
                    "success": False,
                    "latency_ms": latency_ms,
                    "error_type": "http_4xx",
                }

            try:
                body = resp.json()
            except ValueError:
                text_body = resp.text or ""
                if success_checker(resp, text_body):
                    return {"success": True, "latency_ms": latency_ms, "error_type": "ok"}
                return {
                    "success": False,
                    "latency_ms": latency_ms,
                    "error_type": "json_decode_error",
                }

            if success_checker(resp, body):
                return {"success": True, "latency_ms": latency_ms, "error_type": "ok"}
            return {
                "success": False,
                "latency_ms": latency_ms,
                "error_type": "business_invalid",
            }
        except requests.exceptions.Timeout:
            return {
                "success": False,
                "latency_ms": float(self.cfg.request_timeout * 1000),
                "error_type": "timeout",
            }
        except requests.exceptions.RequestException:
            return {
                "success": False,
                "latency_ms": float(self.cfg.request_timeout * 1000),
                "error_type": "request_exception",
            }

    def _aggregate_sli_records(
        self, *, name: str, records: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        total = max(1, len(records))
        success_count = sum(1 for r in records if r.get("success"))
        latencies = [float(r.get("latency_ms", 0.0)) for r in records]

        def _rate(error_type: str) -> float:
            return (
                sum(1 for r in records if r.get("error_type") == error_type) / float(total)
            )

        metrics: Dict[str, float] = {
            f"{name}_sample_count": float(total),
            f"{name}_success_count": float(success_count),
            f"{name}_success_rate": success_count / float(total),
            f"{name}_latency_ms_p50": self._percentile(latencies, 50),
            f"{name}_latency_ms_p95": self._percentile(latencies, 95),
            f"{name}_latency_ms_p99": self._percentile(latencies, 99),
            f"{name}_timeout_rate": _rate("timeout"),
            f"{name}_http_5xx_rate": _rate("http_5xx"),
            f"{name}_http_4xx_rate": _rate("http_4xx"),
            f"{name}_business_invalid_rate": _rate("business_invalid"),
            f"{name}_json_decode_error_rate": _rate("json_decode_error"),
            f"{name}_request_exception_rate": _rate("request_exception"),
        }
        return metrics

    @staticmethod
    def _percentile(values: List[float], percentile: int) -> float:
        if not values:
            return 0.0
        sorted_vals = sorted(values)
        idx = int(round((len(sorted_vals) - 1) * (percentile / 100.0)))
        idx = max(0, min(len(sorted_vals) - 1, idx))
        return float(sorted_vals[idx])

    # ------------------------------------------------------------------ window
    def collect_window(
        self,
        *,
        phase: str,
        spec: Dict[str, Any],
        target: str,
        collect_raw_data: bool = False,
    ) -> Dict[str, Any]:
        """Collect container-level data for all services at the boundary of a window."""
        stats = self._docker_stats()
        service_states = self._container_states()
        services = list(service_states.keys())
        error_counts = self._log_error_counts(
            services=services,
            since_seconds=self.cfg.window_seconds,
        )

        result = {
            "phase": phase,
            "timestamp": time.time(),
            "target": target,
            "stats": stats,
            "states": service_states,
            "error_log_counts": error_counts,
        }

        # Optionally collect raw logs and traces for more detailed analysis
        if collect_raw_data:
            prioritized = self._prioritize_services_for_logs(services, target)
            raw_logs = self._collect_raw_logs(
                services=prioritized,  # full service set
                since_seconds=self.cfg.window_seconds,
                max_lines_per_service=0,  # 0 => full logs in the window
            )
            traces = self._collect_traces(since_seconds=self.cfg.window_seconds)
            result["raw_logs"] = raw_logs
            result["traces"] = traces

        return result

    def _prioritize_services_for_logs(
        self, services: List[str], target: str
    ) -> List[str]:
        """Prioritize app services for log collection instead of docker ps order."""
        seen = set()
        ordered: List[str] = []

        def _add(svc: str):
            if svc and svc in services and svc not in seen:
                seen.add(svc)
                ordered.append(svc)

        # Always prioritize fault target and gateway.
        _add(target)
        _add("ts-gateway-service")

        # Prefer application services over infra dependencies.
        for svc in services:
            low = svc.lower()
            if (
                "mysql" in low
                or "redis" in low
                or "mongo" in low
                or "nacos" in low
                or svc == "rabbitmq"
            ):
                continue
            _add(svc)

        # Append the remaining services to preserve completeness.
        for svc in services:
            _add(svc)
        return ordered

    def _service_from_container_name(self, name: str) -> Optional[str]:
        if not name.startswith(self.cfg.compose_project):
            return None
        service = name[len(self.cfg.compose_project) :].lstrip("_-")
        if not service:
            return None
        if "-" in service and service.rsplit("-", 1)[-1].isdigit():
            return service.rsplit("-", 1)[0]
        if "_" in service and service.rsplit("_", 1)[-1].isdigit():
            return service.rsplit("_", 1)[0]
        return service

    def _container_name_map(self) -> Dict[str, str]:
        """Build service -> container-name mapping for robust log queries."""
        try:
            cmd = ["docker", "ps", "--format", "{{.Names}}", "-a"]
            out = subprocess.check_output(cmd, text=True, timeout=15)
        except Exception:  # pylint: disable=broad-except
            return {}
        mapping: Dict[str, str] = {}
        for name in out.strip().splitlines():
            svc = self._service_from_container_name(name)
            if svc and svc not in mapping:
                mapping[svc] = name
        return mapping

    def _docker_stats(self) -> Dict[str, Dict[str, float]]:
        try:
            cmd = [
                "docker",
                "stats",
                "--no-stream",
                "--format",
                "{{.Name}}|{{.CPUPerc}}|{{.MemPerc}}",
            ]
            out = subprocess.check_output(cmd, text=True, timeout=20)
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("docker stats failed: %s", exc)
            return {}
        result: Dict[str, Dict[str, float]] = {}
        for line in out.strip().splitlines():
            try:
                name, cpu, mem = line.split("|")
                service = self._service_from_container_name(name)
                if not service:
                    continue
                result[service] = {
                    "cpu_percent": float(cpu.rstrip("%")),
                    "mem_percent": float(mem.rstrip("%")),
                }
            except Exception:  # pylint: disable=broad-except
                continue
        return result

    def _container_states(self) -> Dict[str, str]:
        try:
            cmd = ["docker", "ps", "--format", "{{.Names}}|{{.Status}}", "-a"]
            out = subprocess.check_output(cmd, text=True, timeout=15)
        except Exception:  # pylint: disable=broad-except
            return {}
        states: Dict[str, str] = {}
        for line in out.strip().splitlines():
            try:
                name, status = line.split("|", 1)
                service = self._service_from_container_name(name)
                if not service:
                    continue
                states[service] = status
            except Exception:
                continue
        return states

    def _log_error_counts(
        self, services: List[str], since_seconds: int = 30
    ) -> Dict[str, int]:
        """For each service, count how many ERROR/WARN/Exception lines appear in recent logs."""
        out: Dict[str, int] = {}
        containers = self._container_name_map()
        since = f"{since_seconds}s"
        for svc in services:
            container = containers.get(svc) or f"{self.cfg.compose_project}{svc}-1"
            try:
                proc = subprocess.run(
                    ["docker", "logs", "--since", since, container],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                text = (proc.stdout or "") + (proc.stderr or "")
            except Exception:
                out[svc] = 0
                continue
            cnt = 0
            for line in text.splitlines():
                lower = line.lower()
                if (
                    " error " in lower
                    or "exception" in lower
                    or " warn " in lower
                    or " fatal " in lower
                    or "http 5" in lower
                    or "status 5" in lower
                ):
                    cnt += 1
            out[svc] = cnt
        return out

    def _collect_raw_logs(
        self,
        services: List[str],
        since_seconds: int = 30,
        max_lines_per_service: int = 50,
    ) -> Dict[str, List[str]]:
        """Collect raw log lines from services.

        max_lines_per_service:
            > 0: keep newest N lines
            <= 0: keep all lines in the window
        """
        out: Dict[str, List[str]] = {}
        containers = self._container_name_map()
        since = f"{since_seconds}s"
        for svc in services:
            container = containers.get(svc) or f"{self.cfg.compose_project}{svc}-1"
            try:
                proc = subprocess.run(
                    ["docker", "logs", "--since", since, container],
                    capture_output=True,
                    text=True,
                    timeout=15,
                )
                text = (proc.stdout or "") + (proc.stderr or "")
                lines = [ln for ln in text.splitlines() if ln.strip()]
                if max_lines_per_service > 0:
                    out[svc] = lines[-max_lines_per_service:]
                else:
                    out[svc] = lines
            except Exception:
                out[svc] = []
        return out

    def _collect_traces(self, since_seconds: int = 30) -> List[Dict[str, Any]]:
        """Collect traces from Jaeger API."""
        traces = []
        try:
            # Jaeger API uses microseconds for time bounds.
            end_time = int(time.time() * 1000000)
            start_time = end_time - (since_seconds * 1000000)
            services_resp = http_client.get("http://localhost:16686/api/services", timeout=10)
            if services_resp.status_code != 200:
                return traces
            services_to_query = services_resp.json().get("data", []) or []
            seen_trace_ids: Set[str] = set()
            for service in services_to_query:
                try:
                    url = f"http://localhost:16686/api/traces"
                    params = {
                        "service": service,
                        "start": start_time,
                        "end": end_time,
                        "limit": self.cfg.trace_query_limit_per_service,
                    }
                    response = http_client.get(url, params=params, timeout=10)
                    if response.status_code == 200:
                        data = response.json()
                        for trace in data.get("data", []):
                            trace_id = trace.get("traceID")
                            if trace_id and trace_id in seen_trace_ids:
                                continue
                            if trace_id:
                                seen_trace_ids.add(trace_id)
                            traces.append(trace)
                except Exception as e:
                    logger.warning(f"Failed to query traces for {service}: {e}")
        except Exception as e:
            logger.warning(f"Failed to collect traces: {e}")
        return traces

    # ------------------------------------------------------------------ propagation
    def summarize_propagation(
        self,
        *,
        spec: Dict[str, Any],
        target: str,
        baseline: Dict[str, Any],
        fault: Dict[str, Any],
    ) -> Dict[str, Any]:
        if not baseline or not fault:
            return {"max_depth": 0, "fanout": 0, "affected_services": [target]}
        base_err = baseline.get("error_log_counts") or {}
        fault_err = fault.get("error_log_counts") or {}
        affected: Set[str] = {target}
        delta_map: Dict[str, int] = {}
        for svc, cnt in fault_err.items():
            base_cnt = base_err.get(svc, 0)
            if cnt > base_cnt + 2:  # at least 3 new error lines to count as affected
                affected.add(svc)
                delta_map[svc] = cnt - base_cnt
        # Container state transitions (running → restarting etc.)
        base_states = baseline.get("states") or {}
        fault_states = fault.get("states") or {}
        state_changes = {
            svc: {"before": base_states.get(svc), "after": fault_states.get(svc)}
            for svc in fault_states
            if base_states.get(svc) != fault_states.get(svc)
        }
        for svc in state_changes:
            affected.add(svc)

        # Propagation depth via topology BFS from target
        max_depth = self._compute_depth(target, affected)
        return {
            "max_depth": max_depth,
            "fanout": len(affected) - 1,
            "affected_services": sorted(affected),
            "error_log_deltas": delta_map,
            "state_changes": state_changes,
        }

    def _compute_depth(self, target: str, affected: Set[str]) -> int:
        if not self.topology:
            return 0
        services = self.topology.get("services", {})
        if target not in services:
            return 0
        from collections import deque

        depth = {target: 0}
        q = deque([target])
        while q:
            node = q.popleft()
            # outgoing callees
            for call in services.get(node, {}).get("calls") or []:
                tgt = call.get("target")
                if tgt and tgt not in depth:
                    depth[tgt] = depth[node] + 1
                    q.append(tgt)
            # incoming callers (bidirectional propagation)
            for other, info in services.items():
                if other == node or other in depth:
                    continue
                for c in info.get("calls") or []:
                    if c.get("target") == node:
                        depth[other] = depth[node] + 1
                        q.append(other)
                        break
        return max((depth[s] for s in affected if s in depth), default=0)
