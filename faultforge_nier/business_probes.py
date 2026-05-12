"""Business data-quality collectors for FaultForge ASE NIER.

Produces semantic count and distribution JSD metrics from runtime observations
of Train-Ticket business data. These metrics feed into the telemetry-only
PRISM pipeline and are separate from invariant checks (which are hidden labels).

All metrics produced here are in ALLOWED_BUSINESS_METRICS and use the
business.csv long-table schema: (timestamp, window, metric, value, unit, source).
Source field is always 'business_data_quality' for counts and
'business_entity_distribution' for distribution JSDs.

Design rules (from telemetry-first redesign):
- Never depend on invariant IDs, fault spec metadata, or PRISM verdicts.
- Only observe business state through API responses.
- Produce only metrics in ALLOWED_BUSINESS_METRICS.
"""

from __future__ import annotations

import json
import logging
import math
from collections import Counter
from typing import Any

from . import http_client
from .telemetry_contract import ALLOWED_BUSINESS_METRICS

logger = logging.getLogger(__name__)

VALID_STATUS_CODES = {0, 1}
PAYMENT_STATES = {"NOTPAY", "PAYING", "PAID"}
ORDER_STATUSES = {0, 1, 2, 3, 4, 5, 6}

BUSINESS_DATA_QUALITY_CALLS = {
    "order_query": {
        "method": "POST",
        "path": "/api/v1/orderservice/order/query",
        "description": "Query orders for the current user",
    },
    "payment_query": {
        "method": "GET",
        "path": "/api/v1/paymentservice/payment",
        "description": "Query payments for the current user",
    },
    "contacts_query": {
        "method": "GET",
        "path": "/api/v1/contactservice/contacts/account",
        "description": "Query contacts for the current user",
    },
    "config_query": {
        "method": "GET",
        "path": "/api/v1/configservice/configs",
        "description": "Query configuration values",
    },
}


def compute_jsd(p: dict[str, float], q: dict[str, float]) -> float:
    """Compute Jensen-Shannon Divergence between two distributions."""
    all_keys = set(p.keys()) | set(q.keys())
    if not all_keys:
        return 0.0
    total_p = sum(p.values()) or 1.0
    total_q = sum(q.values()) or 1.0
    p_norm = {k: p.get(k, 0.0) / total_p for k in all_keys}
    q_norm = {k: q.get(k, 0.0) / total_q for k in all_keys}
    m = {k: 0.5 * (p_norm[k] + q_norm[k]) for k in all_keys}
    jsd = 0.5 * _kl(p_norm, m) + 0.5 * _kl(q_norm, m)
    return min(1.0, max(0.0, jsd))


def _kl(p: dict[str, float], q: dict[str, float]) -> float:
    result = 0.0
    for k, v in p.items():
        if v > 0 and q.get(k, 0) > 0:
            result += v * math.log(v / q[k])
    return result


class BusinessDataQualityCollector:
    """Collects business data-quality metrics from the Train-Ticket API."""

    def __init__(self, gateway_url: str, token: str | None = None, timeout: int = 8):
        self.gateway_url = gateway_url.rstrip("/")
        self.token = token
        self.timeout = timeout

    def _headers(self) -> dict[str, str]:
        if self.token:
            return {"Authorization": f"Bearer {self.token}"}
        return {}

    def collect_order_counts(self, account_id: str | None = None) -> dict[str, float]:
        """Collect order-related semantic counts and distribution."""
        results: dict[str, float] = {}
        auth = self._headers()
        try:
            payload = {}
            if account_id:
                payload["loginId"] = account_id
                payload["buyDate"] = None
            r = http_client.post(
                f"{self.gateway_url}/api/v1/orderservice/order/query",
                json=payload,
                headers=auth,
                timeout=self.timeout,
            )
            if r.ok:
                body = r.json()
                data = body.get("data", [])
                if isinstance(data, list):
                    results["invalid_order_status_count"] = float(
                        sum(
                            1
                            for o in data
                            if isinstance(o, dict)
                            and o.get("status") not in ORDER_STATUSES
                            and o.get("status") is not None
                        )
                    )
                    illegal_transitions = 0
                    for o in data:
                        if isinstance(o, dict) and o.get("status") is not None:
                            try:
                                status = int(o.get("status", -1))
                                coach = o.get("coachNumber")
                                seat = o.get("seatNumber")
                                if status in (4, 5) and coach is None:
                                    illegal_transitions += 1
                            except (ValueError, TypeError):
                                illegal_transitions += 1
                    results["illegal_order_transition_count"] = float(
                        illegal_transitions
                    )
                    status_counter: Counter = Counter()
                    for o in data:
                        if isinstance(o, dict):
                            status_counter[str(o.get("status", "unknown"))] += 1
                    total_orders = sum(status_counter.values()) or 1
                    status_dist = {
                        k: v / total_orders for k, v in status_counter.items()
                    }
                    results["_order_status_distribution"] = status_dist
        except Exception as exc:
            logger.warning("order_query failed: %s", exc)
        return results

    def collect_payment_counts(self, account_id: str | None = None) -> dict[str, float]:
        """Collect payment-related semantic counts and distribution."""
        results: dict[str, float] = {}
        auth = self._headers()
        try:
            path = f"/api/v1/paymentservice/payment"
            r = http_client.get(
                f"{self.gateway_url}{path}",
                headers=auth,
                timeout=self.timeout,
            )
            if r.ok:
                body = r.json()
                data = body.get("data", [])
                if isinstance(data, list):
                    payments_without_order = 0
                    payment_counter: Counter = Counter()
                    for p in data:
                        if isinstance(p, dict):
                            payment_counter[str(p.get("state", "unknown"))] += 1
                            if p.get("orderId") is None or p.get("orderId") == "":
                                payments_without_order += 1
                    results["payment_without_order_count"] = float(
                        payments_without_order
                    )
                    total_payments = sum(payment_counter.values()) or 1
                    state_dist = {
                        k: v / total_payments for k, v in payment_counter.items()
                    }
                    results["_payment_state_distribution"] = state_dist
        except Exception as exc:
            logger.warning("payment_query failed: %s", exc)
        return results

    def collect_seat_counts(self) -> dict[str, float]:
        """Collect seat assignment consistency counts."""
        results: dict[str, float] = {}
        auth = self._headers()
        try:
            r = http_client.get(
                f"{self.gateway_url}/api/v1/orderservice/order/query",
                headers=auth,
                timeout=self.timeout,
            )
            if r.ok:
                body = r.json()
                data = body.get("data", [])
                if isinstance(data, list):
                    seat_assignments: Counter = Counter()
                    for o in data:
                        if isinstance(o, dict):
                            seat_key = (
                                str(o.get("trainNumber", "")),
                                str(o.get("carriageNumber", "")),
                                str(o.get("seatNumber", "")),
                            )
                            if all(seat_key):
                                seat_assignments[seat_key] += 1
                    results["duplicate_seat_assignment_count"] = float(
                        sum(1 for k, v in seat_assignments.items() if v > 1)
                    )
        except Exception as exc:
            logger.warning("seat_query failed: %s", exc)
        return results

    def collect_config_counts(self) -> dict[str, float]:
        """Collect configuration consistency counts."""
        results: dict[str, float] = {}
        auth = self._headers()
        try:
            r = http_client.get(
                f"{self.gateway_url}/api/v1/configservice/configs",
                headers=auth,
                timeout=self.timeout,
            )
            if r.ok:
                body = r.json()
                data = body.get("data", [])
                if isinstance(data, list):
                    missing_keys = 0
                    out_of_range = 0
                    for c in data:
                        if isinstance(c, dict):
                            if c.get("value") is None or c.get("value") == "":
                                missing_keys += 1
                            try:
                                val_str = str(c.get("value", ""))
                                if val_str.startswith(("-", "0")) and "." in val_str:
                                    val = float(val_str)
                                    if val < -1000 or val > 1000:
                                        out_of_range += 1
                            except (ValueError, TypeError):
                                pass
                    results["config_missing_key_count"] = float(missing_keys)
                    results["config_out_of_range_count"] = float(out_of_range)
        except Exception as exc:
            logger.warning("config_query failed: %s", exc)
        return results

    def collect_all(
        self,
        account_id: str | None = None,
        baseline_distributions: dict[str, dict[str, float]] | None = None,
    ) -> dict[str, float]:
        """Collect all business data-quality metrics.

        If baseline_distributions is provided, computes distribution JSDs
        against the baseline distributions for order status and payment state.

        Returns a dict of metric_name -> value, all in ALLOWED_BUSINESS_METRICS.
        Only metrics that were successfully collected are included.
        """
        results: dict[str, float] = {}

        order_counts = self.collect_order_counts(account_id)
        results.update(
            {k: v for k, v in order_counts.items() if k in ALLOWED_BUSINESS_METRICS}
        )

        payment_counts = self.collect_payment_counts(account_id)
        results.update(
            {k: v for k, v in payment_counts.items() if k in ALLOWED_BUSINESS_METRICS}
        )

        seat_counts = self.collect_seat_counts()
        results.update(
            {k: v for k, v in seat_counts.items() if k in ALLOWED_BUSINESS_METRICS}
        )

        config_counts = self.collect_config_counts()
        results.update(
            {k: v for k, v in config_counts.items() if k in ALLOWED_BUSINESS_METRICS}
        )

        if baseline_distributions:
            order_dist = order_counts.get("_order_status_distribution", {})
            if (
                order_dist
                and "order_status_distribution_jsd" in ALLOWED_BUSINESS_METRICS
            ):
                baseline_order = baseline_distributions.get(
                    "order_status_distribution", {}
                )
                if baseline_order:
                    results["order_status_distribution_jsd"] = compute_jsd(
                        baseline_order, order_dist
                    )

            payment_dist = payment_counts.get("_payment_state_distribution", {})
            if (
                payment_dist
                and "payment_state_distribution_jsd" in ALLOWED_BUSINESS_METRICS
            ):
                baseline_payment = baseline_distributions.get(
                    "payment_state_distribution", {}
                )
                if baseline_payment:
                    results["payment_state_distribution_jsd"] = compute_jsd(
                        baseline_payment, payment_dist
                    )

        return results
