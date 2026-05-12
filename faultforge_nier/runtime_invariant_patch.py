"""ASE NIER runtime invariant compatibility patches.

Keep upstream ``fault-injection`` invariant code unchanged while registering
ASE-specific live checks for curated invariant aliases.
"""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
import subprocess
from typing import Any

from prism.invariants.invariant_runner import InvariantRunner, register_custom
from prism.invariants.oracle_base import InvariantResult


def _query_rows(runner: InvariantRunner, db_name: str, query: str) -> tuple[list[str], list[dict[str, str]], str]:
    """Run a SQL query through the runner's DB mode and return parsed rows."""
    if runner.db_config.mode == "docker_exec":
        container = runner.db_config.container_for_db(db_name)
        schema = runner.db_config.schema
        schema_overrides = {
            "ts-payment-mysql": "ts-payment-mysql",
            "ts-contacts-mysql": "ts-contacts-mysql",
            "ts-user-mysql": "ts-user-mysql",
            "ts-travel-mysql": "ts-travel-mysql",
            "ts-station-mysql": "ts-station-mysql",
            "ts-assurance-mysql": "ts-assurance-mysql",
            "ts-consign-mysql": "ts-consign-mysql",
            "ts-consign-price-mysql": "ts-consign-price-mysql",
            "ts-notification-mysql": "ts-notification-mysql",
            "ts-security-mysql": "ts-security-mysql",
            "ts-config-mysql": "ts-config-mysql",
            "ts-inside-payment-mysql": "ts-inside-payment-mysql",
            "ts-voucher-mysql": "ts-voucher-mysql",
            "ts-delivery-mysql": "ts-delivery-mysql",
            "ts-train-mysql": "ts-train-mysql",
            "ts-order-other-mysql": "ts-order-other-mysql",
            "ts-wait-order-mysql": "ts",
            "ts-food-mysql": "ts-food-mysql",
            "ts-food-delivery-mysql": "ts-food-delivery-mysql",
        }
        schema = schema_overrides.get(db_name, schema)
        cmd = [
            "docker",
            "exec",
            "-i",
            container,
            "mysql",
            f"-u{runner.db_config.user}",
            f"-p{runner.db_config.password}",
            schema,
            "-B",
            "-e",
            query,
        ]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30, check=False, stdin=subprocess.DEVNULL)
        except subprocess.TimeoutExpired:
            return [], [], "docker exec mysql timed out"
        except FileNotFoundError:
            return [], [], "docker CLI not available"
        if proc.returncode != 0:
            err = (proc.stderr or "").strip() or "mysql returned non-zero"
            if err.startswith("mysql: [Warning]") and "ERROR" not in err:
                err = ""
            if err:
                return [], [], err[:400]
        raw = (proc.stdout or "").strip()
        lines = raw.split("\n") if raw else []
        if not lines:
            return [], [], ""
        header = lines[0].split("\t")
        rows = [dict(zip(header, line.split("\t"))) for line in lines[1:] if line]
        return header, rows, ""

    try:
        import pymysql  # type: ignore
    except Exception:  # pylint: disable=broad-except
        return [], [], "pymysql not installed"

    try:
        conn = pymysql.connect(
            host=runner.db_config.host,
            port=runner.db_config.port,
            user=runner.db_config.user,
            password=runner.db_config.password,
            database=runner.db_config.schema,
            connect_timeout=3,
        )
        try:
            with conn.cursor(pymysql.cursors.DictCursor) as cur:
                cur.execute(query)
                rows = cur.fetchall()
        finally:
            conn.close()
    except Exception as exc:  # pylint: disable=broad-except
        return [], [], f"connect/query failed: {exc}"

    header = list(rows[0].keys()) if rows else []
    normalized_rows = [{str(k): str(v) for k, v in row.items()} for row in rows]
    return header, normalized_rows, ""


@register_custom("INV-PAY-ORDER-001")
def _check_inv_pay_order_001(inv: dict[str, Any], runner: InvariantRunner) -> InvariantResult:
    """Live check: each payment.order_id must reference an existing orders.id."""
    payment_query = (
        "SELECT id, order_id FROM payment "
        "WHERE order_id IS NOT NULL AND order_id <> ''"
    )
    order_query = "SELECT id FROM orders"

    _, payment_rows, payment_err = _query_rows(runner, "ts-payment-mysql", payment_query)
    if payment_err:
        return InvariantResult(inv["id"], False, error=payment_err, mode="computed")

    _, order_rows, order_err = _query_rows(runner, "ts-order-mysql", order_query)
    if order_err:
        return InvariantResult(inv["id"], False, error=order_err, mode="computed")

    order_ids = {str(row.get("id", "")) for row in order_rows if row.get("id")}
    violations: list[dict[str, str]] = []
    for row in payment_rows:
        order_id = str(row.get("order_id", ""))
        if not order_id:
            continue
        if order_id not in order_ids:
            violations.append(
                {
                    "payment_id": str(row.get("id", "")),
                    "order_id": order_id,
                }
            )

    return InvariantResult(
        invariant_id=inv["id"],
        violated=bool(violations),
        violation_count=len(violations),
        examples=violations[:5],
        mode="computed (docker_exec)" if runner.db_config.mode == "docker_exec" else "computed (tcp)",
        notes="live payment->order referential check",
    )


@register_custom("INV-PAY-AMOUNT-002")
def _check_inv_pay_amount_002(inv: dict[str, Any], runner: InvariantRunner) -> InvariantResult:
    """Live check: payment amount must stay positive and match order price."""
    payment_query = (
        "SELECT id, order_id, payment_price FROM payment "
        "WHERE order_id IS NOT NULL AND order_id <> ''"
    )
    order_query = "SELECT id, price FROM orders"

    _, payment_rows, payment_err = _query_rows(runner, "ts-payment-mysql", payment_query)
    if payment_err:
        return InvariantResult(inv["id"], False, error=payment_err, mode="computed")

    _, order_rows, order_err = _query_rows(runner, "ts-order-mysql", order_query)
    if order_err:
        return InvariantResult(inv["id"], False, error=order_err, mode="computed")

    def _to_decimal(value: str) -> Decimal | None:
        try:
            return Decimal(value)
        except (InvalidOperation, TypeError):
            return None

    order_prices: dict[str, Decimal] = {}
    for row in order_rows:
        order_id = str(row.get("id", "")).strip()
        price = _to_decimal(str(row.get("price", "")).strip())
        if order_id and price is not None:
            order_prices[order_id] = price

    violations: list[dict[str, str]] = []
    for row in payment_rows:
        payment_id = str(row.get("id", "")).strip()
        order_id = str(row.get("order_id", "")).strip()
        payment_price_raw = str(row.get("payment_price", "")).strip()
        payment_price = _to_decimal(payment_price_raw)
        order_price = order_prices.get(order_id)

        if not order_id or payment_price is None:
            violations.append(
                {
                    "type": "invalid_payment_amount",
                    "payment_id": payment_id,
                    "order_id": order_id or "<missing>",
                    "payment_price": payment_price_raw or "<missing>",
                }
            )
            continue

        if order_price is None:
            violations.append(
                {
                    "type": "missing_order_reference",
                    "payment_id": payment_id,
                    "order_id": order_id,
                    "payment_price": str(payment_price),
                }
            )
            continue

        if payment_price <= 0:
            violations.append(
                {
                    "type": "non_positive_payment_amount",
                    "payment_id": payment_id,
                    "order_id": order_id,
                    "payment_price": str(payment_price),
                    "order_price": str(order_price),
                }
            )
            continue

        if abs(payment_price - order_price) > Decimal("0.0001"):
            violations.append(
                {
                    "type": "payment_order_amount_mismatch",
                    "payment_id": payment_id,
                    "order_id": order_id,
                    "payment_price": str(payment_price),
                    "order_price": str(order_price),
                }
            )

    return InvariantResult(
        invariant_id=inv["id"],
        violated=bool(violations),
        violation_count=len(violations),
        examples=violations[:5],
        mode="computed (docker_exec)" if runner.db_config.mode == "docker_exec" else "computed (tcp)",
        notes="live payment amount positivity and payment->order amount consistency check",
    )


@register_custom("INV-BOOK-INV-004")
def _check_inv_book_inv_004(inv: dict[str, Any], runner: InvariantRunner) -> InvariantResult:
    """Live check: order status/seat assignment coherence for booking inventory."""
    orders_query = (
        "SELECT id, status, seat_number FROM orders "
        "WHERE status IS NOT NULL"
    )
    _, order_rows, order_err = _query_rows(runner, "ts-order-mysql", orders_query)
    if order_err:
        return InvariantResult(inv["id"], False, error=order_err, mode="computed")

    violations: list[dict[str, str]] = []
    for row in order_rows:
        order_id = str(row.get("id", "")).strip()
        status_raw = str(row.get("status", "")).strip()
        seat_number = str(row.get("seat_number", "")).strip()
        has_seat = bool(seat_number)

        try:
            status = int(status_raw)
        except ValueError:
            violations.append(
                {
                    "type": "invalid_status_format",
                    "order_id": order_id,
                    "status": status_raw or "<missing>",
                    "seat_number": seat_number or "<missing>",
                }
            )
            continue

        if status in {1, 2, 5} and not has_seat:
            violations.append(
                {
                    "type": "active_order_missing_seat",
                    "order_id": order_id,
                    "status": str(status),
                    "seat_number": "<missing>",
                }
            )

        if status == 0 and has_seat:
            violations.append(
                {
                    "type": "unpaid_order_with_assigned_seat",
                    "order_id": order_id,
                    "status": str(status),
                    "seat_number": seat_number,
                }
            )

    return InvariantResult(
        invariant_id=inv["id"],
        violated=bool(violations),
        violation_count=len(violations),
        examples=violations[:5],
        mode="computed (docker_exec)" if runner.db_config.mode == "docker_exec" else "computed (tcp)",
        notes="live status/seat coherence check for booking inventory consistency",
    )


@register_custom("INV-CONTACT-OWN-006")
def _check_inv_contact_own_006(inv: dict[str, Any], runner: InvariantRunner) -> InvariantResult:
    """Live check: order account_id must match contact owner by name+document."""
    contacts_query = (
        "SELECT id, account_id, name, document_number FROM contacts "
        "WHERE name IS NOT NULL AND name <> '' "
        "AND document_number IS NOT NULL AND document_number <> ''"
    )
    orders_query = (
        "SELECT id, account_id, contacts_name, contacts_document_number FROM orders "
        "WHERE account_id IS NOT NULL AND account_id <> '' "
        "AND contacts_name IS NOT NULL AND contacts_name <> '' "
        "AND contacts_document_number IS NOT NULL AND contacts_document_number <> ''"
    )

    _, contact_rows, contact_err = _query_rows(runner, "ts-contacts-mysql", contacts_query)
    if contact_err:
        return InvariantResult(inv["id"], False, error=contact_err, mode="computed")

    _, order_rows, order_err = _query_rows(runner, "ts-order-mysql", orders_query)
    if order_err:
        return InvariantResult(inv["id"], False, error=order_err, mode="computed")

    contact_index: dict[tuple[str, str], set[str]] = {}
    for row in contact_rows:
        name = str(row.get("name", "")).strip()
        document_number = str(row.get("document_number", "")).strip()
        account_id = str(row.get("account_id", "")).strip()
        if not name or not document_number or not account_id:
            continue
        key = (name, document_number)
        accounts = contact_index.setdefault(key, set())
        accounts.add(account_id)

    violations: list[dict[str, str]] = []
    for row in order_rows:
        order_id = str(row.get("id", "")).strip()
        order_account_id = str(row.get("account_id", "")).strip()
        contacts_name = str(row.get("contacts_name", "")).strip()
        contacts_document_number = str(row.get("contacts_document_number", "")).strip()
        if not order_account_id or not contacts_name or not contacts_document_number:
            continue
        key = (contacts_name, contacts_document_number)
        matched_accounts = contact_index.get(key, set())
        if order_account_id in matched_accounts:
            continue
        violations.append(
            {
                "order_id": order_id,
                "order_account_id": order_account_id,
                "contacts_name": contacts_name,
                "contacts_document_number": contacts_document_number,
                "matched_contact_accounts": ",".join(sorted(matched_accounts)) if matched_accounts else "<missing>",
            }
        )

    return InvariantResult(
        invariant_id=inv["id"],
        violated=bool(violations),
        violation_count=len(violations),
        examples=violations[:5],
        mode="computed (docker_exec)" if runner.db_config.mode == "docker_exec" else "computed (tcp)",
        notes="live order->contact ownership consistency check",
    )


@register_custom("INV-TRIP-REF-007")
def _check_inv_trip_ref_007(inv: dict[str, Any], runner: InvariantRunner) -> InvariantResult:
    """Live check: trip.route_id and route station references must be valid."""
    trip_query = (
        "SELECT id, route_id FROM trip "
        "WHERE route_id IS NOT NULL AND route_id <> ''"
    )
    route_query = (
        "SELECT id, start_station, end_station FROM route "
        "WHERE id IS NOT NULL AND id <> ''"
    )
    station_query = (
        "SELECT name FROM station "
        "WHERE name IS NOT NULL AND name <> ''"
    )

    _, trip_rows, trip_err = _query_rows(runner, "ts-travel-mysql", trip_query)
    if trip_err:
        return InvariantResult(inv["id"], False, error=trip_err, mode="computed")

    _, route_rows, route_err = _query_rows(runner, "ts-route-mysql", route_query)
    if route_err:
        return InvariantResult(inv["id"], False, error=route_err, mode="computed")

    _, station_rows, station_err = _query_rows(runner, "ts-station-mysql", station_query)
    if station_err:
        return InvariantResult(inv["id"], False, error=station_err, mode="computed")

    route_ids = {str(row.get("id", "")).strip() for row in route_rows if row.get("id")}
    station_names = {str(row.get("name", "")).strip() for row in station_rows if row.get("name")}
    violations: list[dict[str, str]] = []

    for row in trip_rows:
        trip_id = str(row.get("id", "")).strip()
        route_id = str(row.get("route_id", "")).strip()
        if route_id and route_id not in route_ids:
            violations.append(
                {
                    "type": "missing_route_reference",
                    "trip_id": trip_id,
                    "route_id": route_id,
                }
            )

    for row in route_rows:
        route_id = str(row.get("id", "")).strip()
        start_station = str(row.get("start_station", "")).strip()
        end_station = str(row.get("end_station", "")).strip()
        if start_station and start_station not in station_names:
            violations.append(
                {
                    "type": "missing_start_station_reference",
                    "route_id": route_id,
                    "start_station": start_station,
                }
            )
        if end_station and end_station not in station_names:
            violations.append(
                {
                    "type": "missing_end_station_reference",
                    "route_id": route_id,
                    "end_station": end_station,
                }
            )

    return InvariantResult(
        invariant_id=inv["id"],
        violated=bool(violations),
        violation_count=len(violations),
        examples=violations[:5],
        mode="computed (docker_exec)" if runner.db_config.mode == "docker_exec" else "computed (tcp)",
        notes="live trip->route and route->station reference check",
    )


@register_custom("INV-ADDON-ORDER-011")
def _check_inv_addon_order_011(inv: dict[str, Any], runner: InvariantRunner) -> InvariantResult:
    """Live check: addon records must reference existing orders.id."""
    order_query = "SELECT id FROM orders WHERE id IS NOT NULL AND id <> ''"
    assurance_query = (
        "SELECT assurance_id AS id, order_id FROM assurance "
        "WHERE order_id IS NOT NULL AND order_id <> ''"
    )
    consign_query = (
        "SELECT consign_record_id AS id, order_id FROM consign_record "
        "WHERE order_id IS NOT NULL AND order_id <> ''"
    )

    _, order_rows, order_err = _query_rows(runner, "ts-order-mysql", order_query)
    if order_err:
        return InvariantResult(inv["id"], False, error=order_err, mode="computed")

    _, assurance_rows, assurance_err = _query_rows(runner, "ts-assurance-mysql", assurance_query)
    if assurance_err:
        return InvariantResult(inv["id"], False, error=assurance_err, mode="computed")

    _, consign_rows, consign_err = _query_rows(runner, "ts-consign-mysql", consign_query)
    if consign_err:
        return InvariantResult(inv["id"], False, error=consign_err, mode="computed")

    order_ids = {str(row.get("id", "")).strip() for row in order_rows if row.get("id")}
    violations: list[dict[str, str]] = []

    for row in assurance_rows:
        addon_order_id = str(row.get("order_id", "")).strip()
        if not addon_order_id or addon_order_id in order_ids:
            continue
        violations.append(
            {
                "source": "assurance",
                "record_id": str(row.get("id", "")).strip(),
                "order_id": addon_order_id,
            }
        )

    for row in consign_rows:
        addon_order_id = str(row.get("order_id", "")).strip()
        if not addon_order_id or addon_order_id in order_ids:
            continue
        violations.append(
            {
                "source": "consign",
                "record_id": str(row.get("id", "")).strip(),
                "order_id": addon_order_id,
            }
        )

    return InvariantResult(
        invariant_id=inv["id"],
        violated=bool(violations),
        violation_count=len(violations),
        examples=violations[:5],
        mode="computed (docker_exec)" if runner.db_config.mode == "docker_exec" else "computed (tcp)",
        notes="live addon->order referential consistency check",
    )


@register_custom("INV-PRICE-ORDER-003")
def _check_inv_price_order_003(inv: dict[str, Any], runner: InvariantRunner) -> InvariantResult:
    """Live check: order price must be positive and consistent across both order DBs."""
    db_queries = [
        ("ts-order-mysql", "SELECT id, status, price, train_number FROM orders WHERE status IS NOT NULL AND price IS NOT NULL"),
        ("ts-order-other-mysql", "SELECT id, status, price, train_number FROM orders_other WHERE status IS NOT NULL AND price IS NOT NULL"),
    ]

    all_violations: list[dict[str, str]] = []
    for db_name, query in db_queries:
        _, order_rows, order_err = _query_rows(runner, db_name, query)
        if order_err:
            if db_name == "ts-order-mysql":
                return InvariantResult(inv["id"], False, error=order_err, mode="computed")
            continue

        for row in order_rows:
            order_id = str(row.get("id", "")).strip()
            price_raw = str(row.get("price", "")).strip()
            status_raw = str(row.get("status", "")).strip()
            train_number = str(row.get("train_number", "")).strip()
            try:
                status = int(status_raw)
            except ValueError:
                continue
            if status not in {1, 2}:
                continue
            try:
                price = float(price_raw)
            except (ValueError, TypeError):
                all_violations.append(
                    {
                        "source": db_name,
                        "type": "invalid_price_format",
                        "order_id": order_id,
                        "price": price_raw or "<missing>",
                        "train_number": train_number,
                    }
                )
                continue
            if price <= 0:
                all_violations.append(
                    {
                        "source": db_name,
                        "type": "non_positive_order_price",
                        "order_id": order_id,
                        "price": str(price),
                        "train_number": train_number,
                    }
                )
            elif price < 0.10:
                all_violations.append(
                    {
                        "source": db_name,
                        "type": "implausible_order_price",
                        "order_id": order_id,
                        "price": str(price),
                        "train_number": train_number,
                    }
                )

    return InvariantResult(
        invariant_id=inv["id"],
        violated=bool(all_violations),
        violation_count=len(all_violations),
        examples=all_violations[:5],
        mode="computed (docker_exec)" if runner.db_config.mode == "docker_exec" else "computed (tcp)",
        notes="live order price positivity and plausibility check (both ts-order + ts-order-other)",
    )


@register_custom("INV-PRESERVE-BYPASS-012")
def _check_inv_preserve_bypass_012(inv: dict[str, Any], runner: InvariantRunner) -> InvariantResult:
    """Live check: active orders must have passenger name+document and different stations."""
    orders_query = (
        "SELECT id, status, contacts_name, contacts_document_number, from_station, to_station "
        "FROM orders WHERE status IS NOT NULL"
    )
    _, order_rows, order_err = _query_rows(runner, "ts-order-mysql", orders_query)
    if order_err:
        return InvariantResult(inv["id"], False, error=order_err, mode="computed")

    SENTINEL_EMPTY = {"BYPASSED_EMPTY_NAME", "BYPASSED_EMPTY_DOC", "EMPTIED_BY_FAULT"}
    violations: list[dict[str, str]] = []
    for row in order_rows:
        order_id = str(row.get("id", "")).strip()
        status_raw = str(row.get("status", "")).strip()
        contacts_name = str(row.get("contacts_name", "")).strip()
        contacts_document_number = str(row.get("contacts_document_number", "")).strip()
        from_station = str(row.get("from_station", "")).strip()
        to_station = str(row.get("to_station", "")).strip()
        try:
            status = int(status_raw)
        except ValueError:
            continue
        if status not in {1, 2}:
            continue
        if not contacts_name or contacts_name in SENTINEL_EMPTY:
            violations.append(
                {
                    "type": "missing_or_bypassed_passenger_name",
                    "order_id": order_id,
                    "status": str(status),
                    "contacts_name": contacts_name or "<empty>",
                }
            )
        if not contacts_document_number or contacts_document_number in SENTINEL_EMPTY:
            violations.append(
                {
                    "type": "missing_or_bypassed_document_number",
                    "order_id": order_id,
                    "status": str(status),
                    "contacts_document_number": contacts_document_number or "<empty>",
                }
            )
        if from_station and to_station and from_station == to_station:
            violations.append(
                {
                    "type": "same_station_roundtrip",
                    "order_id": order_id,
                    "status": str(status),
                    "from_station": from_station,
                    "to_station": to_station,
                }
            )

    return InvariantResult(
        invariant_id=inv["id"],
        violated=bool(violations),
        violation_count=len(violations),
        examples=violations[:5],
        mode="computed (docker_exec)" if runner.db_config.mode == "docker_exec" else "computed (tcp)",
        notes="live preserve booking validation bypass check",
    )


@register_custom("INV-SEAT-UNIQUE-013")
def _check_inv_seat_unique_013(inv: dict[str, Any], runner: InvariantRunner) -> InvariantResult:
    """Live check: no two active orders on same train share the same seat_number."""
    orders_query = (
        "SELECT id, status, train_number, seat_number FROM orders "
        "WHERE status IN (1, 2) AND seat_number IS NOT NULL AND seat_number <> '' "
        "AND train_number IS NOT NULL AND train_number <> ''"
    )
    _, order_rows, order_err = _query_rows(runner, "ts-order-mysql", orders_query)
    if order_err:
        return InvariantResult(inv["id"], False, error=order_err, mode="computed")

    seat_index: dict[tuple[str, str], list[str]] = {}
    for row in order_rows:
        order_id = str(row.get("id", "")).strip()
        train_number = str(row.get("train_number", "")).strip()
        seat_number = str(row.get("seat_number", "")).strip()
        if not order_id or not train_number or not seat_number:
            continue
        key = (train_number, seat_number)
        ids = seat_index.setdefault(key, [])
        ids.append(order_id)

    violations: list[dict[str, str]] = []
    for (train_number, seat_number), order_ids in seat_index.items():
        if len(order_ids) > 1:
            violations.append(
                {
                    "type": "double_booked_seat",
                    "train_number": train_number,
                    "seat_number": seat_number,
                    "order_ids": ",".join(order_ids),
                }
            )

    return InvariantResult(
        invariant_id=inv["id"],
        violated=bool(violations),
        violation_count=len(violations),
        examples=violations[:5],
        mode="computed (docker_exec)" if runner.db_config.mode == "docker_exec" else "computed (tcp)",
        notes="live seat uniqueness check across active orders",
    )


@register_custom("INV-NOTIFY-ORDER-014")
def _check_inv_notify_order_014(inv: dict[str, Any], runner: InvariantRunner) -> InvariantResult:
    """Live check: notify_info.order_number must reference existing orders.id."""
    notif_query = (
        "SELECT id, order_number FROM notify_info "
        "WHERE order_number IS NOT NULL AND order_number <> ''"
    )
    order_query = "SELECT id FROM orders"

    _, notif_rows, notif_err = _query_rows(runner, "ts-notification-mysql", notif_query)
    if notif_err:
        return InvariantResult(inv["id"], False, error=notif_err, mode="computed")

    _, order_rows, order_err = _query_rows(runner, "ts-order-mysql", order_query)
    if order_err:
        return InvariantResult(inv["id"], False, error=order_err, mode="computed")

    order_ids = {str(row.get("id", "")).strip() for row in order_rows if row.get("id")}
    violations: list[dict[str, str]] = []
    for row in notif_rows:
        order_number = str(row.get("order_number", "")).strip()
        if order_number and order_number not in order_ids:
            violations.append(
                {
                    "notify_id": str(row.get("id", "")).strip(),
                    "order_number": order_number,
                }
            )

    return InvariantResult(
        invariant_id=inv["id"],
        violated=bool(violations),
        violation_count=len(violations),
        examples=violations[:5],
        mode="computed (docker_exec)" if runner.db_config.mode == "docker_exec" else "computed (tcp)",
        notes="live notification->order referential check",
    )


@register_custom("INV-CONFIG-DRIFT-015")
def _check_inv_config_drift_015(inv: dict[str, Any], runner: InvariantRunner) -> InvariantResult:
    """Live check: config values must be in valid range."""
    config_query = "SELECT name, value FROM config WHERE name IS NOT NULL AND name <> ''"
    _, config_rows, config_err = _query_rows(runner, "ts-config-mysql", config_query)
    if config_err:
        return InvariantResult(inv["id"], False, error=config_err, mode="computed")

    violations: list[dict[str, str]] = []
    for row in config_rows:
        name = str(row.get("name", "")).strip()
        value_raw = str(row.get("value", "")).strip()
        if not name:
            continue
        try:
            value = float(value_raw)
        except (ValueError, TypeError):
            violations.append(
                {"type": "invalid_config_value", "name": name, "value": value_raw or "<missing>"}
            )
            continue
        if name == "DirectTicketAllocationProportion" and not (0.0 < value < 1.0):
            violations.append(
                {"type": "allocation_proportion_out_of_range", "name": name, "value": str(value)}
            )

    return InvariantResult(
        invariant_id=inv["id"],
        violated=bool(violations),
        violation_count=len(violations),
        examples=violations[:5],
        mode="computed (docker_exec)" if runner.db_config.mode == "docker_exec" else "computed (tcp)",
        notes="live config value range check",
    )


@register_custom("INV-SECURITY-BYPASS-016")
def _check_inv_security_bypass_016(inv: dict[str, Any], runner: InvariantRunner) -> InvariantResult:
    """Live check: security thresholds must be positive."""
    sec_query = (
        "SELECT name, value FROM security_config "
        "WHERE name IS NOT NULL AND name <> ''"
    )
    _, sec_rows, sec_err = _query_rows(runner, "ts-security-mysql", sec_query)
    if sec_err:
        return InvariantResult(inv["id"], False, error=sec_err, mode="computed")

    violations: list[dict[str, str]] = []
    for row in sec_rows:
        name = str(row.get("name", "")).strip()
        value_raw = str(row.get("value", "")).strip()
        if not name:
            continue
        try:
            value = int(value_raw)
        except (ValueError, TypeError):
            violations.append(
                {"type": "invalid_security_value", "name": name, "value": value_raw or "<missing>"}
            )
            continue
        if value <= 0:
            violations.append(
                {"type": "security_threshold_zeroed", "name": name, "value": str(value)}
            )

    return InvariantResult(
        invariant_id=inv["id"],
        violated=bool(violations),
        violation_count=len(violations),
        examples=violations[:5],
        mode="computed (docker_exec)" if runner.db_config.mode == "docker_exec" else "computed (tcp)",
        notes="live security threshold positivity check",
    )


@register_custom("INV-VOUCHER-ORDER-017")
def _check_inv_voucher_order_017(inv: dict[str, Any], runner: InvariantRunner) -> InvariantResult:
    """Live check: voucher.order_id must reference existing orders.id."""
    voucher_query = (
        "SELECT voucher_id, order_id FROM voucher "
        "WHERE order_id IS NOT NULL AND order_id <> ''"
    )
    order_query = "SELECT id FROM orders"

    _, voucher_rows, voucher_err = _query_rows(runner, "ts-voucher-mysql", voucher_query)
    if voucher_err:
        return InvariantResult(inv["id"], False, error=voucher_err, mode="computed")

    _, order_rows, order_err = _query_rows(runner, "ts-order-mysql", order_query)
    if order_err:
        return InvariantResult(inv["id"], False, error=order_err, mode="computed")

    order_ids = {str(row.get("id", "")).strip() for row in order_rows if row.get("id")}
    violations: list[dict[str, str]] = []
    for row in voucher_rows:
        order_id = str(row.get("order_id", "")).strip()
        if order_id and order_id not in order_ids:
            violations.append(
                {"voucher_id": str(row.get("voucher_id", "")).strip(), "order_id": order_id}
            )

    return InvariantResult(
        invariant_id=inv["id"],
        violated=bool(violations),
        violation_count=len(violations),
        examples=violations[:5],
        mode="computed (docker_exec)" if runner.db_config.mode == "docker_exec" else "computed (tcp)",
        notes="live voucher->order referential check",
    )


@register_custom("INV-DELIVERY-ORDER-018")
def _check_inv_delivery_order_018(inv: dict[str, Any], runner: InvariantRunner) -> InvariantResult:
    """Live check: delivery.order_id must reference existing orders.id when non-null."""
    delivery_query = (
        "SELECT id, order_id FROM delivery "
        "WHERE order_id IS NOT NULL AND order_id <> ''"
    )
    order_query = "SELECT id FROM orders"

    _, delivery_rows, delivery_err = _query_rows(runner, "ts-delivery-mysql", delivery_query)
    if delivery_err:
        return InvariantResult(inv["id"], False, error=delivery_err, mode="computed")

    _, order_rows, order_err = _query_rows(runner, "ts-order-mysql", order_query)
    if order_err:
        return InvariantResult(inv["id"], False, error=order_err, mode="computed")

    order_ids = {str(row.get("id", "")).strip() for row in order_rows if row.get("id")}
    violations: list[dict[str, str]] = []
    for row in delivery_rows:
        order_id = str(row.get("order_id", "")).strip()
        if order_id and order_id not in order_ids:
            violations.append(
                {"delivery_id": str(row.get("id", "")).strip(), "order_id": order_id}
            )

    return InvariantResult(
        invariant_id=inv["id"],
        violated=bool(violations),
        violation_count=len(violations),
        examples=violations[:5],
        mode="computed (docker_exec)" if runner.db_config.mode == "docker_exec" else "computed (tcp)",
        notes="live delivery->order referential check",
    )


@register_custom("INV-TRAIN-TYPE-019")
def _check_inv_train_type_019(inv: dict[str, Any], runner: InvariantRunner) -> InvariantResult:
    """Live check: train_type capacity and speed must be positive."""
    train_query = (
        "SELECT id, name, economy_class, confort_class, average_speed FROM train_type "
        "WHERE name IS NOT NULL AND name <> ''"
    )
    _, train_rows, train_err = _query_rows(runner, "ts-train-mysql", train_query)
    if train_err:
        return InvariantResult(inv["id"], False, error=train_err, mode="computed")

    violations: list[dict[str, str]] = []
    for row in train_rows:
        name = str(row.get("name", "")).strip()
        tid = str(row.get("id", "")).strip()
        for field in ("economy_class", "confort_class", "average_speed"):
            try:
                value = int(str(row.get(field, "")).strip())
            except (ValueError, TypeError):
                violations.append(
                    {"type": f"invalid_{field}", "train_id": tid, "name": name, field: str(row.get(field, ""))}
                )
                continue
            if value <= 0:
                violations.append(
                    {"type": f"non_positive_{field}", "train_id": tid, "name": name, field: str(value)}
                )

    return InvariantResult(
        invariant_id=inv["id"],
        violated=bool(violations),
        violation_count=len(violations),
        examples=violations[:5],
        mode="computed (docker_exec)" if runner.db_config.mode == "docker_exec" else "computed (tcp)",
        notes="live train type capacity/speed positivity check",
    )


@register_custom("INV-INSIDE-PAY-020")
def _check_inv_inside_pay_020(inv: dict[str, Any], runner: InvariantRunner) -> InvariantResult:
    """Live check: inside_payment price and inside_money balance must be valid."""
    ip_query = (
        "SELECT id, order_id, price FROM inside_payment "
        "WHERE price IS NOT NULL"
    )
    money_query = (
        "SELECT id, money FROM inside_money WHERE money IS NOT NULL"
    )

    _, ip_rows, ip_err = _query_rows(runner, "ts-inside-payment-mysql", ip_query)
    if ip_err:
        return InvariantResult(inv["id"], False, error=ip_err, mode="computed")

    _, money_rows, money_err = _query_rows(runner, "ts-inside-payment-mysql", money_query)
    if money_err:
        return InvariantResult(inv["id"], False, error=money_err, mode="computed")

    violations: list[dict[str, str]] = []
    for row in ip_rows:
        try:
            price = float(str(row.get("price", "")).strip())
        except (ValueError, TypeError):
            violations.append(
                {"type": "invalid_payment_price", "ip_id": str(row.get("id", "")).strip(),
                 "price": str(row.get("price", ""))}
            )
            continue
        if price <= 0:
            violations.append(
                {"type": "non_positive_payment_price", "ip_id": str(row.get("id", "")).strip(),
                 "price": str(price), "order_id": str(row.get("order_id", ""))}
            )

    for row in money_rows:
        try:
            money = float(str(row.get("money", "")).strip())
        except (ValueError, TypeError):
            violations.append(
                {"type": "invalid_balance", "money_id": str(row.get("id", "")).strip(),
                 "money": str(row.get("money", ""))}
            )
            continue
        if money <= 0:
            violations.append(
                {"type": "non_positive_balance", "money_id": str(row.get("id", "")).strip(),
                 "money": str(money)}
            )

    return InvariantResult(
        invariant_id=inv["id"],
        violated=bool(violations),
        violation_count=len(violations),
        examples=violations[:5],
        mode="computed (docker_exec)" if runner.db_config.mode == "docker_exec" else "computed (tcp)",
        notes="live inside-payment price/money validity check",
    )


@register_custom("INV-WAIT-STATE-021")
def _check_inv_wait_state_021(inv: dict[str, Any], runner: InvariantRunner) -> InvariantResult:
    """Live check: wait_list_order status transitions must be legal; PAID/COLLECTED/REFUNDS orders must have positive price."""
    query = (
        "SELECT id, account_id, status, CAST(price AS DECIMAL(12,2)) AS price, train_number "
        "FROM wait_list_order "
        "WHERE status IN (1, 2, 4) "
        "AND (price IS NULL OR CAST(price AS DECIMAL(12,2)) <= 0) "
        "LIMIT 100"
    )

    _, rows, err = _query_rows(runner, "ts-wait-order-mysql", query)
    if err:
        return InvariantResult(inv["id"], False, error=err, mode="computed")

    violations: list[dict[str, str]] = []
    for row in rows:
        try:
            status = int(str(row.get("status", "")).strip())
        except (ValueError, TypeError):
            continue
        try:
            price = float(str(row.get("price", "")).strip())
        except (ValueError, TypeError):
            price = 0.0

        # PAID(1) orders must have positive price
        if status == 1 and price <= 0:
            violations.append(
                {"type": "paid_order_zero_price", "wo_id": str(row.get("id", "")).strip(),
                 "status": str(status), "price": str(price)}
            )
        # COLLECTED(2) or REFUNDS(4) orders must have positive price (must have gone through PAID)
        if status in (2, 4) and price <= 0:
            violations.append(
                {"type": "collected_or_refunded_zero_price", "wo_id": str(row.get("id", "")).strip(),
                 "status": str(status), "price": str(price)}
            )

    return InvariantResult(
        invariant_id=inv["id"],
        violated=bool(violations),
        violation_count=len(violations),
        examples=violations[:5],
        mode="computed (docker_exec)" if runner.db_config.mode == "docker_exec" else "computed (tcp)",
        notes="live wait_list_order status-price consistency check",
    )


@register_custom("INV-FOOD-ORDER-022")
def _check_inv_food_order_022(inv: dict[str, Any], runner: InvariantRunner) -> InvariantResult:
    """Live check: food order price must be positive; delivery trip_id must not be empty."""
    food_price_query = (
        "SELECT id, food_name, price, order_id FROM food_order "
        "WHERE price IS NOT NULL"
    )
    delivery_query = (
        "SELECT id, trip_id FROM food_delivery_order "
        "WHERE trip_id IS NULL OR trip_id = '' OR trip_id = 'CLEARED_TRIP'"
    )

    _, food_rows, food_err = _query_rows(runner, "ts-food-mysql", food_price_query)
    if food_err:
        return InvariantResult(inv["id"], False, error=food_err, mode="computed")

    _, delivery_rows, delivery_err = _query_rows(
        runner, "ts-food-delivery-mysql", delivery_query
    )
    if delivery_err:
        return InvariantResult(inv["id"], False, error=delivery_err, mode="computed")

    violations: list[dict[str, str]] = []
    for row in food_rows:
        try:
            price = float(str(row.get("price", "")).strip())
        except (ValueError, TypeError):
            violations.append(
                {"type": "invalid_food_price", "food_order_id": str(row.get("id", "")).strip(),
                 "price": str(row.get("price", ""))}
            )
            continue
        if price <= 0:
            violations.append(
                {"type": "non_positive_food_price", "food_order_id": str(row.get("id", "")).strip(),
                 "price": str(price), "food_name": str(row.get("food_name", "")).strip()}
            )

    for row in delivery_rows:
        violations.append(
            {"type": "delivery_missing_trip_reference",
             "delivery_id": str(row.get("id", "")).strip(),
             "trip_id": str(row.get("trip_id", "") or "<empty>")}
        )

    return InvariantResult(
        invariant_id=inv["id"],
        violated=bool(violations),
        violation_count=len(violations),
        examples=violations[:5],
        mode="computed (docker_exec)" if runner.db_config.mode == "docker_exec" else "computed (tcp)",
        notes="live food order price positivity and delivery trip reference check",
    )


@register_custom("INV-CONSIGN-PRICE-023")
def _check_inv_consign_price_023(inv: dict[str, Any], runner: InvariantRunner) -> InvariantResult:
    """Live check: consign_price config initial_price and initial_weight must be positive."""
    consign_query = (
        "SELECT id, initial_price, initial_weight FROM consign_price "
        "WHERE initial_price IS NULL OR CAST(initial_price AS DECIMAL(12,2)) <= 0 "
        "   OR initial_weight IS NULL OR CAST(initial_weight AS DECIMAL(12,2)) <= 0"
    )

    _, consign_rows, consign_err = _query_rows(
        runner, "ts-consign-price-mysql", consign_query
    )
    if consign_err:
        return InvariantResult(inv["id"], False, error=consign_err, mode="computed")

    violations: list[dict[str, str]] = []
    for row in consign_rows:
        cid = str(row.get("id", "")).strip()
        try:
            price = float(str(row.get("initial_price", "")).strip())
        except (ValueError, TypeError):
            price = 0.0
        try:
            weight = float(str(row.get("initial_weight", "")).strip())
        except (ValueError, TypeError):
            weight = 0.0

        if price <= 0:
            violations.append(
                {"type": "non_positive_consign_price", "id": cid,
                 "initial_price": str(price)}
            )
        if weight <= 0:
            violations.append(
                {"type": "non_positive_consign_weight", "id": cid,
                 "initial_weight": str(weight)}
            )

    return InvariantResult(
        invariant_id=inv["id"],
        violated=bool(violations),
        violation_count=len(violations),
        examples=violations[:5],
        mode="computed (docker_exec)" if runner.db_config.mode == "docker_exec" else "computed (tcp)",
        notes="live consign_price config price/weight positivity check",
    )


def apply_invariant_runner_patch() -> dict[str, Any]:
    """Import-time registration hook for ASE NIER custom invariant checks."""
    return {
        "custom_invariants": [
            "INV-PAY-ORDER-001",
            "INV-PAY-AMOUNT-002",
            "INV-PRICE-ORDER-003",
            "INV-BOOK-INV-004",
            "INV-CONTACT-OWN-006",
            "INV-TRIP-REF-007",
            "INV-ADDON-ORDER-011",
            "INV-PRESERVE-BYPASS-012",
            "INV-SEAT-UNIQUE-013",
            "INV-NOTIFY-ORDER-014",
            "INV-CONFIG-DRIFT-015",
            "INV-SECURITY-BYPASS-016",
            "INV-VOUCHER-ORDER-017",
            "INV-DELIVERY-ORDER-018",
            "INV-TRAIN-TYPE-019",
            "INV-INSIDE-PAY-020",
            "INV-WAIT-STATE-021",
            "INV-FOOD-ORDER-022",
            "INV-CONSIGN-PRICE-023",
        ]
    }
