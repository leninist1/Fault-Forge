"""ASE NIER runtime schema compatibility patches.

The ASE NIER workspace keeps the upstream ``fault-injection`` implementation
unchanged, but production runs still import its BIFI and PRISM classes.  This
module applies narrow Train-Ticket schema aliases at process startup so curated
business fault families can use domain names that differ slightly from the
legacy injector's table map.
"""

from __future__ import annotations

import subprocess
from typing import Any


def _patch_docker_exec_stdin() -> None:
    """Monkey-patch DockerExecMySQLCursor.execute to use stdin=DEVNULL.

    Without this, docker exec -i inherits a pipe stdin from the parent
    process chain and hangs forever on databases with large output.
    """
    try:
        from injectors.business_fault_injector import DockerExecMySQLCursor
    except ImportError:
        return

    _original_execute = DockerExecMySQLCursor.execute

    def _patched_execute(self, sql: str, params=None):
        import subprocess as _subprocess

        if params:
            for p in params if isinstance(params, (list, tuple)) else [params]:
                sql = sql.replace("%s", f"'{p}'", 1)

        cmd = [
            "docker",
            "exec",
            "-i",
            self.connection.container,
            "mysql",
            f"-u{self.connection.user}",
            f"-p{self.connection.password}",
            self.connection.database,
            "-e",
            sql,
        ]

        result = _subprocess.run(cmd, capture_output=True, text=True, timeout=30, stdin=_subprocess.DEVNULL)
        if result.returncode != 0:
            raise Exception(f"MySQL command failed: {result.stderr}")

        self._last_result = result.stdout
        return self

    DockerExecMySQLCursor.execute = _patched_execute


def apply_database_modifier_schema_patch() -> dict[str, Any]:
    """Patch upstream DatabaseModifierInjector schema aliases for ASE NIER."""
    _patch_docker_exec_stdin()

    from injectors.business_fault_injector import DatabaseModifierInjector

    mapping = DatabaseModifierInjector.SERVICE_DB_MAPPING
    payment = mapping.setdefault(
        "ts-payment-service",
        {
            "db": "ts-payment-mysql",
            "tables": [],
        },
    )
    payment.setdefault("db", "ts-payment-mysql")
    payment.setdefault("schema", "ts-payment-mysql")
    tables = list(payment.get("tables") or [])
    for table in ("payment", "order"):
        if table not in tables:
            tables.append(table)
    payment["tables"] = tables

    aliases = dict(payment.get("table_aliases") or {})
    aliases.setdefault("payments", "payment")
    aliases.setdefault("payment_order", "order")
    payment["table_aliases"] = aliases

    fields = list(payment.get("fields") or [])
    for field in ("id", "order_id", "payment_price", "user_id"):
        if field not in fields:
            fields.append(field)
    payment["fields"] = fields

    field_aliases = dict(payment.get("field_aliases") or {})
    field_aliases.setdefault("orderId", "order_id")
    field_aliases.setdefault("paymentPrice", "payment_price")
    field_aliases.setdefault("amount", "payment_price")
    field_aliases.setdefault("price", "payment_price")
    payment["field_aliases"] = field_aliases

    contacts = mapping.setdefault(
        "ts-contacts-service",
        {
            "db": "ts-contacts-mysql",
            "tables": [],
        },
    )
    contacts.setdefault("db", "ts-contacts-mysql")
    contacts.setdefault("schema", "ts-contacts-mysql")
    contact_tables = list(contacts.get("tables") or [])
    if "contacts" not in contact_tables:
        contact_tables.append("contacts")
    contacts["tables"] = contact_tables

    contact_fields = list(contacts.get("fields") or [])
    for field in (
        "id",
        "account_id",
        "name",
        "document_type",
        "document_number",
        "phone_number",
    ):
        if field not in contact_fields:
            contact_fields.append(field)
    contacts["fields"] = contact_fields

    contact_field_aliases = dict(contacts.get("field_aliases") or {})
    contact_field_aliases.setdefault("accountId", "account_id")
    contact_field_aliases.setdefault("documentType", "document_type")
    contact_field_aliases.setdefault("documentNumber", "document_number")
    contact_field_aliases.setdefault("phoneNumber", "phone_number")
    contacts["field_aliases"] = contact_field_aliases

    route = mapping.setdefault(
        "ts-route-service",
        {
            "db": "ts-route-mysql",
            "tables": [],
        },
    )
    route.setdefault("db", "ts-route-mysql")
    route.setdefault("schema", "ts")
    route_tables = list(route.get("tables") or [])
    for table in ("route",):
        if table not in route_tables:
            route_tables.append(table)
    route["tables"] = route_tables

    route_aliases = dict(route.get("table_aliases") or {})
    route_aliases.setdefault("routes", "route")
    route["table_aliases"] = route_aliases

    travel = mapping.setdefault(
        "ts-travel-service",
        {
            "db": "ts-travel-mysql",
            "tables": [],
        },
    )
    travel.setdefault("db", "ts-travel-mysql")
    travel.setdefault("schema", "ts-travel-mysql")
    travel_tables = list(travel.get("tables") or [])
    if "trip" not in travel_tables:
        travel_tables.append("trip")
    travel["tables"] = travel_tables

    travel_aliases = dict(travel.get("table_aliases") or {})
    travel_aliases.setdefault("trips", "trip")
    travel["table_aliases"] = travel_aliases

    travel_field_aliases = dict(travel.get("field_aliases") or {})
    travel_field_aliases.setdefault("routeId", "route_id")
    travel["field_aliases"] = travel_field_aliases

    assurance = mapping.setdefault(
        "ts-assurance-service",
        {
            "db": "ts-assurance-mysql",
            "tables": [],
        },
    )
    assurance.setdefault("db", "ts-assurance-mysql")
    assurance.setdefault("schema", "ts-assurance-mysql")
    assurance_tables = list(assurance.get("tables") or [])
    for table in ("assurance", "assurance_orders"):
        if table not in assurance_tables:
            assurance_tables.append(table)
    assurance["tables"] = assurance_tables

    assurance_aliases = dict(assurance.get("table_aliases") or {})
    assurance_aliases.setdefault("assurance_orders", "assurance")
    assurance["table_aliases"] = assurance_aliases

    assurance_fields = list(assurance.get("fields") or [])
    for field in ("id", "order_id", "type", "price"):
        if field not in assurance_fields:
            assurance_fields.append(field)
    assurance["fields"] = assurance_fields

    assurance_field_aliases = dict(assurance.get("field_aliases") or {})
    assurance_field_aliases.setdefault("orderId", "order_id")
    assurance["field_aliases"] = assurance_field_aliases

    food_service = mapping.setdefault(
        "ts-food-service",
        {
            "db": "ts-food-mysql",
            "tables": [],
        },
    )
    food_service.setdefault("db", "ts-food-mysql")
    food_service.setdefault("schema", "ts-food-mysql")
    foods_tables = list(food_service.get("tables") or [])
    for table in ("food_order",):
        if table not in foods_tables:
            foods_tables.append(table)
    food_service["tables"] = foods_tables

    foods_fields = list(food_service.get("fields") or [])
    for field in ("id", "food_name", "food_type", "order_id", "price", "station_name", "store_name"):
        if field not in foods_fields:
            foods_fields.append(field)
    food_service["fields"] = foods_fields

    foods_field_aliases = dict(food_service.get("field_aliases") or {})
    foods_field_aliases.setdefault("orderId", "order_id")
    food_service["field_aliases"] = foods_field_aliases

    food_delivery = mapping.setdefault(
        "ts-food-delivery-service",
        {
            "db": "ts-food-delivery-mysql",
            "tables": [],
        },
    )
    food_delivery.setdefault("db", "ts-food-delivery-mysql")
    food_delivery.setdefault("schema", "ts-food-delivery-mysql")
    food_tables = list(food_delivery.get("tables") or [])
    for table in ("food_delivery_order", "food_delivery_orders"):
        if table not in food_tables:
            food_tables.append(table)
    food_delivery["tables"] = food_tables

    food_aliases = dict(food_delivery.get("table_aliases") or {})
    food_aliases.setdefault("food_delivery_orders", "food_delivery_order")
    food_delivery["table_aliases"] = food_aliases

    food_fields = list(food_delivery.get("fields") or [])
    for field in ("id", "created_time", "delivery_fee", "delivery_time", "seat_no",
                  "station_food_store_id", "trip_id"):
        if field not in food_fields:
            food_fields.append(field)
    food_delivery["fields"] = food_fields

    food_field_aliases = dict(food_delivery.get("field_aliases") or {})
    food_field_aliases.setdefault("orderId", "order_id")
    food_delivery["field_aliases"] = food_field_aliases

    consign = mapping.setdefault(
        "ts-consign-service",
        {
            "db": "ts-consign-mysql",
            "tables": [],
        },
    )
    consign.setdefault("db", "ts-consign-mysql")
    consign.setdefault("schema", "ts-consign-mysql")
    consign_tables = list(consign.get("tables") or [])
    for table in ("consign_record", "consign", "consign_orders"):
        if table not in consign_tables:
            consign_tables.append(table)
    consign["tables"] = consign_tables

    consign_aliases = dict(consign.get("table_aliases") or {})
    consign_aliases.setdefault("consign", "consign_record")
    consign_aliases.setdefault("consign_orders", "consign_record")
    consign["table_aliases"] = consign_aliases

    consign_fields = list(consign.get("fields") or [])
    for field in ("id", "order_id", "account_id", "from", "to",
                  "consign_record_id", "consign_record_price", "weight"):
        if field not in consign_fields:
            consign_fields.append(field)
    consign["fields"] = consign_fields

    consign_field_aliases = dict(consign.get("field_aliases") or {})
    consign_field_aliases.setdefault("orderId", "order_id")
    consign_field_aliases.setdefault("accountId", "account_id")
    consign_field_aliases.setdefault("price", "consign_record_price")
    consign["field_aliases"] = consign_field_aliases

    consign_price = mapping.setdefault(
        "ts-consign-price-service",
        {
            "db": "ts-consign-price-mysql",
            "tables": [],
        },
    )
    consign_price.setdefault("db", "ts-consign-price-mysql")
    consign_price.setdefault("schema", "ts-consign-price-mysql")
    cp_tables = list(consign_price.get("tables") or [])
    for table in ("consign_price",):
        if table not in cp_tables:
            cp_tables.append(table)
    consign_price["tables"] = cp_tables

    cp_fields = list(consign_price.get("fields") or [])
    for field in ("id", "initial_price", "initial_weight", "within_price", "beyond_price", "idx"):
        if field not in cp_fields:
            cp_fields.append(field)
    consign_price["fields"] = cp_fields

    seat = mapping.setdefault(
        "ts-seat-service",
        {
            "db": "ts-order-mysql",
            "tables": [],
        },
    )
    seat["db"] = "ts-order-mysql"
    seat.setdefault("schema", "ts")
    seat_tables = list(seat.get("tables") or [])
    if "orders" not in seat_tables:
        seat_tables.append("orders")
    seat["tables"] = seat_tables

    seat_aliases = dict(seat.get("table_aliases") or {})
    seat_aliases.setdefault("seat_inventory", "orders")
    seat["table_aliases"] = seat_aliases

    seat_fields = list(seat.get("fields") or [])
    for field in ("id", "status", "seat_number", "travel_date", "train_number"):
        if field not in seat_fields:
            seat_fields.append(field)
    seat["fields"] = seat_fields

    seat_field_aliases = dict(seat.get("field_aliases") or {})
    seat_field_aliases.setdefault("remaining_seats", "status")
    seat["field_aliases"] = seat_field_aliases

    preserve = mapping.setdefault(
        "ts-preserve-service",
        {
            "db": "ts-order-mysql",
            "tables": [],
        },
    )
    preserve["db"] = "ts-order-mysql"
    preserve.setdefault("schema", "ts")
    preserve_tables = list(preserve.get("tables") or [])
    if "orders" not in preserve_tables:
        preserve_tables.append("orders")
    preserve["tables"] = preserve_tables

    preserve_aliases = dict(preserve.get("table_aliases") or {})
    preserve_aliases.setdefault("preserve_orders", "orders")
    preserve["table_aliases"] = preserve_aliases

    preserve_fields = list(preserve.get("fields") or [])
    for field in ("id", "status", "account_id", "contacts_name", "contacts_document_number", "from_station", "to_station"):
        if field not in preserve_fields:
            preserve_fields.append(field)
    preserve["fields"] = preserve_fields

    price = mapping.setdefault(
        "ts-price-service",
        {
            "db": "ts-price-mysql",
            "tables": [],
        },
    )
    price.setdefault("db", "ts-price-mysql")
    price.setdefault("schema", "ts-price-mysql")
    price_tables = list(price.get("tables") or [])
    for table in ("prices",):
        if table not in price_tables:
            price_tables.append(table)
    price["tables"] = price_tables

    price_aliases = dict(price.get("table_aliases") or {})
    price_aliases.setdefault("price_config", "prices")
    price["table_aliases"] = price_aliases

    price_fields = list(price.get("fields") or [])
    for field in ("id", "train_type", "route_id", "basic_price_rate", "first_class_price_rate"):
        if field not in price_fields:
            price_fields.append(field)
    price["fields"] = price_fields

    price_field_aliases = dict(price.get("field_aliases") or {})
    price_field_aliases.setdefault("basicPriceRate", "basic_price_rate")
    price_field_aliases.setdefault("firstClassPriceRate", "first_class_price_rate")
    price["field_aliases"] = price_field_aliases

    execute = mapping.setdefault(
        "ts-execute-service",
        {
            "db": "ts-order-mysql",
            "tables": [],
        },
    )
    execute["db"] = "ts-order-mysql"
    execute.setdefault("schema", "ts")
    execute_tables = list(execute.get("tables") or [])
    if "orders" not in execute_tables:
        execute_tables.append("orders")
    execute["tables"] = execute_tables

    notification = mapping.setdefault(
        "ts-notification-service",
        {
            "db": "ts-notification-mysql",
            "tables": [],
        },
    )
    notification.setdefault("db", "ts-notification-mysql")
    notification.setdefault("schema", "ts-notification-mysql")
    notif_tables = list(notification.get("tables") or [])
    for table in ("notify_info",):
        if table not in notif_tables:
            notif_tables.append(table)
    notification["tables"] = notif_tables

    notif_fields = list(notification.get("fields") or [])
    for field in ("id", "order_number", "send_status", "email", "username", "start_place", "end_place"):
        if field not in notif_fields:
            notif_fields.append(field)
    notification["fields"] = notif_fields

    notif_field_aliases = dict(notification.get("field_aliases") or {})
    notif_field_aliases.setdefault("orderId", "order_number")
    notification["field_aliases"] = notif_field_aliases

    security = mapping.setdefault(
        "ts-security-service",
        {
            "db": "ts-security-mysql",
            "tables": [],
        },
    )
    security.setdefault("db", "ts-security-mysql")
    security.setdefault("schema", "ts-security-mysql")
    sec_tables = list(security.get("tables") or [])
    for table in ("security_config",):
        if table not in sec_tables:
            sec_tables.append(table)
    security["tables"] = sec_tables

    sec_fields = list(security.get("fields") or [])
    for field in ("id", "name", "value", "description"):
        if field not in sec_fields:
            sec_fields.append(field)
    security["fields"] = sec_fields

    config = mapping.setdefault(
        "ts-config-service",
        {
            "db": "ts-config-mysql",
            "tables": [],
        },
    )
    config_svc = mapping["ts-config-service"]
    config_svc.setdefault("db", "ts-config-mysql")
    config_svc.setdefault("schema", "ts-config-mysql")
    cfg_tables = list(config_svc.get("tables") or [])
    for table in ("config",):
        if table not in cfg_tables:
            cfg_tables.append(table)
    config_svc["tables"] = cfg_tables

    cfg_fields = list(config_svc.get("fields") or [])
    for field in ("name", "value", "description"):
        if field not in cfg_fields:
            cfg_fields.append(field)
    config_svc["fields"] = cfg_fields

    inside_payment = mapping.setdefault(
        "ts-inside-payment-service",
        {
            "db": "ts-inside-payment-mysql",
            "tables": [],
        },
    )
    inside_payment.setdefault("db", "ts-inside-payment-mysql")
    inside_payment.setdefault("schema", "ts-inside-payment-mysql")
    ip_tables = list(inside_payment.get("tables") or [])
    for table in ("inside_payment", "inside_money"):
        if table not in ip_tables:
            ip_tables.append(table)
    inside_payment["tables"] = ip_tables

    ip_fields = list(inside_payment.get("fields") or [])
    for field in ("id", "order_id", "price", "type", "user_id", "money"):
        if field not in ip_fields:
            ip_fields.append(field)
    inside_payment["fields"] = ip_fields

    voucher = mapping.setdefault(
        "ts-voucher-service",
        {
            "db": "ts-voucher-mysql",
            "tables": [],
        },
    )
    voucher.setdefault("db", "ts-voucher-mysql")
    voucher.setdefault("schema", "ts-voucher-mysql")
    v_tables = list(voucher.get("tables") or [])
    if "voucher" not in v_tables:
        v_tables.append("voucher")
    voucher["tables"] = v_tables

    v_fields = list(voucher.get("fields") or [])
    for field in ("voucher_id", "order_id", "price", "travelDate", "trainNumber", "seatClass", "seatNumber", "contactName"):
        if field not in v_fields:
            v_fields.append(field)
    voucher["fields"] = v_fields

    delivery_svc = mapping.setdefault(
        "ts-delivery-service",
        {
            "db": "ts-delivery-mysql",
            "tables": [],
        },
    )
    delivery_svc.setdefault("db", "ts-delivery-mysql")
    delivery_svc.setdefault("schema", "ts-delivery-mysql")
    d_tables = list(delivery_svc.get("tables") or [])
    if "delivery" not in d_tables:
        d_tables.append("delivery")
    delivery_svc["tables"] = d_tables

    d_fields = list(delivery_svc.get("fields") or [])
    for field in ("id", "order_id", "food_name", "station_name", "store_name"):
        if field not in d_fields:
            d_fields.append(field)
    delivery_svc["fields"] = d_fields

    train = mapping.setdefault(
        "ts-train-service",
        {
            "db": "ts-train-mysql",
            "tables": [],
        },
    )
    train.setdefault("db", "ts-train-mysql")
    train.setdefault("schema", "ts-train-mysql")
    t_tables = list(train.get("tables") or [])
    if "train_type" not in t_tables:
        t_tables.append("train_type")
    train["tables"] = t_tables

    t_fields = list(train.get("fields") or [])
    for field in ("id", "name", "economy_class", "confort_class", "average_speed"):
        if field not in t_fields:
            t_fields.append(field)
    train["fields"] = t_fields

    order_other = mapping.setdefault(
        "ts-order-other-service",
        {
            "db": "ts-order-other-mysql",
            "tables": [],
        },
    )
    order_other["db"] = "ts-order-other-mysql"
    order_other["schema"] = "ts-order-other-mysql"
    oo_tables = list(order_other.get("tables") or [])
    if "orders_other" not in oo_tables:
        oo_tables.append("orders_other")
    order_other["tables"] = oo_tables

    oo_fields = list(order_other.get("fields") or [])
    for field in ("id", "status", "price", "train_number", "account_id", "contacts_name", "from_station", "to_station"):
        if field not in oo_fields:
            oo_fields.append(field)
    order_other["fields"] = oo_fields

    wait_order = mapping.setdefault(
        "ts-wait-order-service",
        {
            "db": "ts-wait-order-mysql",
            "tables": [],
        },
    )
    wait_order.setdefault("db", "ts-wait-order-mysql")
    wait_order.setdefault("schema", "ts")
    wo_tables = list(wait_order.get("tables") or [])
    if "wait_list_order" not in wo_tables:
        wo_tables.append("wait_list_order")
    wait_order["tables"] = wo_tables

    wo_fields = list(wait_order.get("fields") or [])
    for field in ("id", "account_id", "train_number", "from_station", "to_station",
                  "seat_type", "travel_time", "status", "price", "wait_util_time",
                  "contacts_name", "contacts_document_number", "contacts_document_type",
                  "contacts_id", "created_time"):
        if field not in wo_fields:
            wo_fields.append(field)
    wait_order["fields"] = wo_fields

    return {
        "ts-payment-service": {
            "schema": payment.get("schema"),
            "tables": payment.get("tables"),
            "table_aliases": payment.get("table_aliases"),
            "fields": payment.get("fields"),
            "field_aliases": payment.get("field_aliases"),
        },
        "ts-contacts-service": {
            "schema": contacts.get("schema"),
            "tables": contacts.get("tables"),
            "table_aliases": contacts.get("table_aliases"),
            "fields": contacts.get("fields"),
            "field_aliases": contacts.get("field_aliases"),
        },
        "ts-route-service": {
            "schema": route.get("schema"),
            "tables": route.get("tables"),
            "table_aliases": route.get("table_aliases"),
        },
        "ts-travel-service": {
            "schema": travel.get("schema"),
            "tables": travel.get("tables"),
            "table_aliases": travel.get("table_aliases"),
            "field_aliases": travel.get("field_aliases"),
        },
        "ts-assurance-service": {
            "schema": assurance.get("schema"),
            "tables": assurance.get("tables"),
            "table_aliases": assurance.get("table_aliases"),
            "fields": assurance.get("fields"),
            "field_aliases": assurance.get("field_aliases"),
        },
        "ts-food-delivery-service": {
            "schema": food_delivery.get("schema"),
            "tables": food_delivery.get("tables"),
            "table_aliases": food_delivery.get("table_aliases"),
            "fields": food_delivery.get("fields"),
            "field_aliases": food_delivery.get("field_aliases"),
        },
        "ts-food-service": {
            "schema": food_service.get("schema"),
            "tables": food_service.get("tables"),
            "fields": food_service.get("fields"),
            "field_aliases": food_service.get("field_aliases"),
        },
        "ts-consign-service": {
            "schema": consign.get("schema"),
            "tables": consign.get("tables"),
            "table_aliases": consign.get("table_aliases"),
            "fields": consign.get("fields"),
            "field_aliases": consign.get("field_aliases"),
        },
        "ts-consign-price-service": {
            "schema": consign_price.get("schema"),
            "tables": consign_price.get("tables"),
            "fields": consign_price.get("fields"),
        },
        "ts-seat-service": {
            "schema": seat.get("schema"),
            "tables": seat.get("tables"),
            "table_aliases": seat.get("table_aliases"),
            "fields": seat.get("fields"),
            "field_aliases": seat.get("field_aliases"),
        },
        "ts-preserve-service": {
            "schema": preserve.get("schema"),
            "tables": preserve.get("tables"),
            "table_aliases": preserve.get("table_aliases"),
            "fields": preserve.get("fields"),
            "field_aliases": preserve.get("field_aliases"),
        },
        "ts-price-service": {
            "schema": price.get("schema"),
            "tables": price.get("tables"),
            "table_aliases": price.get("table_aliases"),
            "fields": price.get("fields"),
            "field_aliases": price.get("field_aliases"),
        },
        "ts-execute-service": {
            "schema": execute.get("schema"),
            "tables": execute.get("tables"),
        },
        "ts-notification-service": {
            "schema": notification.get("schema"),
            "tables": notification.get("tables"),
            "fields": notification.get("fields"),
            "field_aliases": notification.get("field_aliases"),
        },
        "ts-security-service": {
            "schema": security.get("schema"),
            "tables": security.get("tables"),
            "fields": security.get("fields"),
        },
        "ts-config-service": {
            "schema": config_svc.get("schema"),
            "tables": config_svc.get("tables"),
            "fields": config_svc.get("fields"),
        },
        "ts-inside-payment-service": {
            "schema": inside_payment.get("schema"),
            "tables": inside_payment.get("tables"),
            "fields": inside_payment.get("fields"),
        },
        "ts-voucher-service": {
            "schema": voucher.get("schema"),
            "tables": voucher.get("tables"),
            "fields": voucher.get("fields"),
        },
        "ts-delivery-service": {
            "schema": delivery_svc.get("schema"),
            "tables": delivery_svc.get("tables"),
            "fields": delivery_svc.get("fields"),
        },
        "ts-train-service": {
            "schema": train.get("schema"),
            "tables": train.get("tables"),
            "fields": train.get("fields"),
        },
        "ts-order-other-service": {
            "schema": order_other.get("schema"),
            "tables": order_other.get("tables"),
            "fields": order_other.get("fields"),
        },
        "ts-wait-order-service": {
            "schema": wait_order.get("schema"),
            "tables": wait_order.get("tables"),
            "fields": wait_order.get("fields"),
        },
    }
