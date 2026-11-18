"""SQLite mock-bank database and auditable tools for the Gate-4 agent.

All data is fictional.  No real customer, card, transfer, or credential is used.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DB_PATH = ROOT / "data" / "gate4" / "agent.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS customers (
    customer_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    email TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS cards (
    card_id TEXT PRIMARY KEY,
    customer_id TEXT NOT NULL,
    status TEXT NOT NULL,
    delivery_status TEXT NOT NULL,
    last4 TEXT NOT NULL,
    activated_at TEXT
);
CREATE TABLE IF NOT EXISTS transfers (
    transfer_id TEXT PRIMARY KEY,
    customer_id TEXT NOT NULL,
    amount REAL NOT NULL,
    currency TEXT NOT NULL,
    status TEXT NOT NULL,
    recipient TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS disputes (
    dispute_id TEXT PRIMARY KEY,
    transaction_id TEXT NOT NULL,
    customer_id TEXT NOT NULL,
    reason TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS tickets (
    ticket_id TEXT PRIMARY KEY,
    customer_id TEXT NOT NULL,
    intent TEXT NOT NULL,
    issue TEXT NOT NULL,
    status TEXT NOT NULL,
    priority TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS audit_events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    customer_id TEXT NOT NULL,
    tool TEXT NOT NULL,
    args_json TEXT NOT NULL,
    result_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def connect(path: Path = DB_PATH) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def seed(path: Path = DB_PATH, reset: bool = False) -> None:
    conn = connect(path)
    if reset:
        conn.executescript(
            """
            DROP TABLE IF EXISTS audit_events;
            DROP TABLE IF EXISTS tickets;
            DROP TABLE IF EXISTS disputes;
            DROP TABLE IF EXISTS transfers;
            DROP TABLE IF EXISTS cards;
            DROP TABLE IF EXISTS customers;
            """
        )
    conn.executescript(SCHEMA)
    if conn.execute("SELECT COUNT(*) FROM customers").fetchone()[0] == 0:
        conn.executemany(
            "INSERT INTO customers(customer_id, name, email) VALUES (?, ?, ?)",
            [
                ("CUST-1001", "Alice Demo", "alice@example.test"),
                ("CUST-1002", "Bob Demo", "bob@example.test"),
                ("CUST-1003", "Carol Demo", "carol@example.test"),
            ],
        )
        conn.executemany(
            "INSERT INTO cards(card_id, customer_id, status, delivery_status, last4, activated_at) VALUES (?, ?, ?, ?, ?, ?)",
            [
                ("CARD-1001", "CUST-1001", "active", "delivered", "4242", "2026-10-01T10:00:00+00:00"),
                ("CARD-1002", "CUST-1001", "inactive", "in_transit", "1881", None),
                ("CARD-2001", "CUST-1002", "active", "delivered", "7777", "2026-09-15T09:00:00+00:00"),
            ],
        )
        conn.executemany(
            "INSERT INTO transfers(transfer_id, customer_id, amount, currency, status, recipient, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                ("TXN-2001", "CUST-1001", 250.00, "USD", "pending", "ACME Supplies", "2026-11-15T08:00:00+00:00"),
                ("TXN-2002", "CUST-1001", 90.00, "USD", "pending", "Landlord Demo", "2026-11-16T08:00:00+00:00"),
                ("TXN-2003", "CUST-1001", 120.00, "USD", "completed", "Utility Co", "2026-11-10T08:00:00+00:00"),
                ("TXN-2004", "CUST-1002", 500.00, "USD", "pending", "Other Demo", "2026-11-14T08:00:00+00:00"),
            ],
        )
        conn.execute(
            "INSERT INTO tickets(ticket_id, customer_id, intent, issue, status, priority, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("TCK-1000", "CUST-1001", "general", "Welcome to the fictional demo queue", "open", "low", now()),
        )
    conn.commit()
    conn.close()


class ToolError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message

    def as_dict(self) -> dict[str, str]:
        return {"error": self.code, "message": self.message}


class BankingTools:
    """Tools bound to one fictional customer session."""

    def __init__(self, customer_id: str = "CUST-1001", session_id: str = "SESSION-DEMO", path: Path = DB_PATH):
        self.customer_id = customer_id
        self.session_id = session_id
        seed(path)
        self.conn = connect(path)

    def close(self) -> None:
        self.conn.close()

    def _audit(self, tool: str, args: dict[str, Any], result: dict[str, Any]) -> None:
        self.conn.execute(
            "INSERT INTO audit_events(session_id, customer_id, tool, args_json, result_json, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (self.session_id, self.customer_id, tool, json.dumps(args, ensure_ascii=False), json.dumps(result, ensure_ascii=False), now()),
        )
        self.conn.commit()

    def _card(self, card_id: str) -> sqlite3.Row:
        row = self.conn.execute("SELECT * FROM cards WHERE card_id = ?", (card_id,)).fetchone()
        if row is None:
            raise ToolError("not_found", f"Card {card_id} is not in the fictional database")
        if row["customer_id"] != self.customer_id:
            raise ToolError("ownership_denied", "That card belongs to another fictional customer")
        return row

    def _transfer(self, transfer_id: str) -> sqlite3.Row:
        row = self.conn.execute("SELECT * FROM transfers WHERE transfer_id = ?", (transfer_id,)).fetchone()
        if row is None:
            raise ToolError("not_found", f"Transfer {transfer_id} is not in the fictional database")
        if row["customer_id"] != self.customer_id:
            raise ToolError("ownership_denied", "That transfer belongs to another fictional customer")
        return row

    def get_card_delivery_status(self, card_id: str) -> dict[str, Any]:
        args = {"card_id": card_id}
        try:
            row = self._card(card_id)
            result = {
                "card_id": row["card_id"],
                "delivery_status": row["delivery_status"],
                "status": row["status"],
                "last4": row["last4"],
            }
            self._audit("get_card_delivery_status", args, result)
            return result
        except ToolError as exc:
            self._audit("get_card_delivery_status", args, exc.as_dict())
            raise

    def get_transfer_status(self, transfer_id: str) -> dict[str, Any]:
        args = {"transfer_id": transfer_id}
        try:
            row = self._transfer(transfer_id)
            result = {
                "transfer_id": row["transfer_id"],
                "status": row["status"],
                "amount": row["amount"],
                "currency": row["currency"],
                "recipient": row["recipient"],
            }
            self._audit("get_transfer_status", args, result)
            return result
        except ToolError as exc:
            self._audit("get_transfer_status", args, exc.as_dict())
            raise

    def cancel_transfer(self, transfer_id: str, confirmed: bool = False) -> dict[str, Any]:
        args = {"transfer_id": transfer_id, "confirmed": confirmed}
        try:
            row = self._transfer(transfer_id)
            if row["status"] == "cancelled":
                result = {"status": "already_cancelled", "transfer_id": transfer_id}
                self._audit("cancel_transfer", args, result)
                return result
            if row["status"] != "pending":
                result = {"status": "not_pending", "transfer_id": transfer_id, "current_status": row["status"]}
                self._audit("cancel_transfer", args, result)
                return result
            if not confirmed:
                result = {
                    "status": "confirmation_required",
                    "transfer_id": transfer_id,
                    "amount": row["amount"],
                    "recipient": row["recipient"],
                }
                self._audit("cancel_transfer", args, result)
                return result
            self.conn.execute("UPDATE transfers SET status = 'cancelled' WHERE transfer_id = ?", (transfer_id,))
            self.conn.commit()
            result = {"status": "cancelled", "transfer_id": transfer_id, "amount": row["amount"]}
            self._audit("cancel_transfer", args, result)
            return result
        except ToolError as exc:
            self._audit("cancel_transfer", args, exc.as_dict())
            raise

    def activate_card(self, card_id: str, confirmed: bool = False) -> dict[str, Any]:
        args = {"card_id": card_id, "confirmed": confirmed}
        try:
            row = self._card(card_id)
            if row["status"] == "active":
                result = {"status": "already_active", "card_id": card_id}
                self._audit("activate_card", args, result)
                return result
            if not confirmed:
                result = {"status": "confirmation_required", "card_id": card_id, "last4": row["last4"]}
                self._audit("activate_card", args, result)
                return result
            self.conn.execute(
                "UPDATE cards SET status = 'active', activated_at = ? WHERE card_id = ?",
                (now(), card_id),
            )
            self.conn.commit()
            result = {"status": "activated", "card_id": card_id}
            self._audit("activate_card", args, result)
            return result
        except ToolError as exc:
            self._audit("activate_card", args, exc.as_dict())
            raise

    def create_dispute(self, transaction_id: str, reason: str) -> dict[str, Any]:
        args = {"transaction_id": transaction_id, "reason": reason}
        try:
            row = self._transfer(transaction_id)
            existing = self.conn.execute(
                "SELECT dispute_id FROM disputes WHERE transaction_id = ? AND customer_id = ?",
                (transaction_id, self.customer_id),
            ).fetchone()
            if existing:
                result = {"status": "already_exists", "dispute_id": existing["dispute_id"]}
                self._audit("create_dispute", args, result)
                return result
            dispute_id = f"DSP-{self.conn.execute('SELECT COUNT(*) FROM disputes').fetchone()[0] + 3001}"
            self.conn.execute(
                "INSERT INTO disputes(dispute_id, transaction_id, customer_id, reason, status, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (dispute_id, transaction_id, self.customer_id, reason, "open", now()),
            )
            self.conn.commit()
            result = {"status": "created", "dispute_id": dispute_id, "transaction_id": transaction_id}
            self._audit("create_dispute", args, result)
            return result
        except ToolError as exc:
            self._audit("create_dispute", args, exc.as_dict())
            raise

    def create_ticket(self, intent: str, issue: str, priority: str = "normal") -> dict[str, Any]:
        args = {"intent": intent, "issue": issue, "priority": priority}
        ticket_id = f"TCK-{self.conn.execute('SELECT COUNT(*) FROM tickets').fetchone()[0] + 1001}"
        self.conn.execute(
            "INSERT INTO tickets(ticket_id, customer_id, intent, issue, status, priority, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (ticket_id, self.customer_id, intent, issue, "open", priority, now()),
        )
        self.conn.commit()
        result = {"status": "created", "ticket_id": ticket_id, "intent": intent}
        self._audit("create_ticket", args, result)
        return result

    def handoff_to_human(self, intent: str, reason: str) -> dict[str, Any]:
        args = {"intent": intent, "reason": reason}
        result = self.create_ticket(intent, f"HUMAN HANDOFF: {reason}", priority="high")
        result["status"] = "handoff_created"
        self._audit("handoff_to_human", args, result)
        return result

    def snapshot(self) -> dict[str, list[dict[str, Any]]]:
        def rows(table: str, order: str) -> list[dict[str, Any]]:
            return [dict(r) for r in self.conn.execute(f"SELECT * FROM {table} ORDER BY {order}").fetchall()]

        return {
            "cards": rows("cards", "card_id"),
            "transfers": rows("transfers", "transfer_id"),
            "disputes": rows("disputes", "dispute_id"),
            "tickets": rows("tickets", "ticket_id"),
            "audit_events": rows("audit_events", "event_id DESC"),
        }
