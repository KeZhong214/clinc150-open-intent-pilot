"""Stateful multi-turn customer-service agent for the Gate-4 demo.

The agent uses the real B3-H/B4 router for intent and a deterministic slot /
confirmation state machine for the business workflow.  No LLM is required.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from src.gate4.db import BankingTools, ToolError
from src.gate4.router import RouteResult

CARD_RE = re.compile(r"\bCARD-\d{4,}\b", re.IGNORECASE)
TXN_RE = re.compile(r"\bTXN-\d{4,}\b", re.IGNORECASE)
YES_WORDS = {"yes", "y", "confirm", "confirmed", "go ahead", "do it", "please do", "sure"}
NO_WORDS = {"no", "n", "cancel", "stop", "don't", "do not"}

CARD_INTENTS = {"card_arrival", "activate_my_card"}
TRANSFER_INTENTS = {"transfer_not_received_by_recipient", "transfer_timing", "cancel_transfer"}
DISPUTE_INTENTS = {"card_payment_not_recognised"}


@dataclass
class Turn:
    response: str
    route: RouteResult | None
    tools: list[dict[str, Any]] = field(default_factory=list)
    state: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentState:
    pending_intent: str | None = None
    pending_id_type: str | None = None
    pending_id: str | None = None
    pending_action: str | None = None
    pending_clarification: list[str] = field(default_factory=list)
    awaiting_confirmation: bool = False
    awaiting_clarification: bool = False
    last_route: RouteResult | None = None


def _has_yes(text: str) -> bool:
    return text.strip().lower() in YES_WORDS or any(word in text.lower().split() for word in ("yes", "confirm"))


def _has_no(text: str) -> bool:
    return text.strip().lower() in NO_WORDS or any(word in text.lower().split() for word in ("no", "cancel"))


class BankAgent:
    def __init__(self, router, tools: BankingTools | None = None, *, ambiguity_margin: float = 0.01):
        self.router = router
        self.tools = tools or BankingTools()
        self.state = AgentState()
        self.ambiguity_margin = ambiguity_margin

    def _tool(self, name: str, **kwargs) -> dict[str, Any]:
        fn = getattr(self.tools, name)
        result = fn(**kwargs)
        return {"tool": name, "args": kwargs, "result": result}

    def _handoff(self, intent: str, reason: str) -> dict[str, Any]:
        result = self.tools.handoff_to_human(intent, reason)
        return {"tool": "handoff_to_human", "args": {"intent": intent, "reason": reason}, "result": result}

    def handle(self, text: str) -> Turn:
        text = text.strip()
        if not text:
            return Turn("Please type a message.", None, [], self.state.__dict__)

        if self.state.awaiting_confirmation:
            return self._handle_confirmation(text)
        if self.state.awaiting_clarification:
            return self._handle_clarification(text)
        if self.state.pending_intent and self.state.pending_id_type:
            return self._handle_slot(text)

        route = self.router.route(text)
        self.state.last_route = route
        if not route.accepted:
            tool = self._handoff("OOS", "Router rejected the request as unsupported/unknown")
            return Turn(
                "I’m not confident this is a supported banking request. I’m handing you to a human specialist and creating a ticket.",
                route,
                [tool],
                self.state.__dict__,
            )

        if route.margin < self.ambiguity_margin:
            candidates = [route.top3[0][0], route.top3[1][0]]
            self.state.awaiting_clarification = True
            self.state.pending_clarification = candidates
            return Turn(
                f"I’m not fully sure whether you mean **{candidates[0]}** or **{candidates[1]}**. "
                "Could you clarify in a few words?",
                route,
                [],
                self.state.__dict__,
            )

        return self._dispatch(route, text)

    def _dispatch(self, route: RouteResult, text: str) -> Turn:
        intent = route.intent
        card_id = CARD_RE.search(text)
        txn_id = TXN_RE.search(text)

        if intent in CARD_INTENTS:
            if not card_id:
                self.state.pending_intent = intent
                self.state.pending_id_type = "card"
                return Turn(
                    f"I can help with **{intent}**. Please provide your fictional card ID (for example CARD-1001).",
                    route,
                    [],
                    self.state.__dict__,
                )
            return self._run_card_action(intent, card_id.group(0).upper(), route)

        if intent in TRANSFER_INTENTS:
            if not txn_id:
                self.state.pending_intent = intent
                self.state.pending_id_type = "transfer"
                return Turn(
                    f"I can help with **{intent}**. Please provide your fictional transfer ID (for example TXN-2001).",
                    route,
                    [],
                    self.state.__dict__,
                )
            return self._run_transfer_action(intent, txn_id.group(0).upper(), route)

        if intent in DISPUTE_INTENTS:
            if not txn_id:
                self.state.pending_intent = intent
                self.state.pending_id_type = "dispute_txn"
                return Turn(
                    "I can open a dispute. Please provide the fictional transaction/transfer ID (for example TXN-2001).",
                    route,
                    [],
                    self.state.__dict__,
                )
            return self._run_dispute(intent, txn_id.group(0).upper(), text, route)

        # All other known intents are safe to route to a ticket/handoff; no financial action.
        tool = self.tools.create_ticket(intent, text, priority="normal")
        return Turn(
            f"I’ve logged this as **{intent}** and created ticket **{tool['ticket_id']}** for a specialist. "
            "No financial action was taken.",
            route,
            [{"tool": "create_ticket", "args": {"intent": intent, "issue": text}, "result": tool}],
            self.state.__dict__,
        )

    def _run_card_action(self, intent: str, card_id: str, route: RouteResult) -> Turn:
        try:
            if intent == "activate_my_card":
                result = self.tools.activate_card(card_id, confirmed=False)
                if result.get("status") == "confirmation_required":
                    self.state.pending_intent = intent
                    self.state.pending_action = "activate_card"
                    self.state.pending_id = card_id
                    self.state.awaiting_confirmation = True
                    return Turn(
                        f"Card **{card_id}** is currently inactive. Activating a card is a sensitive action. "
                        "Do you explicitly confirm activation?",
                        route,
                        [{"tool": "activate_card", "args": {"card_id": card_id, "confirmed": False}, "result": result}],
                        self.state.__dict__,
                    )
                return Turn(f"Card **{card_id}** is already active; no change was made.", route,
                            [{"tool": "activate_card", "args": {"card_id": card_id, "confirmed": False}, "result": result}],
                            self.state.__dict__)
            result = self.tools.get_card_delivery_status(card_id)
            return Turn(
                f"Card **{card_id}**: delivery status is **{result['delivery_status']}**, card status is **{result['status']}**.",
                route,
                [{"tool": "get_card_delivery_status", "args": {"card_id": card_id}, "result": result}],
                self.state.__dict__,
            )
        except ToolError as exc:
            tool = self._handoff(intent, f"{exc.code}: {exc.message}")
            return Turn(
                "I couldn’t perform that card action. I’m creating a human handoff ticket and did not make any financial change.",
                route,
                [tool],
                self.state.__dict__,
            )

    def _run_transfer_action(self, intent: str, txn_id: str, route: RouteResult) -> Turn:
        try:
            if intent == "cancel_transfer":
                result = self.tools.cancel_transfer(txn_id, confirmed=False)
                if result.get("status") == "confirmation_required":
                    self.state.pending_intent = intent
                    self.state.pending_action = "cancel_transfer"
                    self.state.pending_id = txn_id
                    self.state.awaiting_confirmation = True
                    return Turn(
                        f"Transfer **{txn_id}** is still pending ({result['amount']:.2f} USD to {result['recipient']}). "
                        "Cancelling it is a sensitive action. Do you explicitly confirm cancellation?",
                        route,
                        [{"tool": "cancel_transfer", "args": {"transfer_id": txn_id, "confirmed": False}, "result": result}],
                        self.state.__dict__,
                    )
                if result.get("status") == "already_cancelled":
                    return Turn(f"Transfer **{txn_id}** was already cancelled; no duplicate change was made.", route,
                                [{"tool": "cancel_transfer", "args": {"transfer_id": txn_id, "confirmed": False}, "result": result}],
                                self.state.__dict__)
                if result.get("status") == "not_pending":
                    return Turn(
                        f"Transfer **{txn_id}** is **{result['current_status']}**, so it cannot be cancelled from this demo.",
                        route,
                        [{"tool": "cancel_transfer", "args": {"transfer_id": txn_id, "confirmed": False}, "result": result}],
                        self.state.__dict__,
                    )
                return Turn(f"Transfer **{txn_id}** was cancelled.", route,
                            [{"tool": "cancel_transfer", "args": {"transfer_id": txn_id, "confirmed": False}, "result": result}],
                            self.state.__dict__)
            result = self.tools.get_transfer_status(txn_id)
            return Turn(
                f"Transfer **{txn_id}** is **{result['status']}**; amount {result['amount']:.2f} {result['currency']} to {result['recipient']}.",
                route,
                [{"tool": "get_transfer_status", "args": {"transfer_id": txn_id}, "result": result}],
                self.state.__dict__,
            )
        except ToolError as exc:
            tool = self._handoff(intent, f"{exc.code}: {exc.message}")
            return Turn(
                "I couldn’t access that transfer. I’m creating a human handoff ticket and did not make any change.",
                route,
                [tool],
                self.state.__dict__,
            )

    def _run_dispute(self, intent: str, txn_id: str, text: str, route: RouteResult) -> Turn:
        try:
            result = self.tools.create_dispute(txn_id, text)
            return Turn(
                f"I opened dispute **{result['dispute_id']}** for **{txn_id}**. A specialist will review it; "
                "this demo did not move real money.",
                route,
                [{"tool": "create_dispute", "args": {"transaction_id": txn_id, "reason": text}, "result": result}],
                self.state.__dict__,
            )
        except ToolError as exc:
            tool = self._handoff(intent, f"{exc.code}: {exc.message}")
            return Turn(
                "I couldn’t open that dispute. I’m handing you to a human specialist.",
                route,
                [tool],
                self.state.__dict__,
            )

    def _handle_slot(self, text: str) -> Turn:
        self.state.awaiting_confirmation = False
        if self.state.pending_id_type in ("card", "dispute_card"):
            match = CARD_RE.search(text)
            if not match:
                return Turn(
                    "Please provide a fictional card ID in the form CARD-1001.",
                    self.state.last_route,
                    [],
                    self.state.__dict__,
                )
            intent = self.state.pending_intent
            pending_type = self.state.pending_id_type
            self.state.pending_intent = None
            self.state.pending_id_type = None
            if pending_type == "card":
                route = self.state.last_route
                assert route is not None
                return self._run_card_action(intent, match.group(0).upper(), route)
        if self.state.pending_id_type in ("transfer", "dispute_txn"):
            match = TXN_RE.search(text)
            if not match:
                return Turn(
                    "Please provide a fictional transfer ID in the form TXN-2001.",
                    self.state.last_route,
                    [],
                    self.state.__dict__,
                )
            intent = self.state.pending_intent
            pending_type = self.state.pending_id_type
            self.state.pending_intent = None
            self.state.pending_id_type = None
            route = self.state.last_route
            assert route is not None
            if pending_type == "transfer":
                return self._run_transfer_action(intent, match.group(0).upper(), route)
            return self._run_dispute(intent, match.group(0).upper(), text, route)
        return Turn("I lost the pending slot; please restate the request.", self.state.last_route, [], self.state.__dict__)

    def _handle_confirmation(self, text: str) -> Turn:
        yes = _has_yes(text)
        no = _has_no(text)
        if not yes and not no:
            return Turn(
                "Please answer **yes** to confirm or **no** to cancel. No action has been taken.",
                self.state.last_route,
                [],
                self.state.__dict__,
            )
        action = self.state.pending_action
        txn = self.state.pending_id
        intent = self.state.pending_intent
        self.state.awaiting_confirmation = False
        self.state.pending_action = None
        self.state.pending_id = None
        self.state.pending_intent = None
        if not yes:
            return Turn("Understood — no action taken.", self.state.last_route, [], self.state.__dict__)
        try:
            if action == "cancel_transfer":
                result = self.tools.cancel_transfer(txn, confirmed=True)
                return Turn(
                    f"Confirmed. Transfer **{txn}** is now **{result['status']}** in the mock database.",
                    self.state.last_route,
                    [{"tool": "cancel_transfer", "args": {"transfer_id": txn, "confirmed": True}, "result": result}],
                    self.state.__dict__,
                )
            if action == "activate_card":
                result = self.tools.activate_card(txn, confirmed=True)
                return Turn(
                    f"Confirmed. Card **{txn}** is now **{result['status']}** in the mock database.",
                    self.state.last_route,
                    [{"tool": "activate_card", "args": {"card_id": txn, "confirmed": True}, "result": result}],
                    self.state.__dict__,
                )
        except ToolError as exc:
            tool = self._handoff(intent or "unknown", f"{exc.code}: {exc.message}")
            return Turn("I couldn’t complete the confirmed action; a human ticket was created.", self.state.last_route, [tool], self.state.__dict__)
        return Turn("Nothing was executed.", self.state.last_route, [], self.state.__dict__)

    def _handle_clarification(self, text: str) -> Turn:
        candidates = self.state.pending_clarification
        lower = text.lower()
        selected = next((c for c in candidates if c.lower() in lower), None)
        if selected is None:
            self.state.awaiting_clarification = False
            tool = self._handoff("ambiguous", "User did not resolve the clarification")
            return Turn(
                "I’m still unsure what you need. I’m creating a human handoff ticket rather than guessing.",
                self.state.last_route,
                [tool],
                self.state.__dict__,
            )
        self.state.awaiting_clarification = False
        self.state.pending_clarification = []
        # Re-dispatch with a synthetic route for the selected candidate; the tools still enforce ownership.
        route = self.state.last_route
        if route is None:
            return Turn("Please restate the request.", None, [], self.state.__dict__)
        route = RouteResult(
            method=route.method,
            intent=selected,
            score=route.score,
            threshold=route.threshold,
            accepted=True,
            top3=route.top3,
            gate_top1=route.gate_top1,
            reject_score=route.reject_score,
            margin=route.margin,
        )
        self.state.last_route = route
        return self._dispatch(route, text)

    def reset(self) -> None:
        self.state = AgentState()


class StubRouter:
    """Deterministic router for tool/state tests; not used for model claims."""

    def __init__(self, intent: str, *, accepted: bool = True, margin: float = 0.2):
        self.intent = intent
        self.accepted = accepted
        self.margin = margin

    def route(self, text: str) -> RouteResult:
        return RouteResult(
            method="STUB",
            intent=self.intent if self.accepted else "OOS",
            score=0.9 if self.accepted else 0.2,
            threshold=0.5,
            accepted=self.accepted,
            top3=[(self.intent if self.accepted else "OOS", 0.9), ("fallback", 0.1), ("other", 0.05)],
            margin=self.margin,
        )
