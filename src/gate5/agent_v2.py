"""Gate-5 customer-service Agent v2.

Adds local policy RAG, natural-language clarification, slot memory and
session-isolated SQLite.  It deliberately keeps the intent router as the
authority; the LLM/KB layer cannot bypass tool permissions.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.gate4.db import BankingTools, ToolError, seed
from src.gate4.router import RouteResult
from src.gate5.kb import KBRetriever

CARD_RE = re.compile(r"\bCARD-\d{4,}\b", re.IGNORECASE)
TXN_RE = re.compile(r"\bTXN-\d{4,}\b", re.IGNORECASE)
YES_WORDS = {"yes", "y", "confirm", "confirmed", "go ahead", "do it", "please do", "sure", "ok", "okay"}
NO_WORDS = {"no", "n", "cancel", "stop", "don't", "do not", "not now"}

CARD_INTENTS = {"card_arrival", "activate_my_card"}
TRANSFER_INTENTS = {"transfer_not_received_by_recipient", "transfer_timing", "cancel_transfer"}
DISPUTE_INTENTS = {"card_payment_not_recognised"}
POLICY_INTENTS = {
    "transfer_fee_charged",
    "card_payment_fee_charged",
    "card_about_to_expire",
    "card_acceptance",
    "top_up_by_bank_transfer_charge",
    "top_up_by_card_charge",
    "fiat_currency_support",
    "exchange_via_app",
}
POLICY_TERMS = (
    "fee",
    "policy",
    "limit",
    "expire",
    "expiry",
    "accepted",
    "acceptance",
    "currency",
    "exchange",
    "how long",
    "human",
    "handoff",
    "complaint",
    "refund",
    "dispute process",
    "beneficiary",
    "identity",
    "statement",
    "balance",
    "replace",
    "process",
    "methods",
    "supported",
    "top-up",
    "top up",
    "charge",
    "unrecognised",
    "not recognise",
    "dispute",
    "how does",
    "what is",
)

DESCRIPTIONS = {
    "card_arrival": "checking whether a card has arrived",
    "activate_my_card": "activating a card",
    "transfer_not_received_by_recipient": "checking whether a transfer reached the recipient",
    "transfer_timing": "checking when a transfer should arrive",
    "cancel_transfer": "cancelling a pending transfer",
    "card_payment_not_recognised": "reporting an unrecognised card payment",
    "card_not_working": "troubleshooting a card that is not working",
    "card_about_to_expire": "checking a card expiry policy",
    "card_acceptance": "checking where a card can be used",
    "transfer_fee_charged": "asking about a transfer fee",
    "card_payment_fee_charged": "asking about a card payment fee",
    "top_up_by_bank_transfer_charge": "asking about a bank-transfer top-up fee",
    "top_up_by_card_charge": "asking about a card top-up fee",
    "fiat_currency_support": "asking about supported currencies",
    "exchange_via_app": "asking about exchanging currency in the app",
    "lost_or_stolen_card": "reporting a lost or stolen card",
    "card_swallowed": "reporting a card captured by a machine",
    "change_pin": "changing a card PIN",
    "get_physical_card": "requesting a physical card",
    "terminate_account": "closing an account",
}

HINTS = {
    "card_arrival": ["arrive", "arrival", "delivery", "tracking", "received my card", "where is my card", "card come"],
    "activate_my_card": ["activat", "turn on", "enable my card", "start using"],
    "transfer_not_received_by_recipient": ["not received", "hasn't arrived", "has not arrived", "recipient", "didn't get", "did not get", "not arrived"],
    "transfer_timing": ["when", "how long", "timing", "delay", "still pending", "processing"],
    "cancel_transfer": ["cancel", "stop the transfer", "call it back"],
    "card_payment_not_recognised": ["dispute", "do not recognise", "don't recognise", "unknown charge", "unrecognised", "not mine"],
    "card_payment_fee_charged": ["payment fee", "card fee", "charged a fee"],
    "transfer_fee_charged": ["transfer fee", "fee for transfer"],
    "card_about_to_expire": ["expire", "expiry", "renew"],
    "card_acceptance": ["accepted", "acceptance", "where can i use"],
}


def human(intent: str) -> str:
    if intent in DESCRIPTIONS:
        return DESCRIPTIONS[intent]
    return intent.replace("_", " ")


def clean_text(text: str) -> str:
    return " ".join(text.lower().replace("’", "'").split())


def has_yes(text: str) -> bool:
    clean = clean_text(text)
    return clean in YES_WORDS or any(w in clean.split() for w in ("yes", "confirm", "sure", "okay", "ok"))


def has_no(text: str) -> bool:
    clean = clean_text(text)
    return clean in NO_WORDS or any(w in clean.split() for w in ("no", "cancel", "stop"))


@dataclass
class TurnV2:
    response: str
    route: RouteResult | None
    tools: list[dict[str, Any]] = field(default_factory=list)
    citations: list[dict[str, Any]] = field(default_factory=list)
    state: dict[str, Any] = field(default_factory=dict)


@dataclass
class StateV2:
    pending_intent: str | None = None
    pending_slot: str | None = None
    pending_text: str = ""
    pending_action: str | None = None
    pending_id: str | None = None
    awaiting_confirmation: bool = False
    awaiting_clarification: bool = False
    clarification_candidates: list[str] = field(default_factory=list)
    last_route: RouteResult | None = None
    first_route: RouteResult | None = None


class BankAgentV2:
    def __init__(
        self,
        router,
        *,
        session_id: str | None = None,
        customer_id: str = "CUST-1001",
        db_dir: Path = Path("data/gate5/sessions"),
        kb: KBRetriever | None = None,
        llm=None,
        ambiguity_margin: float = 0.01,
    ):
        self.router = router
        self.session_id = session_id or f"S-{uuid.uuid4().hex[:10]}"
        self.db_dir = Path(db_dir)
        self.db_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.db_dir / f"{self.session_id}.db"
        seed(self.db_path)
        self.tools = BankingTools(customer_id=customer_id, session_id=self.session_id, path=self.db_path)
        self.kb = kb or KBRetriever()
        self.llm = llm
        self.ambiguity_margin = ambiguity_margin
        self.state = StateV2()

    def _tool(self, name: str, **kwargs) -> dict:
        fn = getattr(self.tools, name)
        result = fn(**kwargs)
        return {"tool": name, "args": kwargs, "result": result}

    def _handoff(self, intent: str, reason: str) -> dict:
        result = self.tools.handoff_to_human(intent, reason)
        return {"tool": "handoff_to_human", "args": {"intent": intent, "reason": reason}, "result": result}

    def _record_route(self, route: RouteResult) -> None:
        self.state.last_route = route
        if self.state.first_route is None:
            self.state.first_route = route

    def handle(self, text: str) -> TurnV2:
        text = text.strip()
        if not text:
            return TurnV2("Please type your request.", None, [], [], self.state.__dict__)
        if self.state.awaiting_confirmation:
            return self._confirmation(text)
        if self.state.awaiting_clarification:
            return self._clarification(text)
        if self.state.pending_intent and self.state.pending_slot:
            return self._slot(text)
        route = self.router.route(text)
        self._record_route(route)
        if not route.accepted:
            # Policy/FAQ questions can still be answered from the local fictional KB;
            # no tool or financial action is allowed on this path.
            if not CARD_RE.search(text) and not TXN_RE.search(text) and any(term in clean_text(text) for term in POLICY_TERMS):
                body, hits = self.kb.render(text, k=2)
                if hits:
                    return TurnV2("Here is what the **fictional demo policy** says:\n\n" + body, route, [], hits, self.state.__dict__)
            tool = self._handoff("OOS", "Router rejected the request as unsupported/unknown")
            return TurnV2(
                "I’m not confident this is a supported fictional-bank request. I’ll create a human handoff ticket instead of guessing.",
                route,
                [tool],
                [],
                self.state.__dict__,
            )
        if route.margin < self.ambiguity_margin:
            candidates = []
            for name, _ in route.top3[:2]:
                if name not in candidates:
                    candidates.append(name)
            self.state.awaiting_clarification = True
            self.state.clarification_candidates = candidates
            self.state.pending_text = text
            options = " or ".join(f"**{human(c)}**" for c in candidates)
            return TurnV2(
                f"I’m not sure which task you mean: {options}. Which one fits?",
                route,
                [],
                [],
                self.state.__dict__,
            )
        return self._dispatch(route, text)

    def _dispatch(self, route: RouteResult, text: str) -> TurnV2:
        intent = route.intent
        card = CARD_RE.search(text)
        txn = TXN_RE.search(text)

        if intent in POLICY_INTENTS:
            body, hits = self.kb.render(text, k=2)
            if hits:
                return TurnV2(
                    "Here is what the **fictional demo policy** says:\n\n" + body,
                    route,
                    [],
                    hits,
                    self.state.__dict__,
                )
            tool = self._handoff(intent, "No matching fictional policy document")
            return TurnV2(
                "I don’t have a matching fictional policy document, so I’ll hand you to a human specialist.",
                route,
                [tool],
                [],
                self.state.__dict__,
            )

        if not card and not txn and any(term in clean_text(text) for term in POLICY_TERMS):
            body, hits = self.kb.render(text, k=2)
            if hits:
                return TurnV2("Here is what the **fictional demo policy** says:\n\n" + body, route, [], hits, self.state.__dict__)
            tool = self._handoff(intent, "Policy question with no matching fictional document")
            return TurnV2("I don’t have a matching fictional policy document, so I’ll hand you to a human specialist.", route, [tool], [], self.state.__dict__)

        if intent in CARD_INTENTS:
            if not card:
                self.state.pending_intent = intent
                self.state.pending_slot = "card"
                self.state.pending_text = text
                return TurnV2(
                    f"I can help with {human(intent)}. Please provide your fictional card ID, for example CARD-1001.",
                    route,
                    [],
                    [],
                    self.state.__dict__,
                )
            return self._card_action(intent, card.group(0).upper(), route)

        if intent in TRANSFER_INTENTS:
            if not txn:
                self.state.pending_intent = intent
                self.state.pending_slot = "transfer"
                self.state.pending_text = text
                return TurnV2(
                    f"I can help with {human(intent)}. Please provide your fictional transfer ID, for example TXN-2001.",
                    route,
                    [],
                    [],
                    self.state.__dict__,
                )
            return self._transfer_action(intent, txn.group(0).upper(), route)

        if intent in DISPUTE_INTENTS:
            if not txn:
                self.state.pending_intent = intent
                self.state.pending_slot = "dispute"
                self.state.pending_text = text
                return TurnV2(
                    "I can open a fictional dispute. Please provide the transfer/transaction ID, for example TXN-2001.",
                    route,
                    [],
                    [],
                    self.state.__dict__,
                )
            return self._dispute(intent, txn.group(0).upper(), text, route)

        # Safe known intents: ticket only, no financial action.
        if intent in POLICY_INTENTS or intent in {"card_about_to_expire", "card_acceptance"}:
            body, hits = self.kb.render(text, k=2)
            if hits:
                return TurnV2("Here is what the **fictional demo policy** says:\n\n" + body, route, [], hits, self.state.__dict__)

        ticket = self.tools.create_ticket(intent, text, priority="normal")
        return TurnV2(
            f"I’ve logged this as {human(intent)} and created ticket **{ticket['ticket_id']}**. No financial action was taken.",
            route,
            [{"tool": "create_ticket", "args": {"intent": intent, "issue": text}, "result": ticket}],
            [],
            self.state.__dict__,
        )

    def _card_action(self, intent: str, card_id: str, route: RouteResult) -> TurnV2:
        try:
            if intent == "activate_my_card":
                result = self.tools.activate_card(card_id, confirmed=False)
                if result.get("status") == "confirmation_required":
                    self.state.pending_intent = intent
                    self.state.pending_action = "activate_card"
                    self.state.pending_id = card_id
                    self.state.awaiting_confirmation = True
                    return TurnV2(
                        f"Card **{card_id}** is inactive. I can activate it, but this is a sensitive action. "
                        "Do you explicitly confirm activation?",
                        route,
                        [{"tool": "activate_card", "args": {"card_id": card_id, "confirmed": False}, "result": result}],
                        [],
                        self.state.__dict__,
                    )
                return TurnV2(f"Card **{card_id}** is already active; no change was made.", route,
                              [{"tool": "activate_card", "args": {"card_id": card_id, "confirmed": False}, "result": result}], [], self.state.__dict__)
            result = self.tools.get_card_delivery_status(card_id)
            delivery_text = result["delivery_status"].replace("_", " ")
            return TurnV2(
                f"Card **{card_id}** has delivery status **{delivery_text}** and card status **{result['status']}**.",
                route,
                [{"tool": "get_card_delivery_status", "args": {"card_id": card_id}, "result": result}],
                [],
                self.state.__dict__,
            )
        except ToolError as exc:
            hs = self._handoff(intent, f"{exc.code}: {exc.message}")
            return TurnV2(
                "I couldn’t access that card. I didn’t change anything, and I’ve created a human handoff ticket.",
                route,
                [hs],
                [],
                self.state.__dict__,
            )

    def _transfer_action(self, intent: str, txn_id: str, route: RouteResult) -> TurnV2:
        try:
            if intent == "cancel_transfer":
                result = self.tools.cancel_transfer(txn_id, confirmed=False)
                if result.get("status") == "confirmation_required":
                    self.state.pending_intent = intent
                    self.state.pending_action = "cancel_transfer"
                    self.state.pending_id = txn_id
                    self.state.awaiting_confirmation = True
                    return TurnV2(
                        f"Transfer **{txn_id}** is pending: {result['amount']:.2f} {result.get('currency', 'USD')} to {result['recipient']}. "
                        "I can cancel it, but this is a sensitive action. Do you explicitly confirm cancellation?",
                        route,
                        [{"tool": "cancel_transfer", "args": {"transfer_id": txn_id, "confirmed": False}, "result": result}],
                        [],
                        self.state.__dict__,
                    )
                if result.get("status") == "already_cancelled":
                    return TurnV2(f"Transfer **{txn_id}** was already cancelled; no duplicate change was made.", route,
                                  [{"tool": "cancel_transfer", "args": {"transfer_id": txn_id, "confirmed": False}, "result": result}], [], self.state.__dict__)
                if result.get("status") == "not_pending":
                    return TurnV2(f"Transfer **{txn_id}** is {result['current_status']}, so it cannot be cancelled in this demo.", route,
                                  [{"tool": "cancel_transfer", "args": {"transfer_id": txn_id, "confirmed": False}, "result": result}], [], self.state.__dict__)
                return TurnV2(f"Transfer **{txn_id}** is now cancelled.", route,
                              [{"tool": "cancel_transfer", "args": {"transfer_id": txn_id, "confirmed": False}, "result": result}], [], self.state.__dict__)
            result = self.tools.get_transfer_status(txn_id)
            return TurnV2(
                f"Transfer **{txn_id}** is **{result['status']}**: {result['amount']:.2f} {result['currency']} to {result['recipient']}.",
                route,
                [{"tool": "get_transfer_status", "args": {"transfer_id": txn_id}, "result": result}],
                [],
                self.state.__dict__,
            )
        except ToolError as exc:
            hs = self._handoff(intent, f"{exc.code}: {exc.message}")
            return TurnV2(
                "I couldn’t access that transfer. I didn’t change anything, and I’ve created a human handoff ticket.",
                route,
                [hs],
                [],
                self.state.__dict__,
            )

    def _dispute(self, intent: str, txn_id: str, text: str, route: RouteResult) -> TurnV2:
        try:
            result = self.tools.create_dispute(txn_id, text)
            return TurnV2(
                f"I opened fictional dispute **{result['dispute_id']}** for **{txn_id}**. A human specialist will review it; no real money moved.",
                route,
                [{"tool": "create_dispute", "args": {"transaction_id": txn_id, "reason": text}, "result": result}],
                [],
                self.state.__dict__,
            )
        except ToolError as exc:
            hs = self._handoff(intent, f"{exc.code}: {exc.message}")
            return TurnV2("I couldn’t open that dispute, so I’ve handed it to a human specialist.", route, [hs], [], self.state.__dict__)

    def _slot(self, text: str) -> TurnV2:
        if self.state.pending_slot == "card":
            match = CARD_RE.search(text)
            if not match:
                return TurnV2("Please provide a fictional card ID such as CARD-1001.", self.state.last_route, [], [], self.state.__dict__)
            intent = self.state.pending_intent
            original = self.state.pending_text
            self.state.pending_intent = None
            self.state.pending_slot = None
            route = self.state.last_route
            assert route is not None
            return self._card_action(intent, match.group(0).upper(), route)
        if self.state.pending_slot in ("transfer", "dispute"):
            match = TXN_RE.search(text)
            if not match:
                return TurnV2("Please provide a fictional transfer ID such as TXN-2001.", self.state.last_route, [], [], self.state.__dict__)
            intent = self.state.pending_intent
            original = self.state.pending_text
            self.state.pending_intent = None
            self.state.pending_slot = None
            route = self.state.last_route
            assert route is not None
            if intent in DISPUTE_INTENTS:
                return self._dispute(intent, match.group(0).upper(), original + " " + text, route)
            return self._transfer_action(intent, match.group(0).upper(), route)
        return TurnV2("Please restate the request.", self.state.last_route, [], [], self.state.__dict__)

    def _confirmation(self, text: str) -> TurnV2:
        if not has_yes(text) and not has_no(text):
            return TurnV2("Please answer yes to confirm or no to cancel. No action has been taken.", self.state.last_route, [], [], self.state.__dict__)
        yes = has_yes(text)
        action = self.state.pending_action
        obj = self.state.pending_id
        self.state.awaiting_confirmation = False
        self.state.pending_action = None
        self.state.pending_id = None
        self.state.pending_intent = None
        if not yes:
            return TurnV2("Understood — no action was taken.", self.state.last_route, [], [], self.state.__dict__)
        try:
            if action == "cancel_transfer":
                result = self.tools.cancel_transfer(obj, confirmed=True)
                return TurnV2(f"Confirmed. Transfer **{obj}** is now **{result['status']}**.", self.state.last_route,
                              [{"tool": "cancel_transfer", "args": {"transfer_id": obj, "confirmed": True}, "result": result}], [], self.state.__dict__)
            if action == "activate_card":
                result = self.tools.activate_card(obj, confirmed=True)
                return TurnV2(f"Confirmed. Card **{obj}** is now **{result['status']}**.", self.state.last_route,
                              [{"tool": "activate_card", "args": {"card_id": obj, "confirmed": True}, "result": result}], [], self.state.__dict__)
        except ToolError as exc:
            hs = self._handoff("confirmation", f"{exc.code}: {exc.message}")
            return TurnV2("I couldn’t complete the confirmed action; a human ticket was created.", self.state.last_route, [hs], [], self.state.__dict__)
        return TurnV2("Nothing was executed.", self.state.last_route, [], [], self.state.__dict__)

    def _clarification(self, text: str) -> TurnV2:
        candidates = self.state.clarification_candidates
        clean = clean_text(text)
        selected = None
        if clean in {"1", "first", "one"} and candidates:
            selected = candidates[0]
        elif clean in {"2", "second", "two"} and len(candidates) > 1:
            selected = candidates[1]
        if selected is None:
            for candidate in candidates:
                hints = HINTS.get(candidate, [])
                if any(h in clean for h in hints):
                    selected = candidate
                    break
        if selected is None:
            # If the user supplied an ID, choose the candidate that can use it.
            if CARD_RE.search(text):
                selected = next((c for c in candidates if c in CARD_INTENTS), None)
            elif TXN_RE.search(text):
                selected = next((c for c in candidates if c in TRANSFER_INTENTS | DISPUTE_INTENTS), None)
        if selected is None:
            self.state.awaiting_clarification = False
            hs = self._handoff("ambiguous", "User did not resolve the natural-language clarification")
            return TurnV2("I’m still unsure, so I’ve created a human handoff ticket instead of guessing.", self.state.last_route, [hs], [], self.state.__dict__)
        self.state.awaiting_clarification = False
        self.state.clarification_candidates = []
        original = self.state.pending_text
        self.state.pending_text = ""
        route = self.state.last_route
        assert route is not None
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
        # Preserve IDs from the original request and the clarification reply.
        return self._dispatch(route, (original + " " + text).strip())

    def snapshot(self):
        return self.tools.snapshot()

    def close(self):
        self.tools.close()
