"""Gate-8 Agent: deterministic router-first orchestration with context fixes."""

from __future__ import annotations

from pathlib import Path

from src.gate4.db import ToolError
from src.gate5.agent_v2 import BankAgentV2, CARD_RE, TXN_RE, TurnV2, clean_text
from src.gate5.kb import KBRetriever
from src.gate8.llm_assist import LLMAssistant
from src.gate5.llm_client import LLMUnavailable


class Gate8Agent(BankAgentV2):
    """Adds context reset/correction/topic-switch and dispute state checks.

    The router remains the authority for intent acceptance.  The LLM assistant,
    when configured, can only suggest explicit slots; it cannot dispatch a tool
    or change an OOS decision.
    """

    def __init__(self, router, *, session_id=None, customer_id="CUST-1001", db_dir: Path = Path("data/gate8/sessions"), kb=None, llm_assistant: LLMAssistant | None = None):
        super().__init__(router, session_id=session_id, customer_id=customer_id, db_dir=db_dir, kb=kb)
        self.assistant = llm_assistant
        self.history: list[dict] = []
        self.context_resets = 0

    @staticmethod
    def _reset_command(text: str) -> bool:
        return clean_text(text) in {"reset", "start over", "restart", "clear context", "new conversation"}

    @staticmethod
    def _cancel_command(text: str) -> bool:
        return clean_text(text) in {"cancel", "never mind", "nevermind", "stop", "forget it", "abort"}

    @staticmethod
    def _topic_switch(text: str) -> bool:
        c = clean_text(text)
        markers = ("policy", "fee", "charge", "how do i", "what is the", "another question", "different question", "transfer fee", "card fee")
        return any(m in c for m in markers) and not CARD_RE.search(text) and not TXN_RE.search(text)

    def _clear_context(self, message: str, text: str) -> TurnV2:
        from src.gate5.agent_v2 import StateV2

        self.state = StateV2()
        self.context_resets += 1
        return TurnV2(message, None, [], [], self.state.__dict__)

    def handle(self, text: str) -> TurnV2:
        text = text.strip()
        if self._reset_command(text):
            turn = self._clear_context("Session context cleared. Your fictional database records were not changed.", text)
            self._remember(text, turn)
            return turn
        if self._cancel_command(text):
            had = bool(self.state.pending_slot or self.state.pending_intent or self.state.awaiting_confirmation or self.state.awaiting_clarification)
            turn = self._clear_context("Cancelled the pending step. No sensitive action was executed." if had else "There is no pending action to cancel.", text)
            self._remember(text, turn)
            return turn

        # A user switching topic while the Agent waits for a missing slot should
        # not have the new question swallowed as a slot value.
        if self.state.pending_slot and self._topic_switch(text):
            self._clear_context("context_switch", text)
        # Correct the object while awaiting confirmation, without executing.
        if self.state.awaiting_confirmation:
            card = CARD_RE.search(text)
            txn = TXN_RE.search(text)
            new_id = (card or txn).group(0).upper() if (card or txn) else None
            if new_id and new_id != self.state.pending_id:
                self.state.pending_id = new_id
                action = self.state.pending_action or "pending action"
                return self._remember(
                    text,
                    TurnV2(
                        f"Updated the pending {action} to **{new_id}**. Please explicitly confirm yes or no. Nothing has been executed.",
                        self.state.last_route,
                        [],
                        [],
                        self.state.__dict__,
                    ),
                )
        # Optional constrained slot extraction. It can only add an explicit
        # fictional ID; it cannot bypass the router or execute a tool.
        if self.state.pending_slot and self.assistant and self.assistant.available:
            try:
                suggested = self.assistant.parse_slots(text, ["card_arrival", "activate_my_card", "transfer_timing", "cancel_transfer", "card_payment_not_recognised"])
                trusted = " ".join([text, self.state.pending_text, *[h.get("user", "") for h in self.history[-6:]]]).upper()
                if suggested.get("card_id") and not CARD_RE.search(text) and str(suggested["card_id"]).upper() in trusted:
                    text = text + " " + suggested["card_id"]
                if suggested.get("transfer_id") and not TXN_RE.search(text) and str(suggested["transfer_id"]).upper() in trusted:
                    text = text + " " + suggested["transfer_id"]
            except LLMUnavailable:
                pass
        turn = super().handle(text)
        self._remember(text, turn)
        return turn

    @staticmethod
    def _is_yes_no(text: str) -> bool:
        from src.gate5.agent_v2 import has_no, has_yes

        return has_yes(text) or has_no(text)

    def _remember(self, text: str, turn: TurnV2) -> TurnV2:
        self.history.append(
            {
                "user": text,
                "response": turn.response,
                "intent": turn.route.intent if turn.route else None,
                "accepted": turn.route.accepted if turn.route else None,
                "score": turn.route.score if turn.route else None,
                "threshold": turn.route.threshold if turn.route else None,
                "tools": turn.tools,
                "citations": turn.citations,
            }
        )
        return turn

    def _dispute(self, intent: str, txn_id: str, text: str, route):
        """Only completed/eligible transactions may be disputed in this demo."""
        try:
            status = self.tools.get_transfer_status(txn_id)
        except ToolError as exc:
            hs = self._handoff(intent, f"{exc.code}: {exc.message}")
            return self._remember(
                text,
                TurnV2("I couldn’t verify that transaction, so no dispute was opened; a human ticket was created.", route, [hs], [], self.state.__dict__),
            )
        status_record = {"tool": "get_transfer_status", "args": {"transfer_id": txn_id}, "result": status}
        if status.get("status") != "completed":
            hs = self._handoff(intent, f"dispute_not_eligible:{status.get('status')}")
            return self._remember(
                text,
                TurnV2(
                    f"Transfer **{txn_id}** is **{status.get('status')}**, so a dispute is not eligible in this demo. A human specialist will review it; no financial action was taken.",
                    route,
                    [status_record, hs],
                    [],
                    self.state.__dict__,
                ),
            )
        turn = super()._dispute(intent, txn_id, text, route)
        turn.tools = [status_record, *turn.tools]
        return turn

    def llm_status(self) -> dict:
        return self.assistant.status() if self.assistant else {"assist_mode": "OFFLINE_ROUTER_DRIVEN", "available": False}
