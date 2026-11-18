"""Generate the Gate-5 fictional bank policy corpus (30 citable documents)."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "knowledge_base" / "fictional_bank" / "policies.jsonl"

DOCS = [
    ("KB-001", "Card delivery tracking", "Card delivery", "You can check the delivery status of a fictional card with its CARD- id. The demo statuses are delivered, in_transit, and not_shipped. A card that is still in transit cannot be activated yet.", ["card", "delivery", "arrival", "tracking"]),
    ("KB-002", "Activating a new card", "Card activation", "A new fictional card starts inactive. Activation is a sensitive action, so the assistant must show the card id and ask for explicit confirmation before changing the status to active.", ["card", "activate", "activation", "confirm"]),
    ("KB-003", "Activation security check", "Card activation", "The demo never asks for a real PIN, password, full card number, or identity document. Only fictional CARD- ids are accepted. A card owned by another demo customer is rejected by the tool layer.", ["card", "security", "ownership", "confirmation"]),
    ("KB-004", "Card not working", "Card troubleshooting", "If a fictional card is not working, the assistant can create a normal-priority ticket. It must not invent a replacement card, a refund, or a delivery date. A human specialist reviews the ticket.", ["card", "not_working", "ticket"]),
    ("KB-005", "Lost or stolen card", "Card safety", "For a lost or stolen fictional card, the assistant creates a high-priority human handoff ticket. The demo does not perform a real block, replacement, or account closure.", ["card", "lost", "stolen", "handoff"]),
    ("KB-006", "Card replacement", "Card replacement", "Card replacement is outside the automated demo. The assistant records the request in a ticket and explains that a human specialist must verify eligibility and delivery details.", ["card", "replacement", "ticket"]),
    ("KB-007", "Changing a card PIN", "Card PIN", "PIN changes are not executed by this demo. The assistant explains the limitation, does not ask for a PIN, and creates a human handoff ticket if the user needs help.", ["pin", "card", "security", "handoff"]),
    ("KB-008", "Transfer status", "Transfers", "A fictional transfer is identified by a TXN- id. The demo statuses are pending, completed, failed, and cancelled. The assistant reads the status from the local SQLite database and does not guess.", ["transfer", "status", "txn"]),
    ("KB-009", "Transfer timing", "Transfers", "A pending transfer has not completed yet. The demo does not promise a real arrival time. It reports the stored status and creates a ticket if the user needs an investigation.", ["transfer", "timing", "pending"]),
    ("KB-010", "Transfer not received", "Transfers", "If a fictional recipient has not received a transfer, the assistant first checks the transfer status by TXN- id. A completed transfer can still be investigated through a ticket; the demo does not move money.", ["transfer", "recipient", "not_received", "ticket"]),
    ("KB-011", "Cancelling a pending transfer", "Transfers", "Only a pending fictional transfer can be cancelled. The assistant shows the transfer id, amount, and recipient, then asks for explicit confirmation. The SQLite status changes to cancelled only after confirmation.", ["transfer", "cancel", "confirmation"]),
    ("KB-012", "Cancellation cut-off", "Transfers", "The demo has no real cut-off clock. A completed, failed, or already cancelled transfer cannot be cancelled again. The assistant reports the current status rather than repeating the action.", ["transfer", "cancel", "cutoff", "idempotent"]),
    ("KB-013", "Transfer fees", "Fees", "The fictional demo does not calculate real transfer fees. Any fee statement is a policy example, not a quote. For a concrete fee dispute, the assistant creates a ticket for a human.", ["fee", "transfer", "policy"]),
    ("KB-014", "Top-up methods", "Top-ups", "The demo may describe top-up by bank transfer or card as fictional examples. It does not initiate a real top-up or collect real payment credentials. The available methods are illustrative only.", ["top_up", "topup", "payment"]),
    ("KB-015", "Pending top-up", "Top-ups", "A pending fictional top-up has not been posted to the demo balance. The assistant explains that status and offers a ticket if the user wants the case reviewed.", ["top_up", "pending", "balance"]),
    ("KB-016", "Failed top-up", "Top-ups", "A failed fictional top-up did not change the demo balance. The assistant must not claim that money was credited. It can create a ticket with the fictional reference.", ["top_up", "failed", "balance"]),
    ("KB-017", "Cash withdrawal status", "Cash withdrawals", "Cash withdrawal records are fictional. The assistant can explain the status shown in the demo database, but it cannot reverse a withdrawal or promise a refund.", ["cash", "withdrawal", "status"]),
    ("KB-018", "Declined cash withdrawal", "Cash withdrawals", "A declined fictional withdrawal is not a real bank decision. The demo can log the user's concern and hand off to a human; it must not invent a reason or a guaranteed reversal.", ["cash", "declined", "handoff"]),
    ("KB-019", "Exchange rates", "Foreign exchange", "Exchange rates in the demo are illustrative and not live market quotes. The assistant must not give a real FX recommendation or promise a specific converted amount.", ["exchange", "rate", "fx", "currency"]),
    ("KB-020", "Card payment fee", "Card payments", "A card payment fee shown in the demo is a fictional example. The assistant can explain the record and create a dispute ticket, but it cannot waive a real fee.", ["card", "payment", "fee", "dispute"]),
    ("KB-021", "Unrecognised card payment", "Disputes", "For an unrecognised fictional card payment, the assistant collects the TXN- id, explains the demo dispute process, and creates a dispute record. It does not issue a provisional credit.", ["card", "payment", "unrecognised", "dispute"]),
    ("KB-022", "Dispute process", "Disputes", "The demo dispute flow is: identify the transaction, record a reason, create a dispute record, and hand the case to a human reviewer. No real money is moved and no outcome is guaranteed.", ["dispute", "process", "transaction"]),
    ("KB-023", "Dispute timelines", "Disputes", "The demo does not have real dispute timelines or regulatory deadlines. Any timeline shown is a fictional teaching example. The assistant should say that a human specialist will review the case.", ["dispute", "timeline", "review"]),
    ("KB-024", "Refund status", "Refunds", "Refund records in this demo are illustrative. The assistant can check a fictional transfer or payment record and create a ticket, but it cannot promise a refund date or amount.", ["refund", "status", "ticket"]),
    ("KB-025", "Account balance", "Accounts", "The demo may show a fictional balance summary, but it is not a real account statement. The assistant must not claim that a balance is final or that a pending item has posted.", ["balance", "account", "statement"]),
    ("KB-026", "Beneficiary checks", "Transfers", "A beneficiary name in the demo is fictional. The assistant must not claim to verify a real beneficiary or guarantee that a transfer reached the intended person.", ["beneficiary", "transfer", "verification"]),
    ("KB-027", "Spending and transfer limits", "Limits", "Limit information in the demo is a fictional policy example, not a personalised limit. The assistant must not approve an exception or increase a limit automatically.", ["limit", "spending", "transfer"]),
    ("KB-028", "Identity verification", "Security", "The demo does not perform real identity verification and never asks for a password, PIN, full card number, or government id. A request that needs identity checks goes to a human ticket.", ["identity", "verification", "security"]),
    ("KB-029", "Human handoff", "Support", "A human handoff creates a high-priority fictional ticket. The assistant explains why it is escalating and does not pretend that a human has already joined the chat.", ["human", "handoff", "ticket", "escalation"]),
    ("KB-030", "Complaints and security reports", "Support", "Complaints, suspected fraud, and security reports are logged as high-priority tickets. The demo does not block a real card or contact real authorities; it only records the fictional case.", ["complaint", "fraud", "security", "ticket"]),
]


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as handle:
        for doc_id, title, category, text, tags in DOCS:
            handle.write(
                json.dumps(
                    {
                        "doc_id": doc_id,
                        "title": title,
                        "version": "1.0",
                        "source_id": "FICTIONAL-BANK-DEMO-2026",
                        "category": category,
                        "text": text,
                        "citation": f"{doc_id} v1.0 · FICTIONAL-BANK-DEMO-2026",
                        "tags": tags,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    print(f"wrote {len(DOCS)} docs to {OUT}")
    print("GATE5_KB_GEN_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
