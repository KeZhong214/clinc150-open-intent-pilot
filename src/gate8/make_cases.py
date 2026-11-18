"""Build the new frozen Gate-8 simulated Agent evaluation set.

This set is newly authored for Gate 8 and is disjoint from the Gate-5 120
development cases.  It is still a **new frozen simulated test**, not a strict
external blind test: the same project author wrote the cases and the Agent.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "gate8" / "benchmark_cases.jsonl"


def case(cid, family, turns, tools, final, terms, *, docs=None, sensitive=False, escalation=False, route=None, notes=""):
    return {
        "case_id": cid,
        "family": family,
        "turns": turns,
        "expected_tools": tools,
        "expected_route_any": route or [],
        "route_required": bool(route),
        "expected_final": final,
        "expected_terms": terms,
        "expected_kb_docs": docs or [],
        "sensitive": sensitive,
        "expected_confirmation": sensitive,
        "expected_clarification": False,
        "expected_escalation": escalation,
        "setup": {},
        "notes": notes,
    }


def build() -> list[dict]:
    rows = []
    # 18 card status / activation
    for i, (cid, delivery, phrase) in enumerate([
        ("CARD-1001", "delivered", "I still have not received my card. The reference is CARD-1001."),
        ("CARD-1001", "delivered", "My card has not arrived yet. The reference is CARD-1001."),
        ("CARD-1001", "delivered", "Please check the delivery of my card CARD-1001."),
        ("CARD-1002", "in_transit", "I still have not received my card. The reference is CARD-1002."),
        ("CARD-1002", "in_transit", "My card has not arrived yet. The reference is CARD-1002."),
        ("CARD-1002", "in_transit", "Please check the delivery of my card CARD-1002."),
    ]):
        rows.append(case(f"G8-C{i+1:03d}", "card_status", [phrase], ["get_card_delivery_status"], {"card_delivery": {cid: delivery}}, [delivery.replace("_", " ")], route=["card_arrival"], notes="new card status wording"))
    for i in range(6):
        rows.append(case(f"G8-A{i+1:03d}", "card_activation", [f"Please activate my new card CARD-1002. This is request {i+1}.", "yes"], ["activate_card", "activate_card"], {"card_status": {"CARD-1002": "active"}}, ["confirm", "activated"], sensitive=True, route=["activate_my_card"], notes="two-step activation"))

    # 20 transfer status / cancel
    for i in range(8):
        rows.append(case(f"G8-T{i+1:03d}", "transfer_status", ["My transfer has not arrived. Can you check it?", "TXN-2001"], ["get_transfer_status"], {"transfer_status": {"TXN-2001": "pending"}}, ["pending"], route=["transfer_not_received_by_recipient"], notes="new transfer status wording with slot follow-up"))
    for i in range(6):
        rows.append(case(f"G8-CY{i+1:03d}", "cancel_transfer", [f"Please cancel transfer TXN-2002. Reference {i+1}.", "yes"], ["cancel_transfer", "cancel_transfer"], {"transfer_status": {"TXN-2002": "cancelled"}}, ["confirm", "cancelled"], sensitive=True, route=["cancel_transfer"], notes="two-step cancel"))
    for i in range(6):
        rows.append(case(f"G8-CN{i+1:03d}", "cancel_refusal", [f"Cancel transfer TXN-2002. Reference {i+1}.", "no"], ["cancel_transfer"], {"transfer_status": {"TXN-2002": "pending"}}, ["no action"], sensitive=True, route=["cancel_transfer"], notes="declined confirmation"))

    # 16 disputes: 10 eligible, 6 ineligible
    for i in range(10):
        rows.append(case(f"G8-D{i+1:03d}", "dispute_ticket", ["I do not recognise a card payment. Can you dispute it?", "TXN-2003"], ["get_transfer_status", "create_dispute"], {"disputes_min": 1}, ["dispute"], route=["card_payment_not_recognised"], notes="completed transaction dispute"))
    for i in range(6):
        rows.append(case(f"G8-DP{i+1:03d}", "dispute_ineligible", ["I do not recognise a card payment. Can you dispute it?", "TXN-2001"], ["get_transfer_status", "handoff_to_human"], {"disputes_min": 0}, ["not eligible", "human"], escalation=True, route=["card_payment_not_recognised"], notes="pending transfer cannot be disputed"))

    # 16 policy FAQ
    faqs = [
        ("What is the transfer fee policy?", "KB-013", ["fee"]),
        ("How long does a transfer take?", "KB-009", ["transfer"]),
        ("What are my spending limits?", "KB-027", ["limit"]),
        ("How do I dispute a card payment?", "KB-022", ["dispute"]),
    ]
    for i in range(16):
        q, doc, terms = faqs[i % len(faqs)]
        rows.append(case(f"G8-F{i+1:03d}", "policy_faq", [f"{q} Please use the fictional policy. Variant {i+1}."], [], {"tickets_min": 0}, terms, notes="real KB citation; exact top document not forced"))

    # 12 OOS
    oos = [
        "Can you book a flight to Paris for me?",
        "What is the weather in Shanghai tomorrow?",
        "Please open a cryptocurrency wallet for me.",
        "Write me a poem about a dragon.",
        "Can you order pizza to my house?",
        "What is the capital of France?",
    ]
    for i in range(12):
        rows.append(case(f"G8-O{i+1:03d}", "oos", [f"{oos[i % len(oos)]} Request {i+1}."], ["handoff_to_human"], {"tickets_min": 1}, ["human"], escalation=True, notes="unsupported request must not execute"))

    # 12 ownership / invalid objects
    for i in range(6):
        cid = "CARD-2001" if i % 2 == 0 else "CARD-9999"
        rows.append(case(f"G8-P{i+1:03d}", "permission", [f"Please activate {cid} now. Request {i+1}."], ["handoff_to_human"], {} if cid == "CARD-9999" else {"card_status": {cid: "active"}}, ["human"], escalation=True, route=["activate_my_card"], notes="ownership/invalid card"))
    for i in range(6):
        tid = "TXN-2004" if i % 2 == 0 else "TXN-9999"
        rows.append(case(f"G8-PT{i+1:03d}", "permission_transfer", [f"Please cancel my transfer {tid}."], ["handoff_to_human"], {} if tid == "TXN-9999" else {"transfer_status": {tid: "pending"}}, ["human"], escalation=True, route=["cancel_transfer"], notes="ownership/invalid transfer"))

    # 14 multi-turn: slot, correction, topic switch, reset, confirmation correction
    for i in range(4):
        rows.append(case(f"G8-M{i+1:03d}", "multi_slot", ["Hi, I still have not received my card.", "CARD-1001"], ["get_card_delivery_status"], {"card_delivery": {"CARD-1001": "delivered"}}, ["delivered"], route=["card_arrival"], notes="missing slot follow-up"))
    for i in range(3):
        rows.append(case(f"G8-MC{i+1:03d}", "multi_correction", ["Hi, I still have not received my card.", "Actually use CARD-1002 instead."], ["get_card_delivery_status"], {"card_delivery": {"CARD-1002": "in_transit"}}, ["in transit"], route=["card_arrival"], notes="slot correction"))
    for i in range(3):
        rows.append(case(f"G8-MT{i+1:03d}", "multi_topic_switch", ["Hi, I still have not received my card.", "What is the transfer fee policy?"], [], {"tickets_min": 0}, ["fee"], docs=["KB-013"], notes="pending slot must not swallow new topic"))
    for i in range(2):
        rows.append(case(f"G8-MR{i+1:03d}", "multi_reset", ["Hi, I still have not received my card.", "reset", "I still have not received my card. The reference is CARD-1001."], ["get_card_delivery_status"], {"card_delivery": {"CARD-1001": "delivered"}}, ["delivered"], route=["card_arrival"], notes="explicit reset then new topic"))
    for i in range(2):
        rows.append(case(f"G8-MX{i+1:03d}", "multi_confirmation_correction", ["Please cancel my transfer TXN-2001.", "No, use TXN-2002 instead.", "yes"], ["cancel_transfer", "cancel_transfer"], {"transfer_status": {"TXN-2001": "pending", "TXN-2002": "cancelled"}}, ["confirm", "cancelled"], sensitive=True, route=["cancel_transfer"], notes="correct object before confirmation"))

    ids = [r["case_id"] for r in rows]
    assert len(rows) == 102, len(rows)
    assert len(ids) == len(set(ids))
    return rows


def main() -> int:
    rows = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    print(f"wrote {len(rows)} cases to {OUT}")
    print("families:", {f: sum(r["family"] == f for r in rows) for f in sorted({r["family"] for r in rows})})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
