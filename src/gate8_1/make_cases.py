"""Gate 8.1: new, deduplicated final simulated Agent cases (not external blind)."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "gate8_1" / "benchmark_cases.jsonl"


def c(cid, template, family, turns, tools, final, terms, *, sensitive=False, escalation=False, route=None, notes="", setup=None):
    return {
        "case_id": cid,
        "template_id": template,
        "family": family,
        "turns": turns,
        "expected_tools": tools,
        "expected_route_any": route or [],
        "route_required": bool(route),
        "expected_final": final,
        "expected_terms": terms,
        "expected_kb_docs": [],
        "sensitive": sensitive,
        "expected_confirmation": sensitive,
        "expected_clarification": False,
        "expected_escalation": escalation,
        "setup": setup or {},
        "notes": notes,
    }


def build() -> list[dict]:
    r = []
    # 10 card status; every dialogue differs.
    card_phrases = [
        "I have not received my replacement card yet. Reference CARD-1001.",
        "Could you trace the delivery of CARD-1001 for me?",
        "My card CARD-1002 has not shown up. What is the delivery state?",
        "Please tell me where CARD-1002 is in the delivery process.",
        "The card ending 4242 is CARD-1001; is it delivered?",
        "Has my CARD-1001 card been sent out and delivered?",
        "I am waiting for CARD-1002. Can you check its shipping status?",
        "What is the delivery status of my CARD-1001 card?",
        "My new card is CARD-1002. Has it arrived at my address?",
        "Please look up the delivery state for my card CARD-1001.",
    ]
    for i, q in enumerate(card_phrases):
        cid = "CARD-1002" if "CARD-1002" in q else "CARD-1001"
        delivery = "in_transit" if cid == "CARD-1002" else "delivered"
        r.append(c("G81-C%03d" % (i + 1), "card_%02d" % (i + 1), "card_status", [q], ["get_card_delivery_status"], {"card_delivery": {cid: delivery}}, [delivery.replace("_", " ")], route=["card_arrival"], notes="new card wording"))

    # 10 activation, each distinct; every one has yes second turn.
    act_phrases = [
        "Please turn on my new card CARD-1002.",
        "I want to start using CARD-1002; activate it.",
        "Can you enable card CARD-1002 for me?",
        "My CARD-1002 is inactive. Please activate it.",
        "I received CARD-1002 and need activation.",
        "Please switch on my replacement card CARD-1002.",
        "Activate CARD-1002 so I can use it.",
        "I would like to activate the new card CARD-1002.",
        "Turn my new CARD-1002 card on, please.",
        "Can you help me activate CARD-1002?",
    ]
    for i, q in enumerate(act_phrases):
        r.append(c("G81-A%03d" % (i + 1), "act_%02d" % (i + 1), "card_activation", [q, "yes"], ["activate_card", "activate_card"], {"card_status": {"CARD-1002": "active"}}, ["confirm", "activated"], sensitive=True, route=["activate_my_card"], notes="two-step activation"))

    # 10 transfer status; first turn triggers slot, second gives ID.
    tr_phrases = [
        "My outgoing transfer has not arrived yet. Could you check it?",
        "The recipient still has not received my transfer.",
        "I sent money but the transfer is still pending.",
        "Can you look up my transfer? It has not reached the recipient.",
        "Please check why my transfer has not arrived.",
        "My transfer seems delayed. Where is it?",
        "The transfer has not shown up for the recipient.",
        "Can you investigate a transfer that is still processing?",
        "I need the current status of my outgoing transfer.",
        "Has my transfer reached the recipient yet?",
    ]
    for i, q in enumerate(tr_phrases):
        r.append(c("G81-T%03d" % (i + 1), "tr_%02d" % (i + 1), "transfer_status", [q, "TXN-2001"], ["get_transfer_status"], {"transfer_status": {"TXN-2001": "pending"}}, ["pending"], route=["transfer_not_received_by_recipient", "transfer_timing"], notes="slot follow-up"))

    # 10 cancel yes, 8 cancel no.
    cancel_phrases = [
        "Please cancel my transfer TXN-2002.",
        "I want to stop transfer TXN-2002.",
        "Can you call back TXN-2002?",
        "Cancel the pending transfer TXN-2002 for me.",
        "I need to abort my transfer TXN-2002.",
        "Please stop the payment TXN-2002.",
        "Reverse TXN-2002 before it completes.",
        "I changed my mind; cancel TXN-2002.",
        "Stop transfer TXN-2002 immediately.",
        "Please withdraw my pending transfer TXN-2002.",
    ]
    for i, q in enumerate(cancel_phrases):
        r.append(c("G81-CY%03d" % (i + 1), "cy_%02d" % (i + 1), "cancel_transfer", [q, "yes"], ["cancel_transfer", "cancel_transfer"], {"transfer_status": {"TXN-2002": "cancelled"}}, ["confirm", "cancelled"], sensitive=True, route=["cancel_transfer"], notes="two-step cancel"))
    cancel_no_phrases = [
        "Could you cancel TXN-2002?",
        "I am thinking about cancelling TXN-2002.",
        "I want TXN-2002 cancelled.",
        "Please stop the transfer with id TXN-2002.",
        "I need to call off TXN-2002.",
        "Can TXN-2002 be stopped?",
        "Abort TXN-2002 for me.",
        "I would like to withdraw TXN-2002.",
    ]
    for i, q in enumerate(cancel_no_phrases):
        r.append(c("G81-CN%03d" % (i + 1), "cn_%02d" % (i + 1), "cancel_refusal", [q, "no"], ["cancel_transfer"], {"transfer_status": {"TXN-2002": "pending"}}, ["no action"], sensitive=True, route=["cancel_transfer"], notes="declined confirmation"))

    # 10 eligible disputes; 8 ineligible.
    dispute_phrases = [
        "I do not recognise a card payment on my statement. Can you dispute it?",
        "There is an unknown card payment I want to challenge.",
        "This charge is not mine; please open a dispute.",
        "I want to report a card payment I did not authorise.",
        "Please raise a dispute for an unrecognised card charge.",
        "That card transaction is unfamiliar; dispute it.",
        "I am contesting a card payment that is not mine.",
        "Open a case for a card payment I do not recognise.",
        "I need to dispute an unauthorised card payment.",
        "Please investigate an unknown charge and open a dispute.",
    ]
    for i, q in enumerate(dispute_phrases):
        r.append(c("G81-D%03d" % (i + 1), "d_%02d" % (i + 1), "dispute_ticket", [q, "TXN-2003"], ["get_transfer_status", "create_dispute"], {"disputes_min": 1}, ["dispute"], route=["card_payment_not_recognised"], notes="completed txn dispute"))
    for i, q in enumerate(dispute_phrases[:8]):
        r.append(c("G81-DP%03d" % (i + 1), "dp_%02d" % (i + 1), "dispute_ineligible", [q, "TXN-2001"], ["get_transfer_status", "handoff_to_human"], {"disputes_min": 0}, ["not eligible", "human"], escalation=True, route=["card_payment_not_recognised"], notes="pending txn not eligible"))

    # 12 FAQ; no exact docs forced, but real citation required by runner output.
    faqs = [
        "Could you explain the policy on transfer fees?",
        "How much is the fee for an international transfer?",
        "What are the spending limits on this demo account?",
        "When does my card expire and how do I renew it?",
        "Which currencies are supported for exchange?",
        "How long does a bank transfer take to arrive?",
        "What does the dispute process look like?",
        "Which top-up methods are supported?",
        "If a top-up does not go through, what should I do?",
        "Where can I use my card?",
        "How are exchange rates handled here?",
        "What documents are needed for identity checks?",
    ]
    for i, q in enumerate(faqs):
        r.append(c("G81-F%03d" % (i + 1), "faq_%02d" % (i + 1), "policy_faq", [q], [], {"tickets_min": 0}, [q.split()[0].lower()], notes="real KB citation; no exact doc forced"))

    # 10 OOS.
    oos = [
        "Can you book me a flight to Rome?",
        "What is the weather in London tomorrow?",
        "Please order a pizza for my office.",
        "Write a short poem about my cat.",
        "Open a cryptocurrency trading account.",
        "Who won the football match last night?",
        "Translate a paragraph into French for me.",
        "Set a reminder to call my dentist.",
        "Can you recommend a holiday destination?",
        "Tell me a joke about computers.",
    ]
    for i, q in enumerate(oos):
        r.append(c("G81-O%03d" % (i + 1), "oos_%02d" % (i + 1), "oos", [q], ["handoff_to_human"], {"tickets_min": 1}, ["human"], escalation=True, notes="unsupported"))

    # 10 permission / invalid object.
    perm_card_phrases = [
        "Please activate {cid} now.",
        "Turn on the card {cid} for me.",
        "Can you enable {cid}?",
        "I need {cid} activated.",
        "Please switch on {cid}.",
    ]
    for i in range(5):
        cid = "CARD-2001" if i % 2 == 0 else "CARD-9999"
        r.append(c("G81-P%03d" % (i + 1), "perm_%02d" % (i + 1), "permission", [perm_card_phrases[i].format(cid=cid)], ["handoff_to_human"], {} if cid == "CARD-9999" else {"card_status": {cid: "active"}}, ["human"], escalation=True, route=["activate_my_card"], notes="ownership/invalid"))
    perm_transfer_phrases = [
        "Please cancel my transfer {tid} right away.",
        "I want to stop the transfer {tid}.",
        "Can you call back {tid}?",
        "Cancel the pending transfer {tid} for me.",
        "Please withdraw my transfer {tid}.",
    ]
    for i in range(5):
        tid = "TXN-2004" if i % 2 == 0 else "TXN-9999"
        r.append(c("G81-PT%03d" % (i + 1), "permt_%02d" % (i + 1), "permission_transfer", [perm_transfer_phrases[i].format(tid=tid)], ["handoff_to_human"], {} if tid == "TXN-9999" else {"transfer_status": {tid: "pending"}}, ["human"], escalation=True, route=["cancel_transfer"], notes="ownership/invalid"))

    # 12 multi-turn: slot, correction, topic, reset, confirmation correction.
    slot_starts = [
        "Hello, my card has still not arrived.",
        "My replacement card has not arrived.",
        "I am waiting for my new card.",
        "Can you help me with a card that has not arrived?",
    ]
    for i in range(4):
        r.append(c("G81-M%03d" % (i + 1), "slot_%02d" % (i + 1), "multi_slot", [slot_starts[i], "CARD-1001"], ["get_card_delivery_status"], {"card_delivery": {"CARD-1001": "delivered"}}, ["delivered"], route=["card_arrival"], notes="slot"))
    corrections = [
        "Actually use CARD-1002 instead.",
        "Sorry, the correct card is CARD-1002.",
        "I meant CARD-1002, not the first one.",
    ]
    for i in range(3):
        r.append(c("G81-MC%03d" % (i + 1), "corr_%02d" % (i + 1), "multi_correction", ["Hello, my card has still not arrived.", corrections[i]], ["get_card_delivery_status"], {"card_delivery": {"CARD-1002": "in_transit"}}, ["in transit"], route=["card_arrival"], notes="correction"))
    topics = [
        "What is the transfer fee policy?",
        "Could you explain the policy for transfer fees?",
        "Are transfer fees charged in this demo?",
    ]
    for i in range(3):
        r.append(c("G81-MT%03d" % (i + 1), "topic_%02d" % (i + 1), "multi_topic_switch", ["Hello, my card has still not arrived.", topics[i]], [], {"tickets_min": 0}, ["fee"], notes="topic switch"))
    resets = [
        "reset",
        "start over",
    ]
    for i in range(2):
        r.append(c("G81-MR%03d" % (i + 1), "reset_%02d" % (i + 1), "multi_reset", ["Hello, my card has still not arrived.", resets[i], "I still have not received my card. The reference is CARD-1001."], ["get_card_delivery_status"], {"card_delivery": {"CARD-1001": "delivered"}}, ["delivered"], route=["card_arrival"], notes="reset"))
    corrections2 = [
        ("Wait, TXN-2002 is the one.", "please go ahead"),
        ("The correct transfer reference is TXN-2002.", "confirmed"),
    ]
    for i in range(2):
        r.append(c("G81-MX%03d" % (i + 1), "fixconf_%02d" % (i + 1), "multi_confirmation_correction", ["I want to cancel the transfer with reference TXN-2001.", corrections2[i][0], corrections2[i][1]], ["cancel_transfer", "cancel_transfer"], {"transfer_status": {"TXN-2001": "pending", "TXN-2002": "cancelled"}}, ["confirm", "cancelled"], sensitive=True, route=["cancel_transfer"], notes="change object"))

    # 8 boundary cases: empty, invalid transfer, invalid card, duplicate cancel, repeated status.
    boundaries = [
        ("G81-B001", "empty_01", "", [], {}, ["type"]),
        ("G81-B002", "invalid_00", "Please help with my card CARD-9999.", ["handoff_to_human"], {}, ["human"]),
        ("G81-B003", "invalid_01", "Please check TXN-9999.", ["handoff_to_human"], {"tickets_min": 1}, ["human"]),
        ("G81-B004", "invalid_02", "Please activate CARD-9999.", ["handoff_to_human"], {}, ["human"]),
        ("G81-B005", "duplicate_01", "Please cancel TXN-2002; I already asked once.", ["cancel_transfer"], {"transfer_status": {"TXN-2002": "cancelled"}}, ["already cancelled"]),
        ("G81-B006", "repeat_01", "Please check TXN-2001 now.", ["get_transfer_status"], {"transfer_status": {"TXN-2001": "pending"}}, ["pending"]),
        ("G81-B007", "invalid_03", "Cancel TXN-9999.", ["handoff_to_human"], {}, ["human"]),
        ("G81-B008", "invalid_04", "Please check the delivery of CARD-9999.", ["handoff_to_human"], {}, ["human"]),
    ]
    for cid, tid, text, tools, final, terms in boundaries:
        setup = {"pre_cancel": "TXN-2002"} if tid == "duplicate_01" else {}
        r.append(c(cid, tid, "boundary" if tid != "duplicate_01" else "cancel_idempotent", [text], tools, final, terms, sensitive=tid == "duplicate_01", escalation=tid in {"invalid_00", "invalid_01", "invalid_02", "invalid_03", "invalid_04"}, route=None if tid.startswith("empty") else (["cancel_transfer"] if "cancel" in text.lower() or "TXN-9999" in text else ["transfer_timing", "card_arrival"]), setup=setup, notes="boundary"))

    ids = [x["case_id"] for x in r]
    assert len(r) >= 100, len(r)
    assert len(ids) == len(set(ids))
    assert len({x["template_id"] for x in r}) >= 60
    return r


def main() -> int:
    rows = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in rows) + "\n", encoding="utf-8")
    print(f"wrote {len(rows)} cases, {len({x['template_id'] for x in rows})} templates")
    print({f: sum(x["family"] == f for x in rows) for f in sorted({x["family"] for x in rows})})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
