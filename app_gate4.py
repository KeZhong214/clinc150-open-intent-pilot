"""Gate-4 Streamlit demo: stateful customer-service agent with SQLite tools.

Run:
    streamlit run app_gate4.py --server.headless true --server.port 8501
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
DB_PATH = ROOT / "data" / "gate4" / "agent.db"
sys.path.insert(0, str(ROOT))

from src.gate4.agent import BankAgent  # noqa: E402
from src.gate4.db import BankingTools, seed  # noqa: E402
from src.gate4.router import load_b3h_router, load_b4_router, load_gte  # noqa: E402

DEMOS = {
    "Card delivery": ["Where is my card? I still have not received CARD-1001."],
    "Transfer status": ["My transfer TXN-2001 has not arrived yet."],
    "Cancel transfer + confirm": ["Please cancel transfer TXN-2002.", "yes"],
    "Unknown → handoff": ["How do I buy Bitcoin?"],
    "Non-owned card → handoff": ["I still have not received my card CARD-2001."],
    "Ambiguous → clarify → status": [
        "My transfer TXN-2001 has not arrived yet.",
        "transfer_not_received_by_recipient",
        "TXN-2001",
    ],
}


@st.cache_resource
def encoder():
    return load_gte()


@st.cache_resource
def b3h():
    return load_b3h_router(encoder())


@st.cache_resource
def b4():
    return load_b4_router(encoder())


def new_session(method: str) -> None:
    old = st.session_state.get("agent")
    if old is not None:
        try:
            old.tools.close()
        except Exception:
            pass
    seed(DB_PATH, reset=True)
    tools = BankingTools(session_id=f"STREAMLIT-{method}", path=DB_PATH)
    router = b3h() if method == "B3-H" else b4()
    st.session_state.agent = BankAgent(router, tools)
    st.session_state.messages = []
    st.session_state.method = method
    st.session_state.turn_counter = 0


def ensure_session(method: str) -> None:
    if "agent" not in st.session_state or st.session_state.get("method") != method:
        new_session(method)


def send(text: str) -> None:
    agent: BankAgent = st.session_state.agent
    st.session_state.messages.append({"role": "user", "content": text})
    turn = agent.handle(text)
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": turn.response,
            "route": {
                "method": turn.route.method if turn.route else None,
                "intent": turn.route.intent if turn.route else None,
                "accepted": turn.route.accepted if turn.route else None,
                "score": round(turn.route.score, 6) if turn.route else None,
                "threshold": round(turn.route.threshold, 6) if turn.route else None,
                "top3": turn.route.top3 if turn.route else [],
                "gate_top1": turn.route.gate_top1 if turn.route else None,
                "reject_score": turn.route.reject_score if turn.route else None,
            },
            "tools": turn.tools,
        }
    )
    st.session_state.turn_counter += 1


def run_demo(name: str, method: str) -> None:
    new_session(method)
    for text in DEMOS[name]:
        send(text)


def main() -> None:
    st.set_page_config(page_title="Gate 4 Customer Agent", layout="wide")
    st.title("Gate 4 · Customer Service Agent")
    st.caption("Fictional bank · SQLite tools · B3-H / B4-Full router switch · stateful multi-turn")

    with st.sidebar:
        method = st.radio("Router", ["B3-H", "B4-Full"], index=0)
        st.caption("B3-H is the stable Gate-3 baseline. B4-Full is the Gate-4 trained gate + reject head.")
        if st.button("Reset session + DB", type="secondary"):
            new_session(method)
            st.rerun()
        st.divider()
        st.write("**Replay demos**")
        for name in DEMOS:
            if st.button(name, use_container_width=True):
                run_demo(name, method)
                st.rerun()
        st.divider()
        st.caption("All card/transfer IDs are fake. No real bank, card, or user data is used.")
        st.caption("The agent only performs sensitive actions after an explicit yes.")

    ensure_session(method)
    agent: BankAgent = st.session_state.agent

    chat_col, data_col = st.columns([1.35, 1.0], gap="large")
    with chat_col:
        st.subheader("Conversation")
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg["role"] == "assistant":
                    route = msg.get("route") or {}
                    if route.get("method"):
                        st.caption(
                            f"Router **{route['method']}** · intent **{route['intent']}** · "
                            f"score {route['score']} · threshold {route['threshold']} · accepted {route['accepted']}"
                        )
                        if route.get("top3"):
                            st.dataframe(
                                pd.DataFrame(
                                    [{"rank": i + 1, "intent": name, "score": round(score, 5)} for i, (name, score) in enumerate(route["top3"])]
                                ),
                                hide_index=True,
                                use_container_width=True,
                            )
                    if msg.get("tools"):
                        with st.expander(f"Tool timeline ({len(msg['tools'])} call(s))", expanded=False):
                            st.json(msg["tools"])
        prompt = st.chat_input("Ask a banking request (e.g. Where is CARD-1001?)")
        if prompt:
            send(prompt)
            st.rerun()
        st.caption("The reply is template-generated from tool results; no LLM is allowed to invent a database result.")

    with data_col:
        st.subheader("Business state")
        snapshot = agent.tools.snapshot()
        tabs = st.tabs(["Cards", "Transfers", "Tickets", "Disputes", "Audit"])
        with tabs[0]:
            st.dataframe(pd.DataFrame(snapshot["cards"]), hide_index=True, use_container_width=True)
        with tabs[1]:
            st.dataframe(pd.DataFrame(snapshot["transfers"]), hide_index=True, use_container_width=True)
        with tabs[2]:
            st.dataframe(pd.DataFrame(snapshot["tickets"]), hide_index=True, use_container_width=True)
        with tabs[3]:
            st.dataframe(pd.DataFrame(snapshot["disputes"]), hide_index=True, use_container_width=True)
        with tabs[4]:
            audit = pd.DataFrame(snapshot["audit_events"])
            st.dataframe(audit.head(15), hide_index=True, use_container_width=True)

        st.subheader("Router comparison")
        compare = st.toggle("Compare B3-H vs B4-Full on the last user message", value=False)
        if compare and st.session_state.messages:
            last_user = next((m["content"] for m in reversed(st.session_state.messages) if m["role"] == "user"), None)
            if last_user:
                rows = []
                for label, router in (("B3-H", b3h()), ("B4-Full", b4())):
                    r = router.route(last_user)
                    rows.append(
                        {
                            "router": label,
                            "intent": r.intent,
                            "accepted": r.accepted,
                            "score": round(r.score, 5),
                            "threshold": round(r.threshold, 5),
                            "top1": r.top3[0][0] if r.top3 else "",
                        }
                    )
                st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
                st.caption("Same user message, same support cache; B4 uses its trained gate/reject head.")


if __name__ == "__main__":
    main()
