"""Gate-8 Streamlit demo: router-driven bank Agent with visible decisions."""

from __future__ import annotations

import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import gc
import sys
import uuid
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.gate5.kb import KBRetriever  # noqa: E402
from src.gate5.llm_client import LLMClient  # noqa: E402
from src.gate8.agent import Gate8Agent  # noqa: E402
from src.gate8.routers import ROUTER_NAMES, build  # noqa: E402

DEMOS = {
    "1. Card delivery": [
        "I still have not received my card. The reference is CARD-1001.",
    ],
    "2. Activate card (confirm)": [
        "Please activate my new card CARD-1002.",
        "yes",
    ],
    "3. Transfer status (slot)": [
        "My transfer has not arrived. Can you check it?",
        "TXN-2001",
    ],
    "4. Cancel transfer (confirm)": [
        "Please cancel my transfer TXN-2002.",
        "yes",
    ],
    "5. Policy citation": [
        "What is the policy on transfer fees?",
    ],
    "6. Dispute completed txn": [
        "I do not recognise a card payment. Can you dispute it?",
        "TXN-2003",
    ],
    "7. Multi-turn correction": [
        "Hi, I still have not received my card.",
        "Actually use CARD-1002 instead.",
    ],
    "8. OOS handoff": [
        "Can you book a flight to Paris for me?",
    ],
    "9. Ownership / overreach": [
        "Please activate CARD-2001 now.",
    ],
}


@st.cache_resource
def kb():
    return KBRetriever()


@st.cache_resource
def router_cache():
    return {}


def get_router(name: str):
    cache = router_cache()
    if cache.get("name") != name:
        cache.clear()
        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass
        router, meta = build(name)
        cache.update({"name": name, "router": router, "meta": meta})
    return cache["router"], cache["meta"]


def get_agent(name: str) -> Gate8Agent:
    sessions = st.session_state.setdefault("sessions", {})
    if name not in sessions:
        router, meta = get_router(name)
        sessions[name] = Gate8Agent(
            router,
            session_id=f"G8-{uuid.uuid4().hex[:10]}",
            db_dir=ROOT / "data" / "gate8" / "ui_sessions",
            kb=kb(),
        )
        st.session_state.setdefault("messages", {})[name] = []
        st.session_state.setdefault("initial", {})[name] = sessions[name].snapshot()
    return sessions[name]


def reset(name: str) -> None:
    sessions = st.session_state.setdefault("sessions", {})
    old = sessions.pop(name, None)
    if old is not None:
        old.close()
    st.session_state.setdefault("messages", {})[name] = []
    st.session_state.setdefault("initial", {})[name] = None
    get_agent(name)


def send(name: str, text: str) -> None:
    agent = get_agent(name)
    msgs = st.session_state.setdefault("messages", {})[name]
    msgs.append({"role": "user", "content": text})
    turn = agent.handle(text)
    route = turn.route
    msgs.append(
        {
            "role": "assistant",
            "content": turn.response,
            "route": {
                "method": route.method if route else None,
                "intent": route.intent if route else None,
                "accepted": route.accepted if route else None,
                "score": round(route.score, 6) if route else None,
                "threshold": round(route.threshold, 6) if route else None,
                "top3": route.top3 if route else [],
                "gate_top1": route.gate_top1 if route else None,
            },
            "tools": turn.tools,
            "citations": turn.citations,
            "state": turn.state,
        }
    )


def run_demo(name: str, prompt: str) -> None:
    reset(name)
    for text in DEMOS[prompt]:
        send(name, text)
    st.session_state["last_demo"] = prompt


def main() -> None:
    st.set_page_config(page_title="Gate 8 Bank Agent", page_icon="🏦", layout="wide")
    st.title("Gate 8 · Bank Customer-Service Agent")
    st.caption("Offline Router-driven Agent · real Gate7 RejectOnly routers · RAG citations · SQLite tools · session isolation")
    llm_status = LLMClient().status()
    if "pending_method" in st.session_state:
        st.session_state["router_choice"] = st.session_state.pop("pending_method")

    with st.sidebar:
        st.header("Model & run mode")
        name = st.radio("Intent router", ROUTER_NAMES, index=0, key="router_choice")
        st.caption("One encoder is loaded at a time to stay within 8 GB. Switching may reload the model.")
        st.markdown("**Mode**: Offline Router-driven Agent")
        st.caption("LLM-assisted mode is optional and constrained; it is NOT RUN without API credentials.")
        st.json(llm_status, expanded=False)
        if st.button("Reset session + fictional DB", use_container_width=True):
            reset(name)
            st.rerun()
        st.divider()
        st.header("8+ live demos")
        for label in DEMOS:
            if st.button(label, use_container_width=True):
                run_demo(name, label)
                st.rerun()
        st.divider()
        st.caption("All data is fictional. No real bank, credential, card number, or money is used.")

    router, meta = get_router(name)
    agent = get_agent(name)
    msgs = st.session_state.setdefault("messages", {})[name]
    chat_col, side_col = st.columns([1.45, 1.0], gap="large")

    with chat_col:
        st.subheader("Conversation")
        if not msgs:
            st.info("Pick a demo on the left, or type a request. Try `Hi, I still have not received my card. The reference is CARD-1001.`")
        for msg in msgs:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg["role"] != "assistant":
                    continue
                r = msg.get("route") or {}
                c1, c2, c3, c4 = st.columns(4)
                score = f"{r['score']:.4f}" if r.get("score") is not None else "—"
                threshold = f"{r['threshold']:.4f}" if r.get("threshold") is not None else "—"
                c1.markdown(f"**Intent**  \n`{r.get('intent') or '—'}`")
                c2.markdown(f"**Accepted**  \n`{r.get('accepted')}`")
                c3.markdown(f"**Score**  \n`{score}`")
                c4.markdown(f"**Threshold**  \n`{threshold}`")
                if r.get("top3"):
                    st.dataframe(
                        pd.DataFrame(
                            [{"rank": i + 1, "intent": n, "score": round(float(s), 5)} for i, (n, s) in enumerate(r["top3"])]
                        ),
                        hide_index=True,
                        use_container_width=True,
                    )
                if msg.get("citations"):
                    with st.expander(f"KB citations ({len(msg['citations'])})", expanded=True):
                        for c in msg["citations"]:
                            st.markdown(f"**{c['doc_id']}** · {c['title']} · {c['citation']}")
                            st.caption(c["text"])
                if msg.get("tools"):
                    with st.expander(f"Tool timeline ({len(msg['tools'])} calls)", expanded=True):
                        for t in msg["tools"]:
                            st.markdown(f"**{t['tool']}** `{t['args']}`")
                            st.json(t["result"])
        prompt = st.chat_input("Ask about a fictional card, transfer, dispute, fee, or policy...")
        if prompt:
            send(name, prompt)
            st.rerun()

    with side_col:
        st.subheader("Model decision provenance")
        st.json(
            {
                "router": meta["name"],
                "encoder": meta["encoder"],
                "threshold": meta["threshold"],
                "threshold_source": meta["threshold_source"],
                "checkpoint_sha256": meta["checkpoint_sha256"],
                "adapter_sha256": meta["adapter_sha256"],
                "mode": "OFFLINE_ROUTER_DRIVEN",
            },
            expanded=False,
        )
        st.subheader("Business state")
        snap = agent.snapshot()
        tabs = st.tabs(["Cards", "Transfers", "Tickets", "Disputes", "Audit"])
        with tabs[0]:
            st.dataframe(pd.DataFrame(snap["cards"]), hide_index=True, use_container_width=True)
        with tabs[1]:
            st.dataframe(pd.DataFrame(snap["transfers"]), hide_index=True, use_container_width=True)
        with tabs[2]:
            st.dataframe(pd.DataFrame(snap["tickets"]), hide_index=True, use_container_width=True)
        with tabs[3]:
            st.dataframe(pd.DataFrame(snap["disputes"]), hide_index=True, use_container_width=True)
        with tabs[4]:
            st.dataframe(pd.DataFrame(snap["audit_events"]).head(25), hide_index=True, use_container_width=True)
        st.subheader("Policy KB")
        q = st.text_input("Search fictional policy", value="transfer fee")
        if q:
            hits = kb().search(q, k=3)
            st.dataframe(
                pd.DataFrame([{"doc_id": h["doc_id"], "title": h["title"], "score": round(h["score"], 4)} for h in hits]),
                hide_index=True,
                use_container_width=True,
            )


if __name__ == "__main__":
    main()
