"""Gate-6 Streamlit Agent: real Qwen3-Embedding LoRA routers + RAG + SQLite."""

from __future__ import annotations

import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import sys
import uuid
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "gate5"
sys.path.insert(0, str(ROOT))

from src.gate5.agent_v2 import BankAgentV2  # noqa: E402
from src.gate5.kb import KBRetriever  # noqa: E402
from src.gate5.llm_client import LLMClient  # noqa: E402
from src.gate6.router_qwen import load_qwen_router  # noqa: E402

ROUTERS = ["Qwen-Plain-B3H", "Qwen-Plain-RejectOnly", "Qwen-Plain-B4Full"]


@st.cache_resource
def qwen_cache():
    return {"encoders": {}, "routers": {}}


@st.cache_resource
def kb():
    return KBRetriever()


def get_router(method: str):
    cache = qwen_cache()
    if method not in cache["routers"]:
        cache["routers"][method] = load_qwen_router(method, encoder_cache=cache["encoders"])
    return cache["routers"][method]


def get_agent(method: str) -> BankAgentV2:
    sessions = st.session_state.setdefault("sessions", {})
    if method not in sessions:
        agent = BankAgentV2(
            get_router(method),
            session_id=f"G6-{method.replace('-', '_')}-{uuid.uuid4().hex[:8]}",
            db_dir=ROOT / "data" / "gate6" / "sessions",
            kb=kb(),
        )
        sessions[method] = agent
        st.session_state.setdefault("messages", {})[method] = []
    return sessions[method]


def reset_method(method: str) -> None:
    sessions = st.session_state.setdefault("sessions", {})
    old = sessions.pop(method, None)
    if old is not None:
        old.close()
    st.session_state.setdefault("messages", {})[method] = []
    get_agent(method)


def send(method: str, text: str) -> None:
    agent = get_agent(method)
    msgs = st.session_state.setdefault("messages", {}).setdefault(method, [])
    msgs.append({"role": "user", "content": text})
    turn = agent.handle(text)
    msgs.append(
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
            },
            "tools": turn.tools,
            "citations": turn.citations,
        }
    )


DEMOS = [
    ("Card status · Qwen-B3H", "B001", "Qwen-Plain-B3H"),
    ("Card status · Qwen-RejectOnly", "B003", "Qwen-Plain-RejectOnly"),
    ("Transfer status · Qwen-B4Full", "B036", "Qwen-Plain-B4Full"),
    ("Cancel transfer · Qwen-RejectOnly", "B045", "Qwen-Plain-RejectOnly"),
    ("FAQ transfer fee · Qwen-B4Full", "B076", "Qwen-Plain-B4Full"),
    ("Ownership handoff · Qwen-B4Full", "B111", "Qwen-Plain-B4Full"),
    ("Ambiguous transfer · Qwen-B3H", "B106", "Qwen-Plain-B3H"),
]


def run_demo(case_id: str, method: str) -> None:
    st.session_state["pending_method"] = method
    cases = {x["case_id"]: x for x in map(json.loads, (DATA / "benchmark_cases.jsonl").read_text(encoding="utf-8").splitlines())}
    case = cases[case_id]
    reset_method(method)
    agent = get_agent(method)
    if case.get("setup", {}).get("pre_cancel"):
        agent.tools.cancel_transfer(case["setup"]["pre_cancel"], confirmed=True)
    for text in case["turns"]:
        send(method, text)


def main() -> None:
    st.set_page_config(page_title="Gate 6 Qwen Agent", layout="wide")
    st.title("Gate 6 · Qwen3-Embedding Customer Agent")
    st.caption("Real Qwen 0.6B LoRA encoding · B3-H / RejectOnly / B4-Full · RAG · SQLite tools · session isolation")
    llm = LLMClient()
    if "pending_method" in st.session_state:
        st.session_state["router_choice"] = st.session_state.pop("pending_method")

    with st.sidebar:
        method = st.radio("Qwen router", ROUTERS, index=0, key="router_choice")
        st.caption("All three routers use the real Qwen Plain LoRA adapter; the frozen encoder is evaluated in the benchmark but not loaded here to stay within 8 GB.")
        if st.button("Reset this session + DB", use_container_width=True):
            reset_method(method)
            st.rerun()
        st.divider()
        st.write("**Replay demos**")
        for label, case_id, demo_method in DEMOS:
            if st.button(label, use_container_width=True):
                run_demo(case_id, demo_method)
                st.rerun()
        st.divider()
        st.write("**LLM API**")
        st.json(llm.status(), expanded=False)
        st.caption("API mode remains NOT RUN unless credentials exist; no fake calls.")
        st.write("**KB**")
        st.caption(f"{len(kb().docs)} fictional policy documents; TF-IDF retrieval with citable IDs.")

    agent = get_agent(method)
    msgs = st.session_state.setdefault("messages", {}).setdefault(method, [])
    chat_col, data_col = st.columns([1.4, 1.0], gap="large")

    with chat_col:
        st.subheader("Conversation")
        for msg in msgs:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg["role"] == "assistant":
                    r = msg.get("route") or {}
                    if r.get("method"):
                        st.caption(f"Router **{r['method']}** · intent `{r['intent']}` · score {r['score']} · threshold {r['threshold']} · accepted {r['accepted']}")
                        if r.get("top3"):
                            st.dataframe(pd.DataFrame([{"rank": i + 1, "intent": n, "score": round(float(s), 5)} for i, (n, s) in enumerate(r["top3"])]), hide_index=True, use_container_width=True)
                    if msg.get("citations"):
                        with st.expander(f"Policy citations ({len(msg['citations'])})", expanded=True):
                            for c in msg["citations"]:
                                st.markdown(f"**{c['doc_id']}** · {c['title']} · {c['citation']}")
                                st.caption(c["text"])
                    if msg.get("tools"):
                        with st.expander(f"Tool timeline ({len(msg['tools'])} calls)", expanded=False):
                            st.json(msg["tools"])
        prompt = st.chat_input("Ask about a fictional card, transfer, fee, dispute, or policy...")
        if prompt:
            send(method, prompt)
            st.rerun()

    with data_col:
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
            st.dataframe(pd.DataFrame(snap["audit_events"]).head(20), hide_index=True, use_container_width=True)

        st.subheader("Router comparison")
        if st.toggle("Compare Qwen-B3H / RejectOnly / B4Full on the last user message", value=False):
            last = next((m["content"] for m in reversed(msgs) if m["role"] == "user"), None)
            if last:
                rows = []
                for name in ROUTERS:
                    try:
                        r = get_router(name).route(last)
                        rows.append({"router": name, "intent": r.intent, "accepted": r.accepted, "score": round(r.score, 5), "threshold": round(r.threshold, 5), "top1": r.top3[0][0]})
                    except Exception as exc:  # noqa: BLE001
                        rows.append({"router": name, "intent": f"ERROR: {exc}", "accepted": None, "score": None, "threshold": None, "top1": ""})
                st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

        st.subheader("Policy retrieval")
        q = st.text_input("Search fictional policy KB", value="transfer fee")
        if q:
            hits = kb().search(q, k=3)
            st.dataframe(pd.DataFrame([{"doc_id": h["doc_id"], "title": h["title"], "score": round(h["score"], 4)} for h in hits]), hide_index=True, use_container_width=True)


if __name__ == "__main__":
    main()
