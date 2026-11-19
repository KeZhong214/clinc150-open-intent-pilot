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
from src.gate8.llm_assist import LLMAssistant  # noqa: E402
from src.gate8_1.routers import ROUTER_NAMES, build  # noqa: E402
from src.gate8_2.deepseek_client import DeepSeekClient  # noqa: E402
from src.gate8_2.grounded_writer import GroundedReplyWriter  # noqa: E402

MODES = ["Offline Router-driven", "DeepSeek-assisted"]

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


@st.cache_resource
def assistant():
    return LLMAssistant()


@st.cache_resource
def deepseek_client():
    return DeepSeekClient()


@st.cache_resource
def deepseek_writer():
    return GroundedReplyWriter(deepseek_client())


def release_sessions() -> dict:
    """Close every session and drop its encoder/router references."""
    import torch

    before = {}
    if torch.cuda.is_available():
        before = {"allocated_mb": round(torch.cuda.memory_allocated() / 1024**2, 1), "reserved_mb": round(torch.cuda.memory_reserved() / 1024**2, 1)}
    sessions = st.session_state.get("sessions", {})
    for agent in list(sessions.values()):
        try:
            agent.close()
        except Exception:
            pass
    sessions.clear()
    st.session_state["messages"] = {}
    st.session_state["initial"] = {}
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    after = {}
    if torch.cuda.is_available():
        after = {"allocated_mb": round(torch.cuda.memory_allocated() / 1024**2, 1), "reserved_mb": round(torch.cuda.memory_reserved() / 1024**2, 1)}
    st.session_state["last_release"] = {"before": before, "after": after}
    return {"before": before, "after": after}


def get_router(name: str):
    cache = router_cache()
    if cache.get("name") != name:
        # Important: sessions hold encoder references; close them *before*
        # dropping the cached router/model, otherwise CUDA memory is not freed.
        release_sessions()
        cache.clear()
        gc.collect()
        try:
            import torch

            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        except Exception:
            pass
        router, meta = build(name)
        cache.update({"name": name, "router": router, "meta": meta})
    return cache["router"], cache["meta"]


def get_agent(name: str, mode: str) -> Gate8Agent:
    key = f"{name}|{mode}"
    sessions = st.session_state.setdefault("sessions", {})
    if key not in sessions:
        router, meta = get_router(name)
        llm = LLMAssistant(deepseek_client()) if mode == "DeepSeek-assisted" else None
        sessions[key] = Gate8Agent(
            router,
            session_id=f"G82-{uuid.uuid4().hex[:10]}",
            db_dir=ROOT / "data" / "gate8_2" / "ui_sessions",
            kb=kb(),
            llm_assistant=llm,
        )
        st.session_state.setdefault("messages", {})[key] = []
        st.session_state.setdefault("initial", {})[key] = sessions[key].snapshot()
    return sessions[key]


def reset(name: str, mode: str) -> None:
    key = f"{name}|{mode}"
    sessions = st.session_state.setdefault("sessions", {})
    old = sessions.pop(key, None)
    if old is not None:
        old.close()
    st.session_state.setdefault("messages", {})[key] = []
    st.session_state.setdefault("initial", {})[key] = None
    get_agent(name, mode)


def send(name: str, mode: str, text: str) -> None:
    agent = get_agent(name, mode)
    key = f"{name}|{mode}"
    msgs = st.session_state.setdefault("messages", {})[key]
    msgs.append({"role": "user", "content": text})
    turn = agent.handle(text)
    route = turn.route
    reply = turn.response
    reply_source = "offline"
    reply_meta = {"status": "offline", "error": None, "elapsed_seconds": 0.0, "usage": {}}
    if mode == "DeepSeek-assisted":
        if not deepseek_writer().available:
            reply_source = "offline_blocked"
            reply_meta = {"status": "NOT RUN_API_BLOCKED", "error": "DEEPSEEK_API_KEY not configured", "elapsed_seconds": 0.0, "usage": {}}
        else:
            rr = deepseek_writer().rewrite(
                user_text=text,
                template=turn.response,
                intent=route.intent if route else None,
                accepted=route.accepted if route else False,
                tool_results=turn.tools,
                allowed_doc_ids=[c.get("doc_id") for c in turn.citations],
                pending_confirmation=bool(turn.state.get("awaiting_confirmation")),
                state=turn.state,
            )
            reply = rr.text
            reply_source = rr.source
            reply_meta = {"status": rr.status, "error": rr.error, "elapsed_seconds": rr.elapsed_seconds, "usage": rr.usage}
    msgs.append(
        {
            "role": "assistant",
            "content": reply,
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
            "reply_source": reply_source,
            "reply_meta": reply_meta,
        }
    )


def run_demo(name: str, mode: str, prompt: str) -> None:
    reset(name, mode)
    for text in DEMOS[prompt]:
        send(name, mode, text)
    st.session_state["last_demo"] = prompt


def main() -> None:
    st.set_page_config(page_title="Gate 8.2 Bank Agent", page_icon="🏦", layout="wide")
    st.title("Gate 8.2 · Bank Customer-Service Agent")
    st.caption("Offline Router-driven vs DeepSeek-assisted · paired Gate8.1 routers · RAG citations · SQLite tools · session isolation")
    ds_status = deepseek_client().status()
    if "pending_method" in st.session_state:
        st.session_state["router_choice"] = st.session_state.pop("pending_method")

    with st.sidebar:
        st.header("Model & run mode")
        name = st.radio("Intent router", ROUTER_NAMES, index=0, key="router_choice")
        mode = st.radio("Reply mode", MODES, index=0, key="reply_mode")
        st.caption("One encoder is loaded at a time to stay within 8 GB. Switching may reload the model.")
        if st.session_state.get("last_release"):
            st.caption(f"Last switch memory: {st.session_state['last_release']}")
        if mode == "DeepSeek-assisted" and not ds_status["available"]:
            st.error("DeepSeek-assisted selected, but DEEPSEEK_API_KEY is not configured. This run is NOT RUN_API_BLOCKED; replies fall back to offline templates.")
        st.markdown(f"**Mode**: {mode}")
        st.caption("Router/OOS and tool execution remain server-side; DeepSeek may only edit grounded wording.")
        st.json(ds_status, expanded=False)
        if st.button("Reset session + fictional DB", use_container_width=True):
            reset(name, mode)
            st.rerun()
        st.divider()
        st.header("8+ live demos")
        for label in DEMOS:
            if st.button(label, use_container_width=True):
                run_demo(name, mode, label)
                st.rerun()
        st.divider()
        st.caption("All data is fictional. No real bank, credential, card number, or money is used.")

    router, meta = get_router(name)
    agent = get_agent(name, mode)
    msgs = st.session_state.setdefault("messages", {})[f"{name}|{mode}"]
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
                rm = msg.get("reply_meta") or {}
                st.caption(f"Reply source: **{msg.get('reply_source', 'offline')}** · status: `{rm.get('status', 'offline')}` · elapsed: {rm.get('elapsed_seconds', 0.0):.3f}s")
                if rm.get("usage"):
                    st.json({"usage": rm["usage"]}, expanded=False)
                if rm.get("error"):
                    st.caption(f"DeepSeek fallback reason: {rm['error']}")
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
            send(name, mode, prompt)
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
