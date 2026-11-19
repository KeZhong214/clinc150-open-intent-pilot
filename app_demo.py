"""Minimal classroom demo entry for the frozen bank Agent.

Only the UI is simplified; routing, tools, confirmations and audit are the same
Gate8Agent / SQLite implementation used by app_gate8.py.
"""

from __future__ import annotations

import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
import uuid
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.gate8.agent import Gate8Agent  # noqa: E402
from src.gate8_1.routers import build  # noqa: E402
from src.gate8_2.chinese_adapter import ChineseTranslationAdapter  # noqa: E402
from src.gate8_2.chinese_templates import chinese_reply  # noqa: E402
from src.gate8_2.deepseek_client import DeepSeekClient  # noqa: E402
from src.gate8_2.grounded_writer import GroundedReplyWriter  # noqa: E402

MODEL = "Qwen-Plain42-RejectOnly"
DEMO_DB = ROOT / "data" / "gate8_2" / "demo_sessions"
DEMO_DB.mkdir(parents=True, exist_ok=True)
CARD_QUERY = "我的卡CARD-1002还没到"
CANCEL_QUERY = "请取消转账 TXN-2002。"
OOS_QUERY = "帮我订一张去巴黎的机票。"


@st.cache_resource
def load_router():
    router, _meta = build(MODEL)
    return router


@st.cache_resource
def deepseek_client():
    return DeepSeekClient()


def new_agent(router) -> Gate8Agent:
    sid = f"demo-{uuid.uuid4().hex[:10]}"
    return Gate8Agent(router, session_id=sid, db_dir=DEMO_DB)


def reset_demo(router) -> None:
    old = st.session_state.get("agent")
    if old is not None:
        old.close()
    st.session_state["agent"] = new_agent(router)
    st.session_state["messages"] = []


def compact_db(agent: Gate8Agent) -> dict:
    snap = agent.snapshot()
    cards = [{"card_id": c["card_id"], "status": c["status"], "delivery": c["delivery_status"]} for c in snap["cards"]]
    transfers = [{"txn": t["transfer_id"], "status": t["status"], "amount": t["amount"]} for t in snap["transfers"]]
    return {"cards": cards, "transfers": transfers}


def assistant_text(turn, original: str, mode: str, translation=None) -> tuple[str, str, dict]:
    template = chinese_reply(turn, original)
    source = "离线中文模板"
    meta = {"status": "offline", "elapsed": 0.0, "usage": {}, "error": None}
    if mode.startswith("DeepSeek"):
        writer = GroundedReplyWriter(deepseek_client())
        if not writer.available:
            source = "离线备用（API 未配置）"
            meta["status"] = "NOT RUN_API_BLOCKED"
        else:
            rr = writer.rewrite(
                user_text=original + "\n[English routing text] " + (translation or original),
                template=template,
                intent=turn.route.intent if turn.route else None,
                accepted=turn.route.accepted if turn.route else False,
                tool_results=turn.tools,
                allowed_doc_ids=[c.get("doc_id") for c in turn.citations],
                pending_confirmation=bool(turn.state.get("awaiting_confirmation")),
                state=turn.state,
                language="zh",
            )
            template = rr.text
            source = "DeepSeek 中文润色" if rr.source == "deepseek" else "安全回退"
            meta = {"status": rr.status, "elapsed": rr.elapsed_seconds, "usage": rr.usage, "error": rr.error}
    return template, source, meta


def send(router, text: str, mode: str) -> None:
    agent: Gate8Agent = st.session_state["agent"]
    st.session_state["messages"].append({"role": "user", "content": text})
    translation = None
    routing_text = text
    if mode.startswith("DeepSeek") and ChineseTranslationAdapter.has_chinese(text):
        tr = ChineseTranslationAdapter(deepseek_client()).translate(text)
        translation = {"status": tr.status, "original": text, "translated": tr.text, "error": tr.error}
        if tr.status != "ok":
            msg = {
                "role": "assistant",
                "content": "中文翻译不可用。课堂备用方案：切换到离线模式，使用已验证的英文业务输入，回复仍为中文。",
                "route": None, "tools": [], "translation": translation,
                "source": "API 翻译失败/未配置", "meta": {"status": tr.status, "error": tr.error},
            }
            st.session_state["messages"].append(msg)
            return
        routing_text = tr.text
    turn = agent.handle(routing_text)
    reply, source, meta = assistant_text(turn, text, mode, routing_text)
    st.session_state["messages"].append({
        "role": "assistant",
        "content": reply,
        "route": turn.route,
        "tools": turn.tools,
        "citations": turn.citations,
        "translation": translation,
        "source": source,
        "meta": meta,
    })


def main() -> None:
    st.set_page_config(page_title="课堂演示 · 银行客服 Agent", page_icon="🏦", layout="wide")
    st.title("少样本开放意图识别 · 银行客服 Agent")
    st.caption("固定演示模型：Qwen Plain42 RejectOnly · 模拟银行 · 所有账户与资金均为虚构")
    try:
        router = load_router()
    except Exception as exc:
        st.error(f"演示模型加载失败：{exc}")
        st.stop()
    if "agent" not in st.session_state:
        reset_demo(router)
    agent: Gate8Agent = st.session_state["agent"]

    with st.sidebar:
        st.header("演示控制")
        mode = st.radio("运行模式", ["离线模式（无 API）", "DeepSeek 增强（需 API）"], index=0)
        st.caption(f"模型：`{MODEL}`")
        if st.button("重置演示（新会话 DB）", use_container_width=True):
            reset_demo(router)
            st.rerun()
        st.divider()
        st.subheader("固定剧情")
        if st.button("① 查询 CARD-1002", use_container_width=True):
            send(router, CARD_QUERY, mode)
            st.rerun()
        if st.button("② 取消 TXN-2002", use_container_width=True):
            send(router, CANCEL_QUERY, mode)
            st.rerun()
        if st.button("③ OOS 备用", use_container_width=True):
            send(router, OOS_QUERY, mode)
            st.rerun()
        st.divider()
        st.subheader("当前模拟 DB")
        db = compact_db(agent)
        st.caption("CARD-1002 / TXN-2002 每次重置后为固定初始状态")
        card = next((c for c in db["cards"] if c["card_id"] == "CARD-1002"), None)
        transfer = next((t for t in db["transfers"] if t["txn"] == "TXN-2002"), None)
        if card:
            st.markdown(f"**CARD-1002**  \n`{card['status']}` / `{card['delivery']}`")
        if transfer:
            st.markdown(f"**TXN-2002**  \n`{transfer['status']}` / {transfer['amount']} USD")
        st.caption("其余虚构账户未变化；确认前 TXN-2002 保持 pending。")
        if mode.startswith("DeepSeek") and not deepseek_client().available:
            st.warning("未配置有效 DEEPSEEK_API_KEY：当前不会发送 API 请求；中文输入会安全回退。")

    for m in st.session_state["messages"]:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m["role"] != "assistant":
                continue
            route = m.get("route")
            if route:
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("意图", route.intent or "—")
                c2.metric("接受", str(route.accepted))
                c3.metric("分数", f"{route.score:.4f}")
                c4.metric("阈值", f"{route.threshold:.4f}")
            if m.get("translation"):
                with st.expander("中文翻译适配（可展开）", expanded=False):
                    st.write("原文：", m["translation"]["original"])
                    st.write("英文：", m["translation"]["translated"] or "—")
                    st.json({"status": m["translation"]["status"], "error": m["translation"]["error"]})
            if m.get("tools"):
                with st.expander(f"工具调用（{len(m['tools'])} 次，可展开）", expanded=False):
                    for t in m["tools"]:
                        st.code(f"{t['tool']} {t['args']}", language="text")
                        st.json(t["result"])
            st.caption(f"回复来源：{m.get('source')} · 状态：{m.get('meta', {}).get('status')}")
            if m.get("meta", {}).get("error"):
                st.caption(f"回退原因：{m['meta']['error']}")

    prompt = st.chat_input("输入中文业务请求；固定演示：CARD-1002 / TXN-2002 / OOS")
    if prompt:
        send(router, prompt, mode)
        st.rerun()

    st.divider()
    st.caption("现场顺序：① → ②（等待确认提示）→ 输入“确认”验证状态变化。OOS 只作备用。")


if __name__ == "__main__":
    main()
