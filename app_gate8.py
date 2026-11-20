"""Gate-8 Streamlit demo: router-driven bank Agent with visible decisions."""

from __future__ import annotations

import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import gc
import hmac
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
from src.gate8_2.chinese_adapter import ChineseTranslationAdapter  # noqa: E402
from src.gate8_2.chinese_templates import chinese_reply  # noqa: E402

MODES = ["Offline Router-driven", "DeepSeek-assisted"]
MODE_LABELS = {
    "Offline Router-driven": "离线客服模式（Offline Router-driven）",
    "DeepSeek-assisted": "DeepSeek 增强模式（DeepSeek-assisted）",
}
DEMO_LABELS = {
    "1. Card delivery": "1. 卡片配送查询",
    "2. Activate card (confirm)": "2. 激活卡片（需确认）",
    "3. Transfer status (slot)": "3. 转账状态（补充编号）",
    "4. Cancel transfer (confirm)": "4. 取消转账（需确认）",
    "5. Policy citation": "5. 政策/费用问答",
    "6. Dispute completed txn": "6. 争议处理",
    "7. Multi-turn correction": "7. 多轮纠正",
    "8. OOS handoff": "8. OOS 转人工",
    "9. Ownership / overreach": "9. 越权/权限拒绝",
}
SOURCE_LABELS = {
    "offline": "离线模板",
    "deepseek": "DeepSeek 润色",
    "fallback": "安全回退",
    "offline_blocked": "离线回退（API 未配置）",
    "translation_blocked": "中文翻译不可用",
    "translation_fallback": "翻译/歧义回退",
    "api_config_error": "API 配置错误",
    "api_auth_error": "API 认证失败",
    "api_billing_error": "API 余额/计费问题",
    "api_rate_error": "API 限流",
    "api_timeout": "API 超时/网络",
}


def classify_api_error(error: str | None) -> str | None:
    e = (error or "").lower()
    if "config_error" in e or "angle brackets" in e or "quote characters" in e or "whitespace" in e:
        return "api_config_error"
    if "http 401" in e or "authentication" in e or "invalid" in e and "api key" in e:
        return "api_auth_error"
    if "http 402" in e or "balance" in e or "billing" in e:
        return "api_billing_error"
    if "http 429" in e or "rate" in e:
        return "api_rate_error"
    if "timeout" in e or "timed out" in e or "network" in e:
        return "api_timeout"
    return None

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


@st.cache_resource
def chinese_translator():
    return ChineseTranslationAdapter(deepseek_client())


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
    translation = None
    routing_text = text
    if mode == "DeepSeek-assisted" and chinese_translator().has_chinese(text):
        tr = chinese_translator().translate(text)
        translation = {
            "status": tr.status,
            "original": text,
            "translated": tr.text,
            "elapsed_seconds": tr.elapsed_seconds,
            "usage": tr.usage,
            "error": tr.error,
        }
        if tr.status != "ok":
            kind = classify_api_error(tr.error)
            if tr.status == "NOT RUN_API_BLOCKED":
                reply = "当前未配置 DeepSeek API，中文翻译不可用。你可以切换到“离线客服模式”，或先按配置指南设置 Key 后再使用 DeepSeek 增强模式。"
                source = "translation_blocked"
            elif tr.status == "CONFIG_ERROR":
                reply = "DeepSeek API Key 格式明显不正确（可能包含尖括号、引号或首尾空格）。请只把真实 Key 本身写入环境变量，然后重启 PowerShell 与 Streamlit；本次没有执行任何工具。"
                source = "api_config_error"
            elif kind == "api_auth_error":
                reply = "DeepSeek API 认证失败（HTTP 401），说明 Key 无效、被撤销或复制不完整。请重新创建/核对 Key，确认没有尖括号、引号或多余字符；本次没有执行任何工具。"
                source = "api_auth_error"
            elif kind == "api_billing_error":
                reply = "DeepSeek 返回余额/计费问题（HTTP 402）。请检查账户余额；本次没有执行任何工具。"
                source = "api_billing_error"
            elif kind == "api_rate_error":
                reply = "DeepSeek 返回限流（HTTP 429）。请停止重试，稍后再试；本次没有执行任何工具。"
                source = "api_rate_error"
            elif kind == "api_timeout":
                reply = "DeepSeek 请求超时或网络不可达。请检查代理/网络或稍后重试；本次没有执行任何工具。"
                source = "api_timeout"
            else:
                reply = "这条中文请求无法可靠翻译或存在歧义。为避免误触发银行业务，尚未执行任何工具；请换一种说法或补充业务编号。"
                source = "translation_fallback"
            msgs.append({
                "role": "assistant",
                "content": reply,
                "route": {"method": None, "intent": None, "accepted": None, "score": None, "threshold": None, "top3": [], "gate_top1": None},
                "tools": [],
                "citations": [],
                "state": {},
                "translation": translation,
                "reply_source": source,
                "reply_meta": {"status": tr.status, "error": tr.error, "elapsed_seconds": tr.elapsed_seconds, "usage": tr.usage},
            })
            return
        routing_text = tr.text
    turn = agent.handle(routing_text)
    route = turn.route
    template = chinese_reply(turn, text)
    reply = template
    reply_source = "offline"
    reply_meta = {"status": "offline", "error": None, "elapsed_seconds": 0.0, "usage": {}}
    if mode == "DeepSeek-assisted":
        ds_cfg = deepseek_client().status()
        if ds_cfg.get("config_error"):
            reply_source = "api_config_error"
            reply_meta = {"status": "CONFIG_ERROR", "error": ds_cfg["config_error"], "elapsed_seconds": 0.0, "usage": {}}
        elif not deepseek_writer().available:
            reply_source = "offline_blocked"
            reply_meta = {"status": "NOT RUN_API_BLOCKED", "error": "DEEPSEEK_API_KEY not configured", "elapsed_seconds": 0.0, "usage": {}}
        else:
            rr = deepseek_writer().rewrite(
                user_text=text + "\n[English routing text] " + routing_text,
                template=template,
                intent=route.intent if route else None,
                accepted=route.accepted if route else False,
                tool_results=turn.tools,
                allowed_doc_ids=[c.get("doc_id") for c in turn.citations],
                pending_confirmation=bool(turn.state.get("awaiting_confirmation")),
                state=turn.state,
                language="zh",
            )
            reply = rr.text
            reply_source = rr.source
            if rr.status != "ok":
                reply_source = classify_api_error(rr.error) or "fallback"
            reply_meta = {"status": rr.status, "error": rr.error, "elapsed_seconds": rr.elapsed_seconds, "usage": rr.usage}
    msgs.append(
        {
            "role": "assistant",
            "content": reply,
            "offline_template": template,
            "translation": translation,
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


def _access_allowed() -> bool:
    """Optional shared-token gate for a temporary public tunnel."""
    expected = os.environ.get("OPENINTENT_ACCESS_TOKEN", "")
    if not expected:
        return True
    supplied = st.query_params.get("token", "")
    return hmac.compare_digest(str(supplied), expected)


def main() -> None:
    st.set_page_config(page_title="Gate 8.2 银行智能客服工作台", page_icon="🏦", layout="wide")
    if not _access_allowed():
        st.error("演示访问令牌缺失或无效。请使用包含 token 的完整链接。")
        st.stop()
    st.title("Gate 8.2 · 银行智能客服工作台")
    st.caption("离线客服模式 / DeepSeek 增强模式 · Gate8.1 配对路由器 · 中文界面 · 真实工具与知识库")
    ds_status = deepseek_client().status()
    if "pending_method" in st.session_state:
        st.session_state["router_choice"] = st.session_state.pop("pending_method")

    with st.sidebar:
        st.header("模型与运行模式")
        default_router = os.environ.get("OPENINTENT_DEFAULT_ROUTER", "GTE-Frozen-RejectOnly")
        default_router = default_router if default_router in ROUTER_NAMES else "GTE-Frozen-RejectOnly"
        default_mode = os.environ.get("OPENINTENT_DEFAULT_MODE", "Offline Router-driven")
        default_mode = default_mode if default_mode in MODES else "Offline Router-driven"
        name = st.radio("意图路由模型（技术标识保持原样）", ROUTER_NAMES, index=ROUTER_NAMES.index(default_router), key="router_choice")
        mode = st.radio("回复模式", MODES, index=MODES.index(default_mode), key="reply_mode", format_func=lambda x: MODE_LABELS[x])
        st.caption("同一时间只加载一个编码器，避免 8GB 显存不足；切换模型会释放旧会话。")
        st.caption("中文输入说明：离线模式为中文固定模板；DeepSeek 增强模式会先翻译成英文，再交给原英文路由模型。")
        if st.session_state.get("last_release"):
            st.caption(f"上次模型切换显存：{st.session_state['last_release']}")
        if mode == "DeepSeek-assisted" and ds_status.get("config_error"):
            st.error(f"DeepSeek API Key 格式错误：{ds_status['config_error']}。请只设置 Key 本身，不要包含尖括号、引号或空格。")
        elif mode == "DeepSeek-assisted" and not ds_status["available"]:
            st.error("已选择 DeepSeek 增强模式，但未配置 DEEPSEEK_API_KEY。当前为 NOT RUN_API_BLOCKED，回复会回退到离线中文模板。")
        st.markdown(f"**当前模式**：{MODE_LABELS[mode]}")
        st.caption("意图/OOS、业务工具、确认和权限始终由本地 Agent 决定；DeepSeek 只负责翻译和中文措辞。")
        st.json(ds_status, expanded=False)
        if st.button("重置当前会话与虚构数据库", use_container_width=True):
            reset(name, mode)
            st.rerun()
        st.divider()
        st.header("现场演示")
        for label in DEMOS:
            if st.button(DEMO_LABELS[label], use_container_width=True):
                run_demo(name, mode, label)
                st.rerun()
        st.divider()
        st.caption("所有业务数据均为模拟；没有真实银行、客户、卡号或资金。")

    router, meta = get_router(name)
    agent = get_agent(name, mode)
    msgs = st.session_state.setdefault("messages", {})[f"{name}|{mode}"]
    chat_col, side_col = st.columns([1.45, 1.0], gap="large")

    with chat_col:
        st.subheader("客服对话")
        if not msgs:
            st.info("从左侧选择一个演示，或直接输入中文/英文请求。例如：`我的银行卡 CARD-1001 还没收到。`")
        for msg in msgs:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg["role"] != "assistant":
                    continue
                r = msg.get("route") or {}
                c1, c2, c3, c4 = st.columns(4)
                score = f"{r['score']:.4f}" if r.get("score") is not None else "—"
                threshold = f"{r['threshold']:.4f}" if r.get("threshold") is not None else "—"
                c1.markdown(f"**意图**  \n`{r.get('intent') or '—'}`")
                c2.markdown(f"**是否接受**  \n`{r.get('accepted')}`")
                c3.markdown(f"**分数**  \n`{score}`")
                c4.markdown(f"**阈值**  \n`{threshold}`")
                if msg.get("translation"):
                    tr = msg["translation"]
                    with st.expander(f"中文翻译适配（{tr.get('status')}）", expanded=False):
                        st.write("中文原文：", tr.get("original"))
                        st.write("翻译后英文：", tr.get("translated") or "—")
                        st.json({"status": tr.get("status"), "elapsed_seconds": tr.get("elapsed_seconds"), "usage": tr.get("usage"), "error": tr.get("error")})
                if r.get("top3"):
                    st.dataframe(
                        pd.DataFrame(
                            [{"排名": i + 1, "意图": n, "分数": round(float(s), 5)} for i, (n, s) in enumerate(r["top3"])]
                        ),
                        hide_index=True,
                        use_container_width=True,
                    )
                rm = msg.get("reply_meta") or {}
                st.caption(f"回复来源：**{SOURCE_LABELS.get(msg.get('reply_source', 'offline'), msg.get('reply_source'))}** · 状态：`{rm.get('status', 'offline')}` · 耗时：{rm.get('elapsed_seconds', 0.0):.3f}s")
                if rm.get("usage"):
                    st.json({"服务端 usage": rm["usage"]}, expanded=False)
                if rm.get("error"):
                    st.caption(f"安全回退原因：{rm['error']}")
                if msg.get("offline_template") and msg.get("reply_source") in {"deepseek", "fallback"}:
                    with st.expander("离线中文模板（对照）", expanded=False):
                        st.write(msg["offline_template"])
                if msg.get("citations"):
                    with st.expander(f"知识库引用（{len(msg['citations'])} 条）", expanded=True):
                        for c in msg["citations"]:
                            st.markdown(f"**{c['doc_id']}** · {c['title']} · {c['citation']}")
                            st.caption(c["text"])
                if msg.get("tools"):
                    with st.expander(f"工具调用记录（{len(msg['tools'])} 次）", expanded=True):
                        for t in msg["tools"]:
                            st.markdown(f"**{t['tool']}** `{t['args']}`")
                            st.json(t["result"])
        prompt = st.chat_input("请输入中文银行客服请求，例如：我的卡还没收到 / 帮我取消转账 TXN-2002")
        if prompt:
            send(name, mode, prompt)
            st.rerun()

    with side_col:
        st.subheader("模型决策来源")
        st.json(
            {
                "router": meta["name"],
                "encoder": meta["encoder"],
                "threshold": meta["threshold"],
                "threshold_source": meta["threshold_source"],
                "checkpoint_sha256": meta["checkpoint_sha256"],
                "adapter_sha256": meta["adapter_sha256"],
                "mode": MODE_LABELS[mode],
                "deepseek_model": ds_status["model"],
            },
            expanded=False,
        )
        st.subheader("业务状态")
        snap = agent.snapshot()
        tabs = st.tabs(["银行卡", "转账", "工单", "争议", "审计"])
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
        st.subheader("虚构政策知识库")
        q = st.text_input("检索虚构政策（英文关键词）", value="transfer fee")
        if q:
            hits = kb().search(q, k=3)
            st.dataframe(
                pd.DataFrame([{"文档编号": h["doc_id"], "标题": h["title"], "相关度": round(h["score"], 4)} for h in hits]),
                hide_index=True,
                use_container_width=True,
            )


if __name__ == "__main__":
    main()
