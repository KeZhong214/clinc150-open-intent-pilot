"""Reliable Chinese offline templates built only from Agent facts."""

from __future__ import annotations

from typing import Any

from src.gate5.agent_v2 import TurnV2

DELIVERY = {"delivered": "已送达", "in_transit": "运输中", "not_shipped": "尚未发货"}
STATUS = {"active": "已激活", "inactive": "未激活", "pending": "待处理", "cancelled": "已取消", "completed": "已完成"}
CARD = {"CARD-1001": "CARD-1001", "CARD-1002": "CARD-1002", "CARD-2001": "CARD-2001"}


def _tool_result(turn: TurnV2, name: str) -> dict[str, Any] | None:
    for item in turn.tools:
        if item.get("tool") == name:
            return item.get("result") or {}
    return None


def _ids(text: str) -> str:
    return text


def chinese_reply(turn: TurnV2, user_text: str = "") -> str:
    """A conservative Chinese rendering; never adds a fact not in the turn."""
    route = turn.route
    intent = route.intent if route else None
    tools = turn.tools
    state = turn.state or {}
    if not route or not route.accepted:
        if turn.citations:
            return "已找到以下虚构政策条目（原文为英文，引用编号保留）：\n\n" + "\n".join(
                f"- {c.get('doc_id')}：{c.get('title')}（{c.get('citation')}）" for c in turn.citations
            ) + "\n\n这些内容仅用于课程模拟，不构成真实银行条款。"
        if "handoff_to_human" in [t.get("tool") for t in tools]:
            return "我无法确认这是当前支持的模拟银行业务，已创建人工转接工单；没有执行任何资金操作。"
        if state.get("awaiting_clarification"):
            return "我不太确定你指的是哪项业务。请补充说明是卡片、转账、费用政策还是争议处理。"
        return "我不确定这个请求是否属于当前模拟银行支持范围。请换一种说法，或提供相关业务编号。"

    if state.get("awaiting_confirmation"):
        pid = state.get("pending_id") or ""
        action = state.get("pending_action")
        if action == "cancel_transfer":
            return f"转账 {pid} 当前为待处理状态。取消属于敏感操作，请明确回复“确认”后才能执行；目前尚未取消。"
        if action == "activate_card":
            return f"卡片 {pid} 当前未激活。激活属于敏感操作，请明确回复“确认”后才能执行；目前尚未激活。"
        return f"该操作需要明确确认，当前对象为 {pid}。请回复“确认”或“取消”。"

    card = _tool_result(turn, "get_card_delivery_status")
    if card:
        return f"卡片 {card.get('card_id')} 的配送状态是{ DELIVERY.get(card.get('delivery_status'), card.get('delivery_status')) }，卡片状态是{ STATUS.get(card.get('status'), card.get('status')) }。"

    activated = _tool_result(turn, "activate_card")
    if activated and activated.get("status") == "activated":
        return f"已确认，卡片 {activated.get('card_id')} 已激活。"
    if activated and activated.get("status") == "already_active":
        return f"卡片 {activated.get('card_id')} 已经是激活状态，本次没有重复修改。"

    transfer = _tool_result(turn, "get_transfer_status")
    if transfer:
        return f"转账 {transfer.get('transfer_id')} 当前状态为{ STATUS.get(transfer.get('status'), transfer.get('status')) }，金额 {transfer.get('amount')} {transfer.get('currency', 'USD')}，收款方为 {transfer.get('recipient')}。"
    cancelled = _tool_result(turn, "cancel_transfer")
    if cancelled:
        if cancelled.get("status") == "cancelled":
            return f"已确认，转账 {cancelled.get('transfer_id')} 已取消。"
        if cancelled.get("status") == "already_cancelled":
            return f"转账 {cancelled.get('transfer_id')} 之前已经取消，本次没有重复执行。"
        if cancelled.get("status") == "confirmation_required":
            return f"转账 {cancelled.get('transfer_id')} 当前为待处理状态。取消属于敏感操作，请明确回复“确认”后才能执行；目前尚未取消。"

    dispute = _tool_result(turn, "create_dispute")
    if dispute:
        return f"已创建虚构争议记录 {dispute.get('dispute_id')}，关联交易 {dispute.get('transaction_id')}；不会真实移动资金，后续由人工复核。"

    if "handoff_to_human" in [t.get("tool") for t in tools]:
        return "该请求需要人工复核，已创建虚构转接工单；没有执行未经确认的资金操作。"
    if turn.citations:
        return "根据虚构政策库中的以下条目：\n\n" + "\n".join(
            f"- {c.get('doc_id')}：{c.get('title')}（{c.get('citation')}）" for c in turn.citations
        ) + "\n\n以上仅为课程模拟政策，不是真实银行条款。"
    if state.get("pending_slot"):
        slot = {"card": "银行卡号", "transfer": "交易编号", "dispute": "交易编号"}.get(state.get("pending_slot"), "业务编号")
        return f"请提供虚构的{slot}，例如 CARD-1001 或 TXN-2001。"
    return "请求已处理；请以页面中的业务状态和工具记录为准。"
