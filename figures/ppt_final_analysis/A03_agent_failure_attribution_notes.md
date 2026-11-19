# 图表说明与数据核验

```json
{
  "commit": "4c674cbc19ca3c623535033393878d20b5f04ad9",
  "input": "results/gate8_2_v1_rescore_per_case.csv",
  "filter": "v1_corrected_success == False; n=131",
  "categories": [
    "路由/意图错误",
    "FAQ 事实/引用不足",
    "工具选择/参数"
  ],
  "priority": "1) correction_reason grounding_fail/no_grounding -> FAQ; 2) v1_failure_reason routing -> route; 3) tool_selection_or_args -> tool; else other",
  "counts": {
    "GTE-Frozen-RejectOnly|路由/意图错误": 40,
    "GTE-Frozen-RejectOnly|FAQ 事实/引用不足": 6,
    "GTE-Frozen-RejectOnly|工具选择/参数": 2,
    "Qwen-Frozen-RejectOnly|路由/意图错误": 39,
    "Qwen-Frozen-RejectOnly|FAQ 事实/引用不足": 6,
    "Qwen-Frozen-RejectOnly|工具选择/参数": 3,
    "Qwen-Plain42-RejectOnly|路由/意图错误": 27,
    "Qwen-Plain42-RejectOnly|FAQ 事实/引用不足": 7,
    "Qwen-Plain42-RejectOnly|工具选择/参数": 1
  },
  "failed_by_model": {
    "GTE-Frozen-RejectOnly": 48,
    "Qwen-Frozen-RejectOnly": 48,
    "Qwen-Plain42-RejectOnly": 35
  },
  "partition": "Gate8.2 v1 corrected failures only; single primary label, no multi-label double count",
  "caption": "v1 校正后失败归因：路由/意图错误占 106/131",
  "conclusion": "失败主要是路由/意图错误（106/131），工具选择仅 6 例；FAQ 事实/引用不足 19 例。"
}
```
