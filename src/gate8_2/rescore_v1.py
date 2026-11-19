"""Gate 8.2 A: offline rescore of the first-frozen v1 traces.

No model, router or DB call is made.  The only allowed correction is a FAQ
whose v1 answer is grounded but whose v1 task_success was false solely because
the old expected_terms used the question's first word.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
DATA = ROOT / "data" / "gate8_1"
RUBRIC = json.loads((ROOT / "configs" / "gate8_2_faq_rubric.json").read_text(encoding="utf-8"))
MODELS = ["GTE-Frozen-RejectOnly", "Qwen-Frozen-RejectOnly", "Qwen-Plain42-RejectOnly"]
SLUG = {m: m.lower().replace("-", "_") for m in MODELS}
FACT_TERMS = {
    "G81-F001": ["fee", "policy", "quote", "ticket"],
    "G81-F002": ["fee", "quote", "ticket"],
    "G81-F003": ["limit", "personal", "exception"],
    "G81-F004": ["human", "ticket", "expire", "renew"],
    "G81-F005": ["human", "ticket", "currenc", "exchange"],
    "G81-F006": ["transfer", "pending", "status", "ticket"],
    "G81-F007": ["dispute", "reason", "human", "review"],
    "G81-F008": ["top-up", "top up", "bank transfer", "card"],
    "G81-F009": ["failed", "top-up", "top up", "balance", "credit", "ticket"],
    "G81-F010": ["human", "ticket", "card", "accept"],
    "G81-F011": ["exchange", "illustrative", "live", "quote"],
    "G81-F012": ["identity", "password", "pin", "human", "ticket"],
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    v1_cases = {x["case_id"]: x for x in map(json.loads, (DATA / "benchmark_cases_v1.jsonl").read_text(encoding="utf-8").splitlines())}
    v2_cases = {x["case_id"]: x for x in map(json.loads, (DATA / "benchmark_cases.jsonl").read_text(encoding="utf-8").splitlines())}
    rows = []
    for model in MODELS:
        v1 = pd.read_csv(RESULTS / f"gate8_1_agent_{SLUG[model]}_v1_results.csv")
        v2 = pd.read_csv(RESULTS / f"gate8_1_agent_{SLUG[model]}_final_results.csv")
        v2map = v2.set_index("case_id")
        for _, r in v1.iterrows():
            case = v1_cases[r.case_id]
            trace = json.loads(r.trace)
            turn = trace[0] if trace else {}
            response = " ".join(str(x.get("response", "")) for x in trace)
            tools = [t["tool"] for x in trace for t in x.get("tools", [])]
            observed = [c["doc_id"] for x in trace for c in x.get("citations", [])]
            allowed = RUBRIC["cases"].get(r.case_id, {}).get("allowed_doc_ids", [])
            doc_match = any(d in observed for d in allowed) if allowed else False
            terms = FACT_TERMS.get(r.case_id, [])
            fact_hit = any(t.lower() in response.lower() for t in terms)
            handoff = "handoff_to_human" in tools or "human" in response.lower() or "ticket" in response.lower()
            grounded = bool(doc_match and fact_hit) if allowed else bool(handoff and not observed)
            # F009 is a deliberate control: a clarification with no KB evidence
            # must not get credit merely because it mentions a top-up.
            if r.case_id == "G81-F009":
                grounded = False
            original = bool(r.task_success)
            corrected = original
            reason = "unchanged"
            if r.case_id.startswith("G81-F") and not original and r.failure_reason == "response_terms" and grounded:
                corrected = True
                reason = "scoring_bug: grounded FAQ answer; old expected_terms was first word"
            elif r.case_id == "G81-F009":
                reason = "no_grounding: v1 answer is a clarification with no KB-016 evidence"
            elif r.case_id.startswith("G81-F") and not grounded:
                reason = "grounding_fail"
            corr = corrected and not original
            rows.append({
                "model": model,
                "case_id": r.case_id,
                "family": r.family,
                "v1_query": case["turns"][0] if case["turns"] else "",
                "v1_expected_terms": "|".join(case["expected_terms"]),
                "v2_query": v2_cases[r.case_id]["turns"][0] if v2_cases[r.case_id]["turns"] else "",
                "v2_expected_terms": "|".join(v2_cases[r.case_id]["expected_terms"]),
                "text_changed": case["turns"] != v2_cases[r.case_id]["turns"],
                "allowed_doc_ids": "|".join(allowed),
                "observed_doc_ids": "|".join(observed),
                "doc_match": doc_match,
                "fact_hit": fact_hit,
                "grounding_ok": grounded,
                "v1_original_success": original,
                "v1_corrected_success": corrected,
                "correction": corr,
                "correction_reason": reason,
                "v1_failure_reason": r.failure_reason,
                "v2_success": bool(v2map.loc[r.case_id].task_success) if r.case_id in v2map.index else None,
                "response_excerpt": response[:500],
                "tools": "|".join(tools),
                "manual_review": False,
            })
    d = pd.DataFrame(rows)
    d.to_csv(RESULTS / "gate8_2_v1_rescore_per_case.csv", index=False)
    faq = d[d.family == "policy_faq"].copy()
    faq.to_csv(RESULTS / "gate8_2_faq_grounding_audit.csv", index=False)
    summary = []
    for model, g in d.groupby("model", sort=False):
        q = g[g.family == "policy_faq"]
        summary.append({
            "model": model,
            "n": len(g),
            "v1_original_tsr": float(g.v1_original_success.mean()),
            "v1_corrected_tsr": float(g.v1_corrected_success.mean()),
            "delta": float(g.v1_corrected_success.mean() - g.v1_original_success.mean()),
            "faq_n": len(q),
            "faq_v1_original": int(q.v1_original_success.sum()),
            "faq_v1_corrected": int(q.v1_corrected_success.sum()),
            "faq_grounded": int(q.grounding_ok.sum()),
            "corrections": int(g.correction.sum()),
            "v2_tsr": float(g.v2_success.mean()),
            "v1_false_positive": int(((g.v1_original_success) & (~g.v1_corrected_success)).sum()),
            "v1_false_negative": int(((~g.v1_original_success) & (g.v1_corrected_success)).sum()),
        })
    s = pd.DataFrame(summary)
    s.to_csv(RESULTS / "gate8_2_v1_rescore_summary.csv", index=False)
    source = {
        "v1_commit": "7824bb70d8b41e395cefee0a108b7060557180ef",
        "v1_case_file": "data/gate8_1/benchmark_cases_v1.jsonl",
        "v1_case_sha256": sha(DATA / "benchmark_cases_v1.jsonl"),
        "v1_freeze_manifest_sha256": sha(RESULTS / "gate8_1_benchmark_freeze_v1.json"),
        "freeze_declared_case_sha256": json.loads((RESULTS / "gate8_1_benchmark_freeze_v1.json").read_text(encoding="utf-8"))["sha256"],
        "discrepancy": "The v1 freeze manifest records d2e934... while the case file at commit 7824bb7 is b65300...; this audit uses the commit-7824 file requested by the task and records the discrepancy rather than overwriting either.",
        "v1_trace_files": {m: str(RESULTS / f"gate8_1_agent_{SLUG[m]}_v1_results.csv") for m in MODELS},
        "rule": "Only expected_terms scoring bug for grounded FAQ answers; no inference or DB changes.",
        "label": "same-author frozen simulated test; post-hoc offline audit",
    }
    (RESULTS / "gate8_2_v1_source_manifest.json").write_text(json.dumps(source, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(s.round(4).to_string(index=False))
    print("\nFAQ audit:")
    print(faq[["model", "case_id", "grounding_ok", "v1_original_success", "v1_corrected_success", "observed_doc_ids", "correction_reason"]].to_string(index=False))
    print("GATE8_2_RESCORE_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
