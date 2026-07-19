"""評測主程式：python -m evals.run_eval [--limit N]

流程：對 dataset 每個 case 跑真實 agent，算 5 項指標，印報表並存 JSON。
judge / embeddings = NVIDIA（llama-3.3-70b + bge-m3），與受測的 Qwen 隔離。
"""
from __future__ import annotations

# ⚠️ 必須最先 import config：它會在 import ragas 前注入 vertexai shim
from evals.config import (
    COST_PER_1M_INPUT_TOKENS,
    COST_PER_1M_OUTPUT_TOKENS,
    THRESHOLDS,
    build_judge_and_embeddings,
)

import argparse
import asyncio
import json
import math
import os
from datetime import datetime

from ragas.metrics.collections import AnswerRelevancy, ContextRelevance, Faithfulness

from evals.dataset import EVAL_CASES
from evals.metrics_custom import allergen_violation, task_success, tool_selection_score
from evals.trace import run_case


def _num(result) -> float | None:
    """從 ragas MetricResult 取數值，nan / 失敗回 None。"""
    try:
        v = float(result.value)
        return None if math.isnan(v) else v
    except (TypeError, ValueError, AttributeError):
        return None


async def _score_one(case, judge, embeddings) -> dict:
    trace = await run_case(case)
    answer = trace["answer"]
    contexts = trace["retrieved_contexts"]
    query = trace["query"]

    row: dict = {
        "id": trace["id"],
        "query": query,
        "answer": answer,
        "tool_calls": trace["tool_calls"],
        "n_contexts": len(contexts),
        "latency_seconds": trace["latency_seconds"],
        "model_calls": trace["model_calls"],
        "input_tokens": trace["input_tokens"],
        "output_tokens": trace["output_tokens"],
    }

    # ── ragas 三項（逐項獨立保護；無檢索內容者 N/A） ──
    try:
        row["answer_relevancy"] = _num(await AnswerRelevancy(
            llm=judge, embeddings=embeddings).ascore(user_input=query, response=answer))
    except Exception as e:
        row["answer_relevancy"] = None
        row["answer_relevancy_err"] = str(e)[:120]

    if contexts:
        try:
            row["faithfulness"] = _num(await Faithfulness(llm=judge).ascore(
                user_input=query, response=answer, retrieved_contexts=contexts))
        except Exception as e:
            row["faithfulness"] = None
            row["faithfulness_err"] = str(e)[:120]
        try:
            row["context_relevance"] = _num(await ContextRelevance(llm=judge).ascore(
                user_input=query, retrieved_contexts=contexts))
        except Exception as e:
            row["context_relevance"] = None
            row["context_relevance_err"] = str(e)[:120]
    else:
        row["faithfulness"] = None       # N/A：此 case 未檢索
        row["context_relevance"] = None

    # ── 自製兩項 ──
    ts = tool_selection_score(trace["expected_tools"], trace["tool_calls"])
    row["tool_selection_f1"] = ts["f1"]
    row["tool_exact_match"] = ts["exact_match"]
    row["tool_expected"] = ts["expected"]
    row["tool_actual"] = ts["actual"]

    av = await allergen_violation(answer, trace["allergens"])
    row["has_allergens"] = bool(trace["allergens"])
    row["allergen_violated"] = av["violated"]
    row["allergen_hits"] = av["hits"]

    # ── 第三層：任務成功率（規則式，組合上面已驗證過的訊號） ──
    row["task_success"] = task_success(
        answer=answer, allergen_violated=av["violated"], tool_recall=ts["recall"])

    return row


def _avg(values: list[float | None]) -> float | None:
    nums = [v for v in values if v is not None]
    return sum(nums) / len(nums) if nums else None


def _fmt(v) -> str:
    return "  N/A" if v is None else f"{v:5.2f}"


async def main(limit: int | None) -> None:
    cases = EVAL_CASES[:limit] if limit else EVAL_CASES
    judge, embeddings = build_judge_and_embeddings()

    print(f"\n▶ 評測 {len(cases)} 個 case（judge=NVIDIA llama-3.3-70b, embed=bge-m3）\n")
    rows = []
    for i, case in enumerate(cases, 1):
        print(f"  [{i}/{len(cases)}] {case['id']} ...", flush=True)
        rows.append(await _score_one(case, judge, embeddings))

    # ── 明細表 ──
    print("\n" + "=" * 100)
    print(f"{'case':<24}{'faith':>8}{'ans_rel':>9}{'ctx_rel':>9}{'tool_f1':>9}"
          f"{'allergen':>10}{'success':>9}{'sec':>7}{'tokens':>9}")
    print("-" * 100)
    for r in rows:
        allergen = "-" if not r["has_allergens"] else ("VIOLATED" if r["allergen_violated"] else "ok")
        success = "OK" if r["task_success"] else "FAIL"
        tokens = r["input_tokens"] + r["output_tokens"]
        print(f"{r['id']:<24}{_fmt(r.get('faithfulness')):>8}{_fmt(r.get('answer_relevancy')):>9}"
              f"{_fmt(r.get('context_relevance')):>9}{r['tool_selection_f1']:>9.2f}{allergen:>10}"
              f"{success:>9}{r['latency_seconds']:>7.1f}{tokens:>9}")

    # ── 彙整：第一層（可靠度） ──
    summary = {
        "faithfulness": _avg([r.get("faithfulness") for r in rows]),
        "answer_relevancy": _avg([r.get("answer_relevancy") for r in rows]),
        "context_relevance": _avg([r.get("context_relevance") for r in rows]),
        "tool_selection_accuracy": _avg([r["tool_selection_f1"] for r in rows]),
        "tool_exact_match_rate": _avg([1.0 if r["tool_exact_match"] else 0.0 for r in rows]),
    }
    allergen_cases = [r for r in rows if r["has_allergens"]]
    violations = sum(1 for r in allergen_cases if r["allergen_violated"])
    summary["allergen_violation_rate"] = (
        violations / len(allergen_cases) if allergen_cases else None)

    print("\n" + "=" * 100)
    print("【第一層：可靠度】彙整（平均）與門檻判定：")
    for k in ("faithfulness", "answer_relevancy", "context_relevance",
              "tool_selection_accuracy", "allergen_violation_rate"):
        v = summary[k]
        thr = THRESHOLDS.get(k)
        if v is None:
            verdict = "N/A"
        elif k == "allergen_violation_rate":
            verdict = "PASS" if v <= thr else "FAIL"  # 越低越好
        else:
            verdict = "PASS" if v >= thr else "FAIL"
        print(f"  {k:<28}{_fmt(v)}   門檻={thr}  → {verdict}")
    print(f"  tool_exact_match_rate       {_fmt(summary['tool_exact_match_rate'])}   (參考)")

    # ── 彙整：第三層（單位經濟 / 預估 ROI，不需真實使用者） ──
    n = len(rows)
    success_rate = sum(1 for r in rows if r["task_success"]) / n
    avg_latency = _avg([r["latency_seconds"] for r in rows])
    avg_tokens = _avg([r["input_tokens"] + r["output_tokens"] for r in rows])
    avg_input_tok = _avg([r["input_tokens"] for r in rows])
    avg_output_tok = _avg([r["output_tokens"] for r in rows])

    summary["task_success_rate"] = success_rate
    summary["avg_latency_seconds"] = avg_latency
    summary["avg_tokens_per_task"] = avg_tokens

    print("\n" + "=" * 100)
    print("【第三層：單位經濟 / 預估 ROI】（純測試得出，量是假設值，非真實用量）")
    thr = THRESHOLDS["task_success_rate"]
    verdict = "PASS" if success_rate >= thr else "FAIL"
    print(f"  task_success_rate           {_fmt(success_rate)}   門檻={thr}  → {verdict}")
    print(f"  avg_latency_seconds         {avg_latency:6.1f} 秒/任務")
    print(f"  avg_tokens_per_task         {avg_tokens:8.0f} tokens"
          f"（input≈{avg_input_tok:.0f} / output≈{avg_output_tok:.0f}）")

    if COST_PER_1M_INPUT_TOKENS is not None and COST_PER_1M_OUTPUT_TOKENS is not None:
        avg_cost = (avg_input_tok * COST_PER_1M_INPUT_TOKENS
                    + avg_output_tok * COST_PER_1M_OUTPUT_TOKENS) / 1_000_000
        cost_per_success = avg_cost / success_rate if success_rate > 0 else float("inf")
        summary["avg_cost_usd_per_task"] = avg_cost
        summary["cost_usd_per_successful_task"] = cost_per_success
        print(f"  avg_cost_per_task           ${avg_cost:.4f} USD")
        print(f"  cost_per_successful_task    ${cost_per_success:.4f} USD"
              f"（= 平均成本 ÷ 成功率，失敗會拉高真實單位成本）")
    else:
        print("  cost_per_successful_task    N/A（未設定 NVIDIA NIM 單價，"
              "請填 evals/config.py 的 COST_PER_1M_*_TOKENS 換算金額）")
        summary["avg_cost_usd_per_task"] = None
        summary["cost_usd_per_successful_task"] = None

    # ── 存檔 ──
    out_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(out_dir, f"eval_{ts}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "rows": rows}, f, ensure_ascii=False, indent=2)
    print(f"\n✔ 明細已存：{out_path}\n")


if __name__ == "__main__":
    import sys

    # 背景執行 / 導向檔案時 stdout 會變成區塊緩衝（非行緩衝），導致進度看似卡住、
    # 報表延遲出現。強制行緩衝讓每行即時 flush，不必每次都加 python -u。
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except AttributeError:
        pass

    # psycopg 的 async 模式在 Windows 不支援預設的 ProactorEventLoop，須改 Selector
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="只跑前 N 個 case（冒煙測試用）")
    args = parser.parse_args()
    asyncio.run(main(args.limit))
