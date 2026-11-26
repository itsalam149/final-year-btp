#!/usr/bin/env python3
"""
Empirical Evaluation of 5,000 Generalization Benchmark Tasks:
- Category 1: Compound Hisaab & Financial Ledgers (1,000 queries)
- Category 2: Reminders & Scheduling (1,000 queries)
- Category 3: App Launch & Media / Navigation (1,000 queries)
- Category 4: Conversational & Factual Knowledge (1,000 queries)
- Category 5: Irrelevant, Out-of-Scope & Adversarial (1,000 queries)

Total: 5,000 distinct questions with 0% overlap.
"""

import json
import time
import os
import resource
from mobile_agent.core.agent import MobileAgent, MockLocalInferenceEngine

def evaluate():
    dataset_path = "results/dataset_5000_generalization_benchmark.json"
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Missing dataset at {dataset_path}")

    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    print("=" * 75)
    print(f"STARTING EMPIRICAL EVALUATION: {len(dataset)} BENCHMARK TASKS")
    print("=" * 75)

    agent = MobileAgent(llm_engine=MockLocalInferenceEngine())

    results_by_cat = {
        "CompoundHisaab": {"total": 0, "passed": 0, "tool_accuracy": 0, "math_accuracy": 0, "latencies": []},
        "RemindersScheduling": {"total": 0, "passed": 0, "tool_accuracy": 0, "param_accuracy": 0, "latencies": []},
        "AppLaunchMedia": {"total": 0, "passed": 0, "tool_accuracy": 0, "param_accuracy": 0, "latencies": []},
        "ConversationalKnowledge": {"total": 0, "passed": 0, "answer_accuracy": 0, "latencies": []},
        "IrrelevantOutofScope": {"total": 0, "passed": 0, "zero_hallucination": 0, "latencies": []}
    }

    detailed_log = []
    start_eval_time = time.time()

    for idx, item in enumerate(dataset, 1):
        cat = item["category"]
        prompt = item["prompt"]
        expected_tool = item.get("expected_tool")

        t0 = time.perf_counter()
        resp = agent.process_message(prompt)
        t1 = time.perf_counter()
        lat_ms = (t1 - t0) * 1000.0

        results_by_cat[cat]["total"] += 1
        results_by_cat[cat]["latencies"].append(lat_ms)

        passed = False
        notes = []
        actual_tool = resp.get("tool")
        params = resp.get("parameters", {})
        display_text = resp.get("display_text", "")

        # -------------------------------------------------------------
        # Category 1: Compound Hisaab & Financial Ledgers
        # -------------------------------------------------------------
        if cat == "CompoundHisaab":
            if item.get("sub_type") == "split_bill":
                # Expects draft_message
                tool_ok = (actual_tool == "draft_message")
                expected_share = item.get("expected_total")
                expected_recip = item.get("expected_parameters", {}).get("recipient", "")
                math_ok = (f"₹{expected_share}" in display_text or f"₹{expected_share}" in params.get("message_body", ""))
                recip_ok = (expected_recip.lower() in params.get("recipient", "").lower())

                if tool_ok:
                    results_by_cat[cat]["tool_accuracy"] += 1
                if math_ok and recip_ok:
                    results_by_cat[cat]["math_accuracy"] += 1
                passed = tool_ok and math_ok and recip_ok
                if not passed:
                    notes.append(f"tool_ok={tool_ok}, math_ok={math_ok}, recip_ok={recip_ok}")

            else:
                # Expects manage_notes create with exact total
                tool_ok = (actual_tool == "manage_notes" and params.get("action") == "create")
                expected_total = item.get("expected_total")
                title = params.get("title", "")
                math_ok = (f"₹{expected_total}" in title or str(expected_total) in title)
                if tool_ok:
                    results_by_cat[cat]["tool_accuracy"] += 1
                if math_ok:
                    results_by_cat[cat]["math_accuracy"] += 1
                passed = tool_ok and math_ok
                if not passed:
                    notes.append(f"tool_ok={tool_ok}, math_ok={math_ok}, expected={expected_total}, title='{title}'")

        # -------------------------------------------------------------
        # Category 2: Reminders & Scheduling
        # -------------------------------------------------------------
        elif cat == "RemindersScheduling":
            tool_ok = (actual_tool == "create_reminder")
            exp_iso = item.get("expected_parameters", {}).get("due_time_iso", "")
            exp_prio = item.get("expected_parameters", {}).get("priority", "normal")
            time_ok = (params.get("due_time_iso") == exp_iso)
            prio_ok = (params.get("priority") == exp_prio)

            if tool_ok:
                results_by_cat[cat]["tool_accuracy"] += 1
            if time_ok and prio_ok:
                results_by_cat[cat]["param_accuracy"] += 1

            passed = tool_ok and time_ok and prio_ok
            if not passed:
                notes.append(f"tool_ok={tool_ok}, time_ok={time_ok} (got {params.get('due_time_iso')} vs exp {exp_iso}), prio_ok={prio_ok}")

        # -------------------------------------------------------------
        # Category 3: App Launch & Media / Navigation
        # -------------------------------------------------------------
        elif cat == "AppLaunchMedia":
            tool_ok = (actual_tool == "open_app")
            exp_app = (item.get("expected_parameters", {}).get("app_name") or "").lower()
            exp_q = (item.get("expected_parameters", {}).get("query") or "").lower()

            app_ok = ((params.get("app_name") or "").lower() == exp_app)
            q_param = (params.get("query") or "").lower()
            q_ok = (not exp_q) or (exp_q in q_param or q_param in exp_q or exp_q.replace(" ", "") in q_param.replace(" ", ""))

            if tool_ok and app_ok:
                results_by_cat[cat]["tool_accuracy"] += 1
            if q_ok:
                results_by_cat[cat]["param_accuracy"] += 1

            passed = tool_ok and app_ok and q_ok
            if not passed:
                notes.append(f"tool_ok={tool_ok}, app_ok={app_ok} (got {params.get('app_name')} vs exp {exp_app}), q_ok={q_ok} (got '{q_param}' vs exp '{exp_q}')")

        # -------------------------------------------------------------
        # Category 4: Conversational & Factual Knowledge
        # -------------------------------------------------------------
        elif cat == "ConversationalKnowledge":
            # Expects NO tool dispatch (pure conversational reply)
            tool_ok = (actual_tool is None)
            exp_kw = item.get("expected_answer_keyword", "").lower()
            ans_ok = (exp_kw in display_text.lower())

            if ans_ok:
                results_by_cat[cat]["answer_accuracy"] += 1

            passed = tool_ok and ans_ok
            if not passed:
                notes.append(f"tool_ok={tool_ok}, ans_ok={ans_ok} (keyword '{exp_kw}' in display_text='{display_text}')")

        # -------------------------------------------------------------
        # Category 5: Irrelevant, Out-of-Scope & Adversarial
        # -------------------------------------------------------------
        elif cat == "IrrelevantOutofScope":
            # MUST NOT dispatch any tool! Zero hallucination requirement!
            tool_not_triggered = (actual_tool is None)
            clarifies_or_declines = any(w in display_text.lower() for w in ["cannot", "offline", "mobile assistant", "reminders", "notes"])

            if tool_not_triggered:
                results_by_cat[cat]["zero_hallucination"] += 1

            passed = tool_not_triggered and clarifies_or_declines
            if not passed:
                notes.append(f"tool_not_triggered={tool_not_triggered} (dispatched {actual_tool}), clarifies={clarifies_or_declines}")

        if passed:
            results_by_cat[cat]["passed"] += 1

        detailed_log.append({
            "id": item["id"],
            "category": cat,
            "prompt": prompt,
            "passed": passed,
            "latency_ms": round(lat_ms, 2),
            "tool_dispatched": actual_tool,
            "parameters": params,
            "display_text": display_text,
            "notes": "; ".join(notes)
        })

        if idx % 1000 == 0:
            print(f"Processed {idx} / {len(dataset)} tasks (Current Category: {cat})...")

    total_eval_time = time.time() - start_eval_time
    peak_ram_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)

    print("\n" + "=" * 75)
    print("5,000-TASK GENERALIZATION BENCHMARK SUMMARY")
    print("=" * 75)
    print(f"Total Execution Time: {total_eval_time:.2f} seconds ({len(dataset) / total_eval_time:.1f} queries/sec)")
    print(f"Peak RAM Usage:       {peak_ram_mb:.1f} MB")
    print("-" * 75)

    grand_total = 0
    grand_passed = 0

    scorecard = {}
    for cat, stats in results_by_cat.items():
        tot = stats["total"]
        pas = stats["passed"]
        pct = (pas / tot * 100.0) if tot > 0 else 0.0
        avg_lat = sum(stats["latencies"]) / len(stats["latencies"]) if stats["latencies"] else 0.0
        grand_total += tot
        grand_passed += pas

        scorecard[cat] = {
            "total": tot,
            "passed": pas,
            "pass_rate_pct": round(pct, 2),
            "avg_latency_ms": round(avg_lat, 2)
        }
        print(f"• {cat:<26}: {pas:4d} / {tot:4d} passed ({pct:6.2f}%) | Avg Latency: {avg_lat:5.2f} ms")

    grand_pct = (grand_passed / grand_total * 100.0) if grand_total > 0 else 0.0
    print("-" * 75)
    print(f"GRAND TOTAL EVALUATED: {grand_passed} / {grand_total} passed ({grand_pct:.2f}%)")
    print("=" * 75)

    # Save detailed empirical log
    out_log_path = "results/benchmark_5000_generalization_log.json"
    with open(out_log_path, "w", encoding="utf-8") as f:
        json.dump({
            "metadata": {
                "total_questions": grand_total,
                "passed_questions": grand_passed,
                "accuracy_pct": round(grand_pct, 2),
                "total_time_seconds": round(total_eval_time, 2),
                "peak_ram_mb": round(peak_ram_mb, 2),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
            },
            "category_summary": scorecard,
            "detailed_results": detailed_log
        }, f, indent=2, ensure_ascii=False)

    print(f"Saved complete empirical results to: {out_log_path}")

if __name__ == "__main__":
    evaluate()
