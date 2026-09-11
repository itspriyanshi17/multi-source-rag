"""Runs the full agentic pipeline (Phase 3 + Phase 4) on the labeled test set and scores each
answer for faithfulness, relevancy, and correctness via an LLM judge.

Run with: python -m src.evaluation.generation_eval
"""
import json
from collections import defaultdict
from pathlib import Path

from src.evaluation.llm_judge import judge
from src.generation.agentic_pipeline import answer_question_with_context

TEST_SET_PATH = Path(__file__).resolve().parents[2] / "data" / "eval" / "test_set.json"


def evaluate() -> None:
    cases = json.loads(TEST_SET_PATH.read_text(encoding="utf-8"))

    by_category = defaultdict(list)
    print(f"{'ID':<5} {'Faith':<7} {'Relev':<7} {'Correct':<8} Question")

    for case in cases:
        result = answer_question_with_context(case["question"])
        scores = judge(
            question=case["question"],
            context=result.context_used,
            answer=result.answer,
            expected_answer=case["expected_answer"],
        )
        by_category[case["category"]].append(scores)

        correct = scores.get("correctness")
        correct_str = f"{correct:.2f}" if correct is not None else "n/a"
        print(
            f"{case['id']:<5} {scores['faithfulness']:<7.2f} {scores['relevancy']:<7.2f} "
            f"{correct_str:<8} {case['question'][:50]}"
        )

    print(f"\n{'Category':<15} {'n':<4} {'Faithfulness':<14} {'Relevancy':<12} {'Correctness':<12}")
    for category, scores_list in by_category.items():
        n = len(scores_list)
        avg_faith = sum(s["faithfulness"] for s in scores_list) / n
        avg_relev = sum(s["relevancy"] for s in scores_list) / n
        correctness_vals = [s["correctness"] for s in scores_list if s.get("correctness") is not None]
        avg_correct = sum(correctness_vals) / len(correctness_vals) if correctness_vals else None
        avg_correct_str = f"{avg_correct:.3f}" if avg_correct is not None else "n/a"
        print(f"{category:<15} {n:<4} {avg_faith:<14.3f} {avg_relev:<12.3f} {avg_correct_str:<12}")

    all_scores = [s for scores_list in by_category.values() for s in scores_list]
    n = len(all_scores)
    print(f"\n{'OVERALL':<15} {n:<4} "
          f"{sum(s['faithfulness'] for s in all_scores) / n:<14.3f} "
          f"{sum(s['relevancy'] for s in all_scores) / n:<12.3f}")


if __name__ == "__main__":
    evaluate()
