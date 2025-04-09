import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from deepeval import evaluate
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric,
)
from deepeval.test_case import LLMTestCase
from dotenv import load_dotenv

load_dotenv()

# Create metrics with consistent model
MODEL = "gpt-4o-mini"
metrics = [
    ("answer_relevancy", AnswerRelevancyMetric(model=MODEL)),
    ("faithfulness", FaithfulnessMetric(model=MODEL)),
    ("contextual_precision", ContextualPrecisionMetric(model=MODEL)),
    ("contextual_recall", ContextualRecallMetric(model=MODEL)),
    ("contextual_relevancy", ContextualRelevancyMetric(model=MODEL)),
]

# Read and parse the rag_evaluation.jsonl file
test_cases: List[LLMTestCase] = []
with open("rag_evaluation.jsonl", "r") as file:
    for line in file:
        data = json.loads(line)
        test_case = LLMTestCase(
            input=data["question"],
            actual_output=data["rag_answer"],
            expected_output=data["ground_truth_answer"],
            retrieval_context=data["rag_sources"],
        )
        test_cases.append(test_case)

# Create results directory
results_dir = Path("results/eval-1")
results_dir.mkdir(parents=True, exist_ok=True)

# Common hyperparameters for evaluation
hyperparameters = {
    "model": "gpt-4-turbo",
    "prompt template": """ ## Context (Retrieved Information):{context}\n\n## Few Shot Examples:\n    Example 1:\n    User Query: Which receptor is inhibited by Teprotumumab?\n    Assistant Response: Teprotumumab is a monoclonal inhibitory antibody targeting IGF-1 receptor.\n\n    Example 2:\n    User Query: Does the protein mTOR regulate autophagy?\n    Assistant Response: mammalian target of rapamycin (mTOR)  is a major negative regulator of autophagy.\n\n    Example 3:\n    User Query: Which disease was studied in the CADISS trial?\n    Assistant Response: CADISS was a prospective multicentre randomised-controlled trial in acute (within 7 days of onset) carotid and vertebral artery dissection.\n\n    Example 4:\n    User Query: Is Daprodustat effective for anemia?\n    Assistant Response: Yes. Daprodustat is a hypoxia-inducible factor-prolyl hydroxylase inhibitor for the treatment of anemia of chronic kidney disease.\n\n    Instructions for use:\n    Answer the following biomedical question based on the provided research abstracts.\n    Your answer should be accurate, concise, and based solely on the information provided.\n    If the abstracts don't contain enough information to answer confidently, acknowledge the limitations.\n\n    QUESTION: {query}""",
    "top_k": 10,
    "temperature": 0.2,
    "embedding_model": "all-MiniLM-L12-v2",
}

# Dictionary to store all evaluation results
all_results: Dict[str, Dict[str, Any]] = {"metrics": {}, "summary": {}}

# Run evaluation for each metric
for metric_name, metric in metrics:
    print(f"\nEvaluating {metric_name}...")
    results = evaluate(
        test_cases=test_cases[0:5],
        metrics=[metric],
        hyperparameters=hyperparameters,
    )

    # Calculate metric scores
    metric_scores: List[float] = []
    metric_successes: List[bool] = []
    for test_result in results.test_results:
        if hasattr(test_result, "metrics_data") and test_result.metrics_data:
            metric_data = test_result.metrics_data[0]
            if hasattr(metric_data, "score"):
                metric_scores.append(metric_data.score)
            if hasattr(metric_data, "success"):
                metric_successes.append(metric_data.success)

    overall_score: Optional[float] = (
        sum(metric_scores) / len(metric_scores) if metric_scores else None
    )
    overall_success_rate: Optional[float] = (
        sum(1 for x in metric_successes if x) / len(metric_successes)
        if metric_successes
        else None
    )

    # Store results for this metric
    all_results["metrics"][metric_name] = {
        "overall_score": overall_score,
        "overall_success_rate": overall_success_rate,
        "test_cases": [
            {
                "input": test_case.input,
                "actual_output": test_case.actual_output,
                "expected_output": test_case.expected_output,
                "context": test_case.retrieval_context,
                "evaluation": {
                    "score": test_result.metrics_data[0].score
                    if (
                        hasattr(test_result, "metrics_data")
                        and test_result.metrics_data
                    )
                    else None,
                    "success": test_result.metrics_data[0].success
                    if (
                        hasattr(test_result, "metrics_data")
                        and test_result.metrics_data
                    )
                    else None,
                    "reason": test_result.metrics_data[0].reason
                    if (
                        hasattr(test_result, "metrics_data")
                        and test_result.metrics_data
                    )
                    else None,
                    "threshold": test_result.metrics_data[0].threshold
                    if (
                        hasattr(test_result, "metrics_data")
                        and test_result.metrics_data
                    )
                    else None,
                },
            }
            for test_case, test_result in zip(test_cases[0:5], results.test_results)
        ],
    }

    # Print progress
    score_str = f"{overall_score:.3f}" if overall_score is not None else "N/A"
    success_rate_str = (
        f"{overall_success_rate * 100:.1f}%"
        if overall_success_rate is not None
        else "N/A"
    )
    print(f"{metric_name} Score: {score_str}")
    print(f"{metric_name} Success Rate: {success_rate_str}")

# Calculate overall summary across all metrics
all_scores = [
    metric_data["overall_score"]
    for metric_data in all_results["metrics"].values()
    if metric_data["overall_score"] is not None
]
all_success_rates = [
    metric_data["overall_success_rate"]
    for metric_data in all_results["metrics"].values()
    if metric_data["overall_success_rate"] is not None
]

all_results["summary"] = {
    "total_test_cases": len(test_cases[0:5]),
    "total_metrics_evaluated": len(metrics),
    "average_score_across_metrics": sum(all_scores) / len(all_scores)
    if all_scores
    else None,
    "average_success_rate_across_metrics": sum(all_success_rates)
    / len(all_success_rates)
    if all_success_rates
    else None,
}

# Write results to file
output_file = results_dir / "all_metrics_evaluation.json"
with open(output_file, "w") as f:
    json.dump(all_results, f, indent=2)

print(f"\nEvaluation complete! Results have been saved to {output_file}")
print(
    f"Average Score Across All Metrics: {all_results['summary']['average_score_across_metrics']:.3f}"
)
print(
    f"Average Success Rate Across All Metrics: {all_results['summary']['average_success_rate_across_metrics'] * 100:.1f}%"
)
