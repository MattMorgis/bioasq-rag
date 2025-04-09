import json

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

# Create metrics
answer_relevancy = AnswerRelevancyMetric()
faithfulness = FaithfulnessMetric()
contextual_precision = ContextualPrecisionMetric()
contextual_recall = ContextualRecallMetric()
contextual_relevancy = ContextualRelevancyMetric()
# Read and parse the rag_evaluation.jsonl file
test_cases = []
with open("rag_evaluation.jsonl", "r") as file:
    for line in file:
        data = json.loads(line)

        # Create a test case with the specified fields
        test_case = LLMTestCase(
            input=data["question"],
            actual_output=data["rag_answer"],
            expected_output=data["ground_truth_answer"],
            retrieval_context=data["rag_sources"],
        )
        test_cases.append(test_case)


# Evaluate all test cases with all metrics
evaluate(
    test_cases=test_cases[0:15],
    metrics=[
        contextual_precision,
        answer_relevancy,
        faithfulness,
        contextual_precision,
        contextual_recall,
        contextual_relevancy,
    ],
)
