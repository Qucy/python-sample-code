import os
from dotenv import load_dotenv

# DeepEval (latest) basic metrics examples
from deepeval.metrics import GEval, FaithfulnessMetric, AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase, LLMTestCaseParams


def run_deepeval_examples():
    # Ensure a provider key is set (e.g., OPENAI_API_KEY)
    # DeepEval will use your default provider configuration for LLM-as-judge metrics

    # Test Case 1: GEval Correctness against expected_output
    correctness_metric = GEval(
        name="Correctness",
        criteria="Determine if the 'actual output' is correct based on the 'expected output'.",
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
        threshold=0.5,
        include_reason=True,
    )
    tc1 = LLMTestCase(
        input="What if these shoes don't fit?",
        actual_output="You have 30 days to get a full refund at no extra cost.",
        expected_output="We offer a 30-day full refund at no extra costs.",
    )
    correctness_metric.measure(tc1)
    print(f"GEval Correctness: score={correctness_metric.score:.3f}\nreason={correctness_metric.reason}\n")

    # Test Case 2: Faithfulness — checks grounding of actual_output in context
    faithfulness_metric = FaithfulnessMetric(threshold=0.7, include_reason=True)
    tc2 = LLMTestCase(
        input="Explain the refund policy.",
        actual_output="Customers have 30 days to return items for a full refund.",
        retrieval_context=[
            "All customers are eligible for a 30 day full refund at no extra costs.",
            "Returns must be in original condition.",
        ],
    )
    faithfulness_metric.measure(tc2)
    print(f"Faithfulness: score={faithfulness_metric.score:.3f}\nreason={faithfulness_metric.reason}\n")

    # Test Case 3: Answer Relevancy — checks if output statements are relevant to input
    relevancy_metric = AnswerRelevancyMetric(threshold=0.7, include_reason=True)
    tc3 = LLMTestCase(
        input="Summarize the refund policy in one sentence.",
        actual_output="There is a 30-day refund window for returns at no extra costs.",
    )
    relevancy_metric.measure(tc3)
    print(f"Answer Relevancy: score={relevancy_metric.score:.3f}\nreason={relevancy_metric.reason}\n")


def main():
    load_dotenv()
    # For OpenAI provider, set OPENAI_API_KEY in your environment
    if not os.getenv("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY is not set. LLM-as-judge metrics may fail.")
    run_deepeval_examples()


if __name__ == "__main__":
    main()