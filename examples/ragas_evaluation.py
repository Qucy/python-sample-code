import os
import asyncio
from dotenv import load_dotenv

# Ragas (latest) single-turn evaluation examples
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import AspectCritic, SimpleCriteriaScore, Faithfulness

# Use LangChain OpenAI as the evaluator LLM wrapper
# Install deps: pip install -U ragas langchain-openai
from langchain_openai import ChatOpenAI
from ragas.llms import LangchainLLMWrapper


def _build_evaluator_llm():
    """
    Returns an evaluator LLM for Ragas metrics using OpenAI via LangChain.
    Set OPENAI_API_KEY in your environment. Optionally set OPENAI_MODEL.
    """
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    # Temperature 0 is recommended for evaluation
    chat = ChatOpenAI(model=model, temperature=0)
    return LangchainLLMWrapper(chat)


async def run_ragas_single_turn_examples():
    evaluator_llm = _build_evaluator_llm()

    # Example 1: Faithfulness (RAG-style) — checks grounding in provided contexts
    sample_faithful = SingleTurnSample(
        user_input="When was the first Super Bowl?",
        response="The first Super Bowl was held on Jan 15, 1967.",
        retrieved_contexts=[
            "The first AFL–NFL World Championship Game was played on January 15, 1967, at the Los Angeles Memorial Coliseum.",
            "It later became known as Super Bowl I.",
        ],
    )
    faithfulness_metric = Faithfulness(llm=evaluator_llm)
    faithfulness_score = await faithfulness_metric.single_turn_ascore(sample_faithful)
    print(f"Faithfulness score: {faithfulness_score:.3f}")

    # Example 2: AspectCritic — general-purpose binary judger for a custom aspect
    sample_aspect = SingleTurnSample(
        user_input="Write a short summary of 'Azure OpenAI'.",
        response="Azure OpenAI is a managed service that hosts OpenAI models on Azure.",
    )
    aspect_critic = AspectCritic(
        name="maliciousness",
        definition="Is the submission intended to harm, deceive, or exploit users?",
        llm=evaluator_llm,
    )
    aspect_score = await aspect_critic.single_turn_ascore(sample_aspect)
    print(f"AspectCritic (maliciousness) verdict: {aspect_score} (1=yes, 0=no)")

    # Example 3: SimpleCriteriaScore — free-form rubric that yields a numeric score
    sample_simple = SingleTurnSample(
        user_input="Explain Azure in one sentence.",
        response="Azure is Microsoft's cloud platform for building, deploying, and managing services.",
        reference="Azure is a cloud computing platform and service created by Microsoft.",
    )
    simple_score_metric = SimpleCriteriaScore(
        name="coarse_grained_similarity",
        definition="Score from 0 to 5 based on similarity to the reference.",
        llm=evaluator_llm,
    )
    coarse_score = await simple_score_metric.single_turn_ascore(sample_simple)
    print(f"SimpleCriteriaScore (0-5): {coarse_score}")


def main():
    load_dotenv()
    asyncio.run(run_ragas_single_turn_examples())


if __name__ == "__main__":
    main()