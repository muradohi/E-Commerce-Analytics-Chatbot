import json


def evaluate_answer(query, context, answer, llm):
    """
    Uses the LLM to judge how good the answer is.
    Returns a dict with faithfulness, relevance, and hallucination flags.
    """
    prompt = f"""
You are an evaluation system. Judge the answer based on:

1. Faithfulness: Is the answer supported by the context?
2. Relevance: Does the answer actually address the question?

Return ONLY valid JSON, no extra text:

{{
  "faithfulness": "high" | "medium" | "low",
  "relevance": "high" | "medium" | "low",
  "hallucination": true | false
}}

Context:
{context}

Question:
{query}

Answer:
{answer}
"""

    try:
        result = llm.invoke(prompt).content.strip()
        # Clean up any accidental markdown
        result = result.replace("```json", "").replace("```", "").strip()
        return json.loads(result)
    except json.JSONDecodeError as e:
        print(f"[Evaluation] JSON parse failed: {e}")
        return {
            "faithfulness": "unknown",
            "relevance": "unknown",
            "hallucination": None
        }
    except Exception as e:
        print(f"[Evaluation] Unexpected error: {e}")
        return {
            "faithfulness": "unknown",
            "relevance": "unknown",
            "hallucination": None
        }