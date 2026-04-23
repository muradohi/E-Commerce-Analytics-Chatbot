from langchain_openai import ChatOpenAI
import json

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


def evaluate_answer(query: str, context: str, answer: str):
    prompt = f"""
You are an evaluation system for an AI chatbot.

Evaluate the answer based on:

1. Faithfulness (is it supported by context?)
2. Relevance (does it answer the question?)

Return ONLY valid JSON:

{{
  "faithfulness": "high | medium | low",
  "relevance": "high | medium | low",
  "hallucination": true | false
}}

---

Context:
{context}

Question:
{query}

Answer:
{answer}
"""

    try:
        result = llm.invoke(prompt).content
        return json.loads(result)
    except:
        return {
            "faithfulness": "unknown",
            "relevance": "unknown",
            "hallucination": None
        }