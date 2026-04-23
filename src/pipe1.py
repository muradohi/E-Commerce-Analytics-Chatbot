# core/pipeline.py (part)

from src.llm import get_llm
llm = get_llm()

def synthesize_answer(query, context):
    prompt = f"""
You are an e-commerce assistant.

Use the provided context to answer.

Context:
{context}

Question:
{query}

Rules:
- Be concise
- Use only provided information
- If data is missing, say you don't know
"""

    return llm.invoke(prompt).content