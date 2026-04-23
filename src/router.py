from langchain_openai import ChatOpenAI
import json

def route_query(query: str, cfg):

    llm = ChatOpenAI(
        model=cfg["model"]["name"],
        temperature=0
    )

    prompt = f"""
You are an expert routing system for an e-commerce AI assistant.

Your job is to classify the user query into exactly ONE of the following tools:

1. "pandas"
   - Use when the question requires calculations, aggregations, trends, totals, comparisons, or numerical analysis.
   - Examples: revenue, sales, counts, averages, top products, performance over time.

2. "rag"
   - Use when the question is about product descriptions, features, reviews, or semantic understanding of text.
   - Examples: "What do customers think of X?", "Describe product Y", "What is this product used for?"

3. "hybrid"
   - Use when the question requires BOTH numerical analysis AND explanation or context from text.
   - Examples: "Why did sales drop for product X?", "Compare performance and explain reasons."

---

IMPORTANT RULES:
- Return ONLY valid JSON.
- Do NOT include explanations.
- Do NOT output anything except JSON.
- Always choose the MOST appropriate single route.

---

OUTPUT FORMAT:
{{"route": "pandas" | "rag" | "hybrid"}}

---

EXAMPLES:

Query: What is total revenue last month?
Output: {{"route": "pandas"}}

Query: What do users say about the wireless mouse?
Output: {{"route": "rag"}}

Query: Why did sales drop for product A compared to reviews?
Output: {{"route": "hybrid"}}

Query:
{query}

Output:
"""

    try:
        result = llm.invoke(prompt).content
        return json.loads(result)["route"]
    except:
        return "hybrid"