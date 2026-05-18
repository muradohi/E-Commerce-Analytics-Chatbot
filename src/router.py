def route_query(query, llm):
    """
    Decides whether the query should go to:
    - pandas: questions about numbers/totals/averages
    - rag: questions about customer opinions/reviews
    - hybrid: questions that need both
    """
    prompt = f"""
Classify this question into one of three categories.

- "pandas": Asks for numbers, totals, averages, counts, percentages.
            Example: "What's our total revenue?" or "Top 5 products?"

- "rag": Asks about customer opinions, feedback, themes from reviews.
         Example: "What do customers say about shipping?"

- "hybrid": Needs BOTH numbers AND review insights.
            Example: "Why is our top-selling product getting bad reviews?"

Question: "{query}"

Respond with ONLY ONE word: pandas, rag, or hybrid.
"""

    response = llm.invoke(prompt).content.strip().lower()

    if "pandas" in response:
        return "pandas"
    if "hybrid" in response:
        return "hybrid"
    return "rag"  