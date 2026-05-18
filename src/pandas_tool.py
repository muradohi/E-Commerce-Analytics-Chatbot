import pandas as pd


def run_pandas_query(df, query, llm):
    """
    Asks the LLM to write Pandas code that answers the question,
    then runs that code on the DataFrame.
    Returns the result AND the code used.
    """
    columns = list(df.columns)
    preview = df.head(3).to_string()

    prompt = f"""
You are a Python expert. There is a Pandas DataFrame called `df` 
with these columns: {columns}

Here's a preview:
{preview}

User question: "{query}"

Write ONE line of Pandas code (using the variable `df`) that answers the question.
Return ONLY the code — no explanation, no quotes, no markdown.

Examples:
- "Total revenue" -> df['revenue'].sum()
- "Top 5 products by quantity" -> df.groupby('name')['quantity'].sum().sort_values(ascending=False).head(5)
- "Revenue per category" -> df.groupby('category')['revenue'].sum()
"""

    code = llm.invoke(prompt).content.strip()
    
    # Strip markdown if the LLM wraps the code in backticks
    code = code.replace("```python", "").replace("```", "").strip()

    try:
        # Run the code with df and pd available in scope
        result = eval(code, {"df": df, "pd": pd})
        return result, code
    except Exception as e:
        return f"Error running the query: {e}", code