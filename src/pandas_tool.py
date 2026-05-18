import io
import sys
import pandas as pd


def run_pandas_query(df, query, llm):
    """
    Asks the LLM to write Pandas code that answers the question,
    then runs it on the DataFrame.
    
    Captures stdout to handle functions like df.info() that print
    instead of returning values.
    
    Returns: (result, code) tuple.
    """
    columns = list(df.columns)
    preview = df.head(3).to_string()

    prompt = f"""
You are a Python expert. There is a Pandas DataFrame called `df` 
with these columns: {columns}

Here's a preview:
{preview}

User question: "{query}"

Write ONE line of Pandas code (using `df`) that answers the question.

CRITICAL RULES:
- The code MUST RETURN a value.
- Avoid df.info() and print() — they return None.
- For "top X" or "best X" questions, ALWAYS use groupby + sum/mean + sort + head.
- DO NOT use quantile filters — they return individual rows, not aggregated rankings.

Examples:
- "Total revenue" -> df['revenue'].sum()
- "Top 5 products by revenue" -> df.groupby('name')['revenue'].sum().sort_values(ascending=False).head(5)
- "Top-revenue products" -> df.groupby('name')['revenue'].sum().sort_values(ascending=False).head(5)
- "Worst-selling products" -> df.groupby('name')['quantity'].sum().sort_values().head(5)
- "Revenue per category" -> df.groupby('category')['revenue'].sum()
- "Data overview" -> df.describe()
"""

    code = llm.invoke(prompt).content.strip()
    code = code.replace("```python", "").replace("```", "").strip()

    # Capture stdout in case the LLM uses a print-style function anyway
    captured = io.StringIO()
    
    try:
        sys.stdout = captured
        result = eval(code, {"df": df, "pd": pd})
    except Exception as e:
        sys.stdout = sys.__stdout__
        return f"Error running the query: {e}", code
    finally:
        sys.stdout = sys.__stdout__
    
    # If the code returned None but printed something, use the printed text
    if result is None:
        printed = captured.getvalue().strip()
        if printed:
            return printed, code
        return "The code ran but did not return or print anything.", code
    
    return result, code