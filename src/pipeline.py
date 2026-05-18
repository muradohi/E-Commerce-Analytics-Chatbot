from src.llm import get_llm
from src.router import route_query
from src.data_loader import load_data
from src.pandas_tool import run_pandas_query
from src.rag_tool import retrieve_docs
from src.evaluation import evaluate_answer


def run_pipeline(query, vectordb, cfg, df=None):
    """
    Main entry point. Takes a query, routes it, gets an answer, evaluates it.
    """
    llm = get_llm(cfg)

    # Load data if not already loaded
    if df is None:
        df = load_data(cfg)

    # Step 1: Figure out which tool to use
    route = route_query(query, llm)

    answer = ""
    context = ""

    # =========================================
    # PATH 1: Pandas (number questions)
    # =========================================
    if route == "pandas":
        result, code = run_pandas_query(df, query, llm)
        context = f"Code used:\n{code}\n\nResult:\n{result}"

        prompt = f"""
You are a business analyst. Explain this result clearly for a non-technical user.

Question: {query}
Code used: {code}
Result: {result}

Give a clear, one-paragraph explanation. Do not invent numbers beyond what's shown.
"""
        answer = llm.invoke(prompt).content

    # =========================================
    # PATH 2: RAG (text questions)
    # =========================================
    elif route == "rag":
        docs = retrieve_docs(vectordb, query, top_k=cfg["retrieval"]["top_k"])
        context = "\n\n".join([d.page_content for d in docs])

        prompt = f"""
You are an e-commerce assistant. Answer using ONLY the context below.

Context:
{context}

Question: {query}

Rules:
- Be concise
- Use only information from the context
- If the context doesn't have the answer, say "I don't have enough information"
"""
        answer = llm.invoke(prompt).content

    # =========================================
    # PATH 3: Hybrid (both)
    # =========================================
    else:  # route == "hybrid"
        result, code = run_pandas_query(df, query, llm)

        # Check if Pandas failed or returned empty
        if isinstance(result, str) and result.startswith("Error"):
            sales_context = "Sales data could not be computed for this question."
        elif hasattr(result, 'empty') and result.empty:
            sales_context = "No matching sales records found."
        else:
            sales_context = f"Code used: {code}\nResult:\n{result}"

        # Get review side
        docs = retrieve_docs(vectordb, query, top_k=cfg["retrieval"]["top_k"])
        retrieved_text = "\n\n".join([d.page_content for d in docs])

        context = f"SALES DATA:\n{sales_context}\n\nCUSTOMER REVIEWS:\n{retrieved_text}"

        prompt = f"""
You are a business analyst. Answer the question using BOTH sources below.

SALES DATA:
{sales_context}

CUSTOMER REVIEWS:
{retrieved_text}

Question: {query}

Rules:
- Use both sources where relevant
- If sales data is missing, answer using only the reviews and explicitly note the missing data
- Be clear about what each source tells you
- Do not invent numbers or facts beyond what's shown
"""
        answer = llm.invoke(prompt).content

    # =========================================
    # Evaluation (runs for all paths)
    # =========================================
    evaluation = evaluate_answer(query, context, answer, llm)

    return {
        "query": query,
        "route": route,
        "answer": answer,
        "context": context,
        "evaluation": evaluation
    }