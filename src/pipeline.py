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
        context = context = "\n\n".join(str(d.page_content)for d in docs if d.page_content is not None
)

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
        # Step 1: Run Pandas first to get top products
        result, code = run_pandas_query(df, query, llm)

        # Handle errors
        if isinstance(result, str) and result.startswith("Error"):
            sales_context = "Sales data could not be computed for this question."
            top_products = []
        elif hasattr(result, 'empty') and result.empty:
            sales_context = "No matching sales records found."
            top_products = []
        else:
            sales_context = f"Code used: {code}\nResult:\n{result}"
            # Extract product names from result (if it's a Series indexed by name)
            if hasattr(result, 'index'):
                top_products = result.index.tolist()
            else:
                top_products = []

        # Step 2: Build a focused query that includes the top products
        if top_products:
            focused_query = f"{query} (focus on these products: {', '.join(top_products[:5])})"
        else:
            focused_query = query

        # Step 3: Retrieve reviews using the focused query
        docs = retrieve_docs(vectordb, focused_query, top_k=cfg["retrieval"]["top_k"] * 2)

        # Step 4: Filter to only reviews about top products (if we have them)
        if top_products:
            filtered_docs = [
                d for d in docs
                if any(prod in d.page_content for prod in top_products)
            ]
            # If filtering gave us nothing, fall back to original docs
            if filtered_docs:
                docs = filtered_docs[:cfg["retrieval"]["top_k"]]

        retrieved_text = "\n\n".join(str(d.page_content)for d in docs if d.page_content is not None
)

        context = f"SALES DATA:\n{sales_context}\n\nCUSTOMER REVIEWS:\n{retrieved_text}"

        prompt = f"""
You are a business analyst. Answer the question using BOTH sources below.

SALES DATA:
{sales_context}

CUSTOMER REVIEWS (filtered to top products where possible):
{retrieved_text}

Question: {query}

Rules:
- Use both sources where relevant
- If a top-revenue product has no negative reviews shown, say so explicitly
- Do not invent numbers or fabricate reviews
"""
        answer = llm.invoke(prompt).content

    # =========================================
    # EVALUATION (runs for ALL paths)
    # =========================================
    evaluation = evaluate_answer(query, context, answer, llm)

    return {
        "query": query,
        "route": route,
        "answer": answer,
        "context": context,
        "evaluation": evaluation
    }