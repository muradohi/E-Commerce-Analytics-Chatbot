from src.router import route_query
from src.pandas_tool import load_data, run_pandas_query
from src.rag_tool import retrieve_docs
from src.llm import get_llm
from src.evaluation import evaluate_answer


def run_pipeline(query, vectordb, cfg):

    llm = get_llm(cfg)
    route = route_query(query, cfg)
    df = load_data(cfg)

    answer = None
    context = ""

    # -----------------------
    # PANDAS
    # -----------------------
    if route == "pandas":

        result = run_pandas_query(df, query)

        prompt = f"""
Explain this business result clearly:

{result}

Question: {query}
"""
        answer = llm.invoke(prompt).content

        # convert structured result into context for evaluation
        context = str(result)

    # -----------------------
    # RAG
    # -----------------------
    elif route == "rag":

        docs = retrieve_docs(vectordb, query)
        context = "\n".join([d.page_content for d in docs])

        prompt = f"""
Answer using ONLY this context:

{context}

Question: {query}
"""
        answer = llm.invoke(prompt).content

    # -----------------------
    # HYBRID
    # -----------------------
    else:

        docs = retrieve_docs(vectordb, query)
        context = "\n".join([d.page_content for d in docs])

        result = run_pandas_query(df, query)

        prompt = f"""
Combine structured and unstructured data:

STRUCTURED:
{result}

CONTEXT:
{context}

Question:
{query}
"""
        answer = llm.invoke(prompt).content

        context = f"{result}\n{context}"

    # -----------------------
    # EVALUATION (COMMON FOR ALL)
    # -----------------------
    evaluation = evaluate_answer(query, context, answer)

    return {
        "answer": answer,
        "evaluation": evaluation,
        "route": route
    }