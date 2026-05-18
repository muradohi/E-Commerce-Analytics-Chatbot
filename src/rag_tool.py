from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma


def build_vectorstore(docs, cfg):

    clean_docs = []

    for d in docs:
        if (
            d.page_content is not None
            and isinstance(d.page_content, str)
            and d.page_content.strip()
        ):
            clean_docs.append(d)

    print(f"Original docs: {len(docs)}")
    print(f"Valid docs: {len(clean_docs)}")

    embeddings = OpenAIEmbeddings(
        model=cfg["embedding"]["model"]
    )

    return Chroma.from_documents(
        documents=clean_docs,
        embedding=embeddings,
        persist_directory=cfg["vectorstore"]["persist_dir"]
    )


def retrieve_docs(vectordb, query, top_k=3):
    """
    Given a query, returns the top_k most relevant documents.
    Uses MMR (Maximal Marginal Relevance) to balance relevance + diversity.
    """
    return vectordb.max_marginal_relevance_search(query, k=top_k)