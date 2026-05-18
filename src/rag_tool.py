from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma


def build_vectorstore(docs, cfg):
    """
    Takes a list of documents and builds a searchable vector database.
    This is the "indexing" step — done once before queries can be made.
    """
    embeddings = OpenAIEmbeddings(model=cfg["embedding"]["model"])

    return Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=cfg["vectorstore"]["persist_dir"]
    )


def retrieve_docs(vectordb, query, top_k=3):
    """
    Given a query, returns the top_k most relevant documents.
    Uses MMR (Maximal Marginal Relevance) to balance relevance + diversity.
    """
    return vectordb.max_marginal_relevance_search(query, k=top_k)