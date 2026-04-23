from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma


def build_vectorstore(docs, cfg):
    embeddings = OpenAIEmbeddings(
        model=cfg["embedding"]["model"]
    )

    return Chroma.from_documents(
        docs,
        embedding=embeddings,
        persist_directory=cfg["vectorstore"]["persist_dir"]
    )


def retrieve_docs(vectordb, query):
    return vectordb.max_marginal_relevance_search(query, k=3)