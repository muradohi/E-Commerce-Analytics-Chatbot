# core/llm.py
from langchain_openai import ChatOpenAI

def get_llm(cfg):
    return ChatOpenAI(
        model=cfg["model"]["name"],
        temperature=cfg["model"]["temperature"]
    )