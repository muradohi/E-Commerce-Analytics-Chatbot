from langchain_openai import ChatOpenAI

def get_llm(cfg):
    """
    Creates and returns an LLM instance using the config settings.
    """
    return ChatOpenAI(
        model=cfg["model"]["name"],
        temperature=cfg["model"]["temperature"]
    )