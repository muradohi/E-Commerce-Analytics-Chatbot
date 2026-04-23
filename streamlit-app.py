from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import yaml
import pandas as pd
from langchain_core.documents import Document

from src.pipeline import run_pipeline
from src.rag_tool import build_vectorstore

# ---------------------------
# PAGE CONFIG
# ---------------------------
st.set_page_config(
    page_title="E-Commerce AI Assistant",
    page_icon="🛒",
    layout="centered"
)

st.title("🛒 E-Commerce AI Assistant")
st.caption("Ask questions about your store, sales, and products")

# ---------------------------
# LOAD CONFIG
# ---------------------------
with open("conf/config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

# ---------------------------
# SIDEBAR SETTINGS
# ---------------------------
st.sidebar.title("⚙️ Settings")

cfg["model"]["name"] = st.sidebar.selectbox(
    "Model",
    ["gpt-4o-mini", "gpt-4o"],
    index=0
)

cfg["model"]["temperature"] = st.sidebar.slider(
    "Temperature",
    0.0,
    1.0,
    float(cfg["model"]["temperature"])
)

# ---------------------------
# VECTORSTORE
# ---------------------------
@st.cache_resource
def load_vectordb(cfg):
    products = pd.read_csv(cfg["data"]["products"])
    reviews = pd.read_csv(cfg["data"]["reviews"])

    docs = []

    for _, r in products.iterrows():
        docs.append(Document(
            page_content=f"Product: {r['name']}. Description: {r['description']}",
            metadata={"type": "product"}
        ))

    for _, r in reviews.iterrows():
        docs.append(Document(
            page_content=f"Review: {r['review_text']}",
            metadata={"type": "review"}
        ))

    return build_vectorstore(docs, cfg)

vectordb = load_vectordb(cfg)

# ---------------------------
# CHAT STATE
# ---------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------------------
# DISPLAY CHAT HISTORY
# ---------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------------------------
# INPUT
# ---------------------------
user_query = st.chat_input("Ask something about your store...")

if user_query:

    # user message
    st.session_state.messages.append({
        "role": "user",
        "content": user_query
    })

    with st.chat_message("user"):
        st.markdown(user_query)

    # assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking... 🤔"):

            response = run_pipeline(
                user_query,
                vectordb,
                cfg
            )

        # ---------------------------
        # DISPLAY ANSWER (FIX HERE)
        # ---------------------------
        st.markdown(response["answer"])

        # evaluation
        with st.expander("📊 Evaluation"):
            st.write("Route:", response["route"])
            st.write(response["evaluation"])

    # save assistant message (ONLY answer, not dict)
    st.session_state.messages.append({
        "role": "assistant",
        "content": response["answer"]
    })