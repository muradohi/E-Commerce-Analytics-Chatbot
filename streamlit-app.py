from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import yaml
import pandas as pd
from langchain_core.documents import Document

from src.pipeline import run_pipeline
from src.rag_tool import build_vectorstore

# =========================================
# PAGE CONFIG
# =========================================
st.set_page_config(
    page_title="E-Commerce AI Assistant",
    page_icon="🛒",
    layout="wide"
)

st.title("🛒 E-Commerce AI Assistant")
st.caption("Ask questions about your store, sales, and products")

# =========================================
# LOAD CONFIG
# =========================================
with open("conf/config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

# =========================================
# SIDEBAR SETTINGS
# =========================================
st.sidebar.title("⚙️ Settings")

cfg["model"]["name"] = st.sidebar.selectbox(
    "Model",
    ["gpt-4.1-mini", "gpt-4o"],
    index=0
)

cfg["model"]["temperature"] = st.sidebar.slider(
    "Temperature",
    0.0,
    1.0,
    float(cfg["model"]["temperature"])
)

if st.sidebar.button("🗑️ Clear chat"):
    st.session_state.messages = []
    st.rerun()

st.sidebar.divider()
st.sidebar.caption(
    "Tip: Ask numerical questions (revenue, top products), "
    "review-based questions (customer feedback), or hybrid questions "
    "(top-revenue products with bad reviews)."
)

# =========================================
# VECTORSTORE (built once, cached)
# =========================================
@st.cache_resource
def load_vectordb(cfg):
    """
    Loads product and review data and builds the vector store.
    Cached so it only runs once per session.
    """
    products = pd.read_csv(cfg["data"]["products"])
    reviews = pd.read_csv(cfg["data"]["reviews"])

    # Build product_id -> name lookup so reviews include product context
    product_lookup = dict(zip(products["product_id"], products["name"]))

    docs = []

    # Add product descriptions
    for _, r in products.iterrows():
        docs.append(Document(
            page_content=f"Product: {r['name']}. Description: {r['description']}",
            metadata={"type": "product", "product_id": r["product_id"]}
        ))

    # Add reviews enriched with product name + rating
    for _, r in reviews.iterrows():
        product_name = product_lookup.get(r["product_id"], "Unknown product")
        docs.append(Document(
            page_content=(
                f"Review for {product_name} ({r['rating']} stars): "
                f"{r['review_text']}"
            ),
            metadata={
                "type": "review",
                "product_id": r["product_id"],
                "rating": r["rating"]
            }
        ))

    return build_vectorstore(docs, cfg)


vectordb = load_vectordb(cfg)

# =========================================
# CHAT STATE
# =========================================
if "messages" not in st.session_state:
    st.session_state.messages = []

# =========================================
# DISPLAY CHAT HISTORY
# =========================================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# =========================================
# USER INPUT
# =========================================
user_query = st.chat_input("Ask something about your store...")

if user_query:
    # Save and display the user message
    st.session_state.messages.append({
        "role": "user",
        "content": user_query
    })

    with st.chat_message("user"):
        st.markdown(user_query)

    # Assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking... 🤔"):
            try:
                response = run_pipeline(user_query, vectordb, cfg)
            except Exception as e:
                st.error(f"Something went wrong: {e}")
                response = {
                    "query": user_query,
                    "route": "error",
                    "answer": "Sorry, I couldn't process that question.",
                    "context": "",
                    "evaluation": {}
                }

        # ---- Show the answer
        st.markdown(response["answer"])

        # ---- Show the route taken
        st.caption(f"🧭 Route: **{response['route']}**")

        # ---- Show what was actually used (code or reviews)
        with st.expander("🔍 Show details"):
            if response["route"] == "pandas":
                st.markdown("**Pandas code & result:**")
                st.code(response["context"], language="python")

            elif response["route"] == "rag":
                st.markdown("**Retrieved reviews:**")
                st.text(response["context"])

            elif response["route"] == "hybrid":
                st.markdown("**Sales data + retrieved reviews:**")
                st.text(response["context"])

            else:
                st.text(response["context"] or "No additional context")

        # ---- Show evaluation scores
        with st.expander("📊 Evaluation scores"):
            if response["evaluation"]:
                st.json(response["evaluation"])
            else:
                st.text("No evaluation available.")

    # Save assistant message (only the answer, not the full dict)
    st.session_state.messages.append({
        "role": "assistant",
        "content": response["answer"]
    })