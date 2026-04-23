# app.py

from dotenv import load_dotenv
import os

load_dotenv()

from src.pipeline import run_pipeline
from src.rag_tool import build_vectorstore
from langchain_core.documents import Document
import pandas as pd


# load CSVs
products = pd.read_csv("data/products.csv")
reviews = pd.read_csv("data/reviews.csv")

docs = []

for _, r in products.iterrows():
    docs.append(Document(
        page_content=f"{r['name']}: {r['description']}",
        metadata={"type": "product"}
    ))

for _, r in reviews.iterrows():
    docs.append(Document(
        page_content=r["review_text"],
        metadata={"type": "review"}
    ))

vectordb = build_vectorstore(docs)

while True:
    q = input("Ask: ")
    if q == "exit":
        break

    print(run_pipeline(q, vectordb))