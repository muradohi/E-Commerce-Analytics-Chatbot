import pandas as pd

orders = pd.read_csv("/Users/murad/Desktop/rag_pipeline/data/orders.csv")
products = pd.read_csv("/Users/murad/Desktop/rag_pipeline/data/products.csv")
reviews = pd.read_csv("/Users/murad/Desktop/rag_pipeline/data/reviews.csv")

df = orders.merge(products, on="product_id")

# print(df.head(5))