import pandas as pd

def load_data(cfg):
    products = pd.read_csv(cfg["data"]["products"])
    orders = pd.read_csv(cfg["data"]["orders"])

    df = orders.merge(products, on="product_id")
    df["revenue"] = df["quantity"] * df["price"]

    return df


def run_pandas_query(df, query):

    q = query.lower()

    if "top" in q:
        return df.groupby("name")["quantity"].sum().sort_values(ascending=False).head(5)

    if "revenue" in q:
        return df.groupby("category")["revenue"].sum()

    return df.describe()