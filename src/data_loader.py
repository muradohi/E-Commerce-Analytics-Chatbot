import pandas as pd


def load_data(cfg):
    """
    Loads products and orders, merges them, and adds a revenue column.
    Returns one combined DataFrame.
    """
    products = pd.read_csv(cfg["data"]["products"])
    orders = pd.read_csv(cfg["data"]["orders"])

    # Combine orders + products into one table
    df = orders.merge(products, on="product_id")

    # Add a new column: revenue = price × quantity
    df["revenue"] = df["quantity"] * df["price"]

    return df


def load_reviews(cfg):
    """Loads the customer reviews."""
    return pd.read_csv(cfg["data"]["reviews"])