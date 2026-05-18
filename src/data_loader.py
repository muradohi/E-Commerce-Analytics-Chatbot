import pandas as pd


def load_data(cfg):
    products = pd.read_csv(cfg["data"]["products"])
    orders = pd.read_csv(cfg["data"]["orders"])
    reviews = pd.read_csv(cfg["data"]["reviews"])
    
    # Compute review aggregates per product
    review_stats = reviews.groupby("product_id").agg(
        avg_rating=("rating", "mean"),
        review_count=("rating", "count")
    ).reset_index()
    
    # Merge everything
    df = orders.merge(products, on="product_id")
    df = df.merge(review_stats, on="product_id", how="left")
    df["revenue"] = df["quantity"] * df["price"]
    
    return df


def load_reviews(cfg):
    """Loads the customer reviews."""
    return pd.read_csv(cfg["data"]["reviews"])