import numpy as np
import pandas as pd

products = pd.DataFrame([
    [1, "Wireless Mouse", "Electronics", "Ergonomic wireless mouse with USB receiver"],
    [2, "Mechanical Keyboard", "Electronics", "RGB mechanical keyboard with blue switches"],
    [3, "Water Bottle", "Lifestyle", "1L stainless steel insulated bottle"],
    [4, "Running Shoes", "Sportswear", "Lightweight running shoes for daily training"],
    [5, "Backpack", "Accessories", "Laptop backpack with multiple compartments"],
    [6, "Smart Watch", "Electronics", "Fitness tracking smart watch with heart rate monitor"],
    [7, "Yoga Mat", "Fitness", "Non-slip yoga mat for home workouts"],
    [8, "Desk Lamp", "Home", "LED desk lamp with adjustable brightness"]
], columns=["product_id", "name", "category", "description"])

products.to_csv("data/products.csv", index=False)


np.random.seed(42)

orders = pd.DataFrame({
    "order_id": range(1, 51),
    "product_id": np.random.randint(1, 9, 50),
    "quantity": np.random.randint(1, 5, 50),
    "price": np.random.randint(10, 200, 50),
    "date": pd.date_range("2024-01-01", periods=50, freq="D")
})

orders.to_csv("data/orders.csv", index=False)

reviews = pd.DataFrame([
    [1, "Great mouse, very smooth and responsive", 5],
    [2, "Keyboard is loud but satisfying to type on", 4],
    [3, "Bottle keeps water cold for hours", 5],
    [4, "Shoes are comfortable but run slightly small", 4],
    [5, "Backpack quality is okay for the price", 3],
    [6, "Smartwatch battery life could be better", 3],
    [7, "Yoga mat has good grip and thickness", 5],
    [8, "Lamp brightness is adjustable and useful", 4],
    [1, "Stopped working after 3 months", 2],
    [2, "RGB lights are beautiful", 5],
    [6, "Great for tracking workouts", 5],
    [4, "Very comfortable for running", 5]
], columns=["product_id", "review_text", "rating"])

reviews.to_csv("data/reviews.csv", index=False)
