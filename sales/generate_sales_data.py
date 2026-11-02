"""
Generates a retail sales dataset (SYNTHETIC, Superstore-style) with
deliberately injected missing values and a few duplicate rows so that the
cleaning step in sales_analysis.py has real work to do.
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(7)
N = 6000

products = {
    "Furniture": {"Chairs": 220, "Tables": 380, "Bookcases": 260, "Sofas": 540},
    "Technology": {"Laptops": 820, "Phones": 560, "Printers": 310, "Headphones": 120},
    "Office Supplies": {"Paper": 18, "Binders": 25, "Storage Boxes": 60, "Pens": 9},
}
# margin by category (Technology is best, Furniture is hurt by discounts)
margin = {"Furniture": 0.12, "Technology": 0.22, "Office Supplies": 0.17}

regions = {"North": 0.18, "South": 0.22, "East": 0.25, "West": 0.27, "Central": 0.08}
region_cities = {
    "North": ["Delhi", "Chandigarh", "Dehradun", "Jaipur"],
    "South": ["Bengaluru", "Chennai", "Hyderabad", "Kochi"],
    "East": ["Kolkata", "Bhubaneswar", "Patna", "Guwahati"],
    "West": ["Mumbai", "Pune", "Ahmedabad", "Surat"],
    "Central": ["Bhopal", "Indore", "Nagpur", "Raipur"],
}
segments = {"Consumer": 0.52, "Corporate": 0.30, "Home Office": 0.18}

dates = pd.to_datetime("2024-01-01") + pd.to_timedelta(rng.integers(0, 730, N), unit="D")
# seasonality: more orders in Oct-Dec
keep = rng.random(N) < (0.7 + 0.5 * np.isin(dates.month, [10, 11, 12]))
dates = dates[keep]
n = len(dates)

cust_ids = [f"C{ i:04d}" for i in rng.integers(1, 700, n)]
cat = rng.choice(list(products), n, p=[0.26, 0.24, 0.50])
sub = np.array([rng.choice(list(products[c])) for c in cat])
price = np.array([products[c][s] for c, s in zip(cat, sub)]) * rng.uniform(0.9, 1.1, n)
qty = rng.integers(1, 8, n)
discount = rng.choice([0, 0.05, 0.10, 0.20, 0.30, 0.40], n, p=[0.40, 0.15, 0.15, 0.15, 0.10, 0.05])
region = rng.choice(list(regions), n, p=list(regions.values()))
city = np.array([rng.choice(region_cities[r]) for r in region])
segment = rng.choice(list(segments), n, p=list(segments.values()))

sales = price * qty * (1 - discount)
m = np.array([margin[c] for c in cat])
profit = sales * (m - 0.55 * discount) + rng.normal(0, 0.02, n) * sales
# growth trend over the two years
sales = sales * (1 + 0.25 * (dates - dates.min()).days / 730)
profit = profit * (1 + 0.25 * (dates - dates.min()).days / 730)

df = pd.DataFrame({
    "order_id": [f"ORD-{100000 + i}" for i in range(n)],
    "order_date": dates, "customer_id": cust_ids, "segment": segment,
    "region": region, "city": city, "category": cat, "sub_category": sub,
    "quantity": qty, "unit_price": price.round(2), "discount": discount,
    "sales": sales.round(2), "profit": profit.round(2),
}).sort_values("order_date").reset_index(drop=True)

# ---- inject data-quality problems ----------------------------------------
idx = np.arange(len(df))
df.loc[rng.choice(idx, int(len(df) * 0.03), replace=False), "discount"] = np.nan
df.loc[rng.choice(idx, int(len(df) * 0.02), replace=False), "region"] = np.nan
df.loc[rng.choice(idx, int(len(df) * 0.015), replace=False), "segment"] = np.nan
df.loc[rng.choice(idx, int(len(df) * 0.02), replace=False), "profit"] = np.nan
df.loc[rng.choice(idx, int(len(df) * 0.01), replace=False), "city"] = np.nan
df = pd.concat([df, df.sample(25, random_state=1)], ignore_index=True)  # duplicates

df.to_csv("sales_data.csv", index=False)
print("sales_data.csv", df.shape)