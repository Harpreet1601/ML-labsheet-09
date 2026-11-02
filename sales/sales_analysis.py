"""
Project 2: Sales Data Analysis Dashboard
Steps: load -> EDA -> clean -> analyse (incl. RFM customer segmentation) -> visualise -> findings
Run:   python sales_analysis.py   (needs sales_data.csv in the same folder)
Interactive version: python sales_dashboard_plotly.py  (needs `pip install plotly`)
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns

C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GRID,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.titleweight": "bold", "axes.titlesize": 11, "axes.titlecolor": INK,
    "axes.titlelocation": "left",
})
money = mtick.FuncFormatter(lambda x, _: f"₹{x/1e6:.1f}M" if abs(x) >= 1e6 else f"₹{x/1e3:.0f}K")

# =============================== 1. LOAD ===================================
df = pd.read_csv("sales_data.csv", parse_dates=["order_date"])

# =============================== 2. EDA ====================================
print("=" * 70, "\nSHAPE:", df.shape)
print("\nDTYPES:\n", df.dtypes)
print("\nDUPLICATE ROWS:", int(df.duplicated().sum()))
print("\nMISSING VALUES (before cleaning):\n", df.isna().sum())
print("\nDESCRIBE:\n", df[["quantity", "unit_price", "discount", "sales", "profit"]].describe().round(2).T)
print("\nCATEGORY COUNTS:\n", df.category.value_counts())

# =============================== 3. CLEAN ==================================
df = df.drop_duplicates().reset_index(drop=True)
# region is determined by city -> rebuild the mapping from complete rows
city_region = df.dropna(subset=["city", "region"]).groupby("city").region.agg(lambda s: s.mode()[0])
df["region"] = df.region.fillna(df.city.map(city_region)).fillna("Unknown")
df["city"] = df.city.fillna("Unknown")
# segment: use the customer's most common segment, otherwise "Unknown"
cust_seg = df.dropna(subset=["segment"]).groupby("customer_id").segment.agg(lambda s: s.mode()[0])
df["segment"] = df.segment.fillna(df.customer_id.map(cust_seg)).fillna("Unknown")
# discount: median of the sub-category
df["discount"] = df.discount.fillna(df.groupby("sub_category").discount.transform("median"))
# profit: sales x median margin of the sub-category
margin = (df.profit / df.sales).groupby(df.sub_category).transform("median")
df["profit"] = df.profit.fillna((df.sales * margin).round(2))
# derived
df["month"] = df.order_date.dt.to_period("M").dt.to_timestamp()
df["margin_pct"] = df.profit / df.sales * 100

print("\nMISSING VALUES (after cleaning):\n", df.isna().sum().to_string())
df.to_csv("sales_data_clean.csv", index=False)

# =============================== 4. ANALYSE ================================
total_sales, total_profit = df.sales.sum(), df.profit.sum()
orders, customers = df.order_id.nunique(), df.customer_id.nunique()
monthly = df.groupby("month")[["sales", "profit"]].sum()
sub = df.groupby("sub_category")[["sales", "profit"]].sum().sort_values("profit")
reg = df[df.region != "Unknown"].groupby("region")[["sales", "profit"]].sum().sort_values("sales", ascending=False)
seg = df[df.segment != "Unknown"].groupby("segment")[["sales", "profit"]].sum().sort_values("sales", ascending=False)
df["disc_band"] = pd.cut(df.discount, [-0.01, 0, 0.10, 0.20, 0.50],
                         labels=["0%", "1–10%", "11–20%", "21–40%"])
disc = df.groupby("disc_band", observed=True).apply(
    lambda g: g.profit.sum() / g.sales.sum() * 100, include_groups=False)
heat = df[df.region != "Unknown"].pivot_table(index="category", columns="region",
                                             values="profit", aggfunc="sum")

# --- RFM customer segmentation
snap = df.order_date.max() + pd.Timedelta(days=1)
rfm = df.groupby("customer_id").agg(recency=("order_date", lambda s: (snap - s.max()).days),
                                    frequency=("order_id", "nunique"), monetary=("sales", "sum"))
rfm["R"] = pd.qcut(rfm.recency.rank(method="first"), 5, labels=[5, 4, 3, 2, 1]).astype(int)
rfm["F"] = pd.qcut(rfm.frequency.rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)


def label(r):
    if r.R >= 4 and r.F >= 4: return "Champions"
    if r.F >= 4: return "Loyal"
    if r.R >= 4: return "Recent / potential"
    if r.R <= 2 and r.F >= 3: return "At risk"
    if r.R <= 2: return "Lost"
    return "Needs attention"


rfm["rfm_segment"] = rfm.apply(label, axis=1)
rfm.to_csv("customer_rfm.csv")
rfm_sum = rfm.groupby("rfm_segment").agg(customers=("monetary", "size"), revenue=("monetary", "sum"))
rfm_sum = rfm_sum.sort_values("revenue")

# =============================== 5. VISUALISE ==============================
fig = plt.figure(figsize=(17, 12.5), facecolor="white")
gs = fig.add_gridspec(4, 3, height_ratios=[0.26, 1, 1, 1], hspace=0.5, wspace=0.3,
                      left=0.075, right=0.97, top=0.90, bottom=0.05)
fig.text(0.075, 0.955, "Sales Analysis Dashboard", fontsize=20, fontweight="bold", color=INK)
fig.text(0.075, 0.925, f"{df.order_date.min():%b %Y} to {df.order_date.max():%b %Y} · "
         f"{orders:,} orders · synthetic retail dataset", fontsize=10.5, color=INK2)

ax = fig.add_subplot(gs[0, :]); ax.axis("off")
kp = [("Total sales", f"₹{total_sales/1e6:.2f}M"), ("Total profit", f"₹{total_profit/1e6:.2f}M"),
      ("Profit margin", f"{total_profit/total_sales*100:.1f}%"), ("Orders", f"{orders:,}"),
      ("Customers", f"{customers:,}")]
for i, (lab, val) in enumerate(kp):
    x = i / 5
    ax.add_patch(plt.Rectangle((x + 0.004, 0.05), 0.19, 0.9, transform=ax.transAxes,
                               facecolor="#f6f5f2", edgecolor=GRID, lw=1))
    ax.text(x + 0.015, 0.6, val, transform=ax.transAxes, fontsize=21, fontweight="bold", color=INK)
    ax.text(x + 0.015, 0.2, lab, transform=ax.transAxes, fontsize=10, color=INK2)

# 1 monthly sales & profit
ax = fig.add_subplot(gs[1, 0:2])
ax.plot(monthly.index, monthly.sales, color=C[0], lw=2, marker="o", ms=4, label="Sales")
ax.plot(monthly.index, monthly.profit, color=C[2], lw=2, marker="o", ms=4, label="Profit")
ax.set_title("1 · Monthly sales and profit"); ax.yaxis.set_major_formatter(money)
ax.legend(frameon=False, loc="upper left")
pk = monthly.sales.idxmax()
ax.annotate(f"Best month\n{pk:%b %Y}", xy=(pk, monthly.sales.max()), xytext=(-70, -8),
            textcoords="offset points", fontsize=9, arrowprops=dict(arrowstyle="-", color=INK2))

# 2 sales by customer segment
ax = fig.add_subplot(gs[1, 2])
ax.bar(seg.index, seg.sales, color=C[0], width=0.55)
for x, v in enumerate(seg.sales):
    ax.text(x, v, f"{v/total_sales*100:.0f}%", ha="center", va="bottom", fontsize=9, color=INK2)
ax.set_ylim(0, seg.sales.max()*1.12); ax.set_title("2 · Sales by customer segment"); ax.yaxis.set_major_formatter(money); ax.grid(axis="x", visible=False)

# 3 product performance
ax = fig.add_subplot(gs[2, 0])
ax.barh(sub.index, sub.profit, color=[C[1] if v < 0 else C[0] for v in sub.profit], height=0.62)
ax.set_title("3 · Profit by product sub-category"); ax.xaxis.set_major_formatter(money)
ax.grid(axis="y", visible=False); ax.tick_params(axis="y", labelsize=8.5)

# 4 region grouped bars
ax = fig.add_subplot(gs[2, 1])
w, x = 0.38, np.arange(len(reg))
ax.bar(x - w/2, reg.sales, w, color=C[0], label="Sales")
ax.bar(x + w/2, reg.profit, w, color=C[2], label="Profit")
ax.set_xticks(x); ax.set_xticklabels(reg.index); ax.yaxis.set_major_formatter(money)
ax.set_title("4 · Regional sales and profit"); ax.legend(frameon=False); ax.grid(axis="x", visible=False)

# 5 RFM segmentation
ax = fig.add_subplot(gs[2, 2])
ax.barh(rfm_sum.index, rfm_sum.revenue, color=C[5], height=0.6)
for y, (n, v) in enumerate(zip(rfm_sum.customers, rfm_sum.revenue)):
    ax.text(v, y, f"  {n} customers", va="center", fontsize=8.5, color=INK2)
ax.set_title("5 · Customer segmentation (RFM) – revenue"); ax.xaxis.set_major_formatter(money)
ax.set_xlim(0, rfm_sum.revenue.max() * 1.35); ax.grid(axis="y", visible=False)

# 6 discount impact
ax = fig.add_subplot(gs[3, 0])
ax.bar(disc.index.astype(str), disc.values, color=[C[1] if v < 0 else C[0] for v in disc.values], width=0.55)
for xx, v in enumerate(disc.values):
    ax.text(xx, v, f"{v:.1f}%", ha="center", va="bottom" if v >= 0 else "top", fontsize=9, color=INK2)
ax.axhline(0, color=INK2, lw=0.8); ax.set_ylim(min(disc.min()*1.6, -2), disc.max()*1.15); ax.yaxis.set_major_locator(mtick.MultipleLocator(5))
ax.set_title("6 · Profit margin by discount level"); ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
ax.grid(axis="x", visible=False); ax.set_xlabel("discount given")

# 7 heatmap category x region
ax = fig.add_subplot(gs[3, 1])
sns.heatmap(heat / 1e3, annot=True, fmt=".0f", cmap="Blues", ax=ax, linewidths=0.5,
            cbar_kws={"label": "profit (₹ thousand)", "shrink": 0.8})
ax.set_title("7 · Profit: category × region"); ax.set_xlabel(""); ax.set_ylabel(""); ax.grid(False)

# 8 top cities
ax = fig.add_subplot(gs[3, 2])
cities = df[df.city != "Unknown"].groupby("city").sales.sum().sort_values(ascending=False).head(8)[::-1]
ax.barh(cities.index, cities.values, color=C[0], height=0.62)
ax.set_title("8 · Top 8 cities by sales"); ax.xaxis.set_major_formatter(money); ax.grid(axis="y", visible=False)

fig.savefig("sales_dashboard.png", dpi=150); plt.close(fig)

# =============================== 6. FINDINGS ===============================
best_cat = df.groupby("category").profit.sum().idxmax()
worst_sub = sub.profit.idxmin()
q4 = df[df.month.dt.month.isin([10, 11, 12])].sales.sum() / total_sales * 100
y1 = df[df.order_date.dt.year == 2024].sales.sum(); y2 = df[df.order_date.dt.year == 2025].sales.sum()
champ = rfm_sum.loc["Champions"] if "Champions" in rfm_sum.index else None
findings = f"""KEY FINDINGS - Sales analysis
- Total sales ₹{total_sales/1e6:.2f}M, profit ₹{total_profit/1e6:.2f}M (margin {total_profit/total_sales*100:.1f}%) from {orders:,} orders / {customers} customers.
- Sales grew {((y2/y1)-1)*100:.0f}% from 2024 to 2025; Oct-Dec carries {q4:.0f}% of all sales (seasonal peak).
- {best_cat} is the most profitable category; weakest sub-category by profit: {worst_sub} (₹{sub.profit.min()/1e3:.0f}K).
- Top region by sales: {reg.index[0]} (₹{reg.sales.iloc[0]/1e6:.2f}M); smallest: {reg.index[-1]}.
- Discounts hurt: margin is {disc.iloc[0]:.1f}% with no discount vs {disc.iloc[-1]:.1f}% on 21-40% discounts.
- {seg.index[0]} customers give {seg.sales.iloc[0]/total_sales*100:.0f}% of sales.
- RFM: {int(champ.customers) if champ is not None else 0} 'Champions' generate ₹{(champ.revenue/1e6 if champ is not None else 0):.2f}M; {int(rfm_sum.loc['At risk'].customers) if 'At risk' in rfm_sum.index else 0} customers are 'At risk' and worth a win-back offer.
- Recommendation: cap discounts at ~10%, push high-margin Technology bundles in Q4, win back At-risk customers.
NOTE: dataset is synthetic - numbers illustrate the method.
"""
open("sales_findings.txt", "w").write(findings)
print("\n" + findings)