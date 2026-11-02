"""
OPTIONAL interactive version of the sales dashboard (Plotly).
Install:  pip install plotly
Run:      python sales_dashboard_plotly.py   -> creates sales_dashboard_interactive.html
Run sales_analysis.py first (it creates sales_data_clean.csv).
Hover for values, click legend items to hide/show, drag to zoom.
"""
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots

df = pd.read_csv("sales_data_clean.csv", parse_dates=["order_date", "month"])
df = df[df.region != "Unknown"]

monthly = df.groupby("month")[["sales", "profit"]].sum().reset_index()
sub = df.groupby("sub_category")[["sales", "profit"]].sum().sort_values("profit").reset_index()
reg = df.groupby("region")[["sales", "profit"]].sum().reset_index()
seg = df.groupby("segment").sales.sum().reset_index()
heat = df.pivot_table(index="category", columns="region", values="profit", aggfunc="sum")

fig = make_subplots(
    rows=3, cols=2, vertical_spacing=0.12,
    subplot_titles=("Monthly sales and profit", "Sales by customer segment",
                    "Profit by sub-category", "Regional sales and profit",
                    "Profit: category x region", "Sales by category"))

for name, col in [("Sales", "#2a78d6"), ("Profit", "#1baf7a")]:
    fig.add_scatter(x=monthly.month, y=monthly[name.lower()], name=name, mode="lines+markers",
                    line=dict(color=col, width=2), row=1, col=1)
fig.add_bar(x=seg.segment, y=seg.sales, marker_color="#2a78d6", name="Segment sales", row=1, col=2)
fig.add_bar(y=sub.sub_category, x=sub.profit, orientation="h", marker_color="#2a78d6",
            name="Sub-category profit", row=2, col=1)
fig.add_bar(x=reg.region, y=reg.sales, name="Region sales", marker_color="#2a78d6", row=2, col=2)
fig.add_bar(x=reg.region, y=reg.profit, name="Region profit", marker_color="#1baf7a", row=2, col=2)
fig.add_heatmap(z=heat.values, x=heat.columns, y=heat.index, colorscale="Blues",
                showscale=False, row=3, col=1)
cat = df.groupby("category").sales.sum().reset_index()
fig.add_bar(x=cat.category, y=cat.sales, marker_color="#eb6834", name="Category sales", row=3, col=2)

fig.update_layout(height=1050, template="plotly_white", title_text="Sales Analysis Dashboard (interactive)",
                  barmode="group", showlegend=True, legend=dict(orientation="h", y=-0.05))
fig.write_html("sales_dashboard_interactive.html", include_plotlyjs="cdn")
print("saved sales_dashboard_interactive.html")