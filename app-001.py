import streamlit as st
import pandas as pd
import sqlite3
from datetime import datetime

DB_PATH = "gt_sfa.db"

# =========================================================
# HELPERS
# =========================================================
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.replace("\n", " ", regex=False)
        .str.replace("  ", " ", regex=False)
    )
    return df


def to_iso(date_str: str) -> str:
    return datetime.strptime(date_str, "%d/%m/%Y").strftime("%Y-%m-%d")


def extract_single_date(series: pd.Series, label: str) -> str:
    dates = pd.to_datetime(series, errors="coerce").dt.date.dropna().unique()
    if len(dates) != 1:
        st.error(f"{label} must contain exactly ONE date")
        st.stop()
    return dates[0].strftime("%d/%m/%Y")


def detect_value_column(df: pd.DataFrame) -> str:
    for col in df.columns:
        if "VALUE" in col.upper() and "SOLD" in col.upper():
            return col
    return None


# =========================================================
# DATABASE
# =========================================================
def get_connection():
    return sqlite3.connect(DB_PATH, check_same_thread=False)


def init_db():
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS rep_daily_activity (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        report_date TEXT,
        report_date_iso TEXT,
        sales_rep_id INTEGER,
        sales_rep_name TEXT,
        region_name TEXT,
        sales_target_value_kes REAL,
        customers_on_pjp INTEGER,
        actual_visits INTEGER,
        new_customers_mapped INTEGER,
        time_spent_per_outlet_seconds TEXT,
        UNIQUE (report_date_iso, sales_rep_id)
    );
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS sales_line_entries (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        entry_id INTEGER,
        report_date TEXT,
        report_date_iso TEXT,
        sales_rep_id INTEGER,
        sales_rep_name TEXT,
        customer_id TEXT,
        product_code TEXT,
        product_name TEXT,
        brand_name TEXT,
        value_sold_kes REAL,
        entry_time_raw TEXT
    );
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS daily_orders (
        report_date TEXT,
        report_date_iso TEXT,
        sales_rep_id INTEGER,
        customer_id TEXT,
        order_id TEXT,
        order_value_kes REAL,
        lines_count INTEGER,
        PRIMARY KEY (report_date_iso, sales_rep_id, customer_id)
    );
    """)

    conn.commit()
    conn.close()


# =========================================================
# KPI CALCULATION
# =========================================================
def calculate_kpis(activity, orders):
    customers_ordered = orders["customer_id"].nunique()
    total_sales = orders["order_value_kes"].sum()
    total_lines = orders["lines_count"].sum()

    base = activity["customers_on_pjp"].sum() + activity["new_customers_mapped"].sum()

    return {
        "Customers on PJP": activity["customers_on_pjp"].sum(),
        "Actual Visits": activity["actual_visits"].sum(),
        "New Customers": activity["new_customers_mapped"].sum(),
        "Orders Collected": customers_ordered,
        "Productivity %": (customers_ordered / base * 100) if base else 0,
        "Productivity vs 50%": ((customers_ordered / base) / 0.5 * 100) if base else 0,
        "Total Sales (KES)": total_sales,
        "Avg Basket Value": total_sales / customers_ordered if customers_ordered else 0,
        "LPPC Actual": total_lines / customers_ordered if customers_ordered else 0,
    }


# =========================================================
# STREAMLIT UI
# =========================================================
st.set_page_config(page_title="GT Daily Performance", layout="wide")
st.title("📊 General Trade Daily Performance Dashboard")

# KPI card styling
st.markdown("""
<style>
div[data-testid="stMetric"] {
    background-color: #ffffff;
    border: 1px solid #e6e6e6;
    padding: 14px;
    border-radius: 8px;
    box-shadow: 0 2px 6px rgba(0,0,0,0.05);
}
</style>
""", unsafe_allow_html=True)

init_db()

# =========================================================
# LOAD DATA
# =========================================================
conn = get_connection()
activity = pd.read_sql("SELECT * FROM rep_daily_activity", conn)
orders = pd.read_sql("SELECT * FROM daily_orders", conn)
conn.close()

if activity.empty:
    st.info("No data available.")
    st.stop()

activity["report_date_iso"] = pd.to_datetime(activity["report_date_iso"])
orders["report_date_iso"] = pd.to_datetime(orders["report_date_iso"])

# =========================================================
# FILTERS
# =========================================================
f1, f2, f3 = st.columns(3)

with f1:
    regions = ["All"] + sorted(activity["region_name"].dropna().unique())
    selected_region = st.selectbox("Region", regions)

rep_pool = activity if selected_region == "All" else activity[activity["region_name"] == selected_region]

with f2:
    reps = ["All"] + sorted(rep_pool["sales_rep_name"].dropna().unique())
    selected_rep = st.selectbox("Sales Rep", reps)

with f3:
    date_sel = st.date_input(
        "Day / Date Range",
        value=None,
        min_value=activity["report_date_iso"].min(),
        max_value=activity["report_date_iso"].max(),
    )

af = activity.copy()
of = orders.copy()

if selected_region != "All":
    af = af[af["region_name"] == selected_region]
    of = of[of["sales_rep_id"].isin(af["sales_rep_id"])]

if selected_rep != "All":
    af = af[af["sales_rep_name"] == selected_rep]
    of = of[of["sales_rep_id"].isin(af["sales_rep_id"])]

if date_sel:
    if isinstance(date_sel, tuple):
        start, end = pd.to_datetime(date_sel[0]), pd.to_datetime(date_sel[1])
    else:
        start = end = pd.to_datetime(date_sel)

    af = af[(af["report_date_iso"] >= start) & (af["report_date_iso"] <= end)]
    of = of[(of["report_date_iso"] >= start) & (of["report_date_iso"] <= end)]

# =========================================================
# KPI + LEADERBOARD LAYOUT
# =========================================================
left, right = st.columns([3, 1])

# ---------------- KPI CARDS ----------------
with left:
    kpis = calculate_kpis(af, of)
    kpi_items = list(kpis.items())

    for i in range(0, len(kpi_items), 3):
        cols = st.columns(3)
        for col, (k, v) in zip(cols, kpi_items[i:i+3]):
            col.metric(k, f"{v:,.2f}")

# ---------------- LPPC LEADERBOARD ----------------
with right:
    st.subheader("🏆 LPPC Leaderboard")

    lppc_df = (
        of.groupby("sales_rep_id", as_index=False)
        .agg(
            orders=("customer_id", "nunique"),
            lines=("lines_count", "sum")
        )
    )

    rep_names = af[["sales_rep_id", "sales_rep_name", "region_name"]].drop_duplicates()

    lppc_df = lppc_df.merge(rep_names, on="sales_rep_id", how="left")

    lppc_df["LPPC"] = lppc_df["lines"] / lppc_df["orders"]
    lppc_df = lppc_df.replace([float("inf"), -float("inf")], 0).fillna(0)

    leaderboard = (
        lppc_df
        .sort_values("LPPC", ascending=False)
        [["sales_rep_name", "region_name", "LPPC"]]
        .rename(columns={
            "sales_rep_name": "Sales Rep",
            "region_name": "Region"
        })
    )

    st.dataframe(leaderboard, use_container_width=True)

# =========================================================
# REP SALES TABLE (ORDERED BY SALES)
# =========================================================
st.subheader("Sales Performance by Sales Rep")

sales_table = (
    of.groupby("sales_rep_id", as_index=False)
    .agg(
        total_sales_kes=("order_value_kes", "sum"),
        orders=("customer_id", "nunique")
    )
    .merge(
        af[["sales_rep_id", "sales_rep_name", "region_name"]].drop_duplicates(),
        on="sales_rep_id",
        how="left"
    )
    .sort_values("total_sales_kes", ascending=False)
)

sales_table = sales_table.rename(columns={
    "sales_rep_name": "Sales Rep",
    "region_name": "Region",
    "total_sales_kes": "Total Sales (KES)",
    "orders": "Orders Collected"
})

st.dataframe(sales_table, use_container_width=True)
