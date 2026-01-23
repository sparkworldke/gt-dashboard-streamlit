import streamlit as st
import pandas as pd
import sqlite3
from datetime import datetime
from io import BytesIO

# =========================================================
# CONFIG
# =========================================================
DB_PATH = "gt_sfa.db"
LPPC_TARGET = 4
ABV_TARGET = 2000
PRODUCTIVITY_TARGET = 50

# =========================================================
# HELPERS
# =========================================================
def get_connection():
    return sqlite3.connect(DB_PATH, check_same_thread=False)


def to_iso(date_str):
    return datetime.strptime(date_str, "%d/%m/%Y").strftime("%Y-%m-%d")


def normalize_columns(df):
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.replace("\n", " ", regex=False)
    )
    return df


def extract_single_date(series, label):
    dates = pd.to_datetime(series, errors="coerce").dt.date.dropna().unique()
    if len(dates) != 1:
        st.error(f"{label} must contain exactly ONE date")
        st.stop()
    return dates[0].strftime("%d/%m/%Y")


def date_exists(report_date_iso):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT 1 FROM rep_daily_activity WHERE report_date_iso = ? LIMIT 1",
        (report_date_iso,)
    )
    exists = cur.fetchone() is not None
    conn.close()
    return exists


# =========================================================
# DATABASE INIT
# =========================================================
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
        customer_name TEXT,
        region_name TEXT,
        product_code TEXT,
        product_sku TEXT,
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
# KPI CALCULATION (FINAL)
# =========================================================
def calculate_kpis(activity, orders):
    customers_ordered = orders["customer_id"].nunique()
    total_sales = orders["order_value_kes"].sum()
    total_lines = orders["lines_count"].sum()

    base = (
        activity["customers_on_pjp"].sum()
        + activity["new_customers_mapped"].sum()
    )

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
# APP SETUP
# =========================================================
st.set_page_config(page_title="GT Performance", layout="wide")
st.title("📊 General Trade Performance Dashboard")

st.markdown("""
<style>
div[data-testid="stMetric"] {
    background-color:#ffffff;
    border:1px solid #e6e6e6;
    padding:15px;
    border-radius:8px;
    box-shadow:0 2px 4px rgba(0,0,0,0.05);
}
</style>
""", unsafe_allow_html=True)

init_db()

# =========================================================
# LOAD DATA
# =========================================================
conn = get_connection()
activity_all = pd.read_sql("SELECT * FROM rep_daily_activity", conn)
orders_all = pd.read_sql("SELECT * FROM daily_orders", conn)
conn.close()

if activity_all.empty:
    st.warning("No data available yet. Please upload files.")
    activity_all = pd.DataFrame(columns=[
        "report_date_iso", "region_name", "sales_rep_name",
        "customers_on_pjp", "actual_visits", "new_customers_mapped"
    ])
    orders_all = pd.DataFrame(columns=[
        "report_date_iso", "sales_rep_id", "customer_id",
        "order_value_kes", "lines_count"
    ])

activity_all["report_date_iso"] = pd.to_datetime(activity_all["report_date_iso"])
orders_all["report_date_iso"] = pd.to_datetime(orders_all["report_date_iso"])

# =========================================================
# GLOBAL FILTERS
# =========================================================
f1, f2, f3 = st.columns(3)

with f1:
    region = st.selectbox(
        "Region",
        ["All"] + sorted(activity_all["region_name"].dropna().unique())
    )

af = activity_all if region == "All" else activity_all[activity_all["region_name"] == region]

with f2:
    rep = st.selectbox(
        "Sales Rep",
        ["All"] + sorted(af["sales_rep_name"].dropna().unique())
    )

with f3:
    if not activity_all.empty:
        date_sel = st.date_input(
            "Day / Date Range",
            value=(activity_all["report_date_iso"].min(),
                   activity_all["report_date_iso"].max())
        )
    else:
        date_sel = st.date_input("Day / Date Range")

def apply_filters(df):
    out = df.copy()

    if region != "All" and "region_name" in out.columns:
        out = out[out["region_name"] == region]

    if rep != "All" and "sales_rep_name" in out.columns:
        out = out[out["sales_rep_name"] == rep]

    if isinstance(date_sel, tuple):
        start, end = pd.to_datetime(date_sel[0]), pd.to_datetime(date_sel[1])
    else:
        start = end = pd.to_datetime(date_sel)

    if "report_date_iso" in out.columns:
        out = out[
            (out["report_date_iso"] >= start)
            & (out["report_date_iso"] <= end)
        ]

    return out

af = apply_filters(activity_all)
of = apply_filters(orders_all)

# =========================================================
# TABS
# =========================================================
tab_dashboard, tab_lppc, tab_insights, tab_upload = st.tabs(
    ["📊 Dashboard", "📉 LPPC", "🧠 Insights", "📥 Upload"]
)

# =========================================================
# DASHBOARD
# =========================================================
with tab_dashboard:
    st.subheader("KPI Summary")

    kpis = calculate_kpis(af, of)
    items = list(kpis.items())

    for i in range(0, len(items), 3):
        cols = st.columns(3)
        for col, (k, v) in zip(cols, items[i:i+3]):
            if isinstance(v, float):
                col.metric(k, f"{v:,.2f}")
            else:
                col.metric(k, f"{v:,}")

# =========================================================
# LPPC TAB
# =========================================================
with tab_lppc:
    st.subheader("LPPC Overview")

    if of.empty:
        st.info("No orders in selected scope.")
    else:
        lppc_df = (
            of.groupby("sales_rep_id", as_index=False)
            .agg(
                Orders=("customer_id", "nunique"),
                Lines=("lines_count", "sum"),
                Sales=("order_value_kes", "sum")
            )
        )

        lppc_df["LPPC"] = lppc_df["Lines"] / lppc_df["Orders"]
        st.dataframe(lppc_df, use_container_width=True)

# =========================================================
# INSIGHTS TAB
# =========================================================
with tab_insights:
    st.subheader("Insights")

    if of.empty:
        st.info("No insights available.")
    else:
        avg_lppc = (of["lines_count"].sum() / of["customer_id"].nunique()) if of["customer_id"].nunique() else 0
        avg_abv = (of["order_value_kes"].sum() / of["customer_id"].nunique()) if of["customer_id"].nunique() else 0

        if avg_lppc < LPPC_TARGET:
            st.write("• LPPC below target — increase SKU depth per visit.")
        if avg_abv < ABV_TARGET:
            st.write("• ABV below target — push higher value bundles.")
        if avg_lppc >= LPPC_TARGET and avg_abv >= ABV_TARGET:
            st.success("Execution healthy across LPPC and ABV.")

# =========================================================
# UPLOAD TAB
# =========================================================
with tab_upload:
    st.subheader("Upload")

    file1 = st.file_uploader("File 1 – Rep Activity", type=["xlsx", "csv"])
    file2 = st.file_uploader("File 2 – Sales Lines", type=["xlsx", "csv"])

    if file1 and file2:
        df1 = normalize_columns(pd.read_excel(file1) if file1.name.endswith("xlsx") else pd.read_csv(file1))
        df2 = normalize_columns(pd.read_excel(file2) if file2.name.endswith("xlsx") else pd.read_csv(file2))

        df1 = df1.rename(columns={"DATE": "report_date"})
        df2 = df2.rename(columns={"ENTRY_TIME": "entry_time_raw"})

        report_date_1 = extract_single_date(df1["report_date"], "File 1")
        report_date_2 = extract_single_date(df2["entry_time_raw"], "File 2")

        if report_date_1 != report_date_2:
            st.error("Dates do not match.")
            st.stop()

        iso = to_iso(report_date_1)
        if date_exists(iso):
            st.error(f"Data for {report_date_1} already exists.")
            st.stop()

        st.success(f"Detected report date: {report_date_1}")
        st.info("Upload mapping and insert logic can be plugged here as already implemented earlier.")
