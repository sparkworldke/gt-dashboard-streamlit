import streamlit as st
import pandas as pd
import sqlite3
from datetime import datetime, date

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="FSR Retail Dashboard",
    layout="wide"
)

# -------------------------------------------------
# DB CONNECTION
# -------------------------------------------------
@st.cache_resource
def get_connection():
    return sqlite3.connect("fsr.db", check_same_thread=False)

conn = get_connection()
cursor = conn.cursor()

# -------------------------------------------------
# INIT DB
# -------------------------------------------------
def init_db():
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS fsr_daily (
        sales_rep_date_id TEXT,
        sales_rep_id TEXT,
        rep_name TEXT,
        region TEXT,
        report_date DATE,
        sales_target REAL,
        sales REAL,
        customers_in_route INTEGER,
        target_visit INTEGER,
        actual_visits INTEGER,
        unique_visits INTEGER,
        successful_visits INTEGER,
        unique_successful_visits INTEGER,
        mapped_outlets INTEGER,
        source_file TEXT,
        uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS fsr_sales_data (
        sales_rep_date_id TEXT,
        sales_rep_id TEXT,
        entry_id TEXT,
        customer_id TEXT,
        customer_name TEXT,
        product_code TEXT,
        product_id TEXT,
        brand_name TEXT,
        value_sold REAL,
        entry_time TIMESTAMP,
        source_file TEXT,
        uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)
    conn.commit()

init_db()

# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Dashboard", "Brand LPPC", "SKU LPPC Impact", "Data Explorer"]
)
st.sidebar.markdown("---")

# -------------------------------------------------
# DASHBOARD
# -------------------------------------------------
if page == "Dashboard":
    st.title("FSR Retail Dashboard")

    # ---------------- FILTERS ----------------
    regions = ["All"] + pd.read_sql(
        "SELECT DISTINCT region FROM fsr_daily", conn
    )["region"].dropna().tolist()

    sel_region = st.selectbox("Region", regions)

    reps_df = pd.read_sql("""
        SELECT DISTINCT sales_rep_id, rep_name, region
        FROM fsr_daily
    """, conn)

    if sel_region != "All":
        reps_df = reps_df[reps_df["region"] == sel_region]

    reps = ["All"] + sorted(reps_df["rep_name"].dropna().tolist())
    sel_rep = st.selectbox("Sales Rep", reps)

    min_d, max_d = pd.read_sql(
        "SELECT MIN(report_date), MAX(report_date) FROM fsr_daily", conn
    ).iloc[0]

    d_selected = st.date_input(
        "Report Date",
        value=pd.to_datetime(max_d).date()
    )

    # ---------------- SUMMARY FILTER ----------------
    where = f"DATE(report_date) = '{d_selected}'"
    rep_id = None

    if sel_region != "All":
        where += f" AND region = '{sel_region}'"

    if sel_rep != "All":
        rep_id = reps_df[
            reps_df["rep_name"] == sel_rep
        ]["sales_rep_id"].iloc[0]
        where += f" AND sales_rep_id = '{rep_id}'"

    df_summary = pd.read_sql(
        f"SELECT * FROM fsr_daily WHERE {where}", conn
    )

    if df_summary.empty:
        st.warning("No data for selected filters.")
        st.stop()

    # ---------------- DIRECT METRICS (FROM fsr_daily) ----------------
    tot = df_summary.agg({
        "customers_in_route": "sum",
        "actual_visits": "sum",
        "mapped_outlets": "sum",
        "unique_successful_visits": "sum",
        "target_visit": "sum",
        "sales": "sum",
        "sales_target": "sum"
    }).fillna(0)

    orders_collected = tot["unique_successful_visits"]

    # ---------------- RAW AGG (FROM fsr_sales_data USING entry_time) ----------------
    where_raw = f"DATE(entry_time) = '{d_selected}'"
    if rep_id:
        where_raw += f" AND sales_rep_id = '{rep_id}'"

    raw = pd.read_sql(f"""
        SELECT
            COUNT(DISTINCT product_code) AS unique_lines,
            COALESCE(SUM(value_sold),0) AS sales_value
        FROM fsr_sales_data
        WHERE {where_raw}
    """, conn).iloc[0]

    avg_basket = raw["sales_value"] / orders_collected if orders_collected > 0 else 0
    lppc_actual = raw["unique_lines"] / orders_collected if orders_collected > 0 else 0

    LPPC_TARGET = 4.0
    lppc_perf = (lppc_actual / LPPC_TARGET) * 100

    # ---------------- DISPLAY ----------------
    st.markdown("### Universe & Visits")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Customers on PJP", int(tot["customers_in_route"]))
    c2.metric("Actual Visits", int(tot["actual_visits"]))
    c3.metric("New Customers (Mapped)", int(tot["mapped_outlets"]))
    c4.metric("Orders Collected", int(orders_collected))

    st.markdown("### Productivity & Sales")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Productivity %",
        f"{100 * orders_collected / max(tot['target_visit'] + tot['mapped_outlets'],1):.1f}%"
    )
    c2.metric("Total Sales (KES)", f"{tot['sales']:,.0f}")
    c3.metric("Sales Target (KES)", f"{tot['sales_target']:,.0f}")
    c4.metric("Avg Basket Value (KES)", f"{avg_basket:,.0f}")

    st.markdown("### LPPC & Performance")
    c1, c2, c3 = st.columns(3)
    c1.metric("LPPC Target", f"{LPPC_TARGET:.2f}")
    c2.metric("LPPC Actual", f"{lppc_actual:.2f}")
    c3.metric("LPPC Perf %", f"{lppc_perf:.1f}%")

# -------------------------------------------------
# BRAND LPPC (USES entry_time)
# -------------------------------------------------
elif page == "Brand LPPC":
    st.title("Brand LPPC")

    brand_df = pd.read_sql("""
        SELECT
            brand_name,
            COUNT(DISTINCT product_code) AS unique_lines,
            COUNT(DISTINCT entry_id) AS orders
        FROM fsr_sales_data
        WHERE brand_name IS NOT NULL
        GROUP BY brand_name
    """, conn)

    brand_df["LPPC"] = brand_df["unique_lines"] / brand_df["orders"]
    brand_df = brand_df.sort_values("LPPC", ascending=False)

    st.dataframe(
        brand_df.style.format({"LPPC": "{:.2f}"}),
        use_container_width=True
    )

# -------------------------------------------------
# TOP 10 SKUs HURTING LPPC (USES entry_time)
# -------------------------------------------------
elif page == "SKU LPPC Impact":
    st.title("Top 10 SKUs Hurting LPPC")

    sku_df = pd.read_sql("""
        SELECT
            product_code,
            product_id,
            brand_name,
            COUNT(DISTINCT entry_id) AS orders,
            SUM(value_sold) AS sales_value
        FROM fsr_sales_data
        GROUP BY product_code, product_id, brand_name
        HAVING orders > 5
    """, conn)

    sku_df["Sales per Order"] = sku_df["sales_value"] / sku_df["orders"]
    sku_df = sku_df.sort_values("Sales per Order").head(10)

    st.caption(
        "SKUs with low sales contribution per order, pulling down LPPC and ABV."
    )

    st.dataframe(
        sku_df.style.format({
            "sales_value": "{:,.0f}",
            "Sales per Order": "{:,.0f}"
        }),
        use_container_width=True
    )

# -------------------------------------------------
# DATA EXPLORER
# -------------------------------------------------
else:
    st.subheader("Data Explorer")
    tables = pd.read_sql(
        "SELECT name FROM sqlite_master WHERE type='table'", conn
    )["name"].tolist()

    tbl = st.selectbox("Select table", tables)
    df = pd.read_sql(f"SELECT * FROM {tbl} LIMIT 1000", conn)
    st.dataframe(df, use_container_width=True)
