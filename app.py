import streamlit as st
import pandas as pd
import sqlite3
from datetime import datetime

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
def normalize_columns(df):
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.replace("\n", " ", regex=False)
    )
    return df


def to_iso(date_str):
    return datetime.strptime(date_str, "%d/%m/%Y").strftime("%Y-%m-%d")


def extract_single_date(series, label):
    dates = pd.to_datetime(series, errors="coerce").dt.date.dropna().unique()
    if len(dates) != 1:
        st.error(f"{label} must contain exactly ONE date")
        st.stop()
    return dates[0].strftime("%d/%m/%Y")


def detect_value_column(df):
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
        customer_code TEXT,
        customer_name TEXT,
        region_name TEXT,
        product_code TEXT,
        product_id TEXT,
        product_name TEXT,
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
# INSERT LOGIC
# =========================================================
def insert_file1(df, report_date):
    df["report_date"] = report_date
    df["report_date_iso"] = to_iso(report_date)

    df = df[
        [
            "report_date",
            "report_date_iso",
            "sales_rep_id",
            "sales_rep_name",
            "region_name",
            "sales_target_value_kes",
            "customers_on_pjp",
            "actual_visits",
            "new_customers_mapped",
            "time_spent_per_outlet_seconds",
        ]
    ]

    conn = get_connection()
    df.to_sql("rep_daily_activity", conn, if_exists="append", index=False)
    conn.close()


def insert_file2(df, report_date):
    df["report_date"] = report_date
    df["report_date_iso"] = to_iso(report_date)

    df = df[
        [
            "entry_id",
            "report_date",
            "report_date_iso",
            "sales_rep_id",
            "sales_rep_name",
            "customer_id",
            "customer_code",
            "customer_name",
            "region_name",
            "product_code",
            "product_id",
            "product_name",
            "product_sku",
            "brand_name",
            "value_sold_kes",
            "entry_time_raw",
        ]
    ]

    conn = get_connection()
    df.to_sql("sales_line_entries", conn, if_exists="append", index=False)
    conn.close()


def build_daily_orders(report_date):
    iso = to_iso(report_date)
    conn = get_connection()

    conn.execute(f"""
        INSERT OR IGNORE INTO daily_orders
        SELECT
            '{report_date}',
            '{iso}',
            sales_rep_id,
            customer_id,
            sales_rep_id || '_' || customer_id || '_' || '{iso}',
            SUM(value_sold_kes),
            COUNT(DISTINCT product_code)
        FROM sales_line_entries
        WHERE report_date_iso = '{iso}'
          AND customer_id IS NOT NULL
        GROUP BY sales_rep_id, customer_id
    """)

    conn.commit()
    conn.close()


# =========================================================
# APP SETUP
# =========================================================
st.set_page_config(page_title="GT Performance", layout="wide")
st.title("📊 General Trade Performance Dashboard")

init_db()

# =========================================================
# LOAD DATA
# =========================================================
conn = get_connection()
activity_all = pd.read_sql("SELECT * FROM rep_daily_activity", conn)
orders_all = pd.read_sql("SELECT * FROM daily_orders", conn)
lines_all = pd.read_sql("SELECT * FROM sales_line_entries", conn)
conn.close()

if activity_all.empty:
    st.info("No data available yet.")
    st.stop()

activity_all["report_date_iso"] = pd.to_datetime(activity_all["report_date_iso"])
orders_all["report_date_iso"] = pd.to_datetime(orders_all["report_date_iso"])
lines_all["report_date_iso"] = pd.to_datetime(lines_all["report_date_iso"])

# =========================================================
# FILTERS
# =========================================================
f1, f2, f3 = st.columns(3)

with f1:
    regions = ["All"] + sorted(activity_all["region_name"].dropna().unique())
    selected_region = st.selectbox("Region", regions)

af = activity_all if selected_region == "All" else activity_all[activity_all["region_name"] == selected_region]

with f2:
    reps = ["All"] + sorted(af["sales_rep_name"].dropna().unique())
    selected_rep = st.selectbox("Sales Rep", reps)

with f3:
    date_sel = st.date_input(
        "Date / Date Range",
        value=None,
        min_value=activity_all["report_date_iso"].min(),
        max_value=activity_all["report_date_iso"].max()
    )


def apply_filters(df):
    out = df.copy()

    if selected_region != "All" and "region_name" in out.columns:
        out = out[out["region_name"] == selected_region]

    if selected_rep != "All" and "sales_rep_name" in out.columns:
        out = out[out["sales_rep_name"] == selected_rep]

    if date_sel:
        if isinstance(date_sel, tuple):
            start, end = pd.to_datetime(date_sel[0]), pd.to_datetime(date_sel[1])
        else:
            start = end = pd.to_datetime(date_sel)

        out = out[(out["report_date_iso"] >= start) & (out["report_date_iso"] <= end)]

    return out


af = apply_filters(activity_all)
of = apply_filters(orders_all)
lf = apply_filters(lines_all)

# =========================================================
# TABS
# =========================================================
tab_dashboard, tab_lppc, tab_insights, tab_upload = st.tabs(
    ["📊 Dashboard", "📉 LPPC Trends", "🧠 Insights", "📥 Upload"]
)

# =========================================================
# DASHBOARD TAB (RESTORED LEADERBOARDS)
# =========================================================
with tab_dashboard:
    left, right = st.columns([3, 1])

    # ---------- KPI CARDS ----------
    kpis = {
        "Customers on PJP": int(af["customers_on_pjp"].sum()),
        "Orders Collected": of["customer_id"].nunique(),
        "Productivity %": round(
            (of["customer_id"].nunique() /
             (af["customers_on_pjp"].sum() + af["new_customers_mapped"].sum())) * 100
            if (af["customers_on_pjp"].sum() + af["new_customers_mapped"].sum()) else 0, 1
        ),
        "Total Sales (KES)": of["order_value_kes"].sum(),
        "Avg Basket Value (KES)": (
            of["order_value_kes"].sum() / of["customer_id"].nunique()
            if of["customer_id"].nunique() else 0
        ),
        "LPPC": (
            of["lines_count"].sum() / of["customer_id"].nunique()
            if of["customer_id"].nunique() else 0
        )
    }

    for i in range(0, len(kpis), 3):
        cols = st.columns(3)
        for col, (k, v) in zip(cols, list(kpis.items())[i:i+3]):
            col.metric(k, f"{v:,.2f}" if isinstance(v, float) else v)

    # ---------- LPPC LEADERBOARD (RIGHT) ----------
    with right:
        st.subheader("🏆 LPPC Leaderboard")

        if of.empty:
            st.write("No orders in selected scope.")
        else:
            lppc_df = (
                of.groupby("sales_rep_id", as_index=False)
                .agg(
                    orders=("customer_id", "nunique"),
                    lines=("lines_count", "sum"),
                    sales=("order_value_kes", "sum"),
                )
            )

            rep_dim = af[["sales_rep_id", "sales_rep_name", "region_name"]].drop_duplicates()
            lppc_df = lppc_df.merge(rep_dim, on="sales_rep_id", how="left")

            lppc_df["LPPC"] = (
                lppc_df["lines"] / lppc_df["orders"]
            ).replace([float("inf"), -float("inf")], 0).fillna(0)

            leaderboard = (
                lppc_df.sort_values("LPPC", ascending=False)
                [["region_name", "sales_rep_name", "orders", "LPPC"]]
                .rename(columns={
                    "region_name": "Region",
                    "sales_rep_name": "Sales Rep",
                    "orders": "Orders"
                })
            )

            def style_lppc(s):
                if s.name == "LPPC":
                    return [
                        "background-color:#fee2e2;color:#991b1b;font-weight:700" if v < LPPC_TARGET
                        else "background-color:#dcfce7;color:#065f46;font-weight:700"
                        for v in s
                    ]
                return [""] * len(s)

            st.dataframe(
                leaderboard.style.apply(style_lppc),
                use_container_width=True,
                height=450
            )

            st.caption("LPPC by Region (summary)")
            region_lppc = (
                lppc_df.groupby("region_name", as_index=False)
                .agg(lines=("lines", "sum"), orders=("orders", "sum"))
            )
            region_lppc["LPPC"] = (
                region_lppc["lines"] / region_lppc["orders"]
            ).replace([float("inf"), -float("inf")], 0).fillna(0)

            st.dataframe(
                region_lppc[["region_name", "LPPC"]]
                .rename(columns={"region_name": "Region"})
                .sort_values("LPPC", ascending=False),
                use_container_width=True
            )

    # ---------- SALES LEADERBOARD ----------
    st.subheader("Sales Leaderboard (Highest to Lowest)")

    if of.empty:
        st.write("No sales in selected scope.")
    else:
        sales_table = (
            of.groupby("sales_rep_id", as_index=False)
            .agg(
                total_sales_kes=("order_value_kes", "sum"),
                orders=("customer_id", "nunique"),
                lines=("lines_count", "sum"),
            )
            .merge(
                af[["sales_rep_id", "sales_rep_name", "region_name"]].drop_duplicates(),
                on="sales_rep_id",
                how="left"
            )
        )

        sales_table["ABV"] = (
            sales_table["total_sales_kes"] / sales_table["orders"]
        ).replace([float("inf"), -float("inf")], 0).fillna(0)

        sales_table["LPPC"] = (
            sales_table["lines"] / sales_table["orders"]
        ).replace([float("inf"), -float("inf")], 0).fillna(0)

        sales_table = sales_table.sort_values(
            "total_sales_kes", ascending=False
        ).rename(columns={
            "sales_rep_name": "Sales Rep",
            "region_name": "Region",
            "total_sales_kes": "Total Sales (KES)",
            "orders": "Orders Collected",
        })

        display_cols = ["Region", "Sales Rep", "Total Sales (KES)", "Orders Collected", "ABV", "LPPC"]

        def style_abv_lppc(row):
            styles = [""] * len(row)

            abv_idx = display_cols.index("ABV")
            styles[abv_idx] = (
                "background-color:#fee2e2;color:#991b1b;font-weight:700"
                if row["ABV"] < ABV_TARGET
                else "background-color:#dcfce7;color:#065f46;font-weight:700"
            )

            lppc_idx = display_cols.index("LPPC")
            styles[lppc_idx] = (
                "background-color:#fee2e2;color:#991b1b;font-weight:700"
                if row["LPPC"] < LPPC_TARGET
                else "background-color:#dcfce7;color:#065f46;font-weight:700"
            )

            return styles

        st.dataframe(
            sales_table[display_cols]
            .style.apply(style_abv_lppc, axis=1)
            .format({
                "Total Sales (KES)": "{:,.2f}",
                "ABV": "{:,.2f}",
                "LPPC": "{:,.2f}",
            }),
            use_container_width=True
        )

# =========================================================
# LPPC TRENDS TAB (SKU NAME FROM PRODUCT_SKU)
# =========================================================
with tab_lppc:
    st.subheader("📉 LPPC Deep Dive")

    rep_opts = ["All"] + sorted(af["sales_rep_name"].dropna().unique())
    rep_sel = st.selectbox("Rep Drill-down", rep_opts)

    sku_base = lf.copy()
    sku_base["sku_display"] = sku_base["product_sku"]

    sku_grp = (
        sku_base.groupby(
            ["product_code", "sku_display", "brand_name"],
            as_index=False
        )
        .agg(
            Lines=("product_code", "count"),
            Customers=("customer_id", "nunique")
        )
    )

    sku_grp["LPPC"] = sku_grp["Lines"] / sku_grp["Customers"]

    sku_grp = sku_grp.sort_values(
        ["LPPC", "Customers"],
        ascending=[True, False]
    ).head(20)

    st.dataframe(
        sku_grp.rename(columns={
            "product_code": "Product Code",
            "sku_display": "Product (SKU)",
            "brand_name": "Brand"
        })
        .style.applymap(
            lambda v: "color:red;font-weight:700" if v < LPPC_TARGET else "color:green;font-weight:700",
            subset=["LPPC"]
        )
        .format({"LPPC": "{:.2f}"}),
        use_container_width=True
    )

# =========================================================
# INSIGHTS TAB
# =========================================================
with tab_insights:
    st.subheader("🧠 Performance Insights")

    insights = []

    if kpis["Productivity %"] < PRODUCTIVITY_TARGET:
        insights.append("🔴 Productivity below 50%. Improve conversion on PJP calls.")

    if kpis["LPPC"] < LPPC_TARGET:
        insights.append("🔴 LPPC below target. Push range depth and bundles.")

    if kpis["Avg Basket Value (KES)"] < ABV_TARGET:
        insights.append("🟠 ABV below KES 2,000. Focus on upselling.")

    if insights:
        for i in insights:
            st.write(i)
    else:
        st.success("🟢 All KPIs are within expected thresholds.")

# =========================================================
# UPLOAD TAB
# =========================================================
with tab_upload:
    st.subheader("📥 Upload Daily Files")

    file1 = st.file_uploader("File 1 – Sales by Rep", type=["xlsx", "csv"])
    file2 = st.file_uploader("File 2 – Sales Lines", type=["xlsx", "csv"])

    if file1 and file2:
        df1 = normalize_columns(pd.read_excel(file1) if file1.name.endswith("xlsx") else pd.read_csv(file1))
        df2 = normalize_columns(pd.read_excel(file2) if file2.name.endswith("xlsx") else pd.read_csv(file2))

        df1 = df1.rename(columns={"DATE": "report_date"})
        df2 = df2.rename(columns={"ENTRY_TIME": "entry_time_raw"})

        report_date_1 = extract_single_date(df1["report_date"], "File 1")
        report_date_2 = extract_single_date(df2["entry_time_raw"], "File 2")

        if report_date_1 != report_date_2:
            st.error("File dates do not match")
            st.stop()

        iso = to_iso(report_date_1)
        if date_exists(iso):
            st.error(f"🚫 Data for {report_date_1} already exists.")
            st.stop()

        if st.button("✅ Save to Database"):
            df1 = df1.rename(columns={
                "ID": "sales_rep_id",
                "NAME": "sales_rep_name",
                "REGION": "region_name",
                "CUSTOMERS IN ROUTE": "customers_on_pjp",
                "ACTUAL VISITS": "actual_visits",
                "MAPPED OUTLETS": "new_customers_mapped",
                "TIME SPENT (PER OUTLET)": "time_spent_per_outlet_seconds",
                "SALES TARGET": "sales_target_value_kes",
            })

            value_col = detect_value_column(df2)

            df2 = df2.rename(columns={
                "ENTRY_ID": "entry_id",
                "SALES_REP_ID": "sales_rep_id",
                "SALES_REP": "sales_rep_name",
                "CUSTOMER_ID": "customer_id",
                "CUSTOMER_CODE": "customer_code",
                "CUSTOMER_NAME": "customer_name",
                "CUSTOMER_REGION": "region_name",
                "PRODUCT_CODE": "product_code",
                "PRODUCT_ID": "product_id",
                "PRODUCT_NAME": "product_name",
                "PRODUCT_SKU": "product_sku",
                "BRAND_NAME": "brand_name",
                value_col: "value_sold_kes",
            })

            insert_file1(df1, report_date_1)
            insert_file2(df2, report_date_1)
            build_daily_orders(report_date_1)

            st.success("✅ Data uploaded successfully")
            st.experimental_rerun()
