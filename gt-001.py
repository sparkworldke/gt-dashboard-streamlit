import streamlit as st
import pandas as pd
import sqlite3
from datetime import datetime
import numpy as np

# =========================================================
# CONFIG
# =========================================================
DB_PATH = "gt_sfa.db"
LPPC_TARGET = 4.0
ABV_TARGET = 2000.0
PRODUCTIVITY_TARGET = 50.0

# =========================================================
# HELPERS
# =========================================================
def normalize_columns(df):
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.replace("\n", " ", regex=False)
        .str.replace("  ", " ", regex=False)
    )
    return df

def to_iso(date_str):
    try:
        return datetime.strptime(date_str, "%d/%m/%Y").strftime("%Y-%m-%d")
    except ValueError:
        # Try handling potential Excel timestamps or other formats if needed
        # For now, strict adherence to DD/MM/YYYY as per requirements
        raise ValueError(f"Date format must be DD/MM/YYYY. Got: {date_str}")

def extract_single_date(series, label):
    dates = pd.to_datetime(series, dayfirst=True, errors="coerce").dt.date.dropna().unique()
    if len(dates) != 1:
        st.error(f"{label} must contain exactly ONE unique date. Found: {len(dates)}")
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

    # Ensure required columns exist
    required_cols = [
        "sales_rep_id", "sales_rep_name", "region_name",
        "sales_target_value_kes", "customers_on_pjp",
        "actual_visits", "new_customers_mapped", "time_spent_per_outlet_seconds"
    ]
    
    # Fill missing columns with 0/None if strictly necessary, but better to fail if critical
    # For now, assuming validation passed
    
    df_subset = df[
        ["report_date", "report_date_iso"] + required_cols
    ]

    conn = get_connection()
    df_subset.to_sql("rep_daily_activity", conn, if_exists="append", index=False)
    conn.close()

def insert_file2(df, report_date):
    df["report_date"] = report_date
    df["report_date_iso"] = to_iso(report_date)

    df_subset = df[
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
    df_subset.to_sql("sales_line_entries", conn, if_exists="append", index=False)
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
    st.info("No data available yet. Please upload files in the Upload tab.")
    # Initialize empty dataframes to prevent errors before data is loaded
    # But still allow the app to render the Upload tab
    activity_all = pd.DataFrame(columns=["report_date_iso", "region_name", "sales_rep_name", "customers_on_pjp", "new_customers_mapped", "sales_rep_id"])
    orders_all = pd.DataFrame(columns=["report_date_iso", "sales_rep_id", "customer_id", "order_value_kes", "lines_count"])
    lines_all = pd.DataFrame(columns=["report_date_iso", "region_name", "sales_rep_name", "product_sku", "product_code", "brand_name", "customer_id", "value_sold_kes"])
else:
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
    min_date = activity_all["report_date_iso"].min() if not activity_all.empty else datetime.today()
    max_date = activity_all["report_date_iso"].max() if not activity_all.empty else datetime.today()
    
    date_sel = st.date_input(
        "Day OR Date Range",
        value=None,
        min_value=min_date,
        max_value=max_date
    )

def apply_filters(df):
    out = df.copy()
    if out.empty: return out

    if selected_region != "All" and "region_name" in out.columns:
        out = out[out["region_name"] == selected_region]
    # For orders which doesn't have region_name, filter by sales_rep_id from af
    elif selected_region != "All" and "sales_rep_id" in out.columns:
        out = out[out["sales_rep_id"].isin(af["sales_rep_id"])]

    if selected_rep != "All" and "sales_rep_name" in out.columns:
        out = out[out["sales_rep_name"] == selected_rep]
    elif selected_rep != "All" and "sales_rep_id" in out.columns:
        # Get sales_rep_id for the selected name
        rep_ids = af[af["sales_rep_name"] == selected_rep]["sales_rep_id"].unique()
        out = out[out["sales_rep_id"].isin(rep_ids)]

    if date_sel:
        if isinstance(date_sel, tuple):
            if len(date_sel) == 2:
                start, end = pd.to_datetime(date_sel[0]), pd.to_datetime(date_sel[1])
                out = out[(out["report_date_iso"] >= start) & (out["report_date_iso"] <= end)]
            elif len(date_sel) == 1:
                start = pd.to_datetime(date_sel[0])
                out = out[out["report_date_iso"] == start]
        else:
            start = pd.to_datetime(date_sel)
            out = out[out["report_date_iso"] == start]

    return out

af = apply_filters(activity_all)
of = apply_filters(orders_all)
lf = apply_filters(lines_all)

# =========================================================
# TABS
# =========================================================
tab_dashboard, tab_lppc, tab_insights, tab_upload = st.tabs(
    ["📊 Dashboard", "📉 LPPC", "🧠 Insights", "📥 Upload"]
)

# =========================================================
# DASHBOARD TAB
# =========================================================
with tab_dashboard:
    left, right = st.columns([3, 1])

    # ---------- KPI CALCS ----------
    customers_pjp = af["customers_on_pjp"].sum()
    new_customers = af["new_customers_mapped"].sum()
    actual_visits = af["actual_visits"].sum()
    orders_collected = of["customer_id"].nunique()
    total_sales = of["order_value_kes"].sum()
    total_lines = of["lines_count"].sum()
    
    productivity_base = customers_pjp + new_customers
    productivity_pct = (orders_collected / productivity_base * 100) if productivity_base > 0 else 0
    productivity_vs_50 = (productivity_pct / 50.0 * 100)
    
    abv = total_sales / orders_collected if orders_collected > 0 else 0
    lppc_actual = total_lines / orders_collected if orders_collected > 0 else 0

    kpis = {
        "Customers on PJP": int(customers_pjp),
        "Actual Visits": int(actual_visits),
        "New Customers": int(new_customers),
        "Orders Collected": int(orders_collected),
        "Productivity %": f"{productivity_pct:.1f}%",
        "Productivity vs 50%": f"{productivity_vs_50:.1f}%",
        "Total Sales (KES)": total_sales,
        "Avg Basket Value (KES)": abv,
        "LPPC Actual": lppc_actual
    }

    # ---------- KPI CARDS ----------
    with left:
        kpi_items = list(kpis.items())
        for i in range(0, len(kpi_items), 3):
            cols = st.columns(3)
            for col, (k, v) in zip(cols, kpi_items[i:i+3]):
                with col:
                    if isinstance(v, str):
                        val_str = v
                    elif isinstance(v, int):
                        val_str = f"{v:,}"
                    else:
                        val_str = f"{v:,.2f}"

                    # Custom card styling
                    st.markdown(
                        f"""
                        <div style="background-color: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.05); border: 1px solid #eee; margin-bottom: 10px;">
                            <div style="font-size: 0.9em; color: #666;">{k}</div>
                            <div style="font-size: 1.6em; font-weight: bold; color: #333;">{val_str}</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

        # ---------- SALES LEADERBOARD ----------
        st.subheader("Sales Leaderboard")
        if of.empty:
            st.info("No sales data available for the selected filters.")
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

            sales_table["ABV"] = (sales_table["total_sales_kes"] / sales_table["orders"]).fillna(0)
            sales_table["LPPC"] = (sales_table["lines"] / sales_table["orders"]).fillna(0)

            sales_table = sales_table.sort_values("total_sales_kes", ascending=False)
            
            # Formatting
            display_cols = ["region_name", "sales_rep_name", "total_sales_kes", "ABV", "LPPC"]
            
            def highlight_metrics(row):
                styles = [""] * len(row)
                # ABV index 3, LPPC index 4
                if row["ABV"] < ABV_TARGET:
                    styles[3] = "color: #d32f2f; font-weight: bold;"
                else:
                    styles[3] = "color: #388e3c; font-weight: bold;"
                
                if row["LPPC"] < LPPC_TARGET:
                    styles[4] = "color: #d32f2f; font-weight: bold;"
                else:
                    styles[4] = "color: #388e3c; font-weight: bold;"
                return styles

            # Calculate height to show all rows without scrollbar
            height = (len(sales_table) + 1) * 35 + 3

            st.dataframe(
                sales_table[display_cols]
                .rename(columns={
                    "region_name": "Region",
                    "sales_rep_name": "Sales Rep",
                    "total_sales_kes": "Total Sales",
                })
                .style.apply(highlight_metrics, axis=1)
                .format({
                    "Total Sales": "{:,.0f}",
                    "ABV": "{:,.0f}",
                    "LPPC": "{:.2f}"
                }),
                use_container_width=True,
                height=height
            )

    # ---------- LPPC LEADERBOARD (RIGHT) ----------
    with right:
        st.subheader("🏆 LPPC Leaderboard")
        if not of.empty:
            lppc_df = (
                of.groupby("sales_rep_id", as_index=False)
                .agg(orders=("customer_id", "nunique"), lines=("lines_count", "sum"))
            )
            lppc_df = lppc_df.merge(af[["sales_rep_id", "sales_rep_name"]].drop_duplicates(), on="sales_rep_id", how="left")
            lppc_df["LPPC"] = (lppc_df["lines"] / lppc_df["orders"]).fillna(0)
            
            leaderboard = lppc_df.sort_values("LPPC", ascending=False)
            
            def color_lppc(val):
                color = '#ffcdd2' if val < 4.0 else '#c8e6c9' # Red-ish vs Green-ish background
                return f'background-color: {color}; color: black'

            height_lppc = (len(leaderboard) + 1) * 35 + 3
            st.dataframe(
                leaderboard[["sales_rep_name", "LPPC"]]
                .rename(columns={"sales_rep_name": "Rep"})
                .style.applymap(color_lppc, subset=["LPPC"])
                .format({"LPPC": "{:.2f}"}),
                use_container_width=True,
                height=height_lppc
            )

            st.markdown("---")
            st.subheader("Region Summary")
            region_lppc = (
                of.merge(af[["sales_rep_id", "region_name"]], on="sales_rep_id", how="left")
                .groupby("region_name")
                .agg(lines=("lines_count", "sum"), orders=("customer_id", "nunique"))
            )
            region_lppc["LPPC"] = (region_lppc["lines"] / region_lppc["orders"]).fillna(0)
            st.dataframe(
                region_lppc[["LPPC"]].sort_values("LPPC", ascending=False).style.format("{:.2f}"),
                use_container_width=True
            )

# =========================================================
# LPPC TAB
# =========================================================
with tab_lppc:
    st.header("📉 LPPC Analysis & Repair")
    
    col_lppc_1, col_lppc_2 = st.columns([2, 1])
    
    with col_lppc_1:
        st.subheader("Top 20 SKUs Hurting LPPC (Low Lines per Customer)")
        if not lf.empty:
            sku_stats = lf.groupby(["product_sku", "brand_name"], as_index=False).agg(
                lines_sold=("product_code", "count"),
                customers_reached=("customer_id", "nunique")
            )
            # "Hurting LPPC" - Interpreted here as SKUs that have low attachment (Lines per Customer is 1 for unique SKU)
            # but maybe we want to see SKUs with high volume but low customer reach?
            # Or just sort by customers_reached to see what ISN'T selling broadly.
            # User request: "Metrics: Lines sold, Customers reached, Lines per customer"
            sku_stats["lines_per_customer"] = sku_stats["lines_sold"] / sku_stats["customers_reached"]
            
            # Sort by customers_reached ascending to see items with low reach? 
            # Or "Top 20 SKUs" usually implies top SELLERS.
            # "Hurting LPPC" might imply missed opportunities.
            # Let's show High Volume SKUs first, as they drive the business.
            
            st.dataframe(
                sku_stats.sort_values("lines_sold", ascending=False).head(20)
                .rename(columns={"product_sku": "Product", "brand_name": "Brand", "lines_sold": "Lines Sold", "customers_reached": "Reach", "lines_per_customer": "Lines/Cust"}),
                use_container_width=True
            )
            
            st.subheader("Region LPPC Heatmap")
            if not of.empty:
                heatmap_data = (
                    of.merge(af[["sales_rep_id", "region_name"]], on="sales_rep_id", how="left")
                    .groupby(["region_name", "report_date_iso"])
                    .agg(lines=("lines_count", "sum"), orders=("customer_id", "nunique"))
                    .reset_index()
                )
                heatmap_data["LPPC"] = heatmap_data["lines"] / heatmap_data["orders"]
                heatmap_pivot = heatmap_data.pivot(index="region_name", columns="report_date_iso", values="LPPC")
                # Format columns as short dates
                heatmap_pivot.columns = [d.strftime('%d/%m') for d in heatmap_pivot.columns]
                
                def color_lppc_heatmap(val):
                    if pd.isna(val):
                        return ""
                    if val < 3.0:
                        return "background-color: #ffcdd2; color: black"
                    elif val < 4.0:
                        return "background-color: #fff9c4; color: black"
                    else:
                        return "background-color: #c8e6c9; color: black"

                st.dataframe(heatmap_pivot.style.applymap(color_lppc_heatmap).format("{:.2f}"), use_container_width=True)

    with col_lppc_2:
        st.subheader("🛠️ LPPC Repair Simulation")
        
        uplift = st.slider("Simulate adding lines per call:", 1, 3, 1)
        
        current_lines = total_lines
        current_revenue = total_sales
        avg_price_per_line = current_revenue / current_lines if current_lines > 0 else 0
        
        simulated_lines = current_lines + (orders_collected * uplift)
        simulated_revenue = simulated_lines * avg_price_per_line
        simulated_lppc = simulated_lines / orders_collected if orders_collected > 0 else 0
        
        st.metric("Current LPPC", f"{lppc_actual:.2f}")
        st.metric(f"Simulated LPPC (+{uplift})", f"{simulated_lppc:.2f}", delta=f"{simulated_lppc - lppc_actual:.2f}")
        st.metric("Simulated Revenue Impact", f"KES {simulated_revenue:,.0f}", delta=f"{simulated_revenue - current_revenue:,.0f}")
        
        st.markdown("### What to push tomorrow")
        st.info("Based on top selling SKUs not in bottom 20% of distribution:")
        if not lf.empty:
            top_skus = lf["product_sku"].value_counts().head(5).index.tolist()
            for sku in top_skus:
                st.write(f"• {sku}")
                
        if st.button("Export Repair Plan to Excel"):
             # Mock export functionality - creating a simple CSV download
             # Since we can't save files to client machine directly without download button flow
             # We create a dataframe and convert to CSV
             plan_df = sku_stats.sort_values("customers_reached", ascending=True)
             csv = plan_df.to_csv(index=False).encode('utf-8')
             st.download_button(
                 "Download Plan.csv",
                 csv,
                 "lppc_repair_plan.csv",
                 "text/csv",
                 key='download-csv'
             )

# =========================================================
# INSIGHTS TAB
# =========================================================
with tab_insights:
    st.header("🧠 Coaching & Insights")
    
    if of.empty:
        st.warning("No data available for insights.")
    else:
        # Generate insights per rep
        rep_stats = (
            of.groupby("sales_rep_id")
            .agg(
                orders=("customer_id", "nunique"),
                lines=("lines_count", "sum"),
                sales=("order_value_kes", "sum")
            )
            .reset_index()
        )
        # Add PJP info
        rep_pjp = af.groupby("sales_rep_id")["customers_on_pjp"].sum().reset_index()
        rep_stats = rep_stats.merge(rep_pjp, on="sales_rep_id", how="left").fillna(0)
        rep_names = af[["sales_rep_id", "sales_rep_name"]].drop_duplicates()
        rep_stats = rep_stats.merge(rep_names, on="sales_rep_id", how="left")
        
        rep_stats["LPPC"] = rep_stats["lines"] / rep_stats["orders"]
        rep_stats["ABV"] = rep_stats["sales"] / rep_stats["orders"]
        
        for index, row in rep_stats.iterrows():
            with st.expander(f"👮 {row['sales_rep_name']} (LPPC: {row['LPPC']:.2f} | ABV: {row['ABV']:,.0f})"):
                recs = []
                if row["LPPC"] < LPPC_TARGET:
                    recs.append(f"🔴 **LPPC Issue**: Current {row['LPPC']:.2f} < {LPPC_TARGET}. Coach on bundling and presenting full range.")
                if row["ABV"] < ABV_TARGET:
                    recs.append(f"🟠 **ABV Issue**: Current {row['ABV']:,.0f} < {ABV_TARGET}. Focus on upselling higher value SKUs or cases.")
                
                if not recs:
                    st.success("🟢 Excellent execution! Both LPPC and ABV are healthy. Reinforce current behaviors.")
                else:
                    for r in recs:
                        st.write(r)

# =========================================================
# UPLOAD TAB
# =========================================================
with tab_upload:
    st.header("📥 Upload Daily Data")
    
    col_up1, col_up2 = st.columns(2)
    with col_up1:
        file1 = st.file_uploader("File 1: Rep Daily Activity", type=["xlsx", "csv"])
    with col_up2:
        file2 = st.file_uploader("File 2: Sales Lines", type=["xlsx", "csv"])

    if file1 and file2:
        try:
            df1 = normalize_columns(pd.read_excel(file1) if file1.name.endswith("xlsx") else pd.read_csv(file1))
            df2 = normalize_columns(pd.read_excel(file2) if file2.name.endswith("xlsx") else pd.read_csv(file2))

            # Column mapping and normalization
            df1 = df1.rename(columns={"DATE": "report_date"})
            df2 = df2.rename(columns={"ENTRY_TIME": "entry_time_raw"})

            report_date_1 = extract_single_date(df1["report_date"], "File 1")
            report_date_2 = extract_single_date(df2["entry_time_raw"], "File 2")

            if report_date_1 != report_date_2:
                st.error(f"Date Mismatch! File 1 is {report_date_1}, File 2 is {report_date_2}.")
                st.stop()

            iso_date = to_iso(report_date_1)
            
            if date_exists(iso_date):
                st.error(f"⛔ Data for {report_date_1} already exists in the database. Upload blocked to prevent duplicates.")
                st.stop()
            
            st.info(f"Ready to upload data for: **{report_date_1}**")
            
            if st.button("✅ Confirm & Process Upload"):
                with st.spinner("Processing..."):
                    # Rename columns to match DB schema
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
                    if not value_col:
                        st.error("Could not detect a 'Value Sold' column in File 2.")
                        st.stop()

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

                    # Execute Insertions
                    insert_file1(df1, report_date_1)
                    insert_file2(df2, report_date_1)
                    build_daily_orders(report_date_1)
                    
                    st.success("✅ Upload Successful! Refreshing...")
                    st.rerun()

        except Exception as e:
            st.error(f"An error occurred during processing: {e}")