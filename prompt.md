You are a senior data engineer and analytics product designer building a production-ready Streamlit dashboard for a General Trade (GT) FMCG sales team.

CONTEXT
-------
This is a field sales performance system for FMCG GT operations.
Data comes daily from two Excel/CSV files and must be stored in SQLite.
Each day’s data is standalone and must never roll into other days unless explicitly filtered.

The app must be stable, incremental, and non-destructive.

--------------------------------
DATA SOURCES
--------------------------------

FILE 1 – Rep Daily Activity (one row per sales rep per day)
Primary key: (report_date_iso, sales_rep_id)

Columns include:
- ID → sales_rep_id
- NAME → sales_rep_name
- DATE → report_date (DD/MM/YYYY)
- REGION → region_name
- SALES TARGET
- CUSTOMERS IN ROUTE (PJP)
- ACTUAL VISITS
- UNIQUE VISITS
- SUCCESSFUL VISITS
- MAPPED OUTLETS (new customers)
- TIME SPENT (PER OUTLET)
- FIRST CHECKIN / LAST CHECKIN / LAST CHECKOUT

FILE 2 – Sales Lines (many rows per rep per day)
Foreign key: sales_rep_id
Multiple rows per customer per day must be treated as ONE order.

Columns include:
- ENTRY_ID
- SALES_REP_ID
- SALES_REP
- CUSTOMER_ID
- CUSTOMER_NAME
- CUSTOMER_REGION / REGION_NAME
- PRODUCT_CODE
- PRODUCT_ID
- PRODUCT_SKU (THIS is the product display name)
- BRAND_NAME
- VALUE_SOLD (or equivalent revenue column)
- ENTRY_TIME (date + time)

--------------------------------
DATABASE RULES
--------------------------------

Use SQLite.

Tables:
1) rep_daily_activity
2) sales_line_entries
3) daily_orders (derived)

Rules:
- Convert all dates to:
  - report_date (DD/MM/YYYY)
  - report_date_iso (YYYY-MM-DD)
- One day = standalone snapshot
- Block upload if the same report_date already exists
- daily_orders must aggregate:
  - 1 order per (sales_rep_id, customer_id, report_date)

--------------------------------
GLOBAL FILTERS (TOP OF APP)
--------------------------------

These filters apply to ALL tabs:
1) Region (dynamic)
2) Sales Rep (depends on Region)
3) Day OR Date Range

If no filter is selected:
- Show cumulative business view
- Respect selected date or date range

--------------------------------
TABS (MUST EXIST AND REMAIN STABLE)
--------------------------------

TABS ORDER:
1) Dashboard
2) LPPC
3) Insights
4) Upload

--------------------------------
DASHBOARD TAB
--------------------------------

Use KPI cards (3 per row, white background, subtle shadow).

KPIs MUST INCLUDE EXACTLY:

- Customers on PJP
- Actual Visits
- New Customers
- Orders Collected
- Productivity %
- Productivity vs 50%
- Total Sales (KES)
- Avg Basket Value (KES)
- LPPC Actual

Definitions:
- Productivity % = Orders / (PJP + New Customers)
- Productivity vs 50% = (Productivity / 50%) * 100
- ABV = Total Sales / Orders
- LPPC = Total Lines / Orders

Below KPI cards:
- LPPC Leaderboard by Sales Rep (conditional formatting: <4 red, ≥4 green)
- LPPC Summary by Region
- Sales Leaderboard (Total Sales desc, with ABV & LPPC conditional formatting)

--------------------------------
LPPC TAB
--------------------------------

Purpose: Diagnose and fix LPPC.

Must include:
- Top 20 SKUs hurting LPPC
- Stable scope (respect filters)
- Display SKU using:
  - PRODUCT_CODE
  - PRODUCT_SKU (name)
  - BRAND_NAME

Metrics:
- Lines sold
- Customers reached
- Lines per customer

Additional sections:
- SKU bundling recommendations
- “What to push tomorrow” SKU list (top LPPC drivers)
- LPPC repair simulation (+1 / +2 lines per call)
- Region LPPC heatmap (Region × Date)

Export:
- Export LPPC repair plan to Excel
- NO PDF dependencies (no reportlab)

--------------------------------
INSIGHTS TAB
--------------------------------

Auto-generate coaching notes per sales rep:

Rules:
- If LPPC < target → recommend adding SKUs / bundling
- If ABV < target → recommend upselling
- If both healthy → reinforce good execution

Notes must respect all filters.

--------------------------------
UPLOAD TAB
--------------------------------

Features:
- Upload File 1 + File 2 together
- Validate both contain exactly ONE date
- Dates must match
- Block upload if date already exists
- Normalize column names
- Detect revenue column dynamically
- Insert into SQLite
- Build daily_orders
- Use st.rerun() after successful upload
- NEVER use deprecated Streamlit APIs

--------------------------------
TECHNICAL CONSTRAINTS
--------------------------------

- Streamlit (latest version)
- Python
- SQLite
- pandas
- NO reportlab
- Use st.rerun(), NOT st.experimental_rerun()
- Use .style.format(), NOT DataFrame.format()
- All code must be copy-paste runnable
- Do NOT remove existing functionality unless explicitly instructed

--------------------------------
DELIVERY EXPECTATION
--------------------------------

Always respond with:
- Full working app.py
- No placeholders
- No “unchanged” comments
- No breaking changes
- Clear, production-ready logic

This app is for FMCG GT execution review at:
- Business level
- Region level
- Sales rep level
- Day or date range level
