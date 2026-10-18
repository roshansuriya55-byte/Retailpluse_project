import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="RetailPulse | Retail Intelligence", page_icon="🛍️", layout="wide")
DATA_PATH = Path(__file__).parent / "data" / "online_retail_II.xlsx"

@st.cache_data(show_spinner="Loading Online Retail II workbook...")
def load_data(uploaded_file=None):
    source = uploaded_file if uploaded_file is not None else DATA_PATH
    frames = []
    for sheet in ["Year 2009-2010", "Year 2010-2011"]:
        try:
            part = pd.read_excel(source, sheet_name=sheet)
            frames.append(part)
        except Exception:
            continue
    if not frames:
        part = pd.read_excel(source)
        frames.append(part)
    df = pd.concat(frames, ignore_index=True)
    df.columns = [str(c).strip() for c in df.columns]
    return df

def clean_data(raw):
    df = raw.copy()
    required = ["Invoice", "StockCode", "Description", "Quantity", "InvoiceDate", "Price", "Customer ID", "Country"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError("Required columns missing: " + ", ".join(missing))
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    for c in ["Quantity", "Price", "Customer ID"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["Invoice"] = df["Invoice"].astype("string").str.strip()
    df["StockCode"] = df["StockCode"].astype("string").str.strip()
    df["Description"] = df["Description"].astype("string").str.strip()
    df["Country"] = df["Country"].astype("string").str.strip()
    df = df.dropna(subset=["Invoice", "StockCode", "InvoiceDate", "Quantity", "Price", "Country"])
    # Credit notes/cancellations have invoice numbers beginning with C; negative quantities and prices are excluded from product-sales KPIs.
    df["IsCancellation"] = df["Invoice"].str.upper().str.startswith("C", na=False) | (df["Quantity"] < 0)
    df["Revenue"] = df["Quantity"] * df["Price"]
    sales = df[(~df["IsCancellation"]) & (df["Quantity"] > 0) & (df["Price"] > 0)].copy()
    sales["Customer ID"] = sales["Customer ID"].astype("Int64")
    sales["InvoiceMonth"] = sales["InvoiceDate"].dt.to_period("M").astype(str)
    sales["Year"] = sales["InvoiceDate"].dt.year
    sales["Month"] = sales["InvoiceDate"].dt.month
    sales["DayOfWeek"] = sales["InvoiceDate"].dt.day_name()
    sales["Hour"] = sales["InvoiceDate"].dt.hour
    sales["Revenue"] = sales["Quantity"] * sales["Price"]
    return df, sales

@st.cache_data(show_spinner=False)
def load_and_clean_default():
    raw = load_data()
    return clean_data(raw)

def make_rfm(sales):
    cust = sales.dropna(subset=["Customer ID"]).copy()
    if cust.empty:
        return pd.DataFrame()
    snapshot = cust["InvoiceDate"].max() + pd.Timedelta(days=1)
    rfm = cust.groupby("Customer ID").agg(
        Recency=("InvoiceDate", lambda s: (snapshot - s.max()).days),
        Frequency=("Invoice", "nunique"),
        Monetary=("Revenue", "sum"),
        Quantity=("Quantity", "sum"),
        LastPurchase=("InvoiceDate", "max")
    ).reset_index()
    # Rank customers into relative RFM bands; handle small datasets safely.
    for col, ascending in [("Recency", False), ("Frequency", True), ("Monetary", True)]:
        rfm[col + "_Score"] = pd.qcut(rfm[col].rank(method="first"), 4, labels=[1,2,3,4]).astype(int)
    rfm["RFM_Score"] = rfm["Recency_Score"] + rfm["Frequency_Score"] + rfm["Monetary_Score"]
    rfm["Segment"] = np.select(
        [rfm["RFM_Score"] >= 10, (rfm["Frequency_Score"] >= 3) & (rfm["Monetary_Score"] >= 3),
         rfm["Recency_Score"] <= 2],
        ["Champions", "Loyal / High Value", "At Risk"],
        default="Regular"
    )
    rfm["Avg_Order_Value"] = rfm["Monetary"] / rfm["Frequency"].replace(0, np.nan)
    return rfm

def monthly_series(sales, metric="Revenue"):
    if sales.empty:
        return pd.DataFrame(columns=["Month", metric])
    m = sales.set_index("InvoiceDate").resample("MS")[metric].sum().reset_index()
    m.columns = ["Date", metric]
    return m

def make_forecast(sales, metric="Revenue", horizon=3):
    m = monthly_series(sales, metric)
    if len(m) < 4:
        return m, None, "At least 4 months of data are needed for a forecast."
    m = m.copy()
    m["t"] = np.arange(len(m))
    m["month_num"] = m["Date"].dt.month
    m["year"] = m["Date"].dt.year
    m["lag1"] = m[metric].shift(1)
    m["lag2"] = m[metric].shift(2)
    train = m.dropna().copy()
    if len(train) < 3:
        return m, None, "Not enough history after lag features."
    features = ["t", "month_num", "year", "lag1", "lag2"]
    model = RandomForestRegressor(n_estimators=150, random_state=42, min_samples_leaf=2)
    # Hold out the final 20% chronologically for a basic validation score.
    split = max(2, int(len(train) * 0.8))
    tr, te = train.iloc[:split], train.iloc[split:]
    if len(te):
        model.fit(tr[features], tr[metric])
        pred = model.predict(te[features])
        mae = mean_absolute_error(te[metric], pred)
        rmse = np.sqrt(mean_squared_error(te[metric], pred))
        validation = {"MAE": mae, "RMSE": rmse, "Actual": te[metric].to_numpy(), "Predicted": pred}
    else:
        validation = None
    model.fit(train[features], train[metric])
    history = m.copy()
    future_rows = []
    for i in range(horizon):
        date = history["Date"].max() + pd.offsets.MonthBegin(1)
        t = len(history)
        lag1 = float(history.iloc[-1][metric])
        lag2 = float(history.iloc[-2][metric]) if len(history) >= 2 else lag1
        row = pd.DataFrame([{"t":t, "month_num":date.month, "year":date.year, "lag1":lag1, "lag2":lag2}])
        value = max(0, float(model.predict(row[features])[0]))
        future_rows.append({"Date":date, metric:value, "Type":"Forecast"})
        history = pd.concat([history, pd.DataFrame([{"Date":date, metric:value, "t":t, "month_num":date.month, "year":date.year, "lag1":lag1, "lag2":lag2}])], ignore_index=True)
    historical = m[["Date", metric]].copy()
    historical["Type"] = "Historical"
    return pd.concat([historical, pd.DataFrame(future_rows)], ignore_index=True), validation, None

st.title("🛍️ RetailPulse")
st.caption("Retail analytics • Customer segmentation • Demand forecasting • Inventory planning")
with st.sidebar:
    st.header("Navigation")
    page = st.radio("Choose a view", ["Overview", "1 · Sales Dashboard", "2 · Customer Dashboard", "3 · Forecast Dashboard", "4 · Inventory Dashboard", "12-Step Workflow"])
    st.divider()
    st.caption("Data source: UCI Online Retail II Excel workbook")
    uploaded = st.file_uploader("Optional: upload another Excel workbook", type=["xlsx"])

try:
    if uploaded is not None:
        raw = load_data(uploaded)
        audit, sales = clean_data(raw)
    else:
        audit, sales = load_and_clean_default()
except Exception as e:
    st.error(f"Could not load or clean the workbook: {e}")
    st.info("Check that data/online_retail_II.xlsx exists and contains the two Online Retail II sheets.")
    st.stop()

if sales.empty:
    st.error("No valid sales rows remain after cleaning.")
    st.stop()

rfm = make_rfm(sales)
min_date, max_date = sales["InvoiceDate"].min().date(), sales["InvoiceDate"].max().date()
with st.sidebar:
    date_range = st.date_input("Invoice date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
    countries = sorted(sales["Country"].dropna().unique().tolist())
    selected_countries = st.multiselect("Country filter", countries, default=[])
filtered = sales.copy()
if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
    filtered = filtered[(filtered["InvoiceDate"].dt.date >= date_range[0]) & (filtered["InvoiceDate"].dt.date <= date_range[1])]
if selected_countries:
    filtered = filtered[filtered["Country"].isin(selected_countries)]
if filtered.empty:
    st.warning("No rows match these filters. Change the date range or country selection.")
    st.stop()

revenue = filtered["Revenue"].sum()
orders = filtered["Invoice"].nunique()
customers = filtered["Customer ID"].nunique()
items = filtered["Quantity"].sum()

def kpis():
    a,b,c,d = st.columns(4)
    a.metric("Net Sales Revenue", f"£{revenue:,.0f}")
    b.metric("Invoices", f"{orders:,}")
    c.metric("Unique Customers", f"{customers:,}")
    d.metric("Units Sold", f"{items:,.0f}")

if page == "Overview":
    st.subheader("RetailPulse overview")
    kpis()
    left, right = st.columns(2)
    with left:
        monthly = monthly_series(filtered)
        st.plotly_chart(px.line(monthly, x="Date", y="Revenue", markers=True, title="Monthly revenue trend"), width="stretch")
    with right:
        top = filtered.groupby("Description", dropna=False)["Revenue"].sum().nlargest(10).reset_index()
        st.plotly_chart(px.bar(top, x="Revenue", y="Description", orientation="h", title="Top 10 products by revenue"), width="stretch")
    st.markdown("### Available modules")
    st.write("Sales analysis, RFM customer segmentation, baseline time-series forecasting, inventory recommendations and a 12-step project workflow.")
    st.caption("The workbook is historical transaction data. It does not contain real stock-on-hand or an explicit churn label; inventory and churn-risk outputs are estimates/proxies.")

elif page == "1 · Sales Dashboard":
    st.subheader("1 · Sales Dashboard")
    kpis()
    c1,c2 = st.columns(2)
    monthly = monthly_series(filtered)
    with c1:
        st.plotly_chart(px.line(monthly, x="Date", y="Revenue", markers=True, title="Monthly revenue"), width="stretch")
    with c2:
        country_rev = filtered.groupby("Country")["Revenue"].sum().nlargest(12).sort_values().reset_index()
        st.plotly_chart(px.bar(country_rev, x="Revenue", y="Country", orientation="h", title="Top countries by revenue"), width="stretch")
    c3,c4 = st.columns(2)
    with c3:
        product_rev = filtered.groupby("Description")["Revenue"].sum().nlargest(10).sort_values().reset_index()
        st.plotly_chart(px.bar(product_rev, x="Revenue", y="Description", orientation="h", title="Top products"), width="stretch")
    with c4:
        day = filtered.groupby("DayOfWeek")["Revenue"].sum().reindex(["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]).dropna().reset_index()
        st.plotly_chart(px.bar(day, x="DayOfWeek", y="Revenue", title="Revenue by day of week"), width="stretch")
    st.markdown("#### Sales data preview")
    st.dataframe(filtered[["Invoice","InvoiceDate","Customer ID","Description","Quantity","Price","Revenue","Country"]].head(100), width="stretch")

elif page == "2 · Customer Dashboard":
    st.subheader("2 · Customer Dashboard")
    cust_sales = filtered.dropna(subset=["Customer ID"])
    rfm_filtered = make_rfm(cust_sales)
    if rfm_filtered.empty:
        st.warning("Customer IDs are not available for this selection.")
    else:
        a,b,c,d = st.columns(4)
        a.metric("Customers", f"{len(rfm_filtered):,}")
        b.metric("Average spend/customer", f"£{rfm_filtered['Monetary'].mean():,.2f}")
        c.metric("Champions", f"{(rfm_filtered['Segment']=='Champions').sum():,}")
        d.metric("At-risk proxy", f"{(rfm_filtered['Segment']=='At Risk').sum():,}")
        left,right = st.columns(2)
        with left:
            seg = rfm_filtered.groupby("Segment").size().reset_index(name="Customers")
            st.plotly_chart(px.pie(seg, names="Segment", values="Customers", title="RFM customer segments"), width="stretch")
        with right:
            st.plotly_chart(px.scatter(rfm_filtered, x="Frequency", y="Monetary", color="Segment", size="Avg_Order_Value", hover_data=["Customer ID","Recency"], title="Customer value map"), width="stretch")
        st.markdown("#### Customer RFM table")
        st.dataframe(rfm_filtered.sort_values("Monetary", ascending=False), width="stretch")
        st.caption("Segments are derived from recency, purchase frequency and monetary value. 'At Risk' is an RFM heuristic, not a verified churn label.")

elif page == "3 · Forecast Dashboard":
    st.subheader("3 · Forecast Dashboard")
    metric = st.selectbox("Forecast measure", ["Revenue", "Quantity"], index=0)
    horizon = st.slider("Months to forecast", 1, 6, 3)
    forecast_df, validation, error = make_forecast(filtered, metric, horizon)
    if error:
        st.warning(error)
    else:
        hist = forecast_df[forecast_df["Type"]=="Historical"]
        fut = forecast_df[forecast_df["Type"]=="Forecast"]
        a,b,c = st.columns(3)
        a.metric("History months", f"{len(hist):,}")
        b.metric("Forecast horizon", f"{len(fut)} months")
        c.metric(f"Forecast total ({metric})", f"{fut[metric].sum():,.0f}")
        st.plotly_chart(px.line(forecast_df, x="Date", y=metric, color="Type", markers=True, title=f"Historical and {horizon}-month {metric.lower()} forecast"), width="stretch")
        if validation:
            st.markdown("#### Chronological holdout validation")
            a,b = st.columns(2)
            a.metric("MAE", f"{validation['MAE']:,.2f}")
            b.metric("RMSE", f"{validation['RMSE']:,.2f}")
            st.caption("Baseline model: Random Forest with calendar and lag features. These scores are a simple time-ordered holdout, not a guarantee of future accuracy.")
        st.dataframe(forecast_df, width="stretch")

elif page == "4 · Inventory Dashboard":
    st.subheader("4 · Inventory Dashboard")
    st.info("The source workbook has sales transactions, not stock-on-hand. This page estimates replenishment quantities from recent product demand; it is not actual inventory tracking.")
    lead_days = st.slider("Supplier lead time (days)", 1, 60, 14)
    cover_days = st.slider("Target stock coverage (days)", 7, 90, 30)
    inv = filtered.groupby(["StockCode","Description"], dropna=False).agg(
        Units_Sold=("Quantity","sum"),
        Revenue=("Revenue","sum"),
        Active_Days=("InvoiceDate", lambda s: max(1, s.dt.normalize().nunique())),
        Last_Sale=("InvoiceDate","max")
    ).reset_index()
    inv["Avg_Daily_Demand"] = inv["Units_Sold"] / inv["Active_Days"]
    inv["Estimated_Lead_Time_Demand"] = inv["Avg_Daily_Demand"] * lead_days
    inv["Target_Stock_Coverage"] = inv["Avg_Daily_Demand"] * cover_days
    inv["Suggested_Reorder_Qty"] = np.ceil(inv["Target_Stock_Coverage"]).astype(int)
    inv["Priority"] = np.select([inv["Avg_Daily_Demand"] >= inv["Avg_Daily_Demand"].quantile(.75),
                                 inv["Avg_Daily_Demand"] >= inv["Avg_Daily_Demand"].quantile(.4)],
                                ["High", "Medium"], default="Low")
    a,b,c = st.columns(3)
    a.metric("Products in selected data", f"{len(inv):,}")
    b.metric("High-priority products", f"{(inv['Priority']=='High').sum():,}")
    c.metric("Suggested total units", f"{inv['Suggested_Reorder_Qty'].sum():,.0f}")
    left,right = st.columns(2)
    with left:
        top = inv.nlargest(15,"Suggested_Reorder_Qty").sort_values("Suggested_Reorder_Qty")
        st.plotly_chart(px.bar(top, x="Suggested_Reorder_Qty", y="Description", orientation="h", title="Top suggested replenishment quantities"), width="stretch")
    with right:
        pr = inv.groupby("Priority").size().reset_index(name="Products")
        st.plotly_chart(px.pie(pr, names="Priority", values="Products", title="Replenishment priority (estimated)"), width="stretch")
    st.dataframe(inv.sort_values(["Priority","Suggested_Reorder_Qty"], ascending=[True,False]), width="stretch")
    st.download_button("Download inventory recommendations CSV", inv.to_csv(index=False).encode("utf-8"), "retailpulse_inventory_recommendations.csv", "text/csv")

elif page == "12-Step Workflow":
    st.subheader("RetailPulse — complete 12-step workflow")
    steps = [
        ("1. Collect the dataset", "Use the uploaded UCI Online Retail II Excel workbook with the 2009–2010 and 2010–2011 sheets."),
        ("2. Import the libraries", "Pandas and NumPy for data work; Plotly and Streamlit for visualization; Scikit-learn for baseline ML."),
        ("3. Load the dataset", "Read both Excel sheets and combine them into one transaction table."),
        ("4. Data Cleaning", "Parse dates, handle missing required fields, standardize text, identify cancellations/negative quantities and filter valid sales for sales KPIs."),
        ("5. Feature Engineering", "Create Revenue = Quantity × Price, year, month, weekday and hour features."),
        ("6. EDA", "Explore sales trends, countries, products, order volume and customer purchasing patterns."),
        ("7. Customer Segmentation (RFM Analysis)", "Calculate Recency, Frequency and Monetary values and assign relative customer segments."),
        ("8. Demand Forecasting (Time Series)", "Aggregate sales by month and forecast revenue or quantity with a baseline Random Forest using time and lag features."),
        ("9. Churn Prediction", "This dataset has no explicit churn label. The app shows an RFM-based at-risk proxy; a proper supervised churn model needs a defined future inactivity label and time-based split."),
        ("10. Inventory Optimization", "Estimate average daily product demand and suggested stock coverage/replenishment. Actual on-hand inventory and supplier orders are not in this workbook."),
        ("11. Build Streamlit/Power BI Dashboard", "This app provides Sales, Customer, Forecast and Inventory dashboard pages."),
        ("12. Deploy using Flask/FastAPI/Streamlit", "Run locally with Streamlit; deploy to Streamlit Community Cloud or another host after testing and configuring the workbook path.")
    ]
    for title, detail in steps:
        with st.expander(title, expanded=False):
            st.write(detail)
    st.markdown("#### Current dataset limitations")
    st.warning("The workbook is transaction history only. It does not provide verified churn labels, actual inventory counts, supplier lead times or stockout events. The churn and inventory pages are explicitly labelled as estimates/proxies.")
    st.download_button("Download cleaned sales data CSV", filtered.to_csv(index=False).encode("utf-8"), "retailpulse_cleaned_sales.csv", "text/csv")

st.divider()
st.caption("RetailPulse • Educational project. Validate business assumptions and model performance before using recommendations operationally.")
