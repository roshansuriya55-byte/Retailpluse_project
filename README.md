# RetailPulse Streamlit Project

## Features
- Sales Dashboard
- Customer Dashboard (RFM segmentation)
- Forecast Dashboard (monthly revenue/quantity baseline forecast)
- Inventory Dashboard (sales-based replenishment estimates)
- Complete 12-step workflow section matching the project plan

## Dataset
`data/online_retail_II.xlsx` is the uploaded Online Retail II workbook. The app reads both `Year 2009-2010` and `Year 2010-2011` sheets.

## Run locally (Windows)
```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Important data limitations
- The workbook is transaction history, not an inventory-management dataset.
- It has no verified churn label. Customer "At Risk" is an RFM heuristic, not a supervised churn prediction.
- Inventory outputs are estimated reorder quantities based on historical sales and a user-selected coverage period; actual stock-on-hand and supplier lead time are not provided.
- Forecasting is a baseline Random Forest model with a chronological holdout; validate before interpreting as reliable business forecasts.
- Cancellations/negative quantities are excluded from sales KPIs but remain in the audit data during cleaning.

## Project workflow
1. Collect dataset
2. Import libraries
3. Load dataset
4. Data cleaning
5. Feature engineering
6. EDA
7. Customer segmentation (RFM)
8. Demand forecasting (time series)
9. Churn prediction (RFM risk proxy; verified labels required for a true classifier)
10. Inventory optimization (sales-based estimate)
11. Streamlit/Power BI dashboard
12. Deployment using Flask/FastAPI/Streamlit
