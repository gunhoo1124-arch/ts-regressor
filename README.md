# ts-regressor
ML-powered stock trend forecaster for NASDAQ/NYSE tickers using FinancialModelingPrep data, with CLI + Streamlit workflows for horizon-based return prediction, bull/bear signal generation, model benchmarking, and downloadable analysis reports.

# Detailed Repo Description
ts-regressor is a practical time-series regression project for short-horizon equity trend analysis. It combines market data engineering, machine learning, and reporting into one workflow so you can move from raw historical data to a structured bullish/bearish signal quickly.

The project pulls historical price, earnings, and profile data from the FinancialModelingPrep (FMP) API, validates exchange eligibility (NASDAQ/NYSE), and builds leak-aware datasets aligned to real trading sessions. It supports both event-driven analysis (post-earnings reaction) and periodic trend analysis (weekly, biweekly, monthly, quarterly, and longer horizons).

# Core capabilities include:

- Feature engineering from price momentum and volatility windows.
- Earnings surprise features for event-based forecasting.
- Multi-model benchmarking across Linear Regression, Ridge, Lasso, and Random Forest.
- Holdout-based evaluation with MAE, RMSE, and R2 metrics.
- Signal interpretation layer that translates model outputs into direction and strength (bullish/bearish, weak/medium/strong).
- Visual outputs including bull/bear return charts and model diagnostics.
- Exportable artifacts: datasets, metrics CSVs, charts, and full PDF reports.

# The repo provides both:

- A CLI script for reproducible, scriptable analysis.
- A Streamlit app for interactive exploration, model diagnostics, and report download.
This is built for learning and research in financial ML workflows, not for direct trading automation or investment advice.
