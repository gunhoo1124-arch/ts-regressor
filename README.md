# ts-regressor
GitHub: https://github.com/gunhoo1124-arch/ts-regressor

ML-powered stock trend forecaster for NASDAQ/NYSE tickers using FinancialModelingPrep (FMP) data.

`ts-regressor` is a technical, reproducible analysis stack for short-to-medium horizon return forecasting. It is designed as an end-to-end pipeline: ingest -> clean -> feature engineering -> deterministic train/test split -> multi-model evaluation -> signal construction -> visual/report artifacts.

## 1) System Architecture

The project has two entry points:

- CLI: `ts-regressor.py`
- Interactive dashboard: `streamlit_app.py`

Both entry points share the same modeling core by dynamically importing functions from `ts-regressor.py` in the Streamlit app.

- Core data API layer:
  - `fetch_earnings(symbol, api_key)`
  - `fetch_prices(symbol, api_key)`
  - `fetch_profile(symbol, api_key)`
  - exchange gate via `validate_supported_ticker`
- Dataset constructors:
  - `build_dataset(...)` for post-earnings events
  - `build_biweekly_dataset(..., period_days=...)` for periodic forecasting
- Modeling and scoring:
  - `make_model(...)`
  - `_evaluate_models(...)`
  - `forecast_from_models(...)`
  - `describe_signal(...)`
- Visual/report layer:
  - `save_bull_bear_plot(...)`
  - `build_full_report_pdf(...)` (Streamlit)

## 2) Data Sources and Eligibility

### 2.1 FMP endpoints
All data is retrieved from:
`https://financialmodelingprep.com/stable`

The project calls:

- `earnings`
- `historical-price-eod/full`
- `profile`

API key precedence is:

1. `--api-key`
2. `FMP_API_KEY` environment variable

### 2.2 Exchange validation
The analysis is restricted to symbols whose profile exchange resolves to NASDAQ or NYSE.

- `SUPPORTED_EXCHANGES = ("NASDAQ", "NYSE")`
- If unsupported, execution stops before modeling and prompts the user to pick a listed symbol.

## 3) Problem Formulation

Two modes are implemented:

1. **Earnings mode (`--mode earnings`)**
   - each event maps to the nearest aligned trading-day context
   - target is next trading-day open minus event-session prior close (percentage)

2. **Periodic mode (`--mode biweekly`, default)**
   - every `period_days` trading sessions produce one sample
   - target is `period_days` forward return from period end
   - horizon options in UI/CLI: 5, 10, 21, 63, 126, 252 trading days

## 4) Feature Construction and Labels

### 4.1 Earnings feature set

`EARNINGS_FEATURE_COLS = [
    eps_surprise_pct,
    rev_surprise_pct,
    is_bmo,
    pre_1d_return,
    pre_5d_return,
    pre_20d_vol
]`

For each earnings row:

- `target_return = (open_next - close_prior) / close_prior`
- `eps_surprise_pct = (eps_actual - eps_estimated) / |eps_estimated|`
- `rev_surprise_pct = (revenue_actual - revenue_estimated) / |revenue_estimated|`
- `is_bmo` is binary and captures before/after-market timing context
- `pre_1d_return`, `pre_5d_return` are prior-session momentum measures
- `pre_20d_vol` is trailing 20-trading-session close-return sample std

### 4.2 Periodic feature set

`PERIODIC_FEATURE_COLS` currently mirrors the biweekly/periodic feature list:

- pre-1d return
- pre-5d return
- pre-10d return
- pre-20d vol

Target logic for periodic dataset:

1. pick `cutoff_date` every `period_days`
2. compute forward target date `target_idx = cutoff_idx + period_days`
3. `target_return = (close_target - close_cutoff) / close_cutoff`

Rows without a valid aligned target remain in the dataset with `target_return = null` so they can still be inspected but are excluded from supervised estimation.

### 4.3 Trading-calendar alignment

All date joins are done on trading sessions, not wall time, by building a contiguous index of returned price dates.

- For earnings, event date is converted to prior/next trading date depending on AMC/BMO timing.
- For periodic windows, features and targets are always evaluated at trading-day boundaries.

## 5) Modeling Stack (What Actually Learns)

### 5.1 Model definitions

`MODEL_NAMES = [LinearRegression, Ridge, Lasso, RandomForest]`

- Linear/Ridge/Lasso are wrapped with `StandardScaler` in a `Pipeline`
- RandomForest uses:
  - `n_estimators=300`
  - `random_state=42`

### 5.2 Horizon-aware candidate gating

The candidate set changes by horizon to reduce overfitting/instability on tiny samples:

- `minimum rows`: 
  - `<=21d`: 20
  - `<=63d`: 10
  - `<=126d`: 8
  - `>126d`: 6

- `full-stack rows`:
  - `<=21d`: 40
  - `<=63d`: 16
  - `<=126d`: 12
  - `>126d`: 10

`get_modeling_profile(period_days, sample_size)` returns:

- `analysis_mode = ml_forecast` or `trend_only`
- `selected_models`
- gating message (`No ML forecast` reason)

If rows are insufficient for baseline ML, the app reports Trend-Only regime with regime summary stats.

### 5.3 Train/test protocol

Evaluation is **chronological**, not random:

- 75% train prefix
- 25% holdout suffix

The split is time-safe for forward-looking validation.

### 5.4 Metric set

For each model, the pipeline records:

- MAE
- RMSE
- R2
- Directional accuracy (`mean(sign(pred)=sign(actual))`)
- IC (Pearson corr between prediction and realized returns)

### 5.5 Composite rank score

Models are ranked by:

`score = 0.45 * rank(RMSE) + 0.35 * rank(MAE) + 0.20 * rank(direction_error)`

where `direction_error = 1 - directional_accuracy`.

### 5.6 Forecast ensembling

Top two models by score are selected.
Predictions are combined by inverse-RMSE weighting:

`y_hat = (sum_i pred_i * w_i) / (sum_i w_i), where w_i = 1 / RMSE_i`

This favors lower-error candidates while retaining diversification.

### 5.7 Signal translation

`describe_signal(pred, reference_std, dispersion)` computes:

- sign for bullish/bearish
- magnitude buckets relative to realized volatility (`weak`, `medium`, `strong`)
- confidence damping when model disagreement is high

Output template:

`The trend analysis indicates a <strength> <direction> signal.`

## 6) Trend-Only Mode (fallback logic)

When ML is unavailable for a horizon:

- no forecasting model is trained
- system computes trend regime statistics from the same target series:
  - mean return
  - rolling slope
  - period volatility
  - annualized volatility
  - max drawdown
  - bull ratio (fraction of non-negative periods)

The UI states this explicitly to avoid silent mode switching.

## 7) Streamlit Workbench (UI Design)

`streamlit_app.py` is a guided interface:

1. ask ticker
2. cache and validate exchange and prices
3. build horizon diagnostics across available periods
4. show horizon cards with mode availability and labeled-row checks
5. show signal block in large typography
6. render:
   - core trend chart (bar, green/red by sign)
   - model metrics table
   - feature importance (RF only)
   - SHAP summary (optional dependency)
   - target distribution
   - walk-forward strategy curve
   - raw data
7. generate one-click PDF report

UI state is persisted in `st.session_state` to reduce repeated fetches and improve responsiveness.

## 8) Outputs and Files

With `--save-data` (CLI):

- `<TICKER>_earnings_dataset.csv` or `<TICKER>_weekly/biweekly/..._dataset.csv`
- `<TICKER>_..._model_metrics.csv`
- `<TICKER>_earnings_reaction.png`

From Streamlit (local save / download):

- dataset CSV
- metrics CSV
- trend plot
- full PDF report (portrait, multi-page)

PDF sections include:

1. metadata + analysis settings
2. per-model analysis table and interpretation
3. trend chart + distribution + feature importance + strategy curve
4. dataset snippets
5. conclusion with signal statement and mode

## 9) CLI Usage

```bash
python ts-regressor.py --ticker NVDA --mode biweekly --period-days 10 --start 2025-01-01
python ts-regressor.py --ticker NVDA --mode earnings --start 2022-01-01 --end 2025-01-01 --report markdown --save-data --output-dir outputs
python streamlit_app.py
```

### 9.1 Common arguments

- `--ticker` default: `PLTR`
- `--start` and `--end` in `YYYY-MM-DD`
- `--api-key` or env `FMP_API_KEY`
- `--mode {biweekly,earnings}`
- `--period-days` integer trading-day window for periodic mode
- `--report {text,markdown}`
- `--save-data`
- `--output-dir`

## 10) Runtime Notes and Failure Modes

- `Missing API key` -> set `FMP_API_KEY` or pass `--api-key`
- `402 Payment Required` from API -> plan-access issue, check endpoint entitlement
- unsupported ticker exchange -> filtered before modeling
- insufficient labeled rows -> trend-only mode with explicit warning
- malformed feature column gaps -> rows are dropped with `dropna` before training

## 11) Evolution and Expansion History

This is not a static project; it has been incrementally expanded in response to workflow requirements. Key milestones:

1. CLI-only baseline with earnings and periodic forecasting
2. Added multi-model stack (Linear/Ridge/Lasso/RF)
3. Added periodic trend-only fallback path for low-sample horizons
4. Added exchange validation (NASDAQ/NYSE gate)
5. Added walk-forward strategy simulation and cumulative comparison curves
6. Added Streamlit dashboard with centered prompt + progressive horizon selection UX
7. Added feature-importance, target-distribution, and SHAP diagnostics
8. Added PDF reporting and compact layout fixes
9. Added detailed run output and expanded docs/changelog entries

## 12) Repository History (Git)

```text
git log --oneline --max-count=6
35bac98 Fix bug in OLS coefficient calculation
cb67435 Enhance README with project details and features
b1857e7 Fix formatting in README.md
3a9af7d Remove duplicate entries in README.md
... (continue)
```

For authoritative history use:

```bash
git log --graph --decorate --all --oneline
```

## 13) Quick Technical Validation Checklist

Before publishing a run:

- API key resolves and price rows > 0
- exchange is NASDAQ or NYSE
- dataset is non-empty after label/date alignment
- `sample_size >= minimum_rows` when ML mode expected
- signal mode printed (`ML Forecast` vs `Trend-Only`) matches dataset density
- if report enabled, confirm output path and timestamped filenames



