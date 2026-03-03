"""
Stock trend predictor (earnings reaction or biweekly price trend).

Data source: FinancialModelingPrep (FMP) API.
"""

from __future__ import annotations

import argparse
import os
from bisect import bisect_left, bisect_right
from datetime import date, datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


BASE_URL = "https://financialmodelingprep.com/stable"
SUPPORTED_EXCHANGES = ("NASDAQ", "NYSE")

MODEL_NAMES = ["LinearRegression", "Ridge", "Lasso", "RandomForest"]

EARNINGS_FEATURE_COLS = [
    "eps_surprise_pct",
    "rev_surprise_pct",
    "is_bmo",
    "pre_1d_return",
    "pre_5d_return",
    "pre_20d_vol",
]

BIWEEKLY_FEATURE_COLS = [
    "pre_1d_return",
    "pre_5d_return",
    "pre_10d_return",
    "pre_20d_vol",
]

WEEKLY_FEATURE_COLS = [
    "pre_1d_return",
    "pre_5d_return",
    "pre_10d_return",
    "pre_20d_vol",
]


def make_model(name: str) -> Any:
    if name == "LinearRegression":
        return Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())])
    if name == "Ridge":
        return Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))])
    if name == "Lasso":
        return Pipeline([("scaler", StandardScaler()), ("model", Lasso(alpha=0.001, max_iter=10000))])
    if name == "RandomForest":
        return RandomForestRegressor(n_estimators=300, random_state=42)
    raise ValueError(f"Unknown model name: {name}")


def _parse_date(value: Any) -> Optional[date]:
    if value is None or value == "":
        return None
    try:
        return pd.to_datetime(value).date()
    except Exception:
        return None


def _to_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except Exception:
        return None


def _first_present(row: Dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        if key in row and row[key] not in (None, ""):
            return row[key]
    return None


def _normalize_time(value: Any) -> str:
    if value is None:
        return "amc"
    text = str(value).strip().lower()
    if text == "":
        return "amc"
    if "bmo" in text or "before market open" in text or "premarket" in text:
        return "bmo"
    if "amc" in text or "after market close" in text or "postmarket" in text or "after-hours" in text:
        return "amc"
    return "amc"


def _get_api_key(cli_key: Optional[str]) -> Optional[str]:
    return cli_key or os.getenv("FMP_API_KEY")


def _get_json(url: str, params: Dict[str, Any]) -> Any:
    response = requests.get(url, params=params, timeout=30)
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        status_code = response.status_code
        details = ""
        try:
            payload = response.json()
            if isinstance(payload, dict):
                details_value = _first_present(
                    payload,
                    ["Error Message", "error", "message", "Message", "note"],
                )
                if details_value is not None:
                    details = str(details_value)
        except Exception:
            details = ""

        if status_code == 402:
            msg = (
                "FMP API access denied (402 Payment Required). "
                "Your plan may not include this endpoint or symbol."
            )
            if details:
                msg = f"{msg} Details: {details}"
            raise RuntimeError(msg) from exc

        msg = f"FMP API request failed ({status_code}) for {url}."
        if details:
            msg = f"{msg} Details: {details}"
        raise RuntimeError(msg) from exc

    return response.json()


def fetch_earnings(symbol: str, api_key: str) -> pd.DataFrame:
    url = f"{BASE_URL}/earnings"
    params = {"symbol": symbol, "apikey": api_key}
    data = _get_json(url, params)
    if isinstance(data, dict):
        for key in ("data", "historical", "earnings"):
            if key in data and isinstance(data[key], list):
                data = data[key]
                break
    if not isinstance(data, list):
        raise ValueError("Unexpected earnings response format.")
    return pd.DataFrame(data)


def fetch_prices(symbol: str, api_key: str) -> pd.DataFrame:
    url = f"{BASE_URL}/historical-price-eod/full"
    params = {"symbol": symbol, "apikey": api_key}
    data = _get_json(url, params)
    if isinstance(data, dict):
        if "historical" in data and isinstance(data["historical"], list):
            data = data["historical"]
        elif "data" in data and isinstance(data["data"], list):
            data = data["data"]
    if not isinstance(data, list):
        raise ValueError("Unexpected price response format.")
    return pd.DataFrame(data)



def fetch_profile(symbol: str, api_key: str) -> pd.DataFrame:
    url = f"{BASE_URL}/profile"
    params = {"symbol": symbol, "apikey": api_key}
    data = _get_json(url, params)
    if isinstance(data, dict):
        for key in ("data", "profile", "results"):
            if key in data and isinstance(data[key], list):
                data = data[key]
                break
            if key in data and isinstance(data[key], dict):
                data = [data[key]]
                break
        else:
            data = [data]
    if not isinstance(data, list):
        raise ValueError("Unexpected profile response format.")
    return pd.DataFrame(data)


def _extract_exchange_label(profile_df: pd.DataFrame) -> Optional[str]:
    if profile_df.empty:
        return None
    row = profile_df.iloc[0].to_dict()
    value = _first_present(
        row,
        ["exchangeShortName", "exchange", "exchangeName", "exchangeShort", "stockExchange"],
    )
    if value is None:
        return None
    return str(value).strip()


def is_supported_exchange(exchange_label: Optional[str]) -> bool:
    if not exchange_label:
        return False
    text = exchange_label.upper()
    return any(name in text for name in SUPPORTED_EXCHANGES)


def validate_supported_ticker(symbol: str, api_key: str) -> Tuple[bool, Optional[str]]:
    profile_df = fetch_profile(symbol, api_key)
    exchange_label = _extract_exchange_label(profile_df)
    return is_supported_exchange(exchange_label), exchange_label

def _build_trading_index(prices: pd.DataFrame) -> Tuple[List[date], Dict[date, int]]:
    trading_dates = list(prices.index)
    index_map = {d: i for i, d in enumerate(trading_dates)}
    return trading_dates, index_map


def _last_on_or_before(trading_dates: List[date], target: date) -> Optional[date]:
    pos = bisect_right(trading_dates, target) - 1
    if pos < 0:
        return None
    return trading_dates[pos]


def _prev_trading_date(trading_dates: List[date], index_map: Dict[date, int], target: date) -> Optional[date]:
    idx = index_map.get(target)
    if idx is None:
        idx = bisect_left(trading_dates, target) - 1
    if idx is None or idx - 1 < 0:
        return None
    return trading_dates[idx - 1]


def _next_trading_date(trading_dates: List[date], index_map: Dict[date, int], target: date) -> Optional[date]:
    idx = index_map.get(target)
    if idx is None:
        idx = bisect_right(trading_dates, target)
    if idx is None or idx + 1 >= len(trading_dates):
        return None
    return trading_dates[idx + 1]


def build_dataset(
    earnings_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    earnings_df = earnings_df.copy()
    prices_df = prices_df.copy()

    if "date" not in earnings_df.columns:
        raise ValueError("Earnings data must include a 'date' column.")

    earnings_df["event_date"] = earnings_df["date"].apply(_parse_date)
    earnings_df = earnings_df.dropna(subset=["event_date"])
    earnings_df = earnings_df[
        (earnings_df["event_date"] >= start_date) & (earnings_df["event_date"] <= end_date)
    ]

    prices_df["date"] = prices_df["date"].apply(_parse_date)
    prices_df = prices_df.dropna(subset=["date"])
    prices_df = prices_df.sort_values("date")
    prices_df = prices_df.set_index("date")

    for col in ("open", "high", "low", "close", "volume"):
        if col in prices_df.columns:
            prices_df[col] = pd.to_numeric(prices_df[col], errors="coerce")

    prices_df["close_return"] = prices_df["close"].pct_change()

    trading_dates, index_map = _build_trading_index(prices_df)

    rows: List[Dict[str, Any]] = []

    for _, row in earnings_df.iterrows():
        row_dict = row.to_dict()
        event_date = row_dict.get("event_date")
        if event_date is None:
            continue

        raw_time = _first_present(row_dict, ["time", "timeReported", "reportedTime", "timing"])
        event_time = _normalize_time(raw_time)

        ann_trading_date = _last_on_or_before(trading_dates, event_date)
        if ann_trading_date is None:
            continue

        if event_time == "bmo":
            prior_close_date = _prev_trading_date(trading_dates, index_map, ann_trading_date)
            next_open_date = ann_trading_date
            feature_cutoff = prior_close_date
        else:
            prior_close_date = ann_trading_date
            next_open_date = _next_trading_date(trading_dates, index_map, ann_trading_date)
            feature_cutoff = ann_trading_date

        if prior_close_date is None or next_open_date is None or feature_cutoff is None:
            continue

        try:
            close_prior = prices_df.loc[prior_close_date, "close"]
            open_next = prices_df.loc[next_open_date, "open"]
        except KeyError:
            continue

        if pd.isna(close_prior) or pd.isna(open_next) or close_prior == 0:
            continue

        target_return = (open_next - close_prior) / close_prior

        eps_est = _to_float(_first_present(row_dict, ["epsEstimated", "epsEstimate", "eps_estimated"]))
        eps_act = _to_float(_first_present(row_dict, ["epsActual", "epsActualReported", "eps_actual"]))
        rev_est = _to_float(
            _first_present(row_dict, ["revenueEstimated", "revenueEstimate", "revenue_estimated"])
        )
        rev_act = _to_float(
            _first_present(row_dict, ["revenueActual", "revenueActualReported", "revenue_actual"])
        )

        eps_surprise = eps_act - eps_est if eps_act is not None and eps_est is not None else None
        eps_surprise_pct = (
            eps_surprise / abs(eps_est) if eps_surprise is not None and eps_est not in (None, 0) else None
        )
        rev_surprise = rev_act - rev_est if rev_act is not None and rev_est is not None else None
        rev_surprise_pct = (
            rev_surprise / abs(rev_est) if rev_surprise is not None and rev_est not in (None, 0) else None
        )

        cutoff_idx = index_map.get(feature_cutoff)
        pre_1d_return = None
        pre_5d_return = None
        pre_20d_vol = None

        if cutoff_idx is not None:
            if cutoff_idx - 1 >= 0:
                prev_date = trading_dates[cutoff_idx - 1]
                prev_close = prices_df.loc[prev_date, "close"]
                cutoff_close = prices_df.loc[feature_cutoff, "close"]
                if pd.notna(prev_close) and pd.notna(cutoff_close) and prev_close != 0:
                    pre_1d_return = (cutoff_close - prev_close) / prev_close

            if cutoff_idx - 5 >= 0:
                base_date = trading_dates[cutoff_idx - 5]
                base_close = prices_df.loc[base_date, "close"]
                cutoff_close = prices_df.loc[feature_cutoff, "close"]
                if pd.notna(base_close) and pd.notna(cutoff_close) and base_close != 0:
                    pre_5d_return = (cutoff_close - base_close) / base_close

            if cutoff_idx - 20 >= 1:
                window = prices_df.iloc[cutoff_idx - 20 : cutoff_idx]["close_return"]
                if window.notna().sum() > 1:
                    pre_20d_vol = window.std()

        rows.append(
            {
                "event_date": event_date,
                "event_time": event_time,
                "ann_trading_date": ann_trading_date,
                "prior_close_date": prior_close_date,
                "next_open_date": next_open_date,
                "close_prior": close_prior,
                "open_next": open_next,
                "target_return": target_return,
                "eps_est": eps_est,
                "eps_act": eps_act,
                "eps_surprise": eps_surprise,
                "eps_surprise_pct": eps_surprise_pct,
                "rev_est": rev_est,
                "rev_act": rev_act,
                "rev_surprise": rev_surprise,
                "rev_surprise_pct": rev_surprise_pct,
                "is_bmo": 1 if event_time == "bmo" else 0,
                "pre_1d_return": pre_1d_return,
                "pre_5d_return": pre_5d_return,
                "pre_20d_vol": pre_20d_vol,
            }
        )

    dataset = pd.DataFrame(rows)
    dataset = dataset.sort_values("event_date")
    return dataset


def build_biweekly_dataset(
    prices_df: pd.DataFrame,
    start_date: date,
    end_date: date,
    period_days: int = 10,
) -> pd.DataFrame:
    """
    Build a periodic dataset from price history.

    Each row uses features computed up to the period end (event_date) to predict
    the next period return (forward N trading sessions).
    """
    prices_df = prices_df.copy()

    prices_df["date"] = prices_df["date"].apply(_parse_date)
    prices_df = prices_df.dropna(subset=["date"])
    prices_df = prices_df.sort_values("date")
    prices_df = prices_df.set_index("date")

    for col in ("open", "high", "low", "close", "volume"):
        if col in prices_df.columns:
            prices_df[col] = pd.to_numeric(prices_df[col], errors="coerce")

    prices_df["close_return"] = prices_df["close"].pct_change()

    trading_dates, _ = _build_trading_index(prices_df)
    if not trading_dates:
        return pd.DataFrame()

    start_idx = bisect_left(trading_dates, start_date)

    rows: List[Dict[str, Any]] = []
    cutoff_start = start_idx + period_days - 1
    for cutoff_idx in range(cutoff_start, len(trading_dates), period_days):
        cutoff_date = trading_dates[cutoff_idx]
        if cutoff_date > end_date:
            break

        target_idx = cutoff_idx + period_days
        target_date = trading_dates[target_idx] if target_idx < len(trading_dates) else None
        if target_date is not None and target_date > end_date:
            target_date = None

        close_cutoff = prices_df.loc[cutoff_date, "close"]
        if pd.isna(close_cutoff) or close_cutoff == 0:
            continue

        close_target = None
        target_return = None
        if target_date is not None:
            close_target = prices_df.loc[target_date, "close"]
            if pd.notna(close_target) and close_cutoff != 0:
                target_return = (close_target - close_cutoff) / close_cutoff

        pre_1d_return = None
        pre_5d_return = None
        pre_10d_return = None
        pre_20d_vol = None

        if cutoff_idx - 1 >= 0:
            prev_close = prices_df.iloc[cutoff_idx - 1]["close"]
            if pd.notna(prev_close) and prev_close != 0:
                pre_1d_return = (close_cutoff - prev_close) / prev_close

        if cutoff_idx - 5 >= 0:
            base_close = prices_df.iloc[cutoff_idx - 5]["close"]
            if pd.notna(base_close) and base_close != 0:
                pre_5d_return = (close_cutoff - base_close) / base_close

        biweek_start_idx = cutoff_idx - (period_days - 1)
        if biweek_start_idx >= 0:
            base_close = prices_df.iloc[biweek_start_idx]["close"]
            if pd.notna(base_close) and base_close != 0:
                pre_10d_return = (close_cutoff - base_close) / base_close

        if cutoff_idx - 20 >= 1:
            window = prices_df.iloc[cutoff_idx - 20 : cutoff_idx]["close_return"]
            if window.notna().sum() > 1:
                pre_20d_vol = window.std()

        period_start_date = trading_dates[biweek_start_idx] if biweek_start_idx >= 0 else None

        rows.append(
            {
                "event_date": cutoff_date,
                "period_start_date": period_start_date,
                "period_end_date": cutoff_date,
                "target_end_date": target_date,
                "close_cutoff": close_cutoff,
                "close_target": close_target,
                "target_return": target_return,
                "pre_1d_return": pre_1d_return,
                "pre_5d_return": pre_5d_return,
                "pre_10d_return": pre_10d_return,
                "pre_20d_vol": pre_20d_vol,
            }
        )

    dataset = pd.DataFrame(rows)
    dataset = dataset.sort_values("event_date")
    return dataset


def _metrics_frame(records: List[Dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(records).sort_values("model")


def _evaluate_models(
    df: pd.DataFrame,
    report_markdown: bool,
    show_output: bool = True,
    feature_cols: Optional[List[str]] = None,
    return_models: bool = False,
) -> Any:
    """
    Train/evaluate multiple regressors on the selected return target.

    Technical summary:
    - Features: caller-provided (earnings surprises + timing or biweekly momentum/volatility).
    - Split: time-ordered 75/25 holdout to simulate forward-looking evaluation (no shuffling).
    - Models: linear regression, Ridge, Lasso (with standardization), and Random Forest (raw features).
    - Metrics: MAE, RMSE, and R2 on the holdout set.
    """
    # ML step 1: feature/target assembly.
    # We are modeling a real-valued target y (next-period return) from features X.
    # Formally, we fit a function f: R^p -> R that minimizes a loss on training data.
    # Feature vector: caller provides columns for earnings or biweekly mode.
    feature_cols = feature_cols or EARNINGS_FEATURE_COLS

    model_df = df.dropna(subset=feature_cols + ["target_return"]).copy()
    if model_df.shape[0] < 4:
        print("Not enough rows for modeling after feature/target filtering.")
        empty_metrics = pd.DataFrame()
        if return_models:
            return empty_metrics, {}
        return empty_metrics

    # X is the design matrix (n x p); y is the target vector (n).
    X = model_df[feature_cols].to_numpy()
    y = model_df["target_return"].to_numpy()

    # ML step 2: time-ordered split (avoid look-ahead leakage).
    # We evaluate on later data to approximate forward prediction performance.
    # Time-ordered split to avoid look-ahead leakage.
    split_idx = max(1, int(len(model_df) * 0.75))
    if split_idx >= len(model_df):
        split_idx = len(model_df) - 1

    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # ML step 3: model definitions.
    # - LinearRegression solves min_w ||y - Xw||_2^2.
    # - Ridge solves min_w ||y - Xw||_2^2 + alpha * ||w||_2^2 (L2 regularization).
    # - Lasso solves min_w ||y - Xw||_2^2 + alpha * ||w||_1 (L1 regularization).
    # - RandomForest is an ensemble of decision trees; prediction = average of tree outputs.
    # Linear models are standardized; RandomForest uses raw feature scales.
    models = [(name, make_model(name)) for name in MODEL_NAMES]

    results = []
    fitted_models: Dict[str, Any] = {}
    for name, model in models:
        # ML step 4: training (fit) and prediction (inference).
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        fitted_models[name] = model

        # ML step 5: evaluation metrics.
        # MAE = mean(|y - y_hat|)
        # RMSE = sqrt(mean((y - y_hat)^2))
        # R2 = 1 - (SS_res / SS_tot) measures variance explained (higher is better).
        # MAE/RMSE are in return units; R2 is fit quality on the holdout set.
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = float(np.sqrt(mse))
        r2 = r2_score(y_test, y_pred) if len(y_test) > 1 else np.nan

        results.append({"model": name, "mae": mae, "rmse": rmse, "r2": r2})

    metrics = _metrics_frame(results)
    if show_output:
        if report_markdown:
            print("\nModel Metrics:")
            print(df_to_markdown(metrics, index=False))
        else:
            print("\nModel Metrics:")
            print(metrics.to_string(index=False))

    if return_models:
        return metrics, fitted_models
    return metrics


def df_to_markdown(df: pd.DataFrame, index: bool = False) -> str:
    try:
        return df.to_markdown(index=index)
    except Exception:
        return df.to_string(index=index)


def describe_signal(predicted_return: float, reference_std: float) -> Tuple[str, str, str]:
    direction = "bullish" if predicted_return >= 0 else "bearish"
    magnitude = abs(predicted_return)
    strength: str

    if not np.isfinite(reference_std) or reference_std <= 0:
        if magnitude < 0.01:
            strength = "weak"
        elif magnitude < 0.03:
            strength = "medium"
        else:
            strength = "strong"
    else:
        if magnitude < 0.5 * reference_std:
            strength = "weak"
        elif magnitude < 1.0 * reference_std:
            strength = "medium"
        else:
            strength = "strong"

    statement = f"The trend analysis indicates a {strength} {direction} signal."
    return direction, strength, statement


def forecast_from_models(
    df: pd.DataFrame,
    feature_cols: List[str],
    models: Dict[str, Any],
    metrics: pd.DataFrame,
) -> Tuple[Optional[float], Optional[str], Optional[pd.Series]]:
    if metrics is None or metrics.empty:
        return None, None, None

    metrics_sorted = metrics.sort_values("rmse")
    best_name = metrics_sorted.iloc[0]["model"]
    best_model = models.get(best_name)
    if best_model is None:
        return None, None, None

    train_df = df.dropna(subset=feature_cols + ["target_return"])
    if train_df.empty:
        return None, None, None

    X_full = train_df[feature_cols].to_numpy()
    y_full = train_df["target_return"].to_numpy()
    best_model.fit(X_full, y_full)

    predict_row = df.dropna(subset=feature_cols).tail(1)
    if predict_row.empty:
        return None, best_name, None

    y_pred = float(best_model.predict(predict_row[feature_cols].to_numpy())[0])
    return y_pred, best_name, predict_row.iloc[0]


def save_bull_bear_plot(
    dataset: pd.DataFrame,
    output_dir: str,
    ticker: str,
    title: Optional[str] = None,
) -> str:
    plot_df = dataset.copy()
    plot_df = plot_df.dropna(subset=["target_return"]).sort_values("event_date")
    colors = ["green" if v >= 0 else "red" for v in plot_df["target_return"]]

    fig, ax = plt.subplots(figsize=(10, 5))
    x_vals = np.arange(len(plot_df))
    ax.bar(x_vals, plot_df["target_return"] * 100.0, color=colors, width=0.85)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title(title or f"{ticker} Trend Returns")
    ax.set_ylabel("Return (%)")
    ax.set_xlabel("Period End")
    tick_step = max(1, len(plot_df) // 10)
    tick_idx = x_vals[::tick_step]
    ax.set_xticks(tick_idx)
    ax.set_xticklabels(
        [str(d) for d in plot_df["event_date"].iloc[::tick_step]],
        rotation=45,
        ha="right",
    )

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{ticker}_earnings_reaction.png")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict post-earnings or biweekly trend returns using FMP data."
    )
    parser.add_argument("--ticker", default="PLTR", help="Stock ticker (default: PLTR)")
    parser.add_argument("--start", default="2025-01-01", help="Start date YYYY-MM-DD (default: 2025-01-01)")
    parser.add_argument(
        "--end",
        default=date.today().isoformat(),
        help="End date YYYY-MM-DD (default: today)",
    )
    parser.add_argument("--api-key", default=None, help="FMP API key (or set FMP_API_KEY env var)")
    parser.add_argument(
        "--mode",
        choices=["biweekly", "earnings"],
        default="biweekly",
        help="Analysis mode (default: biweekly)",
    )
    parser.add_argument(
        "--period-days",
        type=int,
        default=10,
        help="Trading-day window for periodic mode (default: 10). Use 5 for weekly.",
    )
    parser.add_argument(
        "--report",
        choices=["text", "markdown"],
        default="text",
        help="Output style (default: text)",
    )
    parser.add_argument(
        "--save-data",
        action="store_true",
        help="Save dataset CSV + bull/bear plot (and metrics CSV) to output-dir",
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory for saved CSV (default: current directory)",
    )

    args = parser.parse_args()

    api_key = _get_api_key(args.api_key)
    if not api_key:
        raise SystemExit("Missing API key. Provide --api-key or set FMP_API_KEY.")

    start_date = _parse_date(args.start)
    end_date = _parse_date(args.end)
    if start_date is None or end_date is None:
        raise SystemExit("Invalid start/end date. Use YYYY-MM-DD.")
    if start_date > end_date:
        raise SystemExit("Start date must be <= end date.")

    report_markdown = args.report == "markdown"
    try:
        is_supported, exchange_label = validate_supported_ticker(args.ticker, api_key)
    except RuntimeError as exc:
        raise SystemExit(str(exc))

    if not is_supported:
        exchange_info = exchange_label if exchange_label else "unknown"
        raise SystemExit(
            f"Ticker {args.ticker} is on unsupported exchange ({exchange_info}). Use NASDAQ/NYSE symbols."
        )
    print(f"Ticker {args.ticker} exchange: {exchange_label}")

    print(f"Fetching historical prices for {args.ticker}...")
    try:
        prices_df = fetch_prices(args.ticker, api_key)
    except RuntimeError as exc:
        raise SystemExit(str(exc))
    print(f"Fetched {len(prices_df)} price rows.")

    if args.mode == "earnings":
        print(f"Fetching earnings for {args.ticker} from {start_date} to {end_date}...")
        try:
            earnings_df = fetch_earnings(args.ticker, api_key)
        except RuntimeError as exc:
            raise SystemExit(str(exc))
        print(f"Fetched {len(earnings_df)} earnings rows (pre-filter).")

        dataset = build_dataset(earnings_df, prices_df, start_date, end_date)
        if dataset.empty:
            raise SystemExit("No usable earnings events after filtering and alignment.")
        feature_cols = EARNINGS_FEATURE_COLS
        plot_title = f"{args.ticker} Post-Earnings Next-Day Open vs Prior Close"
        label = "Events"
    else:
        period_days = max(2, int(args.period_days))
        dataset = build_biweekly_dataset(prices_df, start_date, end_date, period_days=period_days)
        if dataset.empty:
            raise SystemExit("No usable periods after filtering and alignment.")
        feature_cols = BIWEEKLY_FEATURE_COLS if period_days >= 10 else WEEKLY_FEATURE_COLS
        if period_days == 5:
            period_label = "Weekly"
        elif period_days == 10:
            period_label = "Biweekly"
        else:
            period_label = f"{period_days}-Day"
        plot_title = f"{args.ticker} {period_label} Trend Returns"
        label = f"{period_label} periods"

    dataset["bull_bear"] = np.where(
        dataset["target_return"].isna(),
        "unknown",
        np.where(dataset["target_return"] >= 0, "bull", "bear"),
    )

    print(f"{label} after alignment: {len(dataset)}")

    if report_markdown:
        print("\nSample Events:")
        print(df_to_markdown(dataset.head(10), index=False))
    else:
        print("\nSample Events:")
        print(dataset.head(10).to_string(index=False))

    metrics, models = _evaluate_models(
        dataset,
        report_markdown=report_markdown,
        feature_cols=feature_cols,
        return_models=True,
    )

    if args.mode == "biweekly":
        pred, best_name, forecast_row = forecast_from_models(dataset, feature_cols, models, metrics)
        if pred is not None:
            reference_std = dataset["target_return"].dropna().std()
            direction, strength, statement = describe_signal(pred, reference_std)
            print(f"\nForecast ({best_name}): {pred * 100:.2f}% next period return")
            print(statement)
            if forecast_row is not None:
                start_label = forecast_row.get("event_date")
                end_label = forecast_row.get("target_end_date")
                if pd.notna(end_label):
                    print(f"Forecast window: {start_label} -> {end_label}")
                else:
                    print(f"Forecast window: next period after {start_label}")

    if args.save_data:
        os.makedirs(args.output_dir, exist_ok=True)
        if args.mode == "earnings":
            dataset_label = "earnings"
        else:
            if period_days == 5:
                dataset_label = "weekly"
            elif period_days == 10:
                dataset_label = "biweekly"
            else:
                dataset_label = f"{period_days}d"
        dataset_path = os.path.join(args.output_dir, f"{args.ticker}_{dataset_label}_dataset.csv")
        dataset.to_csv(dataset_path, index=False)
        print(f"\nSaved dataset to: {dataset_path}")

        if metrics is not None and not metrics.empty:
            metrics_path = os.path.join(args.output_dir, f"{args.ticker}_{dataset_label}_model_metrics.csv")
            metrics.to_csv(metrics_path, index=False)
            print(f"Saved model metrics to: {metrics_path}")

        plot_path = save_bull_bear_plot(dataset, args.output_dir, args.ticker, title=plot_title)
        print(f"Saved bull/bear plot to: {plot_path}")


if __name__ == "__main__":
    main()