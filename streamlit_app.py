"""
Streamlit UI for the stock trend model.
"""

from __future__ import annotations

import importlib.util
import os
import textwrap
import time
from datetime import date, datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
import streamlit as st


def _load_stonks_module():
    module_path = Path(__file__).with_name("ts-regressor.py")
    spec = importlib.util.spec_from_file_location("stonks_module", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


stonks = _load_stonks_module()


PERIOD_OPTIONS = {
    "Weekly (5 trading days)": 5,
    "Biweekly (10 trading days)": 10,
    "Monthly (21 trading days)": 21,
    "Quarterly (63 trading days)": 63,
    "Semiannual (126 trading days)": 126,
    "Annual (252 trading days)": 252,
}


def period_label_from_days(period_days: int) -> str:
    if period_days == 5:
        return "Weekly"
    if period_days == 10:
        return "Biweekly"
    if period_days == 21:
        return "Monthly"
    if period_days == 63:
        return "Quarterly"
    if period_days == 126:
        return "Semiannual"
    if period_days == 252:
        return "Annual"
    return f"{period_days}-Day"


APP_CSS = """
<style>
html, body, [class*="css"] {
    font-family: Helvetica, Arial, sans-serif;
}

.focus-overlay {
    position: fixed;
    inset: 0;
    background: rgba(8, 12, 20, 0.28);
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
    pointer-events: none;
    z-index: 0;
}

.hero-question {
    text-align: center;
    font-size: 2.15rem;
    font-weight: 700;
    line-height: 1.2;
    margin-top: 14vh;
    margin-bottom: 0.8rem;
    color: #ffffff;
}

.horizon-question {
    text-align: center;
    font-size: 1.6rem;
    font-weight: 700;
    margin-top: 14vh;
    margin-bottom: 0.3rem;
    color: #ffffff;
}

.horizon-sub {
    text-align: center;
    font-size: 1rem;
    color: #c9d3e3;
    margin-bottom: 0.9rem;
}

.signal-block {
    text-align: center;
    margin-top: 0.5rem;
    margin-bottom: 0.2rem;
}

.signal-title {
    font-size: 3rem;
    font-weight: 800;
    line-height: 1.0;
    margin-bottom: 0.35rem;
}

.signal-sub {
    font-size: 1.2rem;
    font-weight: 600;
    color: #c9d3e3;
    margin-bottom: 0.2rem;
}

.signal-text {
    font-size: 1.1rem;
    color: #dce4ef;
    margin-bottom: 0.5rem;
}

.center-download {
    display: flex;
    justify-content: center;
    margin-top: 0.4rem;
    margin-bottom: 1rem;
}

details[data-testid="stExpander"] {
    border: 1px solid #3b4557;
    border-radius: 4px;
    background: #151b25;
}

details[data-testid="stExpander"] > summary {
    font-weight: 650;
}

[data-testid="stAppViewContainer"] {
    background: radial-gradient(1200px 600px at 20% -10%, #2a3040 0%, #11161f 40%, #0d1117 100%);
}
</style>
"""


def build_bull_bear_chart(
    dataset: pd.DataFrame,
    title: str,
    x_label: str,
    figsize: Tuple[float, float] = (10, 5),
):
    plot_df = dataset.dropna(subset=["target_return"]).sort_values("event_date")
    colors = ["#24c16b" if v >= 0 else "#ea4d5a" for v in plot_df["target_return"]]

    fig, ax = plt.subplots(figsize=figsize)
    x_vals = np.arange(len(plot_df))
    ax.bar(x_vals, plot_df["target_return"] * 100.0, color=colors, width=0.85)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title(title)
    ax.set_ylabel("Return (%)")
    ax.set_xlabel(x_label)

    tick_step = max(1, len(plot_df) // 10)
    tick_idx = x_vals[::tick_step]
    ax.set_xticks(tick_idx)
    ax.set_xticklabels(
        [str(d) for d in plot_df["event_date"].iloc[::tick_step]],
        rotation=45,
        ha="right",
    )
    fig.tight_layout()
    return fig


def plot_feature_importance(model, feature_cols: List[str], title: str):
    if not hasattr(model, "feature_importances_"):
        return None
    importances = model.feature_importances_
    order = np.argsort(importances)[::-1]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar([feature_cols[i] for i in order], importances[order], color="#3f8cff")
    ax.set_title(title)
    ax.set_ylabel("Importance")
    ax.set_xlabel("Feature")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    return fig


def plot_target_distribution(series: pd.Series, title: str):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(series * 100.0, bins=20, color="#8892a6", edgecolor="white", alpha=0.9)
    ax.set_title(title)
    ax.set_xlabel("Return (%)")
    ax.set_ylabel("Frequency")
    fig.tight_layout()
    return fig


def plot_strategy_curve(dates: List[pd.Timestamp], strategy_returns: List[float], buyhold_returns: List[float], title: str):
    if len(dates) == 0:
        return None

    cum_strategy = np.cumprod(1 + np.array(strategy_returns))
    cum_buyhold = np.cumprod(1 + np.array(buyhold_returns))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(dates, cum_strategy, label="Strategy (long if pred > 0)", color="#3f8cff")
    ax.plot(dates, cum_buyhold, label="Buy & Hold", color="#f39c34")
    ax.set_title(title)
    ax.set_ylabel("Cumulative Return (x)")
    ax.set_xlabel("Period End")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def walk_forward_predictions(df: pd.DataFrame, feature_cols: List[str], model_name: str):
    model_df = df.dropna(subset=feature_cols + ["target_return"]).copy().reset_index(drop=True)
    if len(model_df) < 6:
        return [], [], []

    start_idx = max(5, int(len(model_df) * 0.75))
    dates: List[pd.Timestamp] = []
    preds: List[float] = []
    actuals: List[float] = []

    for i in range(start_idx, len(model_df)):
        train = model_df.iloc[:i]
        test = model_df.iloc[i : i + 1]
        model = stonks.make_model(model_name)
        model.fit(train[feature_cols].to_numpy(), train["target_return"].to_numpy())
        pred = float(model.predict(test[feature_cols].to_numpy())[0])
        actual = float(test["target_return"].iloc[0])
        dates.append(test["event_date"].iloc[0])
        preds.append(pred)
        actuals.append(actual)

    return dates, preds, actuals


def compute_trend_summary(dataset: pd.DataFrame, period_days: int) -> Dict[str, Any]:
    trend_df = dataset.dropna(subset=["target_return"]).sort_values("event_date").copy()
    if trend_df.empty:
        return {
            "observations": 0,
            "mean_return": np.nan,
            "slope_per_period": np.nan,
            "period_volatility": np.nan,
            "annualized_volatility": np.nan,
            "max_drawdown": np.nan,
            "bull_ratio": np.nan,
            "regime": "insufficient",
        }

    returns = trend_df["target_return"].astype(float).to_numpy()
    n = len(returns)

    mean_return = float(np.mean(returns)) if n > 0 else np.nan
    period_vol = float(np.std(returns, ddof=1)) if n > 1 else np.nan
    annualized_vol = (
        float(period_vol * np.sqrt(252.0 / max(1.0, float(period_days))))
        if np.isfinite(period_vol)
        else np.nan
    )

    slope = np.nan
    if n >= 2:
        x = np.arange(n, dtype=float)
        slope = float(np.polyfit(x, returns, 1)[0])

    cumulative = np.cumprod(1.0 + returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative / running_max) - 1.0
    max_drawdown = float(np.min(drawdown)) if drawdown.size > 0 else np.nan

    bull_ratio = float(np.mean(returns >= 0.0)) if n > 0 else np.nan
    if not np.isfinite(bull_ratio):
        regime = "insufficient"
    elif bull_ratio >= 0.60 and mean_return >= 0:
        regime = "bullish"
    elif bull_ratio <= 0.40 and mean_return < 0:
        regime = "bearish"
    else:
        regime = "mixed"

    return {
        "observations": int(n),
        "mean_return": mean_return,
        "slope_per_period": slope,
        "period_volatility": period_vol,
        "annualized_volatility": annualized_vol,
        "max_drawdown": max_drawdown,
        "bull_ratio": bull_ratio,
        "regime": regime,
    }


def build_horizon_diagnostics(
    prices_df: pd.DataFrame,
    start_date: date,
    end_date: date,
    feature_cols: List[str],
) -> Dict[int, Dict[str, Any]]:
    diagnostics: Dict[int, Dict[str, Any]] = {}

    for option_label, period_days in PERIOD_OPTIONS.items():
        dataset = stonks.build_biweekly_dataset(prices_df, start_date, end_date, period_days=period_days)
        labeled_rows = 0
        if not dataset.empty:
            labeled_rows = int(dataset.dropna(subset=feature_cols + ["target_return"]).shape[0])

        profile = stonks.get_modeling_profile(period_days=period_days, sample_size=labeled_rows)
        minimum_rows = int(profile.get("minimum_rows", 0))
        ml_available = bool(profile.get("ml_available", bool(profile.get("selected_models"))))

        unavailable_reason = ""
        if not ml_available:
            unavailable_reason = (
                f"{period_label_from_days(period_days)} ML unavailable: "
                f"only {labeled_rows} labeled rows (min {minimum_rows})."
            )

        diagnostics[period_days] = {
            "option_label": option_label,
            "period_label": period_label_from_days(period_days),
            "dataset": dataset,
            "labeled_rows": labeled_rows,
            "minimum_rows": minimum_rows,
            "profile": profile,
            "ml_available": ml_available,
            "ml_unavailable_reason": unavailable_reason,
        }

    return diagnostics


def validate_exchange_or_stop(ticker: str, api_key: str) -> str:
    try:
        is_supported, exchange_label = stonks.validate_supported_ticker(ticker, api_key)
    except Exception as exc:
        message = str(exc)
        st.error(f"Could not validate ticker metadata: {message}")
        if "402" in message:
            st.info("FMP key does not have access to this endpoint/plan tier.")
        st.stop()

    if not is_supported:
        exchange_info = exchange_label if exchange_label else "unknown"
        st.error(
            f"{ticker} is on unsupported exchange ({exchange_info}). "
            "Use symbols listed on NASDAQ or NYSE."
        )
        st.stop()

    return exchange_label if exchange_label else "unknown"



def _add_pdf_page_title(ax, title: str, subtitle: str = "") -> None:
    # Compact report header ribbon (portrait layout)
    ax.add_patch(plt.Rectangle((0.0, 0.91), 1.0, 0.09, transform=ax.transAxes, color="#2ea8cf", ec="none"))
    ax.text(0.06, 0.94, "TS Regressor Report", ha="left", va="center", fontsize=10.5, color="white", fontweight="bold")
    ax.text(0.94, 0.94, date.today().strftime("%b %Y"), ha="right", va="center", fontsize=10, color="white")
    ax.text(0.06, 0.875, title, ha="left", va="top", fontsize=18, fontweight="bold", color="#111111")
    if subtitle:
        ax.text(0.06, 0.845, subtitle, ha="left", va="top", fontsize=10.5, color="#333333")


def _write_wrapped_text(
    ax,
    x: float,
    y: float,
    text: str,
    size: int = 12,
    weight: str = "normal",
    color: str = "#111111",
    wrap_chars: int = 92,
    line_spacing: float = 1.05,
) -> float:
    wrapped = textwrap.fill(str(text), width=wrap_chars)
    ax.text(x, y, wrapped, ha="left", va="top", fontsize=size, fontweight=weight, color=color)

    lines = max(1, wrapped.count("\n") + 1)
    line_height = max(0.015, ((size / 72.0) / 11.69) * line_spacing)
    return y - (lines * line_height) - 0.008


def _model_interpretation(row: pd.Series) -> str:
    mae = float(row.get("mae", np.nan))
    rmse = float(row.get("rmse", np.nan))
    r2 = float(row.get("r2", np.nan))
    direction_acc = float(row.get("directional_accuracy", np.nan))
    ic = float(row.get("ic", np.nan))

    r2_text = (
        "insufficient variance explanation"
        if not np.isfinite(r2)
        else ("strong fit" if r2 >= 0.35 else "moderate fit" if r2 >= 0.1 else "weak fit")
    )
    error_text = (
        "lower expected error"
        if np.isfinite(rmse) and rmse <= 0.05
        else "moderate expected error"
        if np.isfinite(rmse) and rmse <= 0.10
        else "higher expected error"
    )

    hit_rate_text = f"{direction_acc:.1%}" if np.isfinite(direction_acc) else "n/a"
    ic_text = f"{ic:.3f}" if np.isfinite(ic) else "n/a"

    return (
        f"Interpretation: {error_text}; {r2_text}. "
        f"MAE ~= {mae:.4f}, RMSE ~= {rmse:.4f}, hit-rate ~= {hit_rate_text}, IC ~= {ic_text}."
    )


def build_full_report_pdf(
    ticker: str,
    exchange_label: str,
    period_label: str,
    start_date: date,
    end_date: date,
    forecast_info: Dict[str, str],
    dataset: pd.DataFrame,
    metrics: pd.DataFrame,
    feature_importance_df: pd.DataFrame,
    strategy_df: pd.DataFrame,
) -> bytes:
    buffer = BytesIO()

    models_used = "n/a"
    if metrics is not None and not metrics.empty:
        models_used = ", ".join(metrics["model"].astype(str).tolist())

    with plt.rc_context({"font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"]}):
        with PdfPages(buffer) as pdf:
            # Page 1: report header/meta
            fig1 = plt.figure(figsize=(8.27, 11.69))
            ax1 = fig1.add_axes([0, 0, 1, 1])
            ax1.axis("off")

            _add_pdf_page_title(ax1, "Stock Trend Analysis Report", "Generated from ts-regressor")
            y = 0.79
            y = _write_wrapped_text(ax1, 0.08, y, f"Ticker Name: {ticker}", size=13, weight="bold")
            y = _write_wrapped_text(ax1, 0.08, y, f"Analysis Period (Date Range): {start_date} to {end_date}", size=12)
            y = _write_wrapped_text(ax1, 0.08, y, f"Models Used: {models_used}", size=12, weight="bold")
            y = _write_wrapped_text(ax1, 0.08, y, f"Exchange: {exchange_label}", size=11)
            y = _write_wrapped_text(ax1, 0.08, y, f"Period Granularity: {period_label}", size=11)
            y = _write_wrapped_text(ax1, 0.08, y, f"Data Points Used: {len(dataset)}", size=11)

            pdf.savefig(fig1)
            plt.close(fig1)

            # Page 2+: per-model analysis
            def new_model_page(title: str):
                fig = plt.figure(figsize=(8.27, 11.69))
                ax = fig.add_axes([0, 0, 1, 1])
                ax.axis("off")
                _add_pdf_page_title(ax, title)
                return fig, ax

            fig2, ax2 = new_model_page("Per-Model Analysis")
            y = 0.79
            if metrics is None or metrics.empty:
                _write_wrapped_text(ax2, 0.08, y, "No model metrics available (insufficient training rows).", size=12)
            else:
                rank_col = "score" if "score" in metrics.columns else "rmse"
                for _, row in metrics.sort_values(rank_col).iterrows():
                    if y < 0.24:
                        pdf.savefig(fig2)
                        plt.close(fig2)
                        fig2, ax2 = new_model_page("Per-Model Analysis (Continued)")
                        y = 0.79

                    model_name = str(row.get("model", "Unknown"))
                    mae = float(row.get("mae", np.nan))
                    rmse = float(row.get("rmse", np.nan))
                    r2 = float(row.get("r2", np.nan))

                    y = _write_wrapped_text(ax2, 0.08, y, f"Model: {model_name}", size=14, weight="bold")
                    y = _write_wrapped_text(ax2, 0.11, y, f"MAE: {mae:.6f}", size=11)
                    y = _write_wrapped_text(ax2, 0.11, y, f"RMSE: {rmse:.6f}", size=11)
                    y = _write_wrapped_text(ax2, 0.11, y, f"R2: {r2:.6f}" if np.isfinite(r2) else "R2: n/a", size=11)
                    y = _write_wrapped_text(ax2, 0.11, y, _model_interpretation(row), size=10, color="#222222", wrap_chars=84)
                    y -= 0.012

            pdf.savefig(fig2)
            plt.close(fig2)

            # Page 3: analysis visuals (directly below model section in report sequence)
            fig3, axes = plt.subplots(2, 2, figsize=(8.27, 11.69))
            fig3.suptitle("Analysis Visuals", fontsize=16, fontweight="bold")

            trend_df = dataset.dropna(subset=["target_return"]).sort_values("event_date")
            ax_t = axes[0, 0]
            if not trend_df.empty:
                x_vals = np.arange(len(trend_df))
                colors = ["#24c16b" if v >= 0 else "#ea4d5a" for v in trend_df["target_return"]]
                ax_t.bar(x_vals, trend_df["target_return"] * 100.0, color=colors, width=0.85)
                ax_t.axhline(0, color="black", linewidth=0.8)
                ax_t.set_title("Core Trend Chart")
                ax_t.set_ylabel("Return (%)")
                tick_step = max(1, len(trend_df) // 8)
                tick_idx = x_vals[::tick_step]
                ax_t.set_xticks(tick_idx)
                ax_t.set_xticklabels([str(d) for d in trend_df["event_date"].iloc[::tick_step]], rotation=45, ha="right", fontsize=7)
            else:
                ax_t.text(0.5, 0.5, "No trend data", ha="center", va="center")
                ax_t.set_axis_off()

            ax_d = axes[0, 1]
            if not trend_df.empty:
                ax_d.hist(trend_df["target_return"] * 100.0, bins=20, color="#8892a6", edgecolor="white", alpha=0.9)
                ax_d.set_title("Target Distribution")
                ax_d.set_xlabel("Return (%)")
                ax_d.set_ylabel("Frequency")
            else:
                ax_d.text(0.5, 0.5, "No distribution data", ha="center", va="center")
                ax_d.set_axis_off()

            ax_f = axes[1, 0]
            if feature_importance_df is not None and not feature_importance_df.empty:
                fi = feature_importance_df.copy().sort_values("importance", ascending=False)
                ax_f.bar(fi["feature"].astype(str).tolist(), fi["importance"].to_numpy(), color="#3f8cff")
                ax_f.set_title("Feature Importance")
                ax_f.tick_params(axis="x", rotation=45)
            else:
                ax_f.text(0.5, 0.5, "No feature-importance data", ha="center", va="center")
                ax_f.set_axis_off()

            ax_s = axes[1, 1]
            if strategy_df is not None and not strategy_df.empty:
                sdf = strategy_df.copy()
                if "event_date" in sdf.columns:
                    sdf["event_date"] = pd.to_datetime(sdf["event_date"])
                strategy_returns = sdf["strategy_return"].to_numpy() if "strategy_return" in sdf.columns else np.array([])
                actual_returns = sdf["actual_return"].to_numpy() if "actual_return" in sdf.columns else np.array([])
                if strategy_returns.size > 0 and actual_returns.size > 0:
                    cum_strategy = np.cumprod(1 + strategy_returns)
                    cum_buyhold = np.cumprod(1 + actual_returns)
                    x = sdf["event_date"] if "event_date" in sdf.columns else np.arange(len(cum_strategy))
                    ax_s.plot(x, cum_strategy, label="Strategy", color="#3f8cff")
                    ax_s.plot(x, cum_buyhold, label="Buy & Hold", color="#f39c34")
                    ax_s.set_title("Strategy vs Buy & Hold")
                    ax_s.legend(fontsize=8)
                else:
                    ax_s.text(0.5, 0.5, "No strategy data", ha="center", va="center")
                    ax_s.set_axis_off()
            else:
                ax_s.text(0.5, 0.5, "No strategy data", ha="center", va="center")
                ax_s.set_axis_off()

            fig3.tight_layout(rect=[0.03, 0.03, 0.97, 0.95], h_pad=1.2, w_pad=0.8)
            pdf.savefig(fig3)
            plt.close(fig3)

            # Page 4: data tables
            fig4 = plt.figure(figsize=(8.27, 11.69))
            ax4 = fig4.add_axes([0, 0, 1, 1])
            ax4.axis("off")
            _add_pdf_page_title(ax4, "Data Tables")

            if metrics is not None and not metrics.empty:
                metrics_display = metrics.copy().round(6)
                tbl1 = ax4.table(
                    cellText=metrics_display.values,
                    colLabels=metrics_display.columns,
                    loc="upper left",
                    cellLoc="center",
                    colLoc="center",
                    bbox=[0.06, 0.60, 0.88, 0.24],
                )
                tbl1.auto_set_font_size(False)
                tbl1.set_fontsize(8)

            sample_cols = [
                c
                for c in [
                    "event_date",
                    "target_return",
                    "pre_1d_return",
                    "pre_5d_return",
                    "pre_period_return",
                    "pre_20d_vol",
                    "bull_bear",
                ]
                if c in dataset.columns
            ]
            sample_df = dataset[sample_cols].head(12).copy() if sample_cols else pd.DataFrame()
            if not sample_df.empty:
                for col in sample_df.columns:
                    if pd.api.types.is_datetime64_any_dtype(sample_df[col]):
                        sample_df[col] = sample_df[col].astype(str)
                sample_df = sample_df.round(6)
                tbl2 = ax4.table(
                    cellText=sample_df.values,
                    colLabels=sample_df.columns,
                    loc="upper left",
                    cellLoc="center",
                    colLoc="center",
                    bbox=[0.06, 0.20, 0.88, 0.34],
                )
                tbl2.auto_set_font_size(False)
                tbl2.set_fontsize(7)

            pdf.savefig(fig4)
            plt.close(fig4)

            # Final page: text-only conclusion
            fig5 = plt.figure(figsize=(8.27, 11.69))
            ax5 = fig5.add_axes([0, 0, 1, 1])
            ax5.axis("off")
            _add_pdf_page_title(ax5, "Conclusion")

            direction = forecast_info.get("direction", "n/a")
            strength = forecast_info.get("strength", "n/a")
            pred_pct = forecast_info.get("pred_pct", "n/a")
            statement = forecast_info.get("statement", "n/a")
            best_model = forecast_info.get("best_model", "n/a")
            analysis_mode = forecast_info.get("analysis_mode", "ml_forecast")

            y = 0.76
            if analysis_mode == "ml_forecast":
                y = _write_wrapped_text(
                    ax5,
                    0.08,
                    y,
                    f"Forecast conclusion: The model ensemble indicates a {strength} {direction} signal with expected next {period_label.lower()} return of {pred_pct}.",
                    size=13,
                    weight="bold",
                    wrap_chars=80,
                )
                y = _write_wrapped_text(ax5, 0.08, y, f"Primary model/ensemble: {best_model}.", size=12)
            else:
                y = _write_wrapped_text(
                    ax5,
                    0.08,
                    y,
                    f"Trend-only conclusion: {strength} {direction} regime identified for the selected horizon.",
                    size=13,
                    weight="bold",
                    wrap_chars=80,
                )
                y = _write_wrapped_text(ax5, 0.08, y, "ML forecast was skipped due to limited labeled samples.", size=12)

            y = _write_wrapped_text(ax5, 0.08, y, f"Statement: {statement}", size=12, wrap_chars=84)
            _write_wrapped_text(
                ax5,
                0.08,
                y,
                "This report is quantitative and historical. Use with risk controls and additional fundamental/contextual analysis.",
                size=11,
                color="#333333",
                wrap_chars=84,
            )

            pdf.savefig(fig5)
            plt.close(fig5)

    buffer.seek(0)
    return buffer.getvalue()



def reset_selection_state(clear_ticker: bool = False) -> None:
    st.session_state.pop("selected_period_days", None)
    st.session_state.pop("selected_period_label", None)
    if clear_ticker:
        st.session_state.pop("analysis_ticker", None)
        st.session_state.pop("desired_ticker", None)
        st.session_state.ticker_input_ready = False


def render_ticker_prompt() -> Optional[str]:
    current_ticker = st.session_state.get("analysis_ticker")
    if current_ticker:
        return str(current_ticker)

    st.markdown("<div class='focus-overlay'></div>", unsafe_allow_html=True)

    if "ticker_input_ready" not in st.session_state:
        st.session_state.ticker_input_ready = False

    center = st.columns([1, 2, 1])[1]
    with center:
        question_placeholder = st.empty()
        question_placeholder.markdown(
            "<div class='hero-question'>What is your desired ticker?</div>",
            unsafe_allow_html=True,
        )

        input_placeholder = st.empty()
        if not st.session_state.ticker_input_ready:
            time.sleep(0.5)
            st.session_state.ticker_input_ready = True

        with input_placeholder.container():
            ticker = st.text_input(
                "Desired ticker",
                value=st.session_state.get("desired_ticker", "NVDA"),
                placeholder="e.g., NVDA, AAPL, MSFT",
                label_visibility="collapsed",
            ).strip().upper()
            run_btn = st.button("Analyze Ticker", use_container_width=True, type="primary")

        st.session_state.desired_ticker = ticker

    if run_btn:
        if not ticker:
            st.error("Enter a ticker symbol.")
            return None
        st.session_state.analysis_ticker = ticker
        reset_selection_state(clear_ticker=False)
        st.rerun()

    return None


def render_horizon_selector(horizon_diagnostics: Dict[int, Dict[str, Any]]) -> Optional[Tuple[int, str]]:
    selected_days = st.session_state.get("selected_period_days")
    if selected_days is not None:
        selected_days = int(selected_days)
        return selected_days, period_label_from_days(selected_days)

    center = st.columns([1, 2, 1])[1]
    with center:
        st.markdown("<div class='horizon-question'>Select analysis horizon</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='horizon-sub'>Mode selection is automatic: ML Forecast when eligible, otherwise Trend-Only.</div>",
            unsafe_allow_html=True,
        )

        options = list(PERIOD_OPTIONS.items())
        for row_start in range(0, len(options), 2):
            col_left, col_right = st.columns(2, gap="small")
            row_options = options[row_start : row_start + 2]

            for idx, (label, days) in enumerate(row_options):
                info = horizon_diagnostics.get(days, {})
                ml_available = bool(info.get("ml_available", False))
                labeled_rows = int(info.get("labeled_rows", 0))
                minimum_rows = int(info.get("minimum_rows", 0))
                mode_hint = "ML Forecast" if ml_available else "Trend-Only"

                col = col_left if idx == 0 else col_right
                with col:
                    if st.button(
                        f"{label} ({mode_hint})",
                        key=f"horizon_{days}",
                        use_container_width=True,
                    ):
                        st.session_state.selected_period_days = days
                        st.session_state.selected_period_label = period_label_from_days(days)
                        st.rerun()

                    if ml_available:
                        st.caption(f"ML ready: {labeled_rows} labeled rows")
                    else:
                        st.caption(
                            f"{period_label_from_days(days)} ML unavailable: only {labeled_rows}/{minimum_rows} rows. "
                            "Trend-Only will be shown."
                        )

        st.info(
            "For 20+ year tests use older tickers (e.g., MSFT, IBM). "
            "PLTR listed in 2020, so pre-2020 history is not available. "
            "For robust quarterly/annual ML on newer tickers, use a multi-ticker panel model."
        )

    return None


def main() -> None:
    st.set_page_config(page_title="Stock Trend Forecaster", layout="wide")
    st.markdown(APP_CSS, unsafe_allow_html=True)

    with st.sidebar:
        st.header("Settings")
        api_key_default = os.getenv("FMP_API_KEY", "")
        api_key = st.text_input("FMP API Key", value=api_key_default, type="password")

        start_date = st.date_input("Start Date", value=date(2000, 1, 1))
        end_date = st.date_input("End Date", value=date.today())
        st.caption("Select horizon from centered buttons after ticker confirmation.")

        save_local = st.checkbox("Save outputs locally", value=False)
        output_dir = st.text_input("Local Output Dir", value=os.getcwd())

        st.markdown("---")
        change_ticker_clicked = st.button(
            "Change Ticker",
            use_container_width=True,
            disabled=not bool(st.session_state.get("analysis_ticker")),
        )
        change_horizon_clicked = st.button(
            "Change Horizon",
            use_container_width=True,
            disabled=st.session_state.get("selected_period_days") is None,
        )

    if change_ticker_clicked:
        reset_selection_state(clear_ticker=True)
        st.rerun()
    if change_horizon_clicked:
        reset_selection_state(clear_ticker=False)
        st.rerun()

    ticker = render_ticker_prompt()
    if ticker is None:
        st.stop()

    if not api_key:
        st.error("Missing API key. Enter it in Settings.")
        st.stop()
    if start_date > end_date:
        st.error("Start date must be on or before end date.")
        st.stop()

    exchange_label = validate_exchange_or_stop(ticker, api_key)

    price_cache_key = (ticker, hash(api_key))
    if st.session_state.get("price_cache_key") != price_cache_key:
        with st.spinner("Fetching prices..."):
            try:
                prices_df = stonks.fetch_prices(ticker, api_key)
            except Exception as exc:
                message = str(exc)
                st.error(f"Price fetch failed: {message}")
                if "402" in message:
                    st.info("FMP returned 402 Payment Required. Upgrade your plan or use a supported endpoint.")
                st.stop()
        st.session_state.price_cache_key = price_cache_key
        st.session_state.price_cache_df = prices_df
    else:
        prices_df = st.session_state.price_cache_df

    if prices_df is None or len(prices_df) == 0:
        st.warning("No price rows returned for this ticker.")
        st.stop()

    if "date" in prices_df.columns:
        available_dates = pd.to_datetime(prices_df["date"], errors="coerce").dropna()
        if not available_dates.empty:
            available_start = available_dates.min().date()
            if start_date < available_start:
                st.info(
                    f"Requested start date {start_date} is earlier than available history ({available_start}). "
                    "Using available history only."
                )

    feature_cols = getattr(stonks, "PERIODIC_FEATURE_COLS", stonks.BIWEEKLY_FEATURE_COLS)

    diag_cache_key = (ticker, str(start_date), str(end_date), len(prices_df))
    if st.session_state.get("horizon_diag_key") != diag_cache_key:
        with st.spinner("Preparing horizon diagnostics..."):
            horizon_diagnostics = build_horizon_diagnostics(prices_df, start_date, end_date, feature_cols)
        st.session_state.horizon_diag_key = diag_cache_key
        st.session_state.horizon_diag_data = horizon_diagnostics
    else:
        horizon_diagnostics = st.session_state.horizon_diag_data

    horizon_selection = render_horizon_selector(horizon_diagnostics)
    if horizon_selection is None:
        st.stop()

    period_days, period_label = horizon_selection
    selected_diag = horizon_diagnostics.get(period_days)
    if not selected_diag:
        st.warning("Selected horizon diagnostics are unavailable. Re-select the horizon.")
        reset_selection_state(clear_ticker=False)
        st.stop()

    dataset = selected_diag.get("dataset", pd.DataFrame()).copy()
    if dataset.empty:
        st.warning("No usable periods after filtering and alignment.")
        st.stop()

    model_profile = dict(selected_diag.get("profile", {}))
    analysis_mode = "ml_forecast" if bool(model_profile.get("ml_available", False)) else "trend_only"
    mode_note = str(selected_diag.get("ml_unavailable_reason", "")).strip() if analysis_mode == "trend_only" else ""

    labeled_rows = int(model_profile.get("sample_size", 0))
    minimum_rows = int(model_profile.get("minimum_rows", 0))

    dataset["bull_bear"] = np.where(
        dataset["target_return"].isna(),
        "unknown",
        np.where(dataset["target_return"] >= 0, "bull", "bear"),
    )

    model_df = dataset.dropna(subset=feature_cols + ["target_return"]).copy()
    trend_summary = compute_trend_summary(dataset, period_days)

    metrics = pd.DataFrame()
    models: Dict[str, Any] = {}
    pred = None
    best_name = None
    forecast_row = None
    forecast_details: Dict[str, Any] = {
        "display_name": "n/a",
        "prediction_dispersion": np.nan,
    }

    if analysis_mode == "ml_forecast":
        metrics, models = stonks._evaluate_models(
            dataset,
            report_markdown=False,
            show_output=False,
            feature_cols=feature_cols,
            return_models=True,
            period_days=period_days,
        )

        pred, best_name, forecast_row, forecast_details = stonks.forecast_from_models(
            dataset,
            feature_cols,
            models,
            metrics,
            return_details=True,
        )

        if pred is None:
            analysis_mode = "trend_only"
            fallback_note = str(model_profile.get("gating_note", "")).strip()
            if fallback_note:
                mode_note = fallback_note

    direction = "neutral"
    strength = "n/a"
    statement = "Insufficient data to derive a forecast signal."
    pred_pct = "n/a"
    prediction_dispersion = float(forecast_details.get("prediction_dispersion", np.nan))
    forecast_display_name = str(forecast_details.get("display_name", "n/a"))

    if analysis_mode == "ml_forecast" and pred is not None:
        reference_std = dataset["target_return"].dropna().std()
        direction, strength, statement = stonks.describe_signal(pred, reference_std, prediction_dispersion)
        pred_pct = f"{pred * 100:.2f}%"
        signal_mode_title = "ML Forecast Mode"
        signal_subline = (
            f"{ticker} | {exchange_label} | {signal_mode_title} | "
            f"Predicted {period_label.lower()} return: {pred_pct}"
        )
    else:
        analysis_mode = "trend_only"
        regime = str(trend_summary.get("regime", "insufficient"))
        mean_return = float(trend_summary.get("mean_return", np.nan))
        annualized_vol = float(trend_summary.get("annualized_volatility", np.nan))
        max_drawdown = float(trend_summary.get("max_drawdown", np.nan))
        slope = float(trend_summary.get("slope_per_period", np.nan))
        bull_ratio = float(trend_summary.get("bull_ratio", np.nan))

        if regime == "bullish":
            direction = "bullish"
            strength = "strong" if bull_ratio >= 0.67 else "medium"
        elif regime == "bearish":
            direction = "bearish"
            strength = "strong" if bull_ratio <= 0.33 else "medium"
        else:
            direction = "neutral"
            strength = "weak"

        mean_text = f"{mean_return * 100:.2f}%" if np.isfinite(mean_return) else "n/a"
        vol_text = f"{annualized_vol * 100:.2f}%" if np.isfinite(annualized_vol) else "n/a"
        dd_text = f"{max_drawdown * 100:.2f}%" if np.isfinite(max_drawdown) else "n/a"
        slope_text = f"{slope * 100:.3f}%" if np.isfinite(slope) else "n/a"

        statement = (
            f"Trend-only summary: regime appears {regime}. "
            f"Mean return {mean_text}, annualized volatility {vol_text}, "
            f"max drawdown {dd_text}, slope per period {slope_text}."
        )
        signal_mode_title = "Trend-Only Mode"
        signal_subline = f"{ticker} | {exchange_label} | {signal_mode_title}"

    signal_color = "#24c16b" if direction == "bullish" else "#ea4d5a" if direction == "bearish" else "#9aa4b2"
    if direction != "neutral":
        signal_label = f"{strength.upper()} {direction.upper()} SIGNAL"
    elif analysis_mode == "ml_forecast":
        signal_label = "NO SIGNAL"
    else:
        signal_label = "TREND-ONLY SIGNAL"

    st.markdown(
        f"""
        <div class="signal-block">
            <div class="signal-title" style="color: {signal_color};">{signal_label}</div>
            <div class="signal-sub">{signal_subline}</div>
            <div class="signal-text">{statement}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    selected_models = model_profile.get("selected_models", [])
    model_stack_text = ", ".join(selected_models) if selected_models else "No eligible models"
    horizon_bucket = str(model_profile.get("horizon_bucket", "n/a"))
    mode_text = "ML Forecast" if analysis_mode == "ml_forecast" else "Trend-Only"

    mode_chip = f"Mode: {mode_text}"
    if analysis_mode == "trend_only":
        mode_chip = f"Mode: {mode_text} ({labeled_rows}/{minimum_rows} rows)"

    st.markdown(
        f"<div style='text-align:center; margin-top:0.1rem; margin-bottom:0.2rem;'>"
        f"<span style='display:inline-block; padding:0.35rem 0.75rem; border:1px solid #3b4557; border-radius:999px; color:#c9d3e3; font-size:0.88rem;'>{mode_chip}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )
    if mode_note:
        st.caption(mode_note)

    feature_importance_df = pd.DataFrame()
    rf_model = None
    rf_model_name: Optional[str] = None

    for candidate_name in ["RandomForest", "RandomForestShallow"]:
        if candidate_name in models:
            rf_model_name = candidate_name
            break

    if rf_model_name and not model_df.empty:
        rf_model = stonks.make_model(rf_model_name)
        rf_model.fit(model_df[feature_cols].to_numpy(), model_df["target_return"].to_numpy())
        if hasattr(rf_model, "feature_importances_"):
            feature_importance_df = pd.DataFrame(
                {
                    "feature": feature_cols,
                    "importance": rf_model.feature_importances_,
                }
            ).sort_values("importance", ascending=False)

    target_stats_df = pd.DataFrame()
    if not model_df.empty:
        target_stats_df = model_df["target_return"].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]).to_frame()
        target_stats_df = target_stats_df.reset_index().rename(columns={"index": "stat", "target_return": "value"})

    trend_summary_df = pd.DataFrame(
        [
            {"metric": "observations", "value": int(trend_summary.get("observations", 0))},
            {"metric": "mean_return_pct", "value": round(float(trend_summary.get("mean_return", np.nan)) * 100.0, 4)},
            {
                "metric": "slope_per_period_pct",
                "value": round(float(trend_summary.get("slope_per_period", np.nan)) * 100.0, 4),
            },
            {
                "metric": "annualized_volatility_pct",
                "value": round(float(trend_summary.get("annualized_volatility", np.nan)) * 100.0, 4),
            },
            {"metric": "max_drawdown_pct", "value": round(float(trend_summary.get("max_drawdown", np.nan)) * 100.0, 4)},
            {"metric": "bull_ratio_pct", "value": round(float(trend_summary.get("bull_ratio", np.nan)) * 100.0, 2)},
            {"metric": "regime", "value": str(trend_summary.get("regime", "insufficient"))},
        ]
    )

    strategy_df = pd.DataFrame()
    dates: List[pd.Timestamp] = []
    preds: List[float] = []
    actuals: List[float] = []
    if best_name is not None and metrics is not None and not metrics.empty:
        dates, preds, actuals = walk_forward_predictions(dataset, feature_cols, best_name)
        if len(dates) > 0:
            strategy_returns = [a if p > 0 else 0.0 for p, a in zip(preds, actuals)]
            strategy_df = pd.DataFrame(
                {
                    "event_date": dates,
                    "predicted_return": preds,
                    "actual_return": actuals,
                    "strategy_return": strategy_returns,
                }
            )

    forecast_info = {
        "direction": direction,
        "strength": strength,
        "statement": statement,
        "pred_pct": pred_pct,
        "analysis_mode": analysis_mode,
        "best_model": forecast_display_name if analysis_mode == "ml_forecast" and pred is not None else "Trend-Only Mode",
    }

    analysis_timestamp = datetime.now().strftime("%m%d%Y_%H%M%S")
    report_filename = f"{ticker}_{period_label.lower()}_{analysis_timestamp}_full_report.pdf"

    report_pdf = build_full_report_pdf(
        ticker=ticker,
        exchange_label=exchange_label,
        period_label=period_label,
        start_date=start_date,
        end_date=end_date,
        forecast_info=forecast_info,
        dataset=dataset,
        metrics=metrics,
        feature_importance_df=feature_importance_df,
        strategy_df=strategy_df,
    )

    tabs = st.tabs(["Summary", "Model / Trend Details", "Data", "Download"])

    with tabs[0]:
        center_chart_title = st.columns([1.3, 1.5, 1.3])[1]
        with center_chart_title:
            st.markdown('<h4 style="text-align:center; margin-bottom:0.2rem;">Core Trend Chart</h4>', unsafe_allow_html=True)

        trend_fig = build_bull_bear_chart(
            dataset,
            title=f"{ticker} {period_label} Trend Returns",
            x_label=f"{period_label} Period End",
            figsize=(6.8, 3.4),
        )
        center_chart = st.columns([1.3, 1.5, 1.3])[1]
        with center_chart:
            st.pyplot(trend_fig, use_container_width=False)

        if analysis_mode == "trend_only":
            st.dataframe(trend_summary_df, use_container_width=True)

    with tabs[1]:
        st.markdown(f"**Horizon Bucket:** {horizon_bucket}  ")
        st.markdown(f"**Model Stack:** {model_stack_text}")

        if analysis_mode == "ml_forecast" and metrics is not None and not metrics.empty:
            st.dataframe(metrics, use_container_width=True)

            detail_col1, detail_col2 = st.columns(2)
            with detail_col1:
                if rf_model is None:
                    st.info("Feature importance is available only when a RandomForest-based model is enabled.")
                else:
                    fi_fig = plot_feature_importance(rf_model, feature_cols, title=f"{period_label} Feature Importance")
                    if fi_fig is not None:
                        st.pyplot(fi_fig, use_container_width=True)
                    if not feature_importance_df.empty:
                        st.dataframe(feature_importance_df, use_container_width=True)

                    st.markdown("**SHAP Summary (optional)**")
                    try:
                        import shap  # type: ignore

                        sample_df = model_df
                        if len(sample_df) > 200:
                            sample_df = sample_df.sample(200, random_state=42)
                        explainer = shap.TreeExplainer(rf_model)
                        shap_values = explainer.shap_values(sample_df[feature_cols])
                        shap_fig = plt.figure(figsize=(7, 4))
                        shap.summary_plot(
                            shap_values,
                            sample_df[feature_cols],
                            feature_names=feature_cols,
                            show=False,
                        )
                        st.pyplot(shap_fig, use_container_width=True)
                    except Exception:
                        st.info("Install `shap` to display SHAP summary: `py -m pip install shap`")

            with detail_col2:
                if model_df.empty:
                    st.info("Insufficient data for distribution analysis.")
                else:
                    dist_fig = plot_target_distribution(
                        model_df["target_return"],
                        title=f"{period_label} Return Distribution",
                    )
                    st.pyplot(dist_fig, use_container_width=True)

                if len(dates) >= 2:
                    strategy_returns = [a if p > 0 else 0.0 for p, a in zip(preds, actuals)]
                    curve_fig = plot_strategy_curve(
                        dates,
                        strategy_returns,
                        actuals,
                        title=f"{period_label} Walk-Forward Strategy vs Buy & Hold",
                    )
                    if curve_fig is not None:
                        st.pyplot(curve_fig, use_container_width=True)
                else:
                    st.info("Not enough walk-forward points for strategy curve.")

            if not target_stats_df.empty:
                st.markdown("**Target Stats**")
                st.dataframe(target_stats_df, use_container_width=True)
        else:
            st.info("Trend-Only mode is active for this horizon.")
            st.dataframe(trend_summary_df, use_container_width=True)
            if not model_df.empty:
                dist_fig = plot_target_distribution(
                    model_df["target_return"],
                    title=f"{period_label} Return Distribution",
                )
                st.pyplot(dist_fig, use_container_width=True)

    with tabs[2]:
        st.dataframe(dataset.head(50), use_container_width=True)
        if analysis_mode == "ml_forecast" and metrics is not None and not metrics.empty:
            st.markdown("**Model Metrics Snapshot**")
            st.dataframe(metrics, use_container_width=True)

    with tabs[3]:
        center_download = st.columns([1, 2, 1])[1]
        with center_download:
            st.download_button(
                "Download Full Analysis Report",
                data=report_pdf,
                file_name=report_filename,
                mime="application/pdf",
                use_container_width=True,
            )

        if save_local:
            os.makedirs(output_dir, exist_ok=True)
            dataset_path = os.path.join(output_dir, f"{ticker}_{period_label.lower()}_dataset.csv")
            dataset.to_csv(dataset_path, index=False)

            metrics_path = None
            if metrics is not None and not metrics.empty:
                metrics_path = os.path.join(output_dir, f"{ticker}_{period_label.lower()}_model_metrics.csv")
                metrics.to_csv(metrics_path, index=False)

            trend_path = stonks.save_bull_bear_plot(
                dataset,
                output_dir,
                ticker,
                title=f"{ticker} {period_label} Trend Returns",
            )

            report_path = os.path.join(output_dir, report_filename)
            with open(report_path, "wb") as f:
                f.write(report_pdf)

            st.subheader("Saved Locally")
            st.write(f"Dataset: {dataset_path}")
            if metrics_path:
                st.write(f"Metrics: {metrics_path}")
            st.write(f"Trend Plot: {trend_path}")
            st.write(f"Full Report: {report_path}")



if __name__ == "__main__":
    main()


