import time
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf


# ============================================================
# Page config
# ============================================================

st.set_page_config(
    page_title="Market Open Analyzer - Free + Today Plan",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# Style
# ============================================================

CUSTOM_CSS = """
<style>
html, body, [class*="css"] {
    direction: rtl;
    text-align: right;
    font-family: Arial, Helvetica, sans-serif;
}
.main-title {
    font-size: 42px;
    font-weight: 800;
    margin-bottom: 0px;
}
.subtitle {
    font-size: 18px;
    color: #666;
    margin-top: 0px;
    margin-bottom: 24px;
}
.card {
    padding: 18px;
    border-radius: 18px;
    background: linear-gradient(135deg, #f8fafc 0%, #eef2ff 100%);
    border: 1px solid #e5e7eb;
    box-shadow: 0px 8px 22px rgba(15, 23, 42, 0.06);
    margin-bottom: 12px;
}
.warning-card {
    padding: 16px;
    border-radius: 16px;
    background: #fff7ed;
    border: 1px solid #fed7aa;
    color: #7c2d12;
    margin-bottom: 12px;
}
.good-card {
    padding: 16px;
    border-radius: 16px;
    background: #ecfdf5;
    border: 1px solid #bbf7d0;
    color: #064e3b;
    margin-bottom: 12px;
}
.small-muted {
    color: #64748b;
    font-size: 14px;
}
div[data-testid="stMetricValue"] {
    direction: ltr;
    text-align: center;
}
div[data-testid="stMetricLabel"] {
    text-align: center;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ============================================================
# Constants
# ============================================================

DEFAULT_TICKERS = {
    "Apple": "AAPL",
    "Nvidia": "NVDA",
    "Tesla": "TSLA",
    "Microsoft": "MSFT",
    "Amazon": "AMZN",
    "Meta": "META",
    "Alphabet / Google": "GOOGL",
    "AMD": "AMD",
    "Micron": "MU",
    "Netflix": "NFLX",
    "Nasdaq-100 ETF": "QQQ",
    "S&P 500 ETF": "SPY",
    "S&P 500 ETF - Vanguard": "VOO",
    "Nasdaq-100 ETF - QQQM": "QQQM",
}

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

CUSTOM_TICKERS_FILE = DATA_DIR / "custom_tickers.txt"


def normalize_ticker(ticker: str) -> str:
    """
    Normalize a user-entered ticker.
    Examples:
    ' aapl ' -> 'AAPL'
    'nasdaq:qqq' -> 'QQQ'
    """
    if ticker is None:
        return ""
    t = str(ticker).strip().upper()
    if ":" in t:
        t = t.split(":")[-1].strip()
    t = t.replace(" ", "")
    return t


def load_custom_tickers() -> list[str]:
    """
    Load custom tickers saved by the user.
    """
    if not CUSTOM_TICKERS_FILE.exists():
        return []

    items = []
    for line in CUSTOM_TICKERS_FILE.read_text(encoding="utf-8", errors="ignore").splitlines():
        t = normalize_ticker(line)
        if t:
            items.append(t)

    return sorted(set(items))


def save_custom_ticker(ticker: str) -> bool:
    """
    Save a custom ticker to local file.
    Returns True if added, False if it already existed or invalid.
    """
    t = normalize_ticker(ticker)
    if not t:
        return False

    existing = set(load_custom_tickers())
    built_in = set(DEFAULT_TICKERS.values())

    if t in existing or t in built_in:
        return False

    existing.add(t)
    CUSTOM_TICKERS_FILE.write_text("\n".join(sorted(existing)) + "\n", encoding="utf-8")
    return True


def get_all_ticker_options() -> dict[str, str]:
    """
    Merge default tickers with user-added custom tickers.
    Keys are display names, values are ticker symbols.
    """
    options = dict(DEFAULT_TICKERS)
    for t in load_custom_tickers():
        options[f"Custom - {t}"] = t
    return options




@dataclass
class AnalyzerConfig:
    days_back: int = 59
    bar_minutes: int = 5
    first_window_minutes: int = 30
    moderate_threshold_pct: float = 0.30
    sharp_threshold_pct: float = 0.80
    continuation_fraction: float = 0.50
    retrace_fraction: float = 0.50
    min_samples: int = 5


# ============================================================
# Data fetching: yfinance
# ============================================================

@st.cache_data(show_spinner=False, ttl=60 * 10)
def fetch_intraday_yfinance(
    ticker: str,
    days_back: int,
    interval_minutes: int,
) -> pd.DataFrame:
    """
    Fetch recent intraday bars with yfinance.
    yfinance intraday data is limited to recent history, so this app is for testing.
    """

    interval = f"{interval_minutes}m"

    # Yahoo/yfinance usually limits 1-minute candles to a short recent window.
    # To avoid download errors, the app automatically limits 1m to 7 days.
    effective_days_back = int(days_back)
    if int(interval_minutes) == 1:
        effective_days_back = min(effective_days_back, 7)

    period = f"{effective_days_back}d"

    df = yf.download(
        tickers=ticker,
        period=period,
        interval=interval,
        auto_adjust=True,
        prepost=False,
        progress=False,
        threads=False,
    )

    if df is None or df.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    # yfinance may return MultiIndex columns.
    if isinstance(df.columns, pd.MultiIndex):
        # Usually columns look like ("Close", "AAPL") or ("AAPL", "Close").
        try:
            if ticker in df.columns.get_level_values(0):
                df = df[ticker]
            elif ticker in df.columns.get_level_values(1):
                df = df.xs(ticker, axis=1, level=1)
            else:
                df.columns = df.columns.get_level_values(-1)
        except Exception:
            df.columns = df.columns.get_level_values(-1)

    rename_map = {
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "adj_close",
        "Volume": "volume",
    }
    df = df.rename(columns=rename_map)

    needed = ["open", "high", "low", "close", "volume"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing columns from yfinance data: {missing}")

    df = df[needed].dropna()

    # Make sure timezone is America/New_York.
    if df.index.tz is None:
        df.index = df.index.tz_localize("America/New_York")
    else:
        df.index = df.index.tz_convert("America/New_York")

    df = df.sort_index()
    return df


# ============================================================
# Market hours
# ============================================================

def filter_regular_market_hours(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    return df.between_time("09:30", "15:59")


def get_latest_trading_day(df: pd.DataFrame) -> pd.DataFrame:
    df = filter_regular_market_hours(df)
    if df.empty:
        return df
    last_day = max(df.index.date)
    return df[df.index.date == last_day]


# ============================================================
# Classification logic
# ============================================================

def classify_opening_magnitude(initial_move_pct: float, cfg: AnalyzerConfig) -> str:
    if initial_move_pct >= cfg.sharp_threshold_pct:
        return "sharp_up"
    if initial_move_pct >= cfg.moderate_threshold_pct:
        return "moderate_up"
    if initial_move_pct <= -cfg.sharp_threshold_pct:
        return "sharp_down"
    if initial_move_pct <= -cfg.moderate_threshold_pct:
        return "moderate_down"
    return "flat_open"


def classify_opening_shape(first_window: pd.DataFrame) -> str:
    if first_window.empty:
        return "unknown"

    open_price = float(first_window.iloc[0]["open"])
    close_price = float(first_window.iloc[-1]["close"])
    high_price = float(first_window["high"].max())
    low_price = float(first_window["low"].min())

    total_range = high_price - low_price
    if total_range <= 0:
        return "quiet_flat"

    body = close_price - open_price
    body_abs = abs(body)
    efficiency = body_abs / total_range
    close_location = (close_price - low_price) / total_range
    range_pct = (total_range / open_price) * 100
    move_pct = (close_price / open_price - 1) * 100

    if abs(move_pct) < 0.30:
        if range_pct >= 0.80:
            return "volatile_chop"
        return "quiet_flat"

    if close_price > open_price:
        if efficiency >= 0.60 and close_location >= 0.75:
            return "clean_up"
        if close_location <= 0.55:
            return "up_rejected"
        return "choppy_up"

    if close_price < open_price:
        if efficiency >= 0.60 and close_location <= 0.25:
            return "clean_down"
        if close_location >= 0.45:
            return "down_rejected"
        return "choppy_down"

    return "unknown"


def classify_volume_relative(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty or "opening_volume" not in results.columns:
        return results

    avg_volume = results["opening_volume"].mean()

    if avg_volume <= 0 or pd.isna(avg_volume):
        results["opening_volume_class"] = "unknown_volume"
        return results

    def classify(v):
        if v >= avg_volume * 1.30:
            return "high_volume"
        if v <= avg_volume * 0.70:
            return "low_volume"
        return "normal_volume"

    results["opening_volume_class"] = results["opening_volume"].apply(classify)
    return results


# ============================================================
# Day analysis
# ============================================================

def analyze_single_day(day_df: pd.DataFrame, cfg: AnalyzerConfig) -> dict | None:
    if day_df.empty:
        return None

    day_df = day_df.sort_index()

    market_open_price = float(day_df.iloc[0]["open"])
    first_window_end = day_df.index[0] + pd.Timedelta(minutes=cfg.first_window_minutes)

    first_window = day_df[day_df.index < first_window_end]
    after_window = day_df[day_df.index >= first_window_end]

    if first_window.empty or after_window.empty:
        return None

    first_window_close = float(first_window.iloc[-1]["close"])
    initial_move_pct = (first_window_close / market_open_price - 1) * 100

    day_close = float(day_df.iloc[-1]["close"])
    day_high = float(day_df["high"].max())
    day_low = float(day_df["low"].min())

    opening_magnitude = classify_opening_magnitude(initial_move_pct, cfg)
    opening_shape = classify_opening_shape(first_window)
    opening_type = f"{opening_magnitude}_{opening_shape}"
    opening_volume = float(first_window["volume"].sum())

    result = {
        "date": str(day_df.index[0].date()),
        "open": market_open_price,
        "first_window_close": first_window_close,
        "initial_move_pct": initial_move_pct,
        "opening_magnitude": opening_magnitude,
        "opening_shape": opening_shape,
        "opening_type": opening_type,
        "opening_volume": opening_volume,
        "day_close": day_close,
        "day_high": day_high,
        "day_low": day_low,
        "eod_change_pct": (day_close / market_open_price - 1) * 100,
    }

    if initial_move_pct > 0:
        first_move_points = first_window_close - market_open_price
        continuation_level = first_window_close + cfg.continuation_fraction * first_move_points
        retrace_level = market_open_price + (1 - cfg.retrace_fraction) * first_move_points

        continuation_hits = after_window[after_window["high"] >= continuation_level]
        retrace_hits = after_window[after_window["low"] <= retrace_level]

        continuation_time = continuation_hits.index.min() if not continuation_hits.empty else None
        retrace_time = retrace_hits.index.min() if not retrace_hits.empty else None

        result["continuation_level"] = continuation_level
        result["retrace_half_level"] = retrace_level

        if continuation_time is not None and retrace_time is not None:
            if continuation_time < retrace_time:
                result["after_result"] = "continued_first"
            elif retrace_time < continuation_time:
                result["after_result"] = "half_retrace_first"
            else:
                result["after_result"] = "both_same_bar_unknown_order"
        elif continuation_time is not None:
            result["after_result"] = "continued_only"
        elif retrace_time is not None:
            result["after_result"] = "half_retrace_only"
        else:
            result["after_result"] = "range_no_clear_move"

    elif initial_move_pct < 0:
        first_move_points = market_open_price - first_window_close
        continuation_level = first_window_close - cfg.continuation_fraction * first_move_points
        retrace_level = market_open_price - (1 - cfg.retrace_fraction) * first_move_points

        continuation_hits = after_window[after_window["low"] <= continuation_level]
        retrace_hits = after_window[after_window["high"] >= retrace_level]

        continuation_time = continuation_hits.index.min() if not continuation_hits.empty else None
        retrace_time = retrace_hits.index.min() if not retrace_hits.empty else None

        result["continuation_level"] = continuation_level
        result["retrace_half_level"] = retrace_level

        if continuation_time is not None and retrace_time is not None:
            if continuation_time < retrace_time:
                result["after_result"] = "continued_down_first"
            elif retrace_time < continuation_time:
                result["after_result"] = "half_rebound_first"
            else:
                result["after_result"] = "both_same_bar_unknown_order"
        elif continuation_time is not None:
            result["after_result"] = "continued_down_only"
        elif retrace_time is not None:
            result["after_result"] = "half_rebound_only"
        else:
            result["after_result"] = "range_no_clear_move"
    else:
        result["after_result"] = "not_tested"

    return result


def analyze_history(df: pd.DataFrame, cfg: AnalyzerConfig) -> pd.DataFrame:
    df = filter_regular_market_hours(df)

    if df.empty:
        return pd.DataFrame()

    rows = []
    for _, day_df in df.groupby(df.index.date):
        analyzed = analyze_single_day(day_df, cfg)
        if analyzed is not None:
            rows.append(analyzed)

    results = pd.DataFrame(rows)

    if not results.empty:
        results = classify_volume_relative(results)
        results["opening_type_with_volume"] = (
            results["opening_type"] + "_" + results["opening_volume_class"]
        )

    return results


# ============================================================
# Summaries
# ============================================================

def probability_summary_by_opening_type(results: pd.DataFrame, min_samples: int = 5) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame()

    counts_per_type = results.groupby("opening_type").size().reset_index(name="total_cases")

    summary = (
        results.groupby(["opening_type", "after_result"])
        .size()
        .reset_index(name="count")
    )

    summary = summary.merge(counts_per_type, on="opening_type", how="left")
    summary = summary[summary["total_cases"] >= min_samples].copy()

    if summary.empty:
        return summary

    summary["probability_pct"] = summary["count"] / summary["total_cases"] * 100
    summary = summary.sort_values(["opening_type", "probability_pct"], ascending=[True, False])
    return summary


def probability_summary_by_type_and_volume(results: pd.DataFrame, min_samples: int = 5) -> pd.DataFrame:
    if results.empty or "opening_type_with_volume" not in results.columns:
        return pd.DataFrame()

    counts_per_type = (
        results.groupby("opening_type_with_volume")
        .size()
        .reset_index(name="total_cases")
    )

    summary = (
        results.groupby(["opening_type_with_volume", "after_result"])
        .size()
        .reset_index(name="count")
    )

    summary = summary.merge(counts_per_type, on="opening_type_with_volume", how="left")
    summary = summary[summary["total_cases"] >= min_samples].copy()

    if summary.empty:
        return summary

    summary["probability_pct"] = summary["count"] / summary["total_cases"] * 100
    summary = summary.sort_values(
        ["opening_type_with_volume", "probability_pct"],
        ascending=[True, False],
    )
    return summary


def eod_summary_by_opening_type(results: pd.DataFrame, min_samples: int = 5) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame()

    grouped = results.groupby("opening_type")

    summary = grouped.agg(
        total_cases=("opening_type", "size"),
        avg_eod_change_pct=("eod_change_pct", "mean"),
        median_eod_change_pct=("eod_change_pct", "median"),
        win_rate_eod_green=("eod_change_pct", lambda s: (s > 0).mean() * 100),
        avg_initial_move_pct=("initial_move_pct", "mean"),
    ).reset_index()

    summary = summary[summary["total_cases"] >= min_samples].copy()
    summary = summary.sort_values("total_cases", ascending=False)
    return summary


# ============================================================
# Current day
# ============================================================

def classify_current_opening(latest_day_df: pd.DataFrame, cfg: AnalyzerConfig) -> dict | None:
    latest_day_df = filter_regular_market_hours(latest_day_df)

    if latest_day_df.empty:
        return None

    latest_day_df = latest_day_df.sort_index()

    market_open_price = float(latest_day_df.iloc[0]["open"])
    current_price = float(latest_day_df.iloc[-1]["close"])
    current_time = latest_day_df.index[-1]

    first_window_end = latest_day_df.index[0] + pd.Timedelta(minutes=cfg.first_window_minutes)
    first_window = latest_day_df[latest_day_df.index < first_window_end]

    if first_window.empty:
        return None

    first_window_close = float(first_window.iloc[-1]["close"])

    initial_move_pct = (first_window_close / market_open_price - 1) * 100
    current_move_pct = (current_price / market_open_price - 1) * 100

    opening_magnitude = classify_opening_magnitude(initial_move_pct, cfg)
    opening_shape = classify_opening_shape(first_window)
    opening_type = f"{opening_magnitude}_{opening_shape}"

    expected_bars = max(1, int(cfg.first_window_minutes / cfg.bar_minutes))
    bars_count = len(first_window)
    is_complete_window = bars_count >= expected_bars

    return {
        "date": str(latest_day_df.index[0].date()),
        "current_time": str(current_time),
        "open_price": market_open_price,
        "current_price": current_price,
        "first_window_close": first_window_close,
        "initial_move_pct": initial_move_pct,
        "current_move_pct": current_move_pct,
        "opening_magnitude": opening_magnitude,
        "opening_shape": opening_shape,
        "opening_type": opening_type,
        "opening_volume": float(first_window["volume"].sum()),
        "bars_count": bars_count,
        "expected_bars": expected_bars,
        "is_complete_window": is_complete_window,
    }


def compare_current_to_history(current_info: dict, summary_by_type: pd.DataFrame) -> pd.DataFrame:
    if current_info is None or summary_by_type.empty:
        return pd.DataFrame()

    opening_type = current_info["opening_type"]
    match = summary_by_type[summary_by_type["opening_type"] == opening_type].copy()

    if match.empty:
        return pd.DataFrame()

    return match.sort_values("probability_pct", ascending=False)


# ============================================================
# Today status + educational trade plan
# ============================================================

def build_today_status(latest_day_df: pd.DataFrame, cfg: AnalyzerConfig) -> dict | None:
    """
    Summarizes what actually happened today / latest trading day.
    This is descriptive only, not a trading recommendation.
    """

    day_df = filter_regular_market_hours(latest_day_df)

    if day_df.empty:
        return None

    day_df = day_df.sort_index()

    open_price = float(day_df.iloc[0]["open"])
    current_price = float(day_df.iloc[-1]["close"])
    day_high = float(day_df["high"].max())
    day_low = float(day_df["low"].min())
    day_volume = float(day_df["volume"].sum())

    first_window_end = day_df.index[0] + pd.Timedelta(minutes=cfg.first_window_minutes)
    first_window = day_df[day_df.index < first_window_end]
    after_window = day_df[day_df.index >= first_window_end]

    if first_window.empty:
        return None

    first_open = float(first_window.iloc[0]["open"])
    first_close = float(first_window.iloc[-1]["close"])
    first_high = float(first_window["high"].max())
    first_low = float(first_window["low"].min())
    first_mid = (first_high + first_low) / 2

    initial_move_pct = (first_close / first_open - 1) * 100
    current_move_pct = (current_price / open_price - 1) * 100

    day_range = day_high - day_low
    if day_range > 0:
        current_position_in_range_pct = (current_price - day_low) / day_range * 100
    else:
        current_position_in_range_pct = np.nan

    opening_direction = "up" if initial_move_pct > 0 else "down" if initial_move_pct < 0 else "flat"

    continuation_level = np.nan
    retrace_half_level = np.nan
    continuation_hit = False
    retrace_half_hit = False
    today_after_status = "not_enough_data_after_opening_window"

    if not after_window.empty and initial_move_pct > 0:
        first_move_points = first_close - first_open
        continuation_level = first_close + cfg.continuation_fraction * first_move_points
        retrace_half_level = first_open + (1 - cfg.retrace_fraction) * first_move_points

        continuation_hits = after_window[after_window["high"] >= continuation_level]
        retrace_hits = after_window[after_window["low"] <= retrace_half_level]

        continuation_time = continuation_hits.index.min() if not continuation_hits.empty else None
        retrace_time = retrace_hits.index.min() if not retrace_hits.empty else None

        continuation_hit = continuation_time is not None
        retrace_half_hit = retrace_time is not None

        if continuation_time is not None and retrace_time is not None:
            if continuation_time < retrace_time:
                today_after_status = "continued_first_today"
            elif retrace_time < continuation_time:
                today_after_status = "half_retrace_first_today"
            else:
                today_after_status = "both_same_bar_unknown_order_today"
        elif continuation_time is not None:
            today_after_status = "continued_only_today"
        elif retrace_time is not None:
            today_after_status = "half_retrace_only_today"
        else:
            today_after_status = "no_clear_followthrough_yet"

    elif not after_window.empty and initial_move_pct < 0:
        first_move_points = first_open - first_close
        continuation_level = first_close - cfg.continuation_fraction * first_move_points
        retrace_half_level = first_open - (1 - cfg.retrace_fraction) * first_move_points

        continuation_hits = after_window[after_window["low"] <= continuation_level]
        retrace_hits = after_window[after_window["high"] >= retrace_half_level]

        continuation_time = continuation_hits.index.min() if not continuation_hits.empty else None
        retrace_time = retrace_hits.index.min() if not retrace_hits.empty else None

        continuation_hit = continuation_time is not None
        retrace_half_hit = retrace_time is not None

        if continuation_time is not None and retrace_time is not None:
            if continuation_time < retrace_time:
                today_after_status = "continued_down_first_today"
            elif retrace_time < continuation_time:
                today_after_status = "half_rebound_first_today"
            else:
                today_after_status = "both_same_bar_unknown_order_today"
        elif continuation_time is not None:
            today_after_status = "continued_down_only_today"
        elif retrace_time is not None:
            today_after_status = "half_rebound_only_today"
        else:
            today_after_status = "no_clear_followthrough_yet"

    return {
        "date": str(day_df.index[0].date()),
        "last_bar_time": str(day_df.index[-1]),
        "open_price": open_price,
        "current_price": current_price,
        "current_move_pct": current_move_pct,
        "day_high": day_high,
        "day_low": day_low,
        "day_volume": day_volume,
        "first_window_open": first_open,
        "first_window_close": first_close,
        "first_window_high": first_high,
        "first_window_low": first_low,
        "first_window_mid": first_mid,
        "initial_move_pct": initial_move_pct,
        "opening_direction": opening_direction,
        "current_position_in_range_pct": current_position_in_range_pct,
        "continuation_level": continuation_level,
        "retrace_half_level": retrace_half_level,
        "continuation_hit": continuation_hit,
        "retrace_half_hit": retrace_half_hit,
        "today_after_status": today_after_status,
    }


def _probability_from_match(match: pd.DataFrame, keys: list[str]) -> float:
    if match is None or match.empty:
        return 0.0
    filtered = match[match["after_result"].isin(keys)]
    if filtered.empty:
        return 0.0
    return float(filtered["probability_pct"].sum())


def build_educational_trade_plan(
    current_info: dict,
    today_status: dict,
    match: pd.DataFrame,
    cfg: AnalyzerConfig,
) -> dict:
    """
    Builds a rule-based educational scenario.
    It intentionally avoids saying 'buy/sell now'.
    """

    no_trade = {
        "bias": "NO_TRADE",
        "title_he": "אין עסקה נקייה כרגע",
        "direction_he": "להמתין",
        "confidence_he": "נמוכה",
        "entry_zone": "אין כניסה",
        "trigger": "להמתין לאישור מחיר ברור",
        "stop": "לא רלוונטי",
        "target_1": "לא רלוונטי",
        "target_2": "לא רלוונטי",
        "time_plan": "לא להחזיק עסקה בלי תוכנית מסודרת",
        "reason": "אין מספיק נתונים או שאין יתרון סטטיסטי ברור.",
        "long_probability": 0.0,
        "short_probability": 0.0,
        "edge_points": 0.0,
    }

    if current_info is None or today_status is None:
        return no_trade

    if not current_info.get("is_complete_window", False):
        out = no_trade.copy()
        out["reason"] = "חלון הפתיחה עדיין לא הושלם. עדיף להמתין עד שיש מספיק נרות."
        return out

    if match is None or match.empty:
        out = no_trade.copy()
        out["reason"] = "אין מספיק היסטוריה לסוג הפתיחה הנוכחי."
        return out

    opening_direction = today_status["opening_direction"]
    first_high = float(today_status["first_window_high"])
    first_low = float(today_status["first_window_low"])
    first_mid = float(today_status["first_window_mid"])
    current_price = float(today_status["current_price"])

    opening_range = max(first_high - first_low, current_price * 0.001)
    buffer_value = current_price * 0.0005

    if opening_direction == "up":
        long_prob = _probability_from_match(match, ["continued_first", "continued_only"])
        short_prob = _probability_from_match(match, ["half_retrace_first", "half_retrace_only"])
    elif opening_direction == "down":
        short_prob = _probability_from_match(match, ["continued_down_first", "continued_down_only"])
        long_prob = _probability_from_match(match, ["half_rebound_first", "half_rebound_only"])
    else:
        long_prob = 0.0
        short_prob = 0.0

    edge = abs(long_prob - short_prob)

    # We require a clear edge. Otherwise no trade.
    if max(long_prob, short_prob) < 50 or edge < 12:
        out = no_trade.copy()
        out["reason"] = (
            f"אין יתרון מספיק ברור: הסתברות לונג {long_prob:.1f}% מול שורט {short_prob:.1f}%."
        )
        out["long_probability"] = long_prob
        out["short_probability"] = short_prob
        out["edge_points"] = edge
        return out

    if long_prob > short_prob:
        entry_low = first_high
        entry_high = first_high + buffer_value
        stop_price = min(first_mid, first_low - buffer_value)
        risk = max(entry_high - stop_price, opening_range * 0.35)
        target_1 = entry_high + risk
        target_2 = entry_high + 2 * risk

        return {
            "bias": "LONG_WATCH",
            "title_he": "תרחיש לימודי: לונג רק אם יש פריצה/ריטסט",
            "direction_he": "לונג",
            "confidence_he": "בינונית" if edge < 25 else "גבוהה יחסית",
            "entry_zone": f"{entry_low:.2f} עד {entry_high:.2f}",
            "trigger": "כניסה רק אם המחיר פורץ מעל הגבוה של חלון הפתיחה או חוזר לבדוק אותו ומחזיק מעליו.",
            "stop": f"מתחת לאמצע/נמוך חלון הפתיחה: בערך {stop_price:.2f}",
            "target_1": f"יעד 1R: בערך {target_1:.2f}",
            "target_2": f"יעד 2R: בערך {target_2:.2f}",
            "time_plan": "עסקת intraday בלבד: לבדוק 30-120 דקות אחרי הטריגר או לסגור לפני סוף המסחר.",
            "reason": f"לפי מקרים דומים: המשך למעלה {long_prob:.1f}% מול תיקון/שורט {short_prob:.1f}%.",
            "long_probability": long_prob,
            "short_probability": short_prob,
            "edge_points": edge,
        }

    entry_high = first_low
    entry_low = first_low - buffer_value
    stop_price = max(first_mid, first_high + buffer_value)
    risk = max(stop_price - entry_low, opening_range * 0.35)
    target_1 = entry_low - risk
    target_2 = entry_low - 2 * risk

    return {
        "bias": "SHORT_WATCH",
        "title_he": "תרחיש לימודי: שורט רק אם יש שבירה/ריטסט",
        "direction_he": "שורט",
        "confidence_he": "בינונית" if edge < 25 else "גבוהה יחסית",
        "entry_zone": f"{entry_low:.2f} עד {entry_high:.2f}",
        "trigger": "כניסה רק אם המחיר שובר מתחת לנמוך של חלון הפתיחה או חוזר לבדוק אותו מלמטה ונכשל.",
        "stop": f"מעל האמצע/גבוה חלון הפתיחה: בערך {stop_price:.2f}",
        "target_1": f"יעד 1R: בערך {target_1:.2f}",
        "target_2": f"יעד 2R: בערך {target_2:.2f}",
        "time_plan": "עסקת intraday בלבד: לבדוק 30-120 דקות אחרי הטריגר או לסגור לפני סוף המסחר.",
        "reason": f"לפי מקרים דומים: המשך/שורט {short_prob:.1f}% מול לונג/תיקון {long_prob:.1f}%.",
        "long_probability": long_prob,
        "short_probability": short_prob,
        "edge_points": edge,
    }



# ============================================================
# Invest now / current moment educational scenario
# ============================================================

def add_realtime_indicators(day_df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds intraday indicators:
    EMA20, EMA50, VWAP, RSI14, MACD, Bollinger Bands,
    relative volume, ATR-like range and day-range position.
    """

    df = filter_regular_market_hours(day_df).copy()

    if df.empty:
        return df

    df = df.sort_index()

    # Trend / mean indicators
    df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()

    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    cumulative_pv = (typical_price * df["volume"]).cumsum()
    cumulative_volume = df["volume"].replace(0, np.nan).cumsum()
    df["vwap"] = cumulative_pv / cumulative_volume

    # RSI 14
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi14"] = 100 - (100 / (1 + rs))

    # MACD 12/26/9
    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    # Bollinger Bands 20, 2 std
    bb_mid = df["close"].rolling(20, min_periods=20).mean()
    bb_std = df["close"].rolling(20, min_periods=20).std()
    df["bb_mid"] = bb_mid
    df["bb_upper"] = bb_mid + 2 * bb_std
    df["bb_lower"] = bb_mid - 2 * bb_std
    df["bb_width_pct"] = ((df["bb_upper"] - df["bb_lower"]) / df["bb_mid"]) * 100
    df["bb_position"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])

    # Volume quality
    df["volume_ma20"] = df["volume"].rolling(20, min_periods=5).mean()
    df["relative_volume"] = df["volume"] / df["volume_ma20"].replace(0, np.nan)

    # ATR-like intraday range
    df["bar_range"] = df["high"] - df["low"]
    df["atr_like"] = df["bar_range"].rolling(14, min_periods=3).mean()

    # Day position
    day_high_expanding = df["high"].expanding().max()
    day_low_expanding = df["low"].expanding().min()
    df["day_range_position"] = (df["close"] - day_low_expanding) / (day_high_expanding - day_low_expanding).replace(0, np.nan)

    return df


def _safe_float(value, default=np.nan) -> float:
    try:
        if pd.isna(value):
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def build_invest_now_plan(latest_day_df: pd.DataFrame, cfg: AnalyzerConfig) -> dict:
    """
    Builds a rules-based educational current-moment scenario.

    This is not investment advice and not a buy/sell instruction.
    It is a structured paper-trading scenario using current available data.
    """

    df = add_realtime_indicators(latest_day_df)

    base = {
        "bias": "NO_TRADE",
        "title_he": "אין עסקה נקייה עכשיו",
        "direction_he": "להמתין",
        "confidence_he": "נמוכה",
        "setup_score": 0,
        "setup_quality": "חלש",
        "current_price": np.nan,
        "current_time": "",
        "trend_state": "לא ידוע",
        "entry_zone": "אין כניסה",
        "trigger": "להמתין לאישור מחיר ברור",
        "stop": "לא רלוונטי",
        "target_1": "לא רלוונטי",
        "target_2": "לא רלוונטי",
        "time_plan": "לא להחזיק עסקה בלי תוכנית מסודרת.",
        "reason": "אין מספיק נתונים או שהשוק לא בכיוון ברור.",
        "filters_summary": "",
        "risk_reward": "לא רלוונטי",
        "distance_from_vwap_pct": np.nan,
        "distance_from_ema20_pct": np.nan,
        "ema20": np.nan,
        "ema50": np.nan,
        "vwap": np.nan,
        "rsi14": np.nan,
        "macd_hist": np.nan,
        "relative_volume": np.nan,
        "bb_position": np.nan,
        "bb_width_pct": np.nan,
        "day_range_position": np.nan,
        "atr_like": np.nan,
        "day_open": np.nan,
        "day_high": np.nan,
        "day_low": np.nan,
        "current_move_pct": np.nan,
    }

    if df.empty or len(df) < 30:
        out = base.copy()
        out["reason"] = "אין מספיק נרות היום כדי לבנות תרחיש סביר. צריך לפחות 30 נרות."
        return out

    last = df.iloc[-1]
    prev = df.iloc[-2] if len(df) >= 2 else last

    current_price = _safe_float(last["close"])
    current_time = str(df.index[-1])
    ema20 = _safe_float(last["ema20"])
    ema50 = _safe_float(last["ema50"])
    vwap = _safe_float(last["vwap"])
    rsi14 = _safe_float(last["rsi14"])
    macd_hist = _safe_float(last["macd_hist"])
    prev_macd_hist = _safe_float(prev["macd_hist"])
    relative_volume = _safe_float(last["relative_volume"])
    bb_position = _safe_float(last["bb_position"])
    bb_width_pct = _safe_float(last["bb_width_pct"])
    day_range_position = _safe_float(last["day_range_position"])
    atr_like = _safe_float(last["atr_like"], current_price * 0.003)

    day_open = _safe_float(df.iloc[0]["open"])
    day_high = _safe_float(df["high"].max())
    day_low = _safe_float(df["low"].min())
    current_move_pct = (current_price / day_open - 1) * 100 if day_open > 0 else np.nan

    last_10_high = _safe_float(df["high"].tail(10).max())
    last_10_low = _safe_float(df["low"].tail(10).min())
    last_20_high = _safe_float(df["high"].tail(20).max())
    last_20_low = _safe_float(df["low"].tail(20).min())

    prev_high = _safe_float(prev["high"])
    prev_low = _safe_float(prev["low"])

    distance_from_vwap_pct = (current_price / vwap - 1) * 100 if vwap > 0 else np.nan
    distance_from_ema20_pct = (current_price / ema20 - 1) * 100 if ema20 > 0 else np.nan

    # Build confluence score. Positive = long bias, negative = short bias.
    score = 0
    filters = []

    if current_price > vwap:
        score += 1
        filters.append("מחיר מעל VWAP")
    elif current_price < vwap:
        score -= 1
        filters.append("מחיר מתחת VWAP")

    if current_price > ema20 > ema50:
        score += 2
        filters.append("EMA20 מעל EMA50 והמחיר מעליהם")
    elif current_price < ema20 < ema50:
        score -= 2
        filters.append("EMA20 מתחת EMA50 והמחיר מתחתיהם")
    elif current_price > ema20:
        score += 1
        filters.append("מחיר מעל EMA20")
    elif current_price < ema20:
        score -= 1
        filters.append("מחיר מתחת EMA20")

    if pd.notna(rsi14):
        if 45 <= rsi14 <= 68:
            score += 1
            filters.append("RSI תומך בלונג ללא קניות יתר קיצוניות")
        elif 32 <= rsi14 <= 55:
            score -= 1
            filters.append("RSI תומך בשורט ללא מכירות יתר קיצוניות")
        elif rsi14 > 75:
            score -= 1
            filters.append("RSI גבוה מדי — חשש לרדיפה בלונג")
        elif rsi14 < 25:
            score += 1
            filters.append("RSI נמוך מדי — חשש לרדיפה בשורט")

    if pd.notna(macd_hist):
        if macd_hist > 0 and macd_hist >= prev_macd_hist:
            score += 1
            filters.append("MACD histogram חיובי ומתחזק")
        elif macd_hist < 0 and macd_hist <= prev_macd_hist:
            score -= 1
            filters.append("MACD histogram שלילי ומתחזק למטה")

    if pd.notna(relative_volume):
        if relative_volume >= 1.20:
            filters.append("ווליום יחסי גבוה — התנועה משמעותית יותר")
        elif relative_volume < 0.70:
            filters.append("ווליום יחסי נמוך — אמינות התנועה חלשה יותר")
            if score > 0:
                score -= 1
            elif score < 0:
                score += 1

    if pd.notna(day_range_position):
        if day_range_position >= 0.70:
            score += 1
            filters.append("המחיר בחלק העליון של הטווח היומי")
        elif day_range_position <= 0.30:
            score -= 1
            filters.append("המחיר בחלק התחתון של הטווח היומי")
        else:
            filters.append("המחיר באמצע הטווח — פחות חד")

    # Over-extension protection
    too_extended_long = (
        pd.notna(distance_from_ema20_pct)
        and pd.notna(distance_from_vwap_pct)
        and distance_from_ema20_pct > 1.20
        and distance_from_vwap_pct > 1.50
    )
    too_extended_short = (
        pd.notna(distance_from_ema20_pct)
        and pd.notna(distance_from_vwap_pct)
        and distance_from_ema20_pct < -1.20
        and distance_from_vwap_pct < -1.50
    )

    if too_extended_long:
        score -= 1
        filters.append("המחיר מתוח מדי מעל EMA/VWAP — עדיף לא לרדוף")
    if too_extended_short:
        score += 1
        filters.append("המחיר מתוח מדי מתחת EMA/VWAP — עדיף לא לרדוף")

    if score >= 5:
        setup_quality = "חזק"
        confidence_he = "גבוהה יחסית"
    elif score >= 3:
        setup_quality = "בינוני"
        confidence_he = "בינונית"
    elif score <= -5:
        setup_quality = "חזק"
        confidence_he = "גבוהה יחסית"
    elif score <= -3:
        setup_quality = "בינוני"
        confidence_he = "בינונית"
    else:
        setup_quality = "חלש / מעורב"
        confidence_he = "נמוכה"

    buffer_value = max(current_price * 0.0005, atr_like * 0.10)
    min_risk = max(current_price * 0.0015, atr_like * 0.50)

    bullish_structure = score >= 3 and current_price > ema20 and current_price > vwap
    bearish_structure = score <= -3 and current_price < ema20 and current_price < vwap

    common_values = {
        "setup_score": int(score),
        "setup_quality": setup_quality,
        "confidence_he": confidence_he,
        "current_price": current_price,
        "current_time": current_time,
        "distance_from_vwap_pct": distance_from_vwap_pct,
        "distance_from_ema20_pct": distance_from_ema20_pct,
        "ema20": ema20,
        "ema50": ema50,
        "vwap": vwap,
        "rsi14": rsi14,
        "macd_hist": macd_hist,
        "relative_volume": relative_volume,
        "bb_position": bb_position,
        "bb_width_pct": bb_width_pct,
        "day_range_position": day_range_position,
        "atr_like": atr_like,
        "day_open": day_open,
        "day_high": day_high,
        "day_low": day_low,
        "current_move_pct": current_move_pct,
        "filters_summary": " | ".join(filters),
    }

    if bullish_structure:
        entry_low = max(min(ema20, vwap), last_10_low)
        entry_high = max(prev_high, current_price)
        trigger_price = max(prev_high, current_price + buffer_value)

        stop_price = min(last_20_low - buffer_value, min(ema20, vwap) - buffer_value)
        risk = max(trigger_price - stop_price, min_risk)

        target_1 = trigger_price + risk
        target_2 = trigger_price + 2 * risk

        out = base.copy()
        out.update(common_values)
        out.update({
            "bias": "LONG_WATCH",
            "title_he": "תרחיש לימודי עכשיו: לונג רק עם אישור",
            "direction_he": "לונג",
            "trend_state": "נטייה עולה לפי שילוב EMA/VWAP/RSI/MACD/ווליום",
            "entry_zone": f"{entry_low:.2f} עד {entry_high:.2f}",
            "trigger": f"כניסה לימודית רק אם יש סגירה/פריצה מעל {trigger_price:.2f}, או ריטסט שמחזיק מעל EMA20/VWAP.",
            "stop": f"מתחת לתמיכה/EMA/VWAP: בערך {stop_price:.2f}",
            "target_1": f"יעד 1R: בערך {target_1:.2f}",
            "target_2": f"יעד 2R: בערך {target_2:.2f}",
            "time_plan": "תרחיש intraday: לבדוק כל 15-30 דקות, לא להשאיר בלי סטופ, ולסגור לפני סוף המסחר אם אין תוכנית אחרת.",
            "reason": "רוב האינדיקטורים תומכים בכיוון עולה. עדיין נדרש טריגר מחיר — לא להיכנס רק בגלל שהמערכת מציגה לונג.",
            "risk_reward": "בערך 1:1 ליעד ראשון ו־1:2 ליעד שני",
        })

        if too_extended_long:
            out["entry_zone"] = f"לא לרדוף עכשיו. עדיף להמתין לריטסט לאזור {max(ema20, vwap):.2f} או לפריצה נקייה מעל {last_10_high:.2f}."
            out["reason"] += " המחיר מתוח יחסית מעל EMA/VWAP ולכן הסיכון לתיקון גבוה יותר."

        return out

    if bearish_structure:
        entry_high = min(max(ema20, vwap), last_10_high)
        entry_low = min(prev_low, current_price)
        trigger_price = min(prev_low, current_price - buffer_value)

        stop_price = max(last_20_high + buffer_value, max(ema20, vwap) + buffer_value)
        risk = max(stop_price - trigger_price, min_risk)

        target_1 = trigger_price - risk
        target_2 = trigger_price - 2 * risk

        out = base.copy()
        out.update(common_values)
        out.update({
            "bias": "SHORT_WATCH",
            "title_he": "תרחיש לימודי עכשיו: שורט רק עם אישור",
            "direction_he": "שורט",
            "trend_state": "נטייה יורדת לפי שילוב EMA/VWAP/RSI/MACD/ווליום",
            "entry_zone": f"{entry_low:.2f} עד {entry_high:.2f}",
            "trigger": f"כניסה לימודית רק אם יש סגירה/שבירה מתחת {trigger_price:.2f}, או ריטסט שנכשל מתחת EMA20/VWAP.",
            "stop": f"מעל התנגדות/EMA/VWAP: בערך {stop_price:.2f}",
            "target_1": f"יעד 1R: בערך {target_1:.2f}",
            "target_2": f"יעד 2R: בערך {target_2:.2f}",
            "time_plan": "תרחיש intraday: לבדוק כל 15-30 דקות, לא להשאיר בלי סטופ, ולסגור לפני סוף המסחר אם אין תוכנית אחרת.",
            "reason": "רוב האינדיקטורים תומכים בכיוון יורד. עדיין נדרש טריגר מחיר — לא להיכנס רק בגלל שהמערכת מציגה שורט.",
            "risk_reward": "בערך 1:1 ליעד ראשון ו־1:2 ליעד שני",
        })

        if too_extended_short:
            out["entry_zone"] = f"לא לרדוף עכשיו. עדיף להמתין לריטסט לאזור {min(ema20, vwap):.2f} או לשבירה נקייה מתחת {last_10_low:.2f}."
            out["reason"] += " המחיר מתוח יחסית מתחת EMA/VWAP ולכן הסיכון לריבאונד גבוה יותר."

        return out

    out = base.copy()
    out.update(common_values)
    out.update({
        "trend_state": "מעורב / דשדוש",
        "reason": (
            "אין מספיק קונפלואנס בין EMA, VWAP, RSI, MACD, ווליום ומיקום בטווח היומי. "
            "במצב כזה עדיף להמתין לפריצה, שבירה, או ריטסט ברור."
        ),
    })

    return out


def plot_candles_with_indicators(df: pd.DataFrame, title: str = "Current chart"):
    """
    Candlestick chart with EMA20, EMA50, VWAP and volume.
    """

    if df.empty:
        return None

    plot_df = add_realtime_indicators(df).tail(160)

    if plot_df.empty:
        return None

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.75, 0.25],
    )

    fig.add_trace(
        go.Candlestick(
            x=plot_df.index,
            open=plot_df["open"],
            high=plot_df["high"],
            low=plot_df["low"],
            close=plot_df["close"],
            name="Price",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["ema20"], mode="lines", name="EMA20"), row=1, col=1)
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["ema50"], mode="lines", name="EMA50"), row=1, col=1)
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["vwap"], mode="lines", name="VWAP"), row=1, col=1)
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["bb_upper"], mode="lines", name="BB Upper", opacity=0.35), row=1, col=1)
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["bb_lower"], mode="lines", name="BB Lower", opacity=0.35), row=1, col=1)

    fig.add_trace(
        go.Bar(
            x=plot_df.index,
            y=plot_df["volume"],
            name="Volume",
            opacity=0.55,
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        title=title,
        height=700,
        xaxis_rangeslider_visible=False,
        margin=dict(l=10, r=10, t=50, b=20),
        template="plotly_white",
    )

    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)

    return fig


# ============================================================
# Plotting
# ============================================================

def plot_candles(df: pd.DataFrame, title: str = "Candlestick Chart"):
    if df.empty:
        return None

    plot_df = df.copy().tail(160)

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.75, 0.25],
    )

    fig.add_trace(
        go.Candlestick(
            x=plot_df.index,
            open=plot_df["open"],
            high=plot_df["high"],
            low=plot_df["low"],
            close=plot_df["close"],
            name="Price",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Bar(
            x=plot_df.index,
            y=plot_df["volume"],
            name="Volume",
            opacity=0.55,
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        title=title,
        height=650,
        xaxis_rangeslider_visible=False,
        margin=dict(l=10, r=10, t=50, b=20),
        template="plotly_white",
    )

    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)

    return fig


def plot_probability_bar(summary: pd.DataFrame, opening_type: str):
    if summary.empty:
        return None

    filtered = summary[summary["opening_type"] == opening_type].copy()

    if filtered.empty:
        return None

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=filtered["after_result"],
            y=filtered["probability_pct"],
            text=filtered["probability_pct"].round(1).astype(str) + "%",
            textposition="auto",
        )
    )

    fig.update_layout(
        title=f"הסתברויות היסטוריות עבור {opening_type}",
        xaxis_title="מה קרה אחרי הפתיחה",
        yaxis_title="הסתברות באחוזים",
        height=420,
        template="plotly_white",
        margin=dict(l=10, r=10, t=50, b=20),
    )
    return fig


# ============================================================
# Files
# ============================================================

def save_outputs(ticker: str, results: pd.DataFrame, summary_by_type: pd.DataFrame,
                 summary_by_volume: pd.DataFrame, eod_summary: pd.DataFrame):
    results_path = DATA_DIR / f"{ticker}_detailed_opening_results.csv"
    summary_path = DATA_DIR / f"{ticker}_summary_by_opening_type.csv"
    volume_path = DATA_DIR / f"{ticker}_summary_by_opening_type_volume.csv"
    eod_path = DATA_DIR / f"{ticker}_eod_summary_by_opening_type.csv"

    results.to_csv(results_path, index=False)
    summary_by_type.to_csv(summary_path, index=False)
    summary_by_volume.to_csv(volume_path, index=False)
    eod_summary.to_csv(eod_path, index=False)

    return {
        "results": results_path,
        "summary_by_type": summary_path,
        "summary_by_volume": volume_path,
        "eod_summary": eod_path,
    }


def load_summary_file(ticker: str) -> pd.DataFrame:
    path = DATA_DIR / f"{ticker}_summary_by_opening_type.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def run_history_for_ticker(ticker: str, cfg: AnalyzerConfig):
    df = fetch_intraday_yfinance(
        ticker=ticker,
        days_back=cfg.days_back,
        interval_minutes=cfg.bar_minutes,
    )

    if df.empty:
        return df, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    results = analyze_history(df, cfg)
    summary_by_type = probability_summary_by_opening_type(results, min_samples=cfg.min_samples)
    summary_by_volume = probability_summary_by_type_and_volume(results, min_samples=cfg.min_samples)
    eod_summary = eod_summary_by_opening_type(results, min_samples=cfg.min_samples)

    save_outputs(
        ticker=ticker,
        results=results,
        summary_by_type=summary_by_type,
        summary_by_volume=summary_by_volume,
        eod_summary=eod_summary,
    )

    return df, results, summary_by_type, summary_by_volume, eod_summary


# ============================================================
# Sidebar
# ============================================================

st.sidebar.title("⚙️ הגדרות")

days_back = st.sidebar.slider(
    "כמה ימים אחורה לבדוק",
    min_value=5,
    max_value=59,
    value=59,
    step=1,
    help="ב־yfinance נתוני intraday מוגבלים לתקופה קצרה. לכן זה עד 59 יום.",
)

bar_minutes = st.sidebar.selectbox(
    "גודל נר",
    options=[1, 5, 15, 30, 60],
    index=1,
    help="נר 1 דקה עובד רק על תקופה קצרה יותר ב-yfinance, לכן האפליקציה תגביל אותו אוטומטית.",
)

first_window_minutes = st.sidebar.selectbox(
    "חלון פתיחה לבדיקה",
    options=[5, 10, 15, 30, 45, 60],
    index=3,
)

moderate_threshold_pct = st.sidebar.number_input(
    "סף לתנועה מתונה באחוזים",
    min_value=0.05,
    max_value=5.0,
    value=0.30,
    step=0.05,
)

sharp_threshold_pct = st.sidebar.number_input(
    "סף לתנועה חדה באחוזים",
    min_value=0.10,
    max_value=10.0,
    value=0.80,
    step=0.05,
)

continuation_fraction = st.sidebar.number_input(
    "המשך תנועה: כמה מהמהלך הראשון",
    min_value=0.10,
    max_value=2.00,
    value=0.50,
    step=0.05,
)

retrace_fraction = st.sidebar.number_input(
    "תיקון: כמה מהמהלך הראשון",
    min_value=0.10,
    max_value=1.00,
    value=0.50,
    step=0.05,
)

min_samples = st.sidebar.slider(
    "מינימום מקרים להצגה",
    min_value=2,
    max_value=30,
    value=5,
    step=1,
)

cfg = AnalyzerConfig(
    days_back=int(days_back),
    bar_minutes=int(bar_minutes),
    first_window_minutes=int(first_window_minutes),
    moderate_threshold_pct=float(moderate_threshold_pct),
    sharp_threshold_pct=float(sharp_threshold_pct),
    continuation_fraction=float(continuation_fraction),
    retrace_fraction=float(retrace_fraction),
    min_samples=int(min_samples),
)

if int(bar_minutes) == 1 and int(days_back) > 7:
    st.sidebar.warning("בחרת נר 1 דקה. בגלל מגבלת yfinance, האפליקציה תשתמש בפועל רק עד 7 ימים אחורה.")


# ============================================================
# Header
# ============================================================

st.markdown('<div class="main-title">📊 Market Open Analyzer - Free</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">גרסה חינמית ללא API Key — yfinance + EMA/VWAP/RSI/MACD/Bollinger/Volume</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="warning-card">
<strong>חשוב:</strong>
זו גרסה חינמית ללימוד ובדיקה. היא לא נותנת 5 שנים אחורה.
בדרך כלל היא מוגבלת לנתוני intraday מהתקופה האחרונה בלבד.
זה לא כלי המלצה לקנייה/מכירה ולא מבצע עסקאות.
</div>
""",
    unsafe_allow_html=True,
)


# ============================================================
# Tabs
# ============================================================

tab_single, tab_all, tab_live, tab_invest_now, tab_help = st.tabs(
    [
        "🔎 מניה אחת",
        "🚀 כמה מניות",
        "⏱️ יום אחרון / כמעט זמן אמת",
        "🎯 השקעה כעת",
        "📘 הסבר",
    ]
)


# ============================================================
# Single ticker
# ============================================================

with tab_single:
    st.subheader("🔎 בדיקת מניה אחת")

    col_a, col_b = st.columns([1, 2])

    with col_a:
        selected_name = st.selectbox(
            "בחר מניה / ETF",
            options=list(DEFAULT_TICKERS.keys()),
            index=0,
        )
        ticker = DEFAULT_TICKERS[selected_name]

        st.markdown(
            f"""
<div class="card">
<h3>{selected_name}</h3>
<p class="small-muted">סימול: <strong>{ticker}</strong></p>
<p>הבדיקה תמשוך נתוני intraday אחרונים מ־yfinance.</p>
</div>
""",
            unsafe_allow_html=True,
        )

        run_single = st.button(f"▶️ הרץ בדיקה על {ticker}", use_container_width=True)

    with col_b:
        st.markdown(
            """
<div class="card">
<h3>מה נקבל?</h3>
<p>סיווג פתיחת המסחר לפי עוצמה וצורה, ואז בדיקה סטטיסטית של מה קרה בהמשך היום.</p>
</div>
""",
            unsafe_allow_html=True,
        )

    if run_single:
        with st.spinner(f"מושך ומנתח נתונים עבור {ticker}..."):
            try:
                df, results, summary_by_type, summary_by_volume, eod_summary = run_history_for_ticker(
                    ticker=ticker,
                    cfg=cfg,
                )

                if results.empty:
                    st.warning("לא נמצאו מספיק נתונים לניתוח. נסה פחות דקות/חלון פתיחה קצר יותר.")
                else:
                    c1, c2, c3, c4 = st.columns(4)
                    with c1:
                        st.metric("ימי מסחר שנותחו", f"{len(results):,}")
                    with c2:
                        st.metric("סוגי פתיחה", f"{results['opening_type'].nunique()}")
                    with c3:
                        st.metric("ממוצע תנועה בפתיחה", f"{results['initial_move_pct'].mean():.2f}%")
                    with c4:
                        st.metric("ממוצע עד סוף יום", f"{results['eod_change_pct'].mean():.2f}%")

                    st.success("הניתוח הושלם ונשמר לקבצי CSV בתיקיית data.")

                    st.markdown("### גרף נרות אחרונים")
                    fig = plot_candles(filter_regular_market_hours(df), title=f"{ticker} — נרות אחרונים")
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)

                    st.markdown("### הסתברויות לפי סוג פתיחה")
                    st.dataframe(summary_by_type, use_container_width=True, hide_index=True)

                    st.markdown("### סיכום סוף יום לפי סוג פתיחה")
                    st.dataframe(eod_summary, use_container_width=True, hide_index=True)

                    st.markdown("### הסתברויות לפי סוג פתיחה + ווליום")
                    st.dataframe(summary_by_volume, use_container_width=True, hide_index=True)

                    st.markdown("### תוצאות יומיות גולמיות")
                    st.dataframe(results.tail(100), use_container_width=True, hide_index=True)

            except Exception as e:
                st.error(f"שגיאה: {e}")


# ============================================================
# Batch
# ============================================================

with tab_all:
    st.subheader("🚀 בדיקה על כמה מניות")

    st.markdown(
        """
<div class="card">
בגלל שזו גרסה חינמית, עדיף לבחור 2-3 מניות בכל פעם כדי לא להעמיס על yfinance.
</div>
""",
        unsafe_allow_html=True,
    )

    selected_batch_names = st.multiselect(
        "בחר מניות",
        options=list(DEFAULT_TICKERS.keys()),
        default=["Apple", "Nvidia", "Tesla"],
    )
    selected_batch_tickers = [DEFAULT_TICKERS[name] for name in selected_batch_names]

    run_all = st.button("▶️ הרץ בדיקה על הבחירה", use_container_width=True)

    if run_all:
        if not selected_batch_tickers:
            st.warning("לא נבחרו מניות.")
        else:
            combined_best = []
            combined_eod = []

            progress = st.progress(0)
            status = st.empty()

            for i, ticker in enumerate(selected_batch_tickers, start=1):
                status.info(f"מריץ {ticker} ({i}/{len(selected_batch_tickers)})...")

                try:
                    df, results, summary_by_type, summary_by_volume, eod_summary = run_history_for_ticker(
                        ticker=ticker,
                        cfg=cfg,
                    )

                    if not summary_by_type.empty:
                        temp = summary_by_type.copy()
                        temp.insert(0, "ticker", ticker)
                        best = (
                            temp.sort_values("probability_pct", ascending=False)
                            .groupby(["ticker", "opening_type"], as_index=False)
                            .head(1)
                        )
                        combined_best.append(best)

                    if not eod_summary.empty:
                        temp_eod = eod_summary.copy()
                        temp_eod.insert(0, "ticker", ticker)
                        combined_eod.append(temp_eod)

                    time.sleep(1)

                except Exception as e:
                    st.error(f"שגיאה ב־{ticker}: {e}")

                progress.progress(i / len(selected_batch_tickers))

            status.success("הבדיקה הסתיימה.")

            if combined_best:
                combined_best_df = pd.concat(combined_best, ignore_index=True)
                combined_best_df.to_csv(DATA_DIR / "ALL_best_probabilities_by_ticker.csv", index=False)

                st.markdown("### התוצאה הכי נפוצה לכל סוג פתיחה בכל מניה")
                st.dataframe(combined_best_df, use_container_width=True, hide_index=True)

            if combined_eod:
                combined_eod_df = pd.concat(combined_eod, ignore_index=True)
                combined_eod_df.to_csv(DATA_DIR / "ALL_eod_summary_by_ticker.csv", index=False)

                st.markdown("### סיכום סוף יום משולב")
                st.dataframe(combined_eod_df, use_container_width=True, hide_index=True)


# ============================================================
# Live / latest day
# ============================================================

with tab_live:
    st.subheader("⏱️ יום אחרון / כמעט זמן אמת")

    st.markdown(
        """
<div class="card">
המסך הזה מושך את הנתונים האחרונים הזמינים ב־yfinance,
בודק את יום המסחר האחרון שקיים בנתונים, ומשווה אותו להיסטוריה הקצרה שנשמרה.
</div>
""",
        unsafe_allow_html=True,
    )

    col_live_1, col_live_2 = st.columns([1, 2])

    with col_live_1:
        live_name = st.selectbox(
            "בחר מניה / ETF",
            options=list(DEFAULT_TICKERS.keys()),
            index=0,
            key="live_selectbox",
        )
        live_ticker = DEFAULT_TICKERS[live_name]

        refresh_now = st.button("🔄 בדוק עכשיו", use_container_width=True)
        build_history_now = st.button("🧠 בנה היסטוריה עכשיו", use_container_width=True)
        auto_refresh = st.checkbox("רענון אוטומטי כל 60 שניות", value=False)

    with col_live_2:
        st.markdown(
            f"""
<div class="card">
<h3>{live_name} — {live_ticker}</h3>
<p>המערכת תבדוק את יום המסחר האחרון בנתונים ותשווה לסוגי פתיחה דומים.</p>
</div>
""",
            unsafe_allow_html=True,
        )

    if build_history_now:
        with st.spinner(f"בונה היסטוריה עבור {live_ticker}..."):
            try:
                run_history_for_ticker(ticker=live_ticker, cfg=cfg)
                st.success("היסטוריה נבנתה ונשמרה.")
            except Exception as e:
                st.error(f"שגיאה בבניית היסטוריה: {e}")

    if refresh_now or auto_refresh:
        try:
            df = fetch_intraday_yfinance(
                ticker=live_ticker,
                days_back=cfg.days_back,
                interval_minutes=cfg.bar_minutes,
            )

            latest_day_df = get_latest_trading_day(df)
            current_info = classify_current_opening(latest_day_df, cfg)

            if current_info is None:
                st.warning("אין מספיק נתונים מהיום האחרון.")
            else:
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.metric("תאריך", current_info["date"])
                with c2:
                    st.metric("מחיר פתיחה", f"{current_info['open_price']:.2f}")
                with c3:
                    st.metric("מחיר אחרון", f"{current_info['current_price']:.2f}")
                with c4:
                    st.metric("סוג פתיחה", current_info["opening_type"])

                if not current_info["is_complete_window"]:
                    st.warning(
                        f"חלון הפתיחה עדיין לא מלא: "
                        f"{current_info['bars_count']}/{current_info['expected_bars']} נרות."
                    )
                else:
                    st.success("חלון הפתיחה הושלם.")

                st.markdown("### מה קרה היום בפועל")
                today_status = build_today_status(latest_day_df, cfg)
                if today_status is not None:
                    t1, t2, t3, t4 = st.columns(4)
                    with t1:
                        st.metric("שינוי מהפתיחה", f"{today_status['current_move_pct']:.2f}%")
                    with t2:
                        st.metric("גבוה יומי", f"{today_status['day_high']:.2f}")
                    with t3:
                        st.metric("נמוך יומי", f"{today_status['day_low']:.2f}")
                    with t4:
                        st.metric("מצב אחרי הפתיחה", today_status["today_after_status"])

                    st.dataframe(
                        pd.DataFrame([today_status]),
                        use_container_width=True,
                        hide_index=True,
                    )

                st.markdown("### פרטי הפתיחה")
                st.json(current_info)

                st.markdown("### גרף יום אחרון")
                fig = plot_candles(latest_day_df, title=f"{live_ticker} — יום אחרון")
                if fig:
                    st.plotly_chart(fig, use_container_width=True)

                summary_by_type = load_summary_file(live_ticker)

                if summary_by_type.empty:
                    st.warning("לא נמצאה היסטוריה שמורה. לחץ על ״בנה היסטוריה עכשיו״.")
                else:
                    match = compare_current_to_history(current_info, summary_by_type)
                    if match.empty:
                        st.warning(f"אין מספיק מקרים היסטוריים עבור: {current_info['opening_type']}")
                    else:
                        st.markdown("### מה קרה בעבר במקרים דומים?")
                        st.dataframe(match, use_container_width=True, hide_index=True)

                        fig_prob = plot_probability_bar(summary_by_type, current_info["opening_type"])
                        if fig_prob:
                            st.plotly_chart(fig_prob, use_container_width=True)

                        st.markdown("### תוכנית לימודית להיום — לא המלצת מסחר")
                        today_status = build_today_status(latest_day_df, cfg)
                        plan = build_educational_trade_plan(current_info, today_status, match, cfg)

                        if plan["bias"] == "LONG_WATCH":
                            st.success(plan["title_he"])
                        elif plan["bias"] == "SHORT_WATCH":
                            st.error(plan["title_he"])
                        else:
                            st.warning(plan["title_he"])

                        p1, p2, p3, p4 = st.columns(4)
                        with p1:
                            st.metric("כיוון", plan["direction_he"])
                        with p2:
                            st.metric("ביטחון", plan["confidence_he"])
                        with p3:
                            st.metric("לונג היסטורי", f"{plan['long_probability']:.1f}%")
                        with p4:
                            st.metric("שורט היסטורי", f"{plan['short_probability']:.1f}%")

                        st.markdown(
                            f"""
<div class="card">
<h3>תוכנית בדמו בלבד</h3>
<p><strong>אזור כניסה:</strong> {plan['entry_zone']}</p>
<p><strong>טריגר:</strong> {plan['trigger']}</p>
<p><strong>סטופ:</strong> {plan['stop']}</p>
<p><strong>יעד ראשון:</strong> {plan['target_1']}</p>
<p><strong>יעד שני:</strong> {plan['target_2']}</p>
<p><strong>זמן החזקה:</strong> {plan['time_plan']}</p>
<p><strong>למה:</strong> {plan['reason']}</p>
<p class="small-muted">זה אינו ייעוץ השקעות ואינו הוראת קנייה/מכירה. להשתמש רק ללמידה ובדמו.</p>
</div>
""",
                            unsafe_allow_html=True,
                        )

        except Exception as e:
            st.error(f"שגיאה: {e}")

        if auto_refresh:
            time.sleep(60)
            st.rerun()




def scan_ticker_list_for_now(
    ticker_options: dict[str, str],
    cfg: AnalyzerConfig,
) -> pd.DataFrame:
    """
    Scan all tickers in the current list and rank current educational scenarios.
    """
    rows = []

    for display_name, ticker in list(ticker_options.items()):
        try:
            df = fetch_intraday_yfinance(
                ticker=ticker,
                days_back=cfg.days_back,
                interval_minutes=cfg.bar_minutes,
            )
            latest_day_df = get_latest_trading_day(df)

            if latest_day_df.empty:
                rows.append({
                    "name": display_name,
                    "ticker": ticker,
                    "direction": "אין נתונים",
                    "bias": "NO_DATA",
                    "score": 0,
                    "quality": "N/A",
                    "current_price": np.nan,
                    "change_from_open_pct": np.nan,
                    "rsi14": np.nan,
                    "relative_volume": np.nan,
                    "entry_zone": "",
                    "trigger": "",
                    "stop": "",
                    "target_1": "",
                    "target_2": "",
                    "reason": "לא נמצאו נתונים זמינים.",
                })
                continue

            plan = build_invest_now_plan(latest_day_df, cfg)

            rows.append({
                "name": display_name,
                "ticker": ticker,
                "direction": plan.get("direction_he", ""),
                "bias": plan.get("bias", ""),
                "score": plan.get("setup_score", 0),
                "quality": plan.get("setup_quality", ""),
                "current_price": plan.get("current_price", np.nan),
                "change_from_open_pct": plan.get("current_move_pct", np.nan),
                "rsi14": plan.get("rsi14", np.nan),
                "relative_volume": plan.get("relative_volume", np.nan),
                "entry_zone": plan.get("entry_zone", ""),
                "trigger": plan.get("trigger", ""),
                "stop": plan.get("stop", ""),
                "target_1": plan.get("target_1", ""),
                "target_2": plan.get("target_2", ""),
                "reason": plan.get("reason", ""),
            })

            time.sleep(0.25)

        except Exception as e:
            rows.append({
                "name": display_name,
                "ticker": ticker,
                "direction": "שגיאה",
                "bias": "ERROR",
                "score": 0,
                "quality": "N/A",
                "current_price": np.nan,
                "change_from_open_pct": np.nan,
                "rsi14": np.nan,
                "relative_volume": np.nan,
                "entry_zone": "",
                "trigger": "",
                "stop": "",
                "target_1": "",
                "target_2": "",
                "reason": str(e)[:200],
            })

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)

    def rank_bias(row):
        bias = row.get("bias", "")
        score = abs(float(row.get("score", 0) or 0))
        if bias in ["LONG_WATCH", "SHORT_WATCH"]:
            return 100 + score
        if bias == "NO_TRADE":
            return 10 + score
        return 0

    out["rank"] = out.apply(rank_bias, axis=1)
    out = out.sort_values(["rank", "score"], ascending=[False, False])
    out = out.drop(columns=["rank"])

    return out


# ============================================================
# Invest now tab
# ============================================================

with tab_invest_now:
    st.subheader("🎯 השקעה כעת — תרחיש לימודי לפי הרגע הנוכחי")

    st.markdown(
        """
<div class="warning-card">
<strong>חשוב:</strong>
המסך הזה אינו ייעוץ השקעות ואינו הוראת קנייה/מכירה.
הוא בונה תרחיש לימודי לפי נתונים זמינים, EMA20, EMA50, VWAP, מגמה וסיכון־סיכוי.
לעבוד בדמו בלבד עד שיש לך שיטה מוכחת.
</div>
""",
        unsafe_allow_html=True,
    )

    col_now_1, col_now_2 = st.columns([1, 2])

    with col_now_1:
        ticker_options_now = get_all_ticker_options()

        new_ticker_input = st.text_input(
            "הוסף סימול מניה לרשימה",
            placeholder="לדוגמה: PLTR / SMCI / MSTR / QQQ",
            key="new_ticker_input",
        )

        add_ticker_clicked = st.button("➕ הוסף לרשימה", use_container_width=True)

        if add_ticker_clicked:
            cleaned_ticker = normalize_ticker(new_ticker_input)
            if not cleaned_ticker:
                st.warning("לא הוקלד סימול.")
            else:
                added = save_custom_ticker(cleaned_ticker)
                if added:
                    st.success(f"{cleaned_ticker} נוסף לרשימה.")
                    st.rerun()
                else:
                    st.info(f"{cleaned_ticker} כבר קיים ברשימה או לא תקין.")

        ticker_options_now = get_all_ticker_options()

        now_name = st.selectbox(
            "בחר מניה / ETF לניתוח עכשיו",
            options=list(ticker_options_now.keys()),
            index=0,
            key="invest_now_selectbox",
        )
        now_ticker = ticker_options_now[now_name]

        now_refresh = st.button("🔄 נתח עכשיו", use_container_width=True)
        scan_now_list = st.button("📋 סרוק את כל הרשימה ומצא הזדמנויות", use_container_width=True)
        now_auto_refresh = st.checkbox("רענון אוטומטי כל 60 שניות", value=False, key="invest_now_auto")

    with col_now_2:
        st.markdown(
            f"""
<div class="card">
<h3>{now_name} — {now_ticker}</h3>
<p>
המערכת תבדוק את הנתונים האחרונים הזמינים ותייצר תרחיש לימודי:
לונג / שורט / המתנה, אזור כניסה, טריגר, סטופ ויעדים.
אפשר גם להוסיף סימול מניה ידנית ולסרוק את כל הרשימה.
</p>
<p class="small-muted">
ב־yfinance הנתונים לא בהכרח בזמן אמת מלא ויכולים להיות מושהים.
</p>
</div>
""",
            unsafe_allow_html=True,
        )

    if scan_now_list:
        with st.spinner("סורק את כל הרשימה ומחפש לונג/שורט אפשריים עכשיו..."):
            scan_results = scan_ticker_list_for_now(
                ticker_options=get_all_ticker_options(),
                cfg=cfg,
            )

        if scan_results.empty:
            st.warning("לא נמצאו תוצאות לסריקה.")
        else:
            opportunities = scan_results[scan_results["bias"].isin(["LONG_WATCH", "SHORT_WATCH"])].copy()
            wait_list = scan_results[scan_results["bias"].eq("NO_TRADE")].copy()

            st.markdown("### מניות / ETF שיש להן תרחיש לונג או שורט עכשיו")

            if opportunities.empty:
                st.warning("כרגע אין מניות עם תרחיש לונג/שורט מספיק נקי לפי הפילטרים.")
            else:
                st.dataframe(
                    opportunities[
                        [
                            "ticker",
                            "direction",
                            "score",
                            "quality",
                            "current_price",
                            "change_from_open_pct",
                            "rsi14",
                            "relative_volume",
                            "entry_zone",
                            "trigger",
                            "stop",
                            "target_1",
                            "target_2",
                        ]
                    ],
                    use_container_width=True,
                    hide_index=True,
                )

            with st.expander("הצג גם מניות שכרגע עדיף להמתין בהן"):
                st.dataframe(
                    wait_list[
                        [
                            "ticker",
                            "direction",
                            "score",
                            "quality",
                            "current_price",
                            "change_from_open_pct",
                            "reason",
                        ]
                    ],
                    use_container_width=True,
                    hide_index=True,
                )

            with st.expander("כל תוצאות הסריקה"):
                st.dataframe(scan_results, use_container_width=True, hide_index=True)

    if now_refresh or now_auto_refresh:
        try:
            df = fetch_intraday_yfinance(
                ticker=now_ticker,
                days_back=cfg.days_back,
                interval_minutes=cfg.bar_minutes,
            )

            latest_day_df = get_latest_trading_day(df)

            if latest_day_df.empty:
                st.warning("אין נתונים זמינים ליום המסחר האחרון.")
            else:
                plan_now = build_invest_now_plan(latest_day_df, cfg)

                if plan_now["bias"] == "LONG_WATCH":
                    st.success(plan_now["title_he"])
                elif plan_now["bias"] == "SHORT_WATCH":
                    st.error(plan_now["title_he"])
                else:
                    st.warning(plan_now["title_he"])

                n1, n2, n3, n4 = st.columns(4)
                with n1:
                    st.metric("מחיר נוכחי", f"{plan_now['current_price']:.2f}" if pd.notna(plan_now["current_price"]) else "N/A")
                with n2:
                    st.metric("שינוי מהפתיחה", f"{plan_now['current_move_pct']:.2f}%" if pd.notna(plan_now["current_move_pct"]) else "N/A")
                with n3:
                    st.metric("כיוון", plan_now["direction_he"])
                with n4:
                    st.metric("ביטחון", plan_now["confidence_he"])

                n5, n6, n7, n8 = st.columns(4)
                with n5:
                    st.metric("EMA20", f"{plan_now['ema20']:.2f}" if pd.notna(plan_now["ema20"]) else "N/A")
                with n6:
                    st.metric("EMA50", f"{plan_now['ema50']:.2f}" if pd.notna(plan_now["ema50"]) else "N/A")
                with n7:
                    st.metric("VWAP", f"{plan_now['vwap']:.2f}" if pd.notna(plan_now["vwap"]) else "N/A")
                with n8:
                    st.metric("ציון איכות", f"{plan_now['setup_score']} / איכות: {plan_now['setup_quality']}")

                n9, n10, n11, n12 = st.columns(4)
                with n9:
                    st.metric("RSI14", f"{plan_now['rsi14']:.1f}" if pd.notna(plan_now["rsi14"]) else "N/A")
                with n10:
                    st.metric("MACD Hist", f"{plan_now['macd_hist']:.4f}" if pd.notna(plan_now["macd_hist"]) else "N/A")
                with n11:
                    st.metric("Relative Volume", f"{plan_now['relative_volume']:.2f}x" if pd.notna(plan_now["relative_volume"]) else "N/A")
                with n12:
                    st.metric("מרחק מ־VWAP", f"{plan_now['distance_from_vwap_pct']:.2f}%" if pd.notna(plan_now["distance_from_vwap_pct"]) else "N/A")

                st.markdown(
                    f"""
<div class="card">
<h3>תרחיש לימודי לפי הרגע הנוכחי</h3>
<p><strong>מצב מגמה:</strong> {plan_now['trend_state']}</p>
<p><strong>אזור כניסה:</strong> {plan_now['entry_zone']}</p>
<p><strong>טריגר כניסה:</strong> {plan_now['trigger']}</p>
<p><strong>סטופ:</strong> {plan_now['stop']}</p>
<p><strong>יעד ראשון:</strong> {plan_now['target_1']}</p>
<p><strong>יעד שני:</strong> {plan_now['target_2']}</p>
<p><strong>יחס סיכון־סיכוי:</strong> {plan_now['risk_reward']}</p>
<p><strong>ציון איכות:</strong> {plan_now['setup_score']} — {plan_now['setup_quality']}</p>
<p><strong>פילטרים:</strong> {plan_now['filters_summary']}</p>
<p><strong>לכמה זמן:</strong> {plan_now['time_plan']}</p>
<p><strong>למה:</strong> {plan_now['reason']}</p>
<p class="small-muted">
זה תרחיש לימודי בלבד. אם אין טריגר — אין עסקה. לא לרדוף אחרי מחיר.
</p>
</div>
""",
                    unsafe_allow_html=True,
                )

                st.markdown("### גרף עם EMA20 / EMA50 / VWAP")
                fig_now = plot_candles_with_indicators(latest_day_df, title=f"{now_ticker} — השקעה כעת")
                if fig_now:
                    st.plotly_chart(fig_now, use_container_width=True)

                st.markdown("### נתונים גולמיים של התרחיש")
                st.dataframe(pd.DataFrame([plan_now]), use_container_width=True, hide_index=True)

        except Exception as e:
            st.error(f"שגיאה: {e}")

        if now_auto_refresh:
            time.sleep(60)
            st.rerun()


# ============================================================
# Help
# ============================================================

with tab_help:
    st.subheader("📘 איך להבין את התוצאות?")

    st.markdown(
        """
<div class="card">
<h3>1. מה זה opening_type?</h3>
<p>זה סוג הפתיחה. לדוגמה:</p>
<ul>
<li><strong>sharp_up_clean_up</strong> — עלייה חדה ונקייה</li>
<li><strong>sharp_up_up_rejected</strong> — עלייה חדה עם דחייה מלמעלה</li>
<li><strong>moderate_down_clean_down</strong> — ירידה מתונה ונקייה</li>
<li><strong>flat_open_volatile_chop</strong> — פתיחה שטוחה אבל תנודתית</li>
</ul>
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown(
        """
<div class="card">
<h3>2. מה זה after_result?</h3>
<ul>
<li><strong>continued_first</strong> — אחרי עלייה בפתיחה, המחיר המשיך קודם למעלה</li>
<li><strong>half_retrace_first</strong> — אחרי עלייה בפתיחה, המחיר קודם תיקן חצי מהעלייה</li>
<li><strong>continued_down_first</strong> — אחרי ירידה בפתיחה, המחיר המשיך קודם למטה</li>
<li><strong>half_rebound_first</strong> — אחרי ירידה בפתיחה, המחיר קודם חזר חצי מהירידה</li>
<li><strong>range_no_clear_move</strong> — לא היה מהלך ברור</li>
</ul>
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown(
        """
<div class="warning-card">
<strong>כלל חשוב:</strong>
גם אם ההיסטוריה אומרת ש־60% מהמקרים קרה משהו,
זה לא אומר שזה יקרה היום. זה רק יתרון סטטיסטי אפשרי, לא ודאות.
</div>
""",
        unsafe_allow_html=True,
    )
