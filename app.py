
import json
import time
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf


# ============================================================
# App setup
# ============================================================

st.set_page_config(page_title="Paper Trading Lab V2", page_icon="🧪", layout="wide")

DATA_DIR = Path("paper_data")
DATA_DIR.mkdir(exist_ok=True)

TRADES_FILE = DATA_DIR / "trades_v2.csv"
TICKERS_FILE = DATA_DIR / "tickers_v2.json"
COSTS_FILE = DATA_DIR / "costs_v2.json"
UNITS_FILE = DATA_DIR / "units_v2.json"
RULES_FILE = DATA_DIR / "rules_v2.json"

NY_TZ = "America/New_York"

DEFAULT_TICKERS = ["QQQ", "SPY", "AAPL", "NVDA", "TSLA", "MSFT", "AMD", "META", "AMZN", "GOOGL", "NFLX", "SMCI", "PLTR", "MSTR"]

DEFAULT_COSTS = {
    "cost_pct_per_side": 0.02,
    "fixed_fee_per_side": 0.0,
    "min_fee_per_side": 0.0,
    "max_cost_to_target_pct": 25.0,
}

DEFAULT_UNITS = {
    "base_unit_dollars": 200.0,
    "max_trade_dollars": 2000.0,
    "score_units": {
        "1": 0.0, "2": 0.0, "3": 1.0, "4": 1.25,
        "5": 1.75, "6": 2.5, "7": 3.5, "8": 5.0,
    },
}

DEFAULT_RULES = {
    "min_hold_fast_minutes": 3,
    "min_hold_half_hour_minutes": 10,
    "cooldown_after_close_minutes": 8,
    "max_new_trades_per_scan": 3,
    "min_profit_r_for_profit_stop": 0.55,
    "emergency_exit_after_minutes": 2,
}

TRADE_COLUMNS = [
    "trade_id", "status", "ticker", "mode", "side", "score",
    "entry_time", "exit_time", "age_minutes",
    "entry_price", "current_price", "exit_price",
    "quantity", "notional",
    "stop_loss", "initial_stop_loss", "profit_stop", "target_reference",
    "highest_price", "lowest_price",
    "entry_cost", "exit_cost", "total_cost",
    "gross_pnl", "net_pnl", "net_pnl_pct",
    "exit_reason", "management_action", "management_reason", "signal_reason",
    "cost_pct_per_side", "fixed_fee_per_side", "min_fee_per_side", "max_cost_to_target_pct",
    "base_unit_dollars", "unit_multiplier",
    "created_settings_snapshot",
]


# ============================================================
# Styling
# ============================================================

st.markdown(
    """
<style>
html, body, [class*="css"] { direction: rtl; text-align: right; }
.title-box {
    background: linear-gradient(135deg,#111827,#1f2937,#374151);
    color:white; padding:24px; border-radius:22px; margin-bottom:16px;
    box-shadow:0 10px 24px rgba(0,0,0,.12);
}
.title-box h1 { margin:0; font-size:34px; }
.title-box p { margin-top:8px; color:#d1d5db; }
.card {
    border:1px solid #e5e7eb; border-radius:18px; padding:14px 16px;
    background:#fff; box-shadow:0 6px 14px rgba(0,0,0,.05); margin:8px 0;
}
.warn { background:#fff7ed; border:1px solid #fed7aa; color:#7c2d12; }
.green-row {
    background:#dcfce7; border:1px solid #86efac; border-radius:14px;
    padding:10px; margin:6px 0; color:#064e3b;
}
.red-row {
    background:#fee2e2; border:1px solid #fca5a5; border-radius:14px;
    padding:10px; margin:6px 0; color:#7f1d1d;
}
.neutral-row {
    background:#f9fafb; border:1px solid #e5e7eb; border-radius:14px;
    padding:10px; margin:6px 0; color:#111827;
}
.small { color:#6b7280; font-size:13px; }
.metric-note { font-size:12px; color:#6b7280; margin-top:-10px; }
</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# File helpers
# ============================================================

def now_ny():
    return pd.Timestamp.now(tz=NY_TZ)

def now_ny_iso():
    return now_ny().isoformat()

def safe_float(x, default=np.nan):
    try:
        if pd.isna(x):
            return float(default)
        return float(x)
    except Exception:
        return float(default)

def normalize_ticker(t):
    t = str(t or "").strip().upper()
    if ":" in t:
        t = t.split(":")[-1]
    return t.replace(" ", "")

def read_json(path, default):
    if not path.exists() or path.stat().st_size == 0:
        return default
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(default, dict):
            merged = json.loads(json.dumps(default))
            for k, v in data.items():
                if isinstance(v, dict) and isinstance(merged.get(k), dict):
                    merged[k].update(v)
                else:
                    merged[k] = v
            return merged
        return data
    except Exception:
        return default

def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

def load_tickers():
    data = read_json(TICKERS_FILE, {"tickers": DEFAULT_TICKERS})
    return sorted(set(normalize_ticker(x) for x in data.get("tickers", DEFAULT_TICKERS) if normalize_ticker(x)))

def save_tickers(tickers):
    write_json(TICKERS_FILE, {"tickers": sorted(set(normalize_ticker(x) for x in tickers if normalize_ticker(x)))})

def load_costs():
    return read_json(COSTS_FILE, DEFAULT_COSTS)

def save_costs(costs):
    write_json(COSTS_FILE, costs)

def load_units():
    return read_json(UNITS_FILE, DEFAULT_UNITS)

def save_units(units):
    write_json(UNITS_FILE, units)

def load_rules():
    return read_json(RULES_FILE, DEFAULT_RULES)

def save_rules(rules):
    write_json(RULES_FILE, rules)

def empty_trades():
    return pd.DataFrame(columns=TRADE_COLUMNS)

def load_trades():
    if not TRADES_FILE.exists() or TRADES_FILE.stat().st_size == 0:
        return empty_trades()
    try:
        df = pd.read_csv(TRADES_FILE)
    except Exception:
        return empty_trades()

    for col in TRADE_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    return df[TRADE_COLUMNS]

def save_trades(df):
    if df is None or df.empty:
        empty_trades().to_csv(TRADES_FILE, index=False)
        return
    for col in TRADE_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    df[TRADE_COLUMNS].to_csv(TRADES_FILE, index=False)

def clear_trades():
    save_trades(empty_trades())


# ============================================================
# Data + indicators
# ============================================================

@st.cache_data(show_spinner=False, ttl=20)
def fetch_1m(ticker, days=7):
    ticker = normalize_ticker(ticker)
    df = yf.download(
        ticker,
        period=f"{min(int(days), 7)}d",
        interval="1m",
        progress=False,
        auto_adjust=True,
        prepost=False,
        threads=False,
    )
    if df is None or df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [str(c[0]).lower() for c in df.columns]
    else:
        df.columns = [str(c).lower() for c in df.columns]

    required = ["open", "high", "low", "close", "volume"]
    if not all(c in df.columns for c in required):
        return pd.DataFrame()

    df = df[required].dropna()

    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC").tz_convert(NY_TZ)
    else:
        df.index = df.index.tz_convert(NY_TZ)

    df = df.between_time("09:30", "16:00")
    return df

def latest_session(df):
    if df is None or df.empty:
        return pd.DataFrame()
    d = df.copy().sort_index()
    last_date = d.index[-1].date()
    return d[d.index.date == last_date]

def add_indicators(df):
    d = df.copy().sort_index()

    for span in [3, 5, 9, 21, 50]:
        d[f"ema{span}"] = d["close"].ewm(span=span, adjust=False).mean()
        d[f"ema{span}_slope"] = d[f"ema{span}"].diff()
        d[f"ema{span}_curv"] = d[f"ema{span}_slope"].diff()

    typical = (d["high"] + d["low"] + d["close"]) / 3
    d["vwap"] = (typical * d["volume"]).cumsum() / d["volume"].replace(0, np.nan).cumsum()
    d["vwap_slope"] = d["vwap"].diff()
    d["vwap_curv"] = d["vwap_slope"].diff()

    delta = d["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    for p in [3, 7, 14]:
        avg_gain = gain.ewm(alpha=1 / p, adjust=False, min_periods=max(2, p // 2)).mean()
        avg_loss = loss.ewm(alpha=1 / p, adjust=False, min_periods=max(2, p // 2)).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        d[f"rsi{p}"] = 100 - (100 / (1 + rs))
        d[f"rsi{p}_slope"] = d[f"rsi{p}"].diff()
        d[f"rsi{p}_curv"] = d[f"rsi{p}_slope"].diff()

    ema12 = d["close"].ewm(span=12, adjust=False).mean()
    ema26 = d["close"].ewm(span=26, adjust=False).mean()
    d["macd"] = ema12 - ema26
    d["macd_signal"] = d["macd"].ewm(span=9, adjust=False).mean()
    d["macd_hist"] = d["macd"] - d["macd_signal"]
    d["macd_hist_slope"] = d["macd_hist"].diff()
    d["macd_hist_curv"] = d["macd_hist_slope"].diff()

    d["range"] = d["high"] - d["low"]
    d["atr3"] = d["range"].rolling(3, min_periods=2).mean()
    d["atr14"] = d["range"].rolling(14, min_periods=5).mean()
    d["vol_ma5"] = d["volume"].rolling(5, min_periods=2).mean()
    d["vol_ma20"] = d["volume"].rolling(20, min_periods=5).mean()
    d["rel_vol5"] = d["volume"] / d["vol_ma5"].replace(0, np.nan)
    d["rel_vol20"] = d["volume"] / d["vol_ma20"].replace(0, np.nan)
    d["mom2_pct"] = (d["close"] / d["close"].shift(2) - 1) * 100
    d["mom5_pct"] = (d["close"] / d["close"].shift(5) - 1) * 100
    d["mom30_pct"] = (d["close"] / d["close"].shift(30) - 1) * 100
    return d


# ============================================================
# Costs and units
# ============================================================

def side_cost(notional, costs):
    variable = abs(float(notional)) * (float(costs["cost_pct_per_side"]) / 100)
    raw = variable + float(costs["fixed_fee_per_side"])
    return float(max(raw, float(costs["min_fee_per_side"])))

def estimate_costs(entry, exit_price, qty, costs):
    entry_notional = abs(float(entry) * float(qty))
    exit_notional = abs(float(exit_price) * float(qty))
    entry_cost = side_cost(entry_notional, costs)
    exit_cost = side_cost(exit_notional, costs)
    return entry_cost, exit_cost, entry_cost + exit_cost

def units_for_score(score, units_cfg):
    score = int(max(1, min(8, int(score))))
    return float(units_cfg["score_units"].get(str(score), 0.0))

def position_size(score, entry, units_cfg):
    unit_mult = units_for_score(score, units_cfg)
    notional = min(float(units_cfg["base_unit_dollars"]) * unit_mult, float(units_cfg["max_trade_dollars"]))
    qty = notional / float(entry) if entry > 0 else 0
    return float(qty), float(notional), float(unit_mult)

def cost_tradeoff(side, entry, target, qty, costs):
    if side == "LONG":
        expected_gross = (target - entry) * qty
    else:
        expected_gross = (entry - target) * qty

    _, _, expected_cost = estimate_costs(entry, target, qty, costs)
    expected_net = expected_gross - expected_cost

    if expected_gross <= 0:
        return False, expected_gross, expected_cost, expected_net, "הרווח הצפוי ליעד לא חיובי."

    ratio = (expected_cost / expected_gross) * 100
    max_ratio = float(costs["max_cost_to_target_pct"])

    if expected_net <= 0:
        return False, expected_gross, expected_cost, expected_net, "לא משתלם אחרי עלויות."
    if ratio > max_ratio:
        return False, expected_gross, expected_cost, expected_net, f"העלות {ratio:.1f}% מהרווח הצפוי — גבוה מדי."

    return True, expected_gross, expected_cost, expected_net, "משתלם אחרי עלויות."

def pnl_for_trade(row, current_price):
    entry = safe_float(row["entry_price"], 0)
    qty = safe_float(row["quantity"], 0)
    costs = {
        "cost_pct_per_side": safe_float(row["cost_pct_per_side"], DEFAULT_COSTS["cost_pct_per_side"]),
        "fixed_fee_per_side": safe_float(row["fixed_fee_per_side"], DEFAULT_COSTS["fixed_fee_per_side"]),
        "min_fee_per_side": safe_float(row["min_fee_per_side"], DEFAULT_COSTS["min_fee_per_side"]),
    }

    if str(row["side"]) == "LONG":
        gross = (float(current_price) - entry) * qty
    else:
        gross = (entry - float(current_price)) * qty

    entry_cost, exit_cost, total_cost = estimate_costs(entry, current_price, qty, costs)
    net = gross - total_cost
    notional = abs(entry * qty)
    net_pct = (net / notional) * 100 if notional > 0 else 0

    return {
        "entry_cost": entry_cost,
        "exit_cost": exit_cost,
        "total_cost": total_cost,
        "gross_pnl": gross,
        "net_pnl": net,
        "net_pnl_pct": net_pct,
    }


# ============================================================
# Signal logic
# ============================================================

def score_side_fast(d, side):
    if len(d) < 3:
        return 0, ["פחות מ־3 נרות"]

    last = d.iloc[-1]
    last3 = d.tail(3)
    close = safe_float(last["close"])
    score = 0
    reasons = []

    green = int((last3["close"] > last3["open"]).sum())
    red = int((last3["close"] < last3["open"]).sum())

    if side == "LONG":
        checks = [
            (close > safe_float(last["ema3"]) > safe_float(last["ema5"]), 2, "מחיר מעל EMA3/5"),
            (close > safe_float(last["ema9"]), 1, "מעל EMA9"),
            (close > safe_float(last["vwap"]), 1, "מעל VWAP"),
            (green >= 2, 1, "2 מתוך 3 נרות ירוקים"),
            (safe_float(last["mom2_pct"], 0) > 0.03, 1, "מומנטום קצר חיובי"),
            (safe_float(last["ema3_slope"], 0) > 0 and safe_float(last["ema5_slope"], 0) > 0, 1, "שיפוע EMA חיובי"),
            (safe_float(last["ema3_curv"], 0) > 0 or safe_float(last["rsi3_slope"], 0) > 0, 1, "שיפור בנגזרת/עקמומיות"),
            (42 <= safe_float(last["rsi3"], 50) <= 82, 1, "RSI3 תומך"),
            (safe_float(last["rel_vol5"], 1) >= 1.05, 1, "ווליום תומך"),
        ]
    else:
        checks = [
            (close < safe_float(last["ema3"]) < safe_float(last["ema5"]), 2, "מחיר מתחת EMA3/5"),
            (close < safe_float(last["ema9"]), 1, "מתחת EMA9"),
            (close < safe_float(last["vwap"]), 1, "מתחת VWAP"),
            (red >= 2, 1, "2 מתוך 3 נרות אדומים"),
            (safe_float(last["mom2_pct"], 0) < -0.03, 1, "מומנטום קצר שלילי"),
            (safe_float(last["ema3_slope"], 0) < 0 and safe_float(last["ema5_slope"], 0) < 0, 1, "שיפוע EMA שלילי"),
            (safe_float(last["ema3_curv"], 0) < 0 or safe_float(last["rsi3_slope"], 0) < 0, 1, "היחלשות בנגזרת/עקמומיות"),
            (18 <= safe_float(last["rsi3"], 50) <= 58, 1, "RSI3 תומך בשורט"),
            (safe_float(last["rel_vol5"], 1) >= 1.05, 1, "ווליום תומך"),
        ]

    for ok, pts, reason in checks:
        if ok:
            score += pts
            reasons.append(reason)

    return int(max(1, min(8, score))), reasons

def score_side_half(d, side):
    if len(d) < 12:
        return 0, ["פחות מדי נרות לחצי שעה"]

    last = d.iloc[-1]
    close = safe_float(last["close"])
    score = 0
    reasons = []

    if side == "LONG":
        checks = [
            (close > safe_float(last["ema9"]) > safe_float(last["ema21"]), 2, "מעל EMA9/21"),
            (close > safe_float(last["ema50"]), 1, "מעל EMA50"),
            (close > safe_float(last["vwap"]), 1, "מעל VWAP"),
            (safe_float(last["ema9_slope"], 0) > 0 and safe_float(last["ema21_slope"], 0) > 0, 1, "שיפוע EMA חיובי"),
            (safe_float(last["ema9_curv"], 0) > 0 or safe_float(last["macd_hist_curv"], 0) > 0, 1, "עקמומיות חיובית"),
            (safe_float(last["macd_hist_slope"], 0) > 0 or safe_float(last["rsi14_slope"], 0) > 0, 1, "אינדיקטורים משתפרים"),
            (safe_float(last["mom30_pct"], 0) > 0.05, 1, "מומנטום 30 דק׳ חיובי"),
            (45 <= safe_float(last["rsi14"], 50) <= 75, 1, "RSI14 תומך"),
            (safe_float(last["rel_vol20"], 1) >= 1, 1, "ווליום תומך"),
        ]
    else:
        checks = [
            (close < safe_float(last["ema9"]) < safe_float(last["ema21"]), 2, "מתחת EMA9/21"),
            (close < safe_float(last["ema50"]), 1, "מתחת EMA50"),
            (close < safe_float(last["vwap"]), 1, "מתחת VWAP"),
            (safe_float(last["ema9_slope"], 0) < 0 and safe_float(last["ema21_slope"], 0) < 0, 1, "שיפוע EMA שלילי"),
            (safe_float(last["ema9_curv"], 0) < 0 or safe_float(last["macd_hist_curv"], 0) < 0, 1, "עקמומיות שלילית"),
            (safe_float(last["macd_hist_slope"], 0) < 0 or safe_float(last["rsi14_slope"], 0) < 0, 1, "אינדיקטורים נחלשים"),
            (safe_float(last["mom30_pct"], 0) < -0.05, 1, "מומנטום 30 דק׳ שלילי"),
            (25 <= safe_float(last["rsi14"], 50) <= 55, 1, "RSI14 תומך בשורט"),
            (safe_float(last["rel_vol20"], 1) >= 1, 1, "ווליום תומך"),
        ]

    for ok, pts, reason in checks:
        if ok:
            score += pts
            reasons.append(reason)

    return int(max(1, min(8, score))), reasons

def make_signal(ticker, mode):
    df = latest_session(fetch_1m(ticker))
    if df.empty:
        return {"signal": "WAIT", "ticker": ticker, "mode": mode, "score": 0, "reason": "אין נתונים"}

    d = add_indicators(df).dropna(subset=["close"])
    if d.empty:
        return {"signal": "WAIT", "ticker": ticker, "mode": mode, "score": 0, "reason": "אין אינדיקטורים"}

    if mode == "מהירה":
        ls, lr = score_side_fast(d, "LONG")
        ss, sr = score_side_fast(d, "SHORT")
        atr = safe_float(d.iloc[-1]["atr3"], safe_float(d.iloc[-1]["close"]) * 0.001)
        stop_mult, target_mult = 0.85, 1.15
    else:
        ls, lr = score_side_half(d, "LONG")
        ss, sr = score_side_half(d, "SHORT")
        atr = safe_float(d.iloc[-1]["atr14"], safe_float(d.iloc[-1]["close"]) * 0.002)
        stop_mult, target_mult = 1.20, 1.90

    entry = safe_float(d.iloc[-1]["close"])
    atr = max(atr, entry * 0.0008)

    if ls >= ss and ls >= 3:
        side, score, reasons = "LONG", ls, lr
        stop = entry - stop_mult * atr
        target = entry + target_mult * atr
    elif ss > ls and ss >= 3:
        side, score, reasons = "SHORT", ss, sr
        stop = entry + stop_mult * atr
        target = entry - target_mult * atr
    else:
        return {"signal": "WAIT", "ticker": ticker, "mode": mode, "score": max(ls, ss), "reason": f"לונג {ls}, שורט {ss} — אין יתרון ברור"}

    return {
        "signal": side,
        "ticker": normalize_ticker(ticker),
        "mode": mode,
        "score": int(score),
        "entry": float(entry),
        "stop": float(stop),
        "target": float(target),
        "reason": " | ".join(reasons),
    }


# ============================================================
# Trade lifecycle
# ============================================================

def trade_age_minutes(row):
    try:
        entry = pd.Timestamp(row["entry_time"])
        if entry.tzinfo is None:
            entry = entry.tz_localize(NY_TZ)
        else:
            entry = entry.tz_convert(NY_TZ)
        return max(0.0, (now_ny() - entry).total_seconds() / 60)
    except Exception:
        return 0.0

def min_hold_for_mode(mode, rules):
    if str(mode) == "מהירה":
        return float(rules["min_hold_fast_minutes"])
    return float(rules["min_hold_half_hour_minutes"])

def has_open_trade(trades, ticker, mode):
    if trades.empty:
        return False
    return bool((trades["status"].eq("OPEN") & trades["ticker"].astype(str).eq(ticker) & trades["mode"].astype(str).eq(mode)).any())

def in_cooldown(trades, ticker, mode, rules):
    if trades.empty:
        return False, ""
    closed = trades[
        trades["status"].eq("CLOSED")
        & trades["ticker"].astype(str).eq(ticker)
        & trades["mode"].astype(str).eq(mode)
    ].copy()
    if closed.empty:
        return False, ""

    closed = closed.dropna(subset=["exit_time"])
    if closed.empty:
        return False, ""

    try:
        last_exit = pd.Timestamp(closed["exit_time"].iloc[-1])
        if last_exit.tzinfo is None:
            last_exit = last_exit.tz_localize(NY_TZ)
        else:
            last_exit = last_exit.tz_convert(NY_TZ)
        minutes = (now_ny() - last_exit).total_seconds() / 60
        needed = float(rules["cooldown_after_close_minutes"])
        if minutes < needed:
            return True, f"Cooldown: נסגרה עסקה לפני {minutes:.1f} דק׳, מחכים {needed:.0f} דק׳."
    except Exception:
        return False, ""

    return False, ""

def open_trade(signal, min_score):
    trades = load_trades()
    costs = load_costs()
    units = load_units()
    rules = load_rules()

    ticker = normalize_ticker(signal["ticker"])
    mode = signal["mode"]
    side = signal["signal"]
    score = int(signal["score"])

    if side not in ["LONG", "SHORT"]:
        return False, f"{ticker}: אין איתות."
    if score < int(min_score):
        return False, f"{ticker}: ניקוד {score} נמוך מהמינימום."
    if has_open_trade(trades, ticker, mode):
        return False, f"{ticker}: כבר יש עסקה פתוחה ב־{mode}."

    cd, msg = in_cooldown(trades, ticker, mode, rules)
    if cd:
        return False, f"{ticker}: {msg}"

    entry = float(signal["entry"])
    stop = float(signal["stop"])
    target = float(signal["target"])

    qty, notional, unit_mult = position_size(score, entry, units)
    if qty <= 0 or notional <= 0:
        return False, f"{ticker}: לפי יוניטים, ניקוד {score} לא מקבל כניסה."

    ok, eg, ec, en, msg = cost_tradeoff(side, entry, target, qty, costs)
    if not ok:
        return False, f"{ticker}: {msg} ברוטו ${eg:.2f}, עלות ${ec:.2f}, נטו ${en:.2f}."

    entry_cost, exit_cost, total_cost_now = estimate_costs(entry, entry, qty, costs)

    row = {
        "trade_id": str(uuid.uuid4()),
        "status": "OPEN",
        "ticker": ticker,
        "mode": mode,
        "side": side,
        "score": score,
        "entry_time": now_ny_iso(),
        "exit_time": "",
        "age_minutes": 0.0,
        "entry_price": entry,
        "current_price": entry,
        "exit_price": np.nan,
        "quantity": qty,
        "notional": notional,
        "stop_loss": stop,
        "initial_stop_loss": stop,
        "profit_stop": np.nan,
        "target_reference": target,
        "highest_price": entry,
        "lowest_price": entry,
        "entry_cost": entry_cost,
        "exit_cost": exit_cost,
        "total_cost": total_cost_now,
        "gross_pnl": 0.0,
        "net_pnl": -total_cost_now,
        "net_pnl_pct": (-total_cost_now / notional) * 100 if notional else 0,
        "exit_reason": "",
        "management_action": "OPENED",
        "management_reason": "נפתחה עסקה אחרי בדיקת ניקוד, עלויות ויוניטים.",
        "signal_reason": signal.get("reason", ""),
        "cost_pct_per_side": costs["cost_pct_per_side"],
        "fixed_fee_per_side": costs["fixed_fee_per_side"],
        "min_fee_per_side": costs["min_fee_per_side"],
        "max_cost_to_target_pct": costs["max_cost_to_target_pct"],
        "base_unit_dollars": units["base_unit_dollars"],
        "unit_multiplier": unit_mult,
        "created_settings_snapshot": json.dumps({"costs": costs, "units": units, "rules": rules}, ensure_ascii=False),
    }

    trades = pd.concat([trades, pd.DataFrame([row])], ignore_index=True)
    save_trades(trades)

    return True, f"{ticker}: נפתחה {side} | {mode} | ניקוד {score} | יוניטים {unit_mult} | נטו צפוי ליעד ${en:.2f}."

def manage_trade(row, df_after_entry):
    rules = load_rules()
    side = str(row["side"])
    mode = str(row["mode"])
    score = int(safe_float(row["score"], 1))
    entry = safe_float(row["entry_price"])
    stop = safe_float(row["stop_loss"])
    initial_stop = safe_float(row["initial_stop_loss"], stop)
    target = safe_float(row["target_reference"])
    age = trade_age_minutes(row)
    min_hold = min_hold_for_mode(mode, rules)

    res = {
        "exit": False,
        "exit_reason": "",
        "stop_loss": stop,
        "profit_stop": safe_float(row.get("profit_stop"), np.nan),
        "target_reference": target,
        "highest_price": safe_float(row.get("highest_price"), entry),
        "lowest_price": safe_float(row.get("lowest_price"), entry),
        "action": "HOLD",
        "reason": "מחזיק, אין שינוי.",
    }

    if df_after_entry is None or df_after_entry.empty:
        return res

    d = add_indicators(df_after_entry).dropna(subset=["close"])
    if d.empty:
        return res

    last = d.iloc[-1]
    current = safe_float(last["close"])
    high_since = max(res["highest_price"], safe_float(d["high"].max(), current))
    low_since = min(res["lowest_price"], safe_float(d["low"].min(), current))
    res["highest_price"] = high_since
    res["lowest_price"] = low_since

    base_risk = abs(entry - initial_stop)
    if base_risk <= 0:
        base_risk = max(entry * 0.001, abs(entry - stop))

    last3 = d.tail(min(3, len(d)))
    green = int((last3["close"] > last3["open"]).sum())
    red = int((last3["close"] < last3["open"]).sum())

    ema5 = safe_float(last["ema5"], current)
    ema5_slope = safe_float(last["ema5_slope"], 0)
    ema5_curv = safe_float(last["ema5_curv"], 0)
    macd_slope = safe_float(last["macd_hist_slope"], 0)

    # Higher score gets more room to run. Lower score is managed tighter.
    if score >= 7:
        trail_r = 0.95
    elif score >= 6:
        trail_r = 0.75
    elif score >= 5:
        trail_r = 0.55
    else:
        trail_r = 0.38

    actions, reasons = [], []

    # Hard stop is allowed immediately. Everything else waits at least min hold.
    if side == "LONG":
        r_now = (current - entry) / base_risk
        best_r = (high_since - entry) / base_risk

        if current <= stop:
            res["exit"] = True
            res["exit_reason"] = "STOP_LOSS"
            actions.append("EXIT_STOP")
            reasons.append("הגיע לסטופ.")

        if age >= min_hold:
            if best_r >= float(rules["min_profit_r_for_profit_stop"]):
                new_profit_stop = max(entry + 0.05 * base_risk, high_since - trail_r * base_risk)
                if not np.isfinite(res["profit_stop"]) or new_profit_stop > res["profit_stop"]:
                    res["profit_stop"] = new_profit_stop
                    actions.append("RAISE_PROFIT_STOP")
                    reasons.append("העסקה ברווח, סטופ רווח עלה.")

            if score >= 6 and r_now >= 0.85 and current > ema5 and ema5_slope > 0:
                new_target = max(target, current + 1.25 * base_risk)
                if new_target > target:
                    res["target_reference"] = new_target
                    actions.append("EXTEND_TARGET")
                    reasons.append("ניקוד גבוה ומומנטום חיובי — נותנים לעסקה לרוץ.")

            if score < 6 and current >= target:
                res["exit"] = True
                res["exit_reason"] = "TARGET_REACHED"
                actions.append("EXIT_TARGET")
                reasons.append("ניקוד לא גבוה, יציאה ביעד.")

            if np.isfinite(res["profit_stop"]) and current <= res["profit_stop"]:
                res["exit"] = True
                res["exit_reason"] = "PROFIT_STOP"
                actions.append("EXIT_PROFIT_STOP")
                reasons.append("חזרה לסטופ רווח.")

            if best_r >= 0.7 and (red >= 2 or ema5_curv < 0 or macd_slope < 0):
                tightened = current - 0.28 * base_risk
                if not np.isfinite(res["profit_stop"]) or tightened > res["profit_stop"]:
                    res["profit_stop"] = tightened
                    actions.append("TIGHTEN_PROFIT_STOP")
                    reasons.append("אחרי רווח יש היחלשות, סטופ רווח הודק.")

        # Emergency exit only after a short minimum, to avoid instant entry/exit.
        if age >= float(rules["emergency_exit_after_minutes"]) and r_now < -0.30 and red >= 2 and current < ema5 and ema5_slope < 0:
            res["exit"] = True
            res["exit_reason"] = "EARLY_EXIT_AGAINST_LONG"
            actions.append("EARLY_EXIT")
            reasons.append("העסקה הולכת חזק נגד לונג, יציאה מוקדמת.")

    else:
        r_now = (entry - current) / base_risk
        best_r = (entry - low_since) / base_risk

        if current >= stop:
            res["exit"] = True
            res["exit_reason"] = "STOP_LOSS"
            actions.append("EXIT_STOP")
            reasons.append("הגיע לסטופ.")

        if age >= min_hold:
            if best_r >= float(rules["min_profit_r_for_profit_stop"]):
                new_profit_stop = min(entry - 0.05 * base_risk, low_since + trail_r * base_risk)
                if not np.isfinite(res["profit_stop"]) or new_profit_stop < res["profit_stop"]:
                    res["profit_stop"] = new_profit_stop
                    actions.append("LOWER_PROFIT_STOP")
                    reasons.append("העסקה ברווח, סטופ רווח ירד.")

            if score >= 6 and r_now >= 0.85 and current < ema5 and ema5_slope < 0:
                new_target = min(target, current - 1.25 * base_risk)
                if new_target < target:
                    res["target_reference"] = new_target
                    actions.append("EXTEND_TARGET")
                    reasons.append("ניקוד גבוה ומומנטום שלילי — נותנים לשורט לרוץ.")

            if score < 6 and current <= target:
                res["exit"] = True
                res["exit_reason"] = "TARGET_REACHED"
                actions.append("EXIT_TARGET")
                reasons.append("ניקוד לא גבוה, יציאה ביעד.")

            if np.isfinite(res["profit_stop"]) and current >= res["profit_stop"]:
                res["exit"] = True
                res["exit_reason"] = "PROFIT_STOP"
                actions.append("EXIT_PROFIT_STOP")
                reasons.append("חזרה לסטופ רווח.")

            if best_r >= 0.7 and (green >= 2 or ema5_curv > 0 or macd_slope > 0):
                tightened = current + 0.28 * base_risk
                if not np.isfinite(res["profit_stop"]) or tightened < res["profit_stop"]:
                    res["profit_stop"] = tightened
                    actions.append("TIGHTEN_PROFIT_STOP")
                    reasons.append("אחרי רווח יש היחלשות, סטופ רווח הודק.")

        if age >= float(rules["emergency_exit_after_minutes"]) and r_now < -0.30 and green >= 2 and current > ema5 and ema5_slope > 0:
            res["exit"] = True
            res["exit_reason"] = "EARLY_EXIT_AGAINST_SHORT"
            actions.append("EARLY_EXIT")
            reasons.append("העסקה הולכת חזק נגד שורט, יציאה מוקדמת.")

    if actions:
        res["action"] = " + ".join(sorted(set(actions)))
        res["reason"] = " ".join(reasons)

    return res

def update_open_trades():
    trades = load_trades()
    messages = []
    if trades.empty:
        return trades, messages

    open_idx = trades.index[trades["status"].eq("OPEN")].tolist()
    for idx in open_idx:
        ticker = str(trades.loc[idx, "ticker"])
        try:
            df = latest_session(fetch_1m(ticker))
            if df.empty:
                continue

            current = safe_float(df.iloc[-1]["close"])
            entry_time = pd.Timestamp(trades.loc[idx, "entry_time"])
            if entry_time.tzinfo is None:
                entry_time = entry_time.tz_localize(NY_TZ)
            else:
                entry_time = entry_time.tz_convert(NY_TZ)

            after_entry = df[df.index >= entry_time]
            if after_entry.empty:
                after_entry = df.tail(5)

            decision = manage_trade(trades.loc[idx], after_entry)

            trades.loc[idx, "age_minutes"] = trade_age_minutes(trades.loc[idx])
            trades.loc[idx, "current_price"] = current
            trades.loc[idx, "stop_loss"] = decision["stop_loss"]
            trades.loc[idx, "profit_stop"] = decision["profit_stop"]
            trades.loc[idx, "target_reference"] = decision["target_reference"]
            trades.loc[idx, "highest_price"] = decision["highest_price"]
            trades.loc[idx, "lowest_price"] = decision["lowest_price"]
            trades.loc[idx, "management_action"] = decision["action"]
            trades.loc[idx, "management_reason"] = decision["reason"]

            pnl = pnl_for_trade(trades.loc[idx], current)
            for k, v in pnl.items():
                trades.loc[idx, k] = v

            if decision["exit"]:
                trades.loc[idx, "status"] = "CLOSED"
                trades.loc[idx, "exit_time"] = now_ny_iso()
                trades.loc[idx, "exit_price"] = current
                trades.loc[idx, "exit_reason"] = decision["exit_reason"]
                messages.append(f"{ticker}: נסגרה עסקה — {decision['exit_reason']} | נטו ${pnl['net_pnl']:.2f}")

        except Exception as e:
            trades.loc[idx, "management_action"] = "ERROR"
            trades.loc[idx, "management_reason"] = str(e)[:180]

    save_trades(trades)
    return trades, messages

def close_trade_manually(trade_id):
    trades = load_trades()
    mask = trades["trade_id"].astype(str).eq(str(trade_id)) & trades["status"].eq("OPEN")
    if trades.empty or not mask.any():
        return False, "העסקה לא נמצאה או כבר סגורה."

    idx = trades.index[mask][0]
    ticker = str(trades.loc[idx, "ticker"])

    try:
        df = latest_session(fetch_1m(ticker))
        current = safe_float(df.iloc[-1]["close"]) if not df.empty else safe_float(trades.loc[idx, "current_price"])
    except Exception:
        current = safe_float(trades.loc[idx, "current_price"])

    pnl = pnl_for_trade(trades.loc[idx], current)
    for k, v in pnl.items():
        trades.loc[idx, k] = v

    trades.loc[idx, "current_price"] = current
    trades.loc[idx, "exit_price"] = current
    trades.loc[idx, "status"] = "CLOSED"
    trades.loc[idx, "exit_time"] = now_ny_iso()
    trades.loc[idx, "exit_reason"] = "MANUAL_CLOSE"
    trades.loc[idx, "management_action"] = "MANUAL_CLOSE"
    trades.loc[idx, "management_reason"] = "נסגר ידנית על ידי המשתמש."

    save_trades(trades)
    return True, f"{ticker}: נסגר ידנית במחיר {current:.2f}. נטו ${pnl['net_pnl']:.2f}"

def scan_and_open(tickers, modes, min_score):
    messages = []
    opened = 0
    max_new = int(load_rules()["max_new_trades_per_scan"])

    for ticker in tickers:
        for mode in modes:
            if opened >= max_new:
                messages.append(f"הגעת למקסימום עסקאות חדשות בסריקה: {max_new}.")
                return messages

            try:
                sig = make_signal(ticker, mode)
                ok, msg = open_trade(sig, min_score)
                if ok:
                    opened += 1
                    messages.append(msg)
                elif "לא משתלם" in msg or "Cooldown" in msg:
                    messages.append(msg)
            except Exception as e:
                messages.append(f"{ticker} | {mode}: שגיאה {str(e)[:100]}")
            time.sleep(0.05)

    return messages


# ============================================================
# Summary + display
# ============================================================

def fmt_price(x):
    return "" if pd.isna(x) else f"{safe_float(x):.2f}"

def fmt_money(x):
    return f"${safe_float(x, 0):,.2f}"

def summary_stats(trades):
    if trades.empty:
        return {
            "opened_count": 0, "open_count": 0, "closed_count": 0,
            "gross_total": 0.0, "entry_cost_total": 0.0,
            "cost_total": 0.0, "net_total": 0.0,
        }

    for col in ["gross_pnl", "entry_cost", "total_cost", "net_pnl"]:
        trades[col] = pd.to_numeric(trades[col], errors="coerce").fillna(0)

    return {
        "opened_count": int(len(trades)),
        "open_count": int(trades["status"].eq("OPEN").sum()),
        "closed_count": int(trades["status"].eq("CLOSED").sum()),
        "gross_total": float(trades["gross_pnl"].sum()),
        "entry_cost_total": float(trades["entry_cost"].sum()),
        "cost_total": float(trades["total_cost"].sum()),
        "net_total": float(trades["net_pnl"].sum()),
    }

def render_summary(trades):
    stats = summary_stats(trades)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("רווח כולל נטו", fmt_money(stats["net_total"]))
    c2.metric("רווח מהעסקאות ברוטו", fmt_money(stats["gross_total"]))
    c3.metric("עלות כניסה כוללת", fmt_money(stats["entry_cost_total"]))
    c4.metric("סך כל העלויות", fmt_money(stats["cost_total"]))

    d1, d2, d3 = st.columns(3)
    d1.metric("כמות עסקאות שנפתחו", stats["opened_count"])
    d2.metric("עסקאות כעת", stats["open_count"])
    d3.metric("עסקאות סגורות", stats["closed_count"])

def render_open_trades(open_trades):
    st.markdown("### עסקאות כעת")

    if open_trades.empty:
        st.info("אין עסקאות פתוחות כרגע.")
        return

    head = st.columns([0.65, .8, .9, .7, .85, .85, .85, .85, .9, .95, .75, .7])
    labels = ["סיים", "מניה", "סוג", "כיוון", "כניסה", "נוכחי", "סטופ", "סטופ רווח", "רווח/הפסד", "זמן כניסה", "גיל דק׳", "ניקוד"]
    for col, label in zip(head, labels):
        col.markdown(f"**{label}**")

    for _, r in open_trades.iterrows():
        pnl = safe_float(r["net_pnl"], 0)
        klass = "green-row" if pnl >= 0 else "red-row"

        st.markdown(f"<div class='{klass}'>", unsafe_allow_html=True)
        row = st.columns([0.65, .8, .9, .7, .85, .85, .85, .85, .9, .95, .75, .7])

        if row[0].button("סיים", key=f"close_{r['trade_id']}"):
            ok, msg = close_trade_manually(str(r["trade_id"]))
            if ok:
                st.success(msg)
                st.rerun()
            else:
                st.error(msg)

        row[1].write(str(r["ticker"]))
        row[2].write(str(r["mode"]))
        row[3].write(str(r["side"]))
        row[4].write(fmt_price(r["entry_price"]))
        row[5].write(fmt_price(r["current_price"]))
        row[6].write(fmt_price(r["stop_loss"]))
        row[7].write(fmt_price(r["profit_stop"]))
        row[8].write(fmt_money(pnl))
        row[9].write(str(r["entry_time"])[:19])
        row[10].write(f"{safe_float(r['age_minutes'], 0):.1f}")
        row[11].write(int(safe_float(r["score"], 0)))

        with st.expander(f"ניהול: {r['ticker']} | {r['mode']} | {str(r['trade_id'])[:8]}"):
            st.write("פעולה אחרונה:", r.get("management_action", ""))
            st.write("סיבה:", r.get("management_reason", ""))
            st.write("למה נכנס:", r.get("signal_reason", ""))
            st.write("עלות כוללת:", fmt_money(r.get("total_cost", 0)))
        st.markdown("</div>", unsafe_allow_html=True)

def render_closed_trades(closed_trades):
    st.markdown("### עסקאות שהסתיימו")

    if closed_trades.empty:
        st.info("אין עסקאות סגורות עדיין.")
        return

    d = closed_trades.sort_values("exit_time", ascending=False).copy().reset_index(drop=True)

    display = pd.DataFrame({
        "מניה": d["ticker"],
        "סוג": d["mode"],
        "כיוון": d["side"],
        "מחיר כניסה": d["entry_price"].map(fmt_price),
        "מחיר יציאה": d["exit_price"].map(fmt_price),
        "סטופ": d["stop_loss"].map(fmt_price),
        "סטופ רווח": d["profit_stop"].map(fmt_price),
        "רווח/הפסד": d["net_pnl"].map(fmt_money),
        "זמן כניסה": d["entry_time"].astype(str).str.slice(0, 19),
        "זמן יציאה": d["exit_time"].astype(str).str.slice(0, 19),
        "ניקוד": d["score"].fillna(0).astype(int),
        "סיבה": d["exit_reason"],
    })

    pnl_values = d["net_pnl"].fillna(0).astype(float).tolist()

    def style_row(row):
        pnl = pnl_values[row.name]
        if pnl >= 0:
            return ["background-color:#dcfce7;color:#064e3b;"] * len(row)
        return ["background-color:#fee2e2;color:#7f1d1d;"] * len(row)

    st.dataframe(display.style.apply(style_row, axis=1), use_container_width=True, hide_index=True)


# ============================================================
# Main UI
# ============================================================

st.markdown(
    """
<div class="title-box">
<h1>🧪 Paper Trading Lab V2</h1>
<p>אפליקציית Paper Trading נקייה: עסקאות מהירות/חצי שעה, עלויות, יוניטים לפי ניקוד, וסיכום רווח נטו.</p>
</div>
""",
    unsafe_allow_html=True,
)

tab_paper, tab_costs, tab_units, tab_rules, tab_help = st.tabs([
    "🧪 Paper Trading",
    "💸 עלויות",
    "📦 יוניטים לפי ניקוד",
    "⚙️ חוקים חכמים",
    "📘 הסבר",
])


with tab_paper:
    st.markdown("<div class='card warn'><strong>בדמו בלבד:</strong> אין חיבור לברוקר ואין כסף אמיתי.</div>", unsafe_allow_html=True)

    trades, update_msgs = update_open_trades()
    trades = load_trades()

    render_summary(trades)

    st.markdown("---")

    a, b, c, d = st.columns([1.4, 1.2, 1.1, 1.1])

    with a:
        tickers = load_tickers()
        selected_tickers = st.multiselect("מניות לסריקה", tickers, default=tickers[:min(8, len(tickers))])
        new_ticker = st.text_input("הוסף מניה", placeholder="לדוגמה: QQQ / NVDA / PLTR")
        if st.button("➕ הוסף מניה", use_container_width=True):
            t = normalize_ticker(new_ticker)
            if t:
                tickers.append(t)
                save_tickers(tickers)
                st.success(f"{t} נוסף.")
                st.rerun()

    with b:
        modes = st.multiselect("סוג השקעה", ["מהירה", "חצי שעה"], default=["מהירה", "חצי שעה"])
        min_score = st.slider("מינימום ניקוד לפתיחה", 1, 8, 4)

    with c:
        run_scan = st.button("▶️ סרוק ופתח עסקאות", use_container_width=True)
        update_now = st.button("🔄 עדכן עסקאות פתוחות", use_container_width=True)
        auto_run = st.checkbox("הרצה כל 30 שניות", value=False)

    with d:
        clear_all = st.button("🧹 ניקוי עסקאות", use_container_width=True)
        st.caption("שינוי עלויות/יוניטים/חוקים משפיע רק על העסקאות הבאות, לא על עסקאות פתוחות.")

    if clear_all:
        clear_trades()
        st.success("נוקה.")
        st.rerun()

    if update_now:
        trades, msgs = update_open_trades()
        for msg in msgs:
            st.info(msg)
        st.success("עודכן.")
        st.rerun()

    if run_scan or auto_run:
        if not selected_tickers or not modes:
            st.warning("בחר מניות וסוג השקעה.")
        else:
            with st.spinner("סורק, בודק ניקוד, עלויות ו־cooldown..."):
                msgs = scan_and_open(selected_tickers, modes, min_score)
                trades, _ = update_open_trades()

            if msgs:
                with st.expander("תוצאות סריקה", expanded=True):
                    for msg in msgs[:80]:
                        if "נפתחה" in msg:
                            st.success(msg)
                        elif "לא משתלם" in msg or "Cooldown" in msg:
                            st.warning(msg)
                        else:
                            st.info(msg)

    trades = load_trades()
    open_trades = trades[trades["status"].eq("OPEN")].copy() if not trades.empty else empty_trades()
    closed_trades = trades[trades["status"].eq("CLOSED")].copy() if not trades.empty else empty_trades()

    render_open_trades(open_trades)
    render_closed_trades(closed_trades)

    if auto_run:
        time.sleep(30)
        st.rerun()


with tab_costs:
    st.subheader("💸 עלויות — משפיע רק על עסקאות חדשות")
    costs = load_costs()

    c1, c2 = st.columns(2)
    with c1:
        cost_pct = st.number_input("עלות משתנה לכל צד (%)", 0.0, 3.0, float(costs["cost_pct_per_side"]), 0.01)
        fixed_fee = st.number_input("עמלה קבועה לכל צד ($)", 0.0, 100.0, float(costs["fixed_fee_per_side"]), 0.10)
    with c2:
        min_fee = st.number_input("מינימום עמלה לכל צד ($)", 0.0, 100.0, float(costs["min_fee_per_side"]), 0.10)
        max_ratio = st.slider("מקסימום עלות מתוך הרווח הצפוי ליעד (%)", 1, 100, int(float(costs["max_cost_to_target_pct"])))

    ex = st.number_input("דוגמה: שווי עסקה ($)", 50.0, 100000.0, 1000.0, 50.0)
    temp_costs = {
        "cost_pct_per_side": cost_pct,
        "fixed_fee_per_side": fixed_fee,
        "min_fee_per_side": min_fee,
        "max_cost_to_target_pct": max_ratio,
    }
    entry_cost = side_cost(ex, temp_costs)
    st.info(f"בדוגמה של ${ex:,.0f}: עלות כניסה ≈ ${entry_cost:.2f}, עלות כניסה+יציאה ≈ ${entry_cost*2:.2f}")

    if st.button("💾 שמור עלויות", use_container_width=True):
        save_costs(temp_costs)
        st.success("נשמר. ישפיע רק על עסקאות חדשות.")


with tab_units:
    st.subheader("📦 יוניטים לפי ניקוד — משפיע רק על עסקאות חדשות")
    units = load_units()

    u1, u2 = st.columns(2)
    with u1:
        base_unit = st.number_input("ערך יוניט אחד ($)", 10.0, 100000.0, float(units["base_unit_dollars"]), 10.0)
    with u2:
        max_trade = st.number_input("מקסימום כסף לעסקה אחת ($)", 10.0, 1000000.0, float(units["max_trade_dollars"]), 50.0)

    score_units = dict(units["score_units"])
    new_score_units = {}
    cols = st.columns(4)

    for score in range(1, 9):
        with cols[(score - 1) % 4]:
            new_score_units[str(score)] = st.number_input(
                f"ניקוד {score}",
                min_value=0.0,
                max_value=50.0,
                value=float(score_units.get(str(score), 0.0)),
                step=0.25,
                key=f"score_unit_{score}",
            )

    preview = []
    for score in range(1, 9):
        mult = float(new_score_units[str(score)])
        dollars = min(base_unit * mult, max_trade)
        preview.append({"ניקוד": score, "יוניטים": mult, "כסף לעסקה": f"${dollars:,.2f}"})
    st.dataframe(pd.DataFrame(preview), use_container_width=True, hide_index=True)

    if st.button("💾 שמור יוניטים", use_container_width=True):
        save_units({"base_unit_dollars": base_unit, "max_trade_dollars": max_trade, "score_units": new_score_units})
        st.success("נשמר. ישפיע רק על עסקאות חדשות.")


with tab_rules:
    st.subheader("⚙️ חוקים חכמים נגד כניסה/יציאה מהירה מדי")
    rules = load_rules()

    r1, r2 = st.columns(2)
    with r1:
        min_hold_fast = st.number_input("מינימום החזקה לעסקה מהירה, בדקות", 0, 60, int(rules["min_hold_fast_minutes"]))
        min_hold_half = st.number_input("מינימום החזקה לעסקת חצי שעה, בדקות", 0, 120, int(rules["min_hold_half_hour_minutes"]))
        cooldown = st.number_input("Cooldown אחרי סגירה, בדקות", 0, 120, int(rules["cooldown_after_close_minutes"]))
    with r2:
        max_new = st.number_input("מקסימום עסקאות חדשות בכל סריקה", 1, 20, int(rules["max_new_trades_per_scan"]))
        profit_r = st.number_input("כמה רווח R לפני הפעלת סטופ רווח", 0.1, 3.0, float(rules["min_profit_r_for_profit_stop"]), 0.05)
        emergency_minutes = st.number_input("מינימום דקות לפני יציאה מוקדמת נגד הכיוון", 0, 30, int(rules["emergency_exit_after_minutes"]))

    st.markdown(
        """
<div class="card">
<strong>מה זה עושה?</strong><br>
האפליקציה לא תיכנס ותצא סתם מהר: היא ממתינה מינימום זמן לפני יציאה רגילה,
שומרת cooldown אחרי סגירה, ומגבילה כמה עסקאות חדשות נפתחות בכל סריקה.
סטופ לוס קשיח עדיין יכול לסגור מיד כדי להגן על ההפסד.
</div>
""",
        unsafe_allow_html=True,
    )

    if st.button("💾 שמור חוקים", use_container_width=True):
        save_rules({
            "min_hold_fast_minutes": int(min_hold_fast),
            "min_hold_half_hour_minutes": int(min_hold_half),
            "cooldown_after_close_minutes": int(cooldown),
            "max_new_trades_per_scan": int(max_new),
            "min_profit_r_for_profit_stop": float(profit_r),
            "emergency_exit_after_minutes": int(emergency_minutes),
        })
        st.success("נשמר.")


with tab_help:
    st.subheader("📘 הסבר")
    st.markdown(
        """
### מה חדש ב־V2?

**סיכום עליון**
- רווח כולל נטו = רווח אחרי כל העלויות.
- רווח מהעסקאות ברוטו = לפני עלויות.
- עלות כניסה כוללת = כמה עלתה הכניסה לכל העסקאות.
- סך כל העלויות = כניסה + יציאה משוערת.
- כמות עסקאות שנפתחו, עסקאות כעת, עסקאות סגורות.

**לא נכנס ויוצא מהר**
- יש מינימום זמן החזקה.
- יש cooldown אחרי סגירה.
- יש מקסימום עסקאות חדשות בכל סריקה.
- יציאה מוקדמת נגד הכיוון קיימת, אבל רק אחרי זמן מינימלי.
- סטופ לוס קשיח עדיין יכול לסגור מיד כדי לא לתת להפסד לברוח.

**שינוי הגדרות**
שינוי עלויות, יוניטים או חוקים משפיע רק על עסקאות חדשות.
עסקה שכבר פתוחה שומרת את ההגדרות שהיו בזמן הפתיחה.

**ניקוד**
הניקוד 1–8 מבוסס לא רק על “מעל/מתחת”, אלא גם על:
- שיפוע EMA/VWAP/RSI
- עקמומיות, כלומר האם השיפוע מתחזק או נחלש
- מומנטום
- ווליום
- מבנה נרות
"""
    )
