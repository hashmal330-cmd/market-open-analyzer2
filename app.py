
import json
import time
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go


# ============================================================
# App setup
# ============================================================

st.set_page_config(page_title="Paper Trading Lab V5.2", page_icon="🧪", layout="wide")

DATA_DIR = Path("paper_data")
DATA_DIR.mkdir(exist_ok=True)

TRADES_FILE = DATA_DIR / "trades_v3.csv"
TICKERS_FILE = DATA_DIR / "tickers_v3.json"
COSTS_FILE = DATA_DIR / "costs_v3.json"
UNITS_FILE = DATA_DIR / "units_v3.json"
RULES_FILE = DATA_DIR / "rules_v3.json"
ACCOUNT_FILE = DATA_DIR / "account_v3.json"
PENDING_FILE = DATA_DIR / "pending_signals_v5.csv"

NY_TZ = "America/New_York"

DEFAULT_TICKERS = [
    "QQQ", "SPY", "IWM", "DIA", "TQQQ", "SQQQ",
    "AAPL", "MSFT", "NVDA", "AMD", "AVGO", "ARM", "INTC", "MU", "MRVL", "SMCI",
    "TSLA", "META", "GOOGL", "AMZN", "NFLX",
    "PLTR", "MSTR", "COIN", "HOOD", "SOFI", "UBER", "SHOP", "SNOW",
    "CRM", "ORCL", "ADBE", "PANW", "CRWD", "BABA",
    "JPM", "BAC", "XOM", "CVX", "LLY", "UNH",
]

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
    "max_open_trades": 6,

    # Profit-taking and protection
    "cycle_net_profit_target": 50.0,
    "min_profit_r_for_profit_stop": 0.45,
    "emergency_exit_after_minutes": 2,
    "breakeven_after_profit_dollars": 4.0,
    "lock_profit_after_net_dollars": 8.0,
    "max_allowed_loss_per_trade_dollars": 20.0,
    "exit_if_profitable_trade_turns_red": True,
    "exit_on_target_when_score_below": 7,
    "profit_giveback_pct": 10.0,
    "min_net_profit_for_giveback": 5.0,
    "confirm_before_entry_seconds": 60,
    "pending_signal_expire_minutes": 6,
}

DEFAULT_ACCOUNT = {
    "starting_balance": 10000.0,
    "cycles_completed": 0,
    "locked_profit": 0.0,
    "last_cycle_closed_at": "",
    "last_cycle_reason": "",
}

TRADE_COLUMNS = [
    "trade_id", "status", "ticker", "mode", "side", "score",
    "entry_time", "exit_time", "duration_minutes", "age_minutes",
    "entry_price", "current_price", "exit_price",
    "quantity", "notional",
    "stop_loss", "initial_stop_loss", "manual_stop_loss", "profit_stop", "target_reference", "breakeven_price",
    "highest_price", "lowest_price", "max_net_pnl_seen",
    "entry_cost", "exit_cost", "total_cost",
    "gross_pnl", "net_pnl", "net_pnl_pct",
    "exit_reason", "exit_reason_he", "management_action", "management_reason", "signal_reason",
    "cost_pct_per_side", "fixed_fee_per_side", "min_fee_per_side", "max_cost_to_target_pct",
    "base_unit_dollars", "unit_multiplier",
    "created_settings_snapshot",
]

PENDING_COLUMNS = [
    "pending_id", "created_at", "ticker", "mode", "side", "score",
    "entry_price", "stop_loss", "target_reference", "reason",
    "status", "last_checked_at", "message",
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
    tickers = sorted(set(normalize_ticker(x) for x in data.get("tickers", DEFAULT_TICKERS) if normalize_ticker(x)))
    if len(tickers) < 20:
        tickers = sorted(set(tickers + DEFAULT_TICKERS))
    return tickers

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

def load_account():
    return read_json(ACCOUNT_FILE, DEFAULT_ACCOUNT)

def save_account(account):
    write_json(ACCOUNT_FILE, account)

def reset_account():
    save_account(DEFAULT_ACCOUNT)

def timestamp_to_ny(ts):
    try:
        out = pd.Timestamp(ts)
        if out.tzinfo is None:
            out = out.tz_localize(NY_TZ)
        else:
            out = out.tz_convert(NY_TZ)
        return out
    except Exception:
        return None

def minutes_between(start, end):
    s = timestamp_to_ny(start)
    e = timestamp_to_ny(end)
    if s is None or e is None:
        return 0.0
    return max(0.0, (e - s).total_seconds() / 60.0)

def exit_reason_he(reason):
    mapping = {
        "STOP_LOSS": "הגענו לסטופ לוס",
        "PROFIT_STOP": "העסקה הייתה ברווח וחזרה לסטופ רווח",
        "TARGET_REACHED": "הגענו ליעד רווח",
        "TARGET_REACHED_SCORE_EXIT": "הגענו ליעד והניקוד לא מצדיק להישאר",
        "EARLY_EXIT_AGAINST_LONG": "יציאה מוקדמת: לונג התחיל לרדת מהר",
        "EARLY_EXIT_AGAINST_SHORT": "יציאה מוקדמת: שורט התחיל לעלות מהר",
        "BREAKEVEN_AFTER_COSTS": "העסקה הייתה ברווח וחזרה לאזור איזון אחרי עלויות",
        "LOCKED_SMALL_PROFIT": "נלקח רווח קטן אחרי עלויות כדי לצמצם סיכון",
        "MAX_LOSS_LIMIT": "הפסד הגיע למגבלת ההפסד לעסקה",
        "MANUAL_CLOSE": "סגירה ידנית",
        "CYCLE_TARGET_50": "מחזור רווח הושלם: נסגר בגלל יעד רווח נטו",
        "PROFIT_GIVEBACK": "הרווח ירד באחוז שהוגדר מהרווח המקסימלי",
    }
    return mapping.get(str(reason), str(reason or ""))


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


def empty_pending():
    return pd.DataFrame(columns=PENDING_COLUMNS)


def load_pending():
    if not PENDING_FILE.exists() or PENDING_FILE.stat().st_size == 0:
        return empty_pending()
    try:
        df = pd.read_csv(PENDING_FILE)
    except Exception:
        return empty_pending()
    for col in PENDING_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    return df[PENDING_COLUMNS]


def save_pending(df):
    if df is None or df.empty:
        empty_pending().to_csv(PENDING_FILE, index=False)
        return
    for col in PENDING_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    df[PENDING_COLUMNS].to_csv(PENDING_FILE, index=False)


def clear_pending():
    save_pending(empty_pending())


def has_pending_signal(ticker, mode):
    pending = load_pending()
    if pending.empty:
        return False
    return bool((pending["status"].astype(str).eq("PENDING") & pending["ticker"].astype(str).eq(str(ticker)) & pending["mode"].astype(str).eq(str(mode))).any())


def add_pending_signal(signal):
    pending = load_pending()
    ticker = normalize_ticker(signal["ticker"])
    mode = str(signal["mode"])
    trades = load_trades()
    if has_open_trade(trades, ticker, mode):
        return False, f"{ticker}: כבר יש עסקה פתוחה ב־{mode}."
    if has_pending_signal(ticker, mode):
        return False, f"{ticker}: כבר יש מועמדת בהמתנה לבדיקה."

    row = {
        "pending_id": str(uuid.uuid4()),
        "created_at": now_ny_iso(),
        "ticker": ticker,
        "mode": mode,
        "side": str(signal["signal"]),
        "score": int(signal.get("score", 0)),
        "entry_price": float(signal.get("entry", np.nan)),
        "stop_loss": float(signal.get("stop", np.nan)),
        "target_reference": float(signal.get("target", np.nan)),
        "reason": str(signal.get("reason", "")),
        "status": "PENDING",
        "last_checked_at": "",
        "message": "נמצאה עסקה משתלמת. מחכים לאישור חוזר לפני כניסה.",
    }
    pending = pd.concat([pending, pd.DataFrame([row])], ignore_index=True)
    save_pending(pending)
    return True, f"{ticker}: נשמרה מועמדת ל־{mode}. נבדוק שוב בעוד דקה לפני כניסה."


def process_pending_signals(min_score, max_new_override=None, max_open_override=None):
    pending = load_pending()
    messages = []
    if pending.empty:
        return messages

    rules = load_rules()
    trades = load_trades()
    max_new = int(max_new_override) if max_new_override is not None else int(rules["max_new_trades_per_scan"])
    max_open = int(max_open_override) if max_open_override is not None else int(rules.get("max_open_trades", 6))
    current_open = 0 if trades.empty else int(trades["status"].eq("OPEN").sum())
    available_slots = max(0, max_open - current_open)
    max_to_open = min(max_new, available_slots)

    confirm_seconds = float(rules.get("confirm_before_entry_seconds", 60))
    expire_minutes = float(rules.get("pending_signal_expire_minutes", 6))
    opened = 0

    for idx in pending.index[pending["status"].astype(str).eq("PENDING")].tolist():
        if opened >= max_to_open:
            break
        created_at = timestamp_to_ny(pending.loc[idx, "created_at"])
        if created_at is None:
            pending.loc[idx, "status"] = "REJECTED"
            pending.loc[idx, "message"] = "זמן יצירה לא תקין."
            continue

        age_seconds = (now_ny() - created_at).total_seconds()
        pending.loc[idx, "last_checked_at"] = now_ny_iso()

        if age_seconds > expire_minutes * 60:
            pending.loc[idx, "status"] = "EXPIRED"
            pending.loc[idx, "message"] = "המועמדת פגה כי עבר יותר מדי זמן."
            messages.append(f"{pending.loc[idx, 'ticker']}: מועמדת פגה.")
            continue
        if age_seconds < confirm_seconds:
            remaining = int(confirm_seconds - age_seconds)
            pending.loc[idx, "message"] = f"בהמתנה לאישור חוזר. נשארו בערך {remaining} שניות."
            continue

        ticker = str(pending.loc[idx, "ticker"])
        mode = str(pending.loc[idx, "mode"])
        original_side = str(pending.loc[idx, "side"])
        original_score = int(safe_float(pending.loc[idx, "score"], 0))

        try:
            new_signal = make_signal(ticker, mode)
        except Exception as e:
            pending.loc[idx, "message"] = f"שגיאה בבדיקה חוזרת: {str(e)[:100]}"
            continue

        new_side = str(new_signal.get("signal", "WAIT"))
        new_score = int(new_signal.get("score", 0))
        if new_side != original_side:
            pending.loc[idx, "status"] = "REJECTED"
            pending.loc[idx, "message"] = f"נדחה: הכיוון השתנה מ־{original_side} ל־{new_side}."
            messages.append(f"{ticker}: לא נכנסנו — הכיוון השתנה אחרי דקה.")
            continue
        if new_score < int(min_score):
            pending.loc[idx, "status"] = "REJECTED"
            pending.loc[idx, "message"] = f"נדחה: הניקוד ירד מ־{original_score} ל־{new_score}."
            messages.append(f"{ticker}: לא נכנסנו — הניקוד ירד אחרי דקה.")
            continue

        ok, msg = open_trade(new_signal, min_score=min_score)
        if ok:
            opened += 1
            pending.loc[idx, "status"] = "OPENED"
            pending.loc[idx, "message"] = "נפתחה עסקה אחרי אישור חוזר של דקה."
            messages.append(f"{msg} | נפתחה אחרי בדיקה חוזרת של דקה.")
        else:
            pending.loc[idx, "status"] = "REJECTED"
            pending.loc[idx, "message"] = msg
            messages.append(f"{ticker}: לא נפתחה אחרי דקה — {msg}")

    save_pending(pending)
    return messages


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


def breakeven_after_costs(row):
    """
    Approximate breakeven price after entry+exit costs.
    Long needs price above entry; short needs price below entry.
    """
    entry = safe_float(row["entry_price"])
    qty = safe_float(row["quantity"])
    if qty <= 0:
        return entry

    costs = {
        "cost_pct_per_side": safe_float(row["cost_pct_per_side"], DEFAULT_COSTS["cost_pct_per_side"]),
        "fixed_fee_per_side": safe_float(row["fixed_fee_per_side"], DEFAULT_COSTS["fixed_fee_per_side"]),
        "min_fee_per_side": safe_float(row["min_fee_per_side"], DEFAULT_COSTS["min_fee_per_side"]),
    }
    _, _, total_cost = estimate_costs(entry, entry, qty, costs)
    buffer_per_share = total_cost / qty

    if str(row["side"]) == "LONG":
        return entry + buffer_per_share
    return entry - buffer_per_share




# ============================================================
# Chart-based stop / target logic
# ============================================================

def linear_slope_per_bar(series: pd.Series, lookback: int = 8) -> float:
    """Approximate price slope per 1-minute bar using linear regression."""
    s = pd.Series(series).dropna().tail(max(3, int(lookback)))
    if len(s) < 3:
        return 0.0
    x = np.arange(len(s), dtype=float)
    y = s.astype(float).values
    try:
        return float(np.polyfit(x, y, 1)[0])
    except Exception:
        return 0.0


def recent_swing_levels(d: pd.DataFrame, lookback: int = 12) -> dict:
    """Recent support/resistance based on last candles."""
    recent = d.tail(max(5, int(lookback)))
    return {
        "support": safe_float(recent["low"].min(), safe_float(d.iloc[-1]["close"])),
        "resistance": safe_float(recent["high"].max(), safe_float(d.iloc[-1]["close"])),
        "last_low": safe_float(recent["low"].iloc[-1], safe_float(d.iloc[-1]["close"])),
        "last_high": safe_float(recent["high"].iloc[-1], safe_float(d.iloc[-1]["close"])),
    }


def chart_based_stop_target(d: pd.DataFrame, side: str, mode: str) -> dict:
    """
    Stop/TP calculated from the chart:
    - Stop: recent swing low/high plus buffer, not a random number.
    - TP: slope projection discounted by 20%, with a minimum RR check.
    """
    d = d.dropna(subset=["close"]).copy()
    if d.empty:
        return {"stop": np.nan, "target": np.nan, "reason": "אין נתוני גרף"}

    last = d.iloc[-1]
    entry = safe_float(last["close"])

    if mode == "מהירה":
        lookback = 10
        projection_bars = 5
        min_rr = 1.10
        atr_col = "atr3"
    else:
        lookback = 30
        projection_bars = 30
        min_rr = 1.35
        atr_col = "atr14"

    levels = recent_swing_levels(d, lookback=lookback)
    atr = safe_float(last.get(atr_col), entry * 0.0015)
    atr = max(atr, entry * 0.0008)
    buffer = max(atr * 0.25, entry * 0.00025)

    slope = linear_slope_per_bar(d["close"], lookback=min(lookback, 14))
    discounted_move = abs(slope) * projection_bars * 0.80  # 20% reduction from slope projection
    min_move = atr * (1.0 if mode == "מהירה" else 1.5)
    projected_move = max(discounted_move, min_move)

    if side == "LONG":
        stop = min(levels["support"], levels["last_low"]) - buffer
        risk = max(entry - stop, atr * 0.65)
        stop = entry - risk
        target_from_slope = entry + projected_move
        target_from_rr = entry + risk * min_rr
        target = max(target_from_slope, target_from_rr)
        reason = (
            f"סטופ לפי swing low/support פחות buffer. "
            f"TP לפי שיפוע {slope:.4f} ל־{projection_bars} נרות עם הורדת 20%, "
            f"ובדיקת מינימום RR {min_rr:.2f}."
        )
    else:
        stop = max(levels["resistance"], levels["last_high"]) + buffer
        risk = max(stop - entry, atr * 0.65)
        stop = entry + risk
        target_from_slope = entry - projected_move
        target_from_rr = entry - risk * min_rr
        target = min(target_from_slope, target_from_rr)
        reason = (
            f"סטופ לפי swing high/resistance פלוס buffer. "
            f"TP לפי שיפוע {slope:.4f} ל־{projection_bars} נרות עם הורדת 20%, "
            f"ובדיקת מינימום RR {min_rr:.2f}."
        )

    return {"stop": float(stop), "target": float(target), "slope": float(slope), "projection_bars": int(projection_bars), "reason": reason}


def make_live_trade_chart(ticker: str, row=None):
    """Render only on demand so the app stays responsive."""
    df = latest_session(fetch_1m(ticker))
    if df.empty:
        return None

    d = add_indicators(df).tail(120).copy()
    if d.empty:
        return None

    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=d.index, open=d["open"], high=d["high"], low=d["low"], close=d["close"], name="נרות 1 דקה"))

    for col, label in [("ema3", "EMA3"), ("ema5", "EMA5"), ("ema9", "EMA9"), ("ema21", "EMA21"), ("vwap", "VWAP")]:
        if col in d.columns:
            fig.add_trace(go.Scatter(x=d.index, y=d[col], mode="lines", name=label))

    if row is not None:
        x0, x1 = d.index[0], d.index[-1]
        lines = [
            (safe_float(row.get("entry_price"), np.nan), "כניסה", "dash"),
            (safe_float(row.get("stop_loss"), np.nan), "סטופ", "dot"),
            (safe_float(row.get("target_reference"), np.nan), "TP/יעד", "dashdot"),
            (safe_float(row.get("profit_stop"), np.nan), "סטופ רווח", "longdash"),
        ]
        for value, name, dash in lines:
            if np.isfinite(value):
                fig.add_trace(go.Scatter(x=[x0, x1], y=[value, value], mode="lines", name=name, line=dict(dash=dash)))

    fig.update_layout(
        title=f"{ticker} — גרף חי 1 דקה",
        height=520,
        xaxis_rangeslider_visible=False,
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation="h"),
    )
    return fig


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
    elif ss > ls and ss >= 3:
        side, score, reasons = "SHORT", ss, sr
    else:
        return {"signal": "WAIT", "ticker": ticker, "mode": mode, "score": max(ls, ss), "reason": f"לונג {ls}, שורט {ss} — אין יתרון ברור"}

    chart_plan = chart_based_stop_target(d, side, mode)
    stop = chart_plan["stop"]
    target = chart_plan["target"]

    return {
        "signal": side,
        "ticker": normalize_ticker(ticker),
        "mode": mode,
        "score": int(score),
        "entry": float(entry),
        "stop": float(stop),
        "target": float(target),
        "reason": " | ".join(reasons + [chart_plan["reason"]]),
    }


# ============================================================
# Trade lifecycle
# ============================================================

def trade_age_minutes(row):
    entry = timestamp_to_ny(row.get("entry_time"))
    if entry is None:
        return 0.0
    return max(0.0, (now_ny() - entry).total_seconds() / 60.0)

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
        "duration_minutes": 0.0,
        "age_minutes": 0.0,
        "entry_price": entry,
        "current_price": entry,
        "exit_price": np.nan,
        "quantity": qty,
        "notional": notional,
        "stop_loss": stop,
        "initial_stop_loss": stop,
        "manual_stop_loss": np.nan,
        "profit_stop": np.nan,
        "target_reference": target,
        "breakeven_price": np.nan,
        "highest_price": entry,
        "lowest_price": entry,
        "max_net_pnl_seen": -total_cost_now,
        "entry_cost": entry_cost,
        "exit_cost": exit_cost,
        "total_cost": total_cost_now,
        "gross_pnl": 0.0,
        "net_pnl": -total_cost_now,
        "net_pnl_pct": (-total_cost_now / notional) * 100 if notional else 0,
        "exit_reason": "",
        "exit_reason_he": "",
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
    row["breakeven_price"] = breakeven_after_costs(row)

    trades = pd.concat([trades, pd.DataFrame([row])], ignore_index=True)
    save_trades(trades)

    return True, f"{ticker}: נפתחה {side} | {mode} | ניקוד {score} | יוניטים {unit_mult} | נטו צפוי ליעד ${en:.2f}."


def update_trade_stop(trade_id, new_stop):
    trades = load_trades()
    if trades.empty:
        return False, "אין עסקאות."

    mask = trades["trade_id"].astype(str).eq(str(trade_id)) & trades["status"].eq("OPEN")
    if not mask.any():
        return False, "העסקה לא נמצאה או כבר סגורה."

    idx = trades.index[mask][0]
    side = str(trades.loc[idx, "side"])
    current = safe_float(trades.loc[idx, "current_price"])
    new_stop = float(new_stop)

    if side == "LONG" and new_stop >= current:
        return False, "בלונג הסטופ צריך להיות מתחת למחיר הנוכחי."
    if side == "SHORT" and new_stop <= current:
        return False, "בשורט הסטופ צריך להיות מעל המחיר הנוכחי."

    trades.loc[idx, "stop_loss"] = new_stop
    trades.loc[idx, "manual_stop_loss"] = new_stop
    trades.loc[idx, "management_action"] = "MANUAL_STOP_UPDATE"
    trades.loc[idx, "management_reason"] = f"הסטופ עודכן ידנית ל־{new_stop:.2f}."
    save_trades(trades)
    return True, "הסטופ עודכן."


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
    current_max_net_seen = safe_float(row.get("max_net_pnl_seen"), safe_float(row.get("net_pnl"), 0))

    res = {
        "exit": False,
        "exit_reason": "",
        "stop_loss": stop,
        "profit_stop": safe_float(row.get("profit_stop"), np.nan),
        "target_reference": target,
        "highest_price": safe_float(row.get("highest_price"), entry),
        "lowest_price": safe_float(row.get("lowest_price"), entry),
        "max_net_pnl_seen": current_max_net_seen,
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

    current_pnl = pnl_for_trade(row, current)
    current_net = current_pnl["net_pnl"]
    res["max_net_pnl_seen"] = max(current_max_net_seen, current_net)

    # יציאה לפי ירידה מהרווח המקסימלי.
    # דוגמה: שיא רווח 20$, ירידה מוגדרת 10% => יוצאים אם ירד ל־18$ או פחות.
    giveback_pct = float(rules.get("profit_giveback_pct", 10.0))
    min_profit_for_giveback = float(rules.get("min_net_profit_for_giveback", 5.0))
    peak_profit = float(res["max_net_pnl_seen"])
    if peak_profit >= min_profit_for_giveback:
        allowed_drop = peak_profit * (giveback_pct / 100.0)
        if current_net <= peak_profit - allowed_drop:
            res["exit"] = True
            res["exit_reason"] = "PROFIT_GIVEBACK"
            res["action"] = "EXIT_PROFIT_GIVEBACK"
            res["reason"] = (
                f"הרווח ירד ב־{giveback_pct:.1f}% או יותר מהרווח המקסימלי. "
                f"שיא רווח נטו: ${peak_profit:.2f}, רווח נוכחי: ${current_net:.2f}."
            )
            return res

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

    breakeven = safe_float(row.get("breakeven_price"), breakeven_after_costs(row))

    if score >= 7:
        trail_r = 0.70
    elif score >= 6:
        trail_r = 0.55
    elif score >= 5:
        trail_r = 0.40
    else:
        trail_r = 0.28

    actions, reasons = [], []

    if current_net <= -abs(float(rules["max_allowed_loss_per_trade_dollars"])):
        res["exit"] = True
        res["exit_reason"] = "MAX_LOSS_LIMIT"
        actions.append("EXIT_MAX_LOSS")
        reasons.append("הפסד נטו הגיע למגבלת ההפסד לעסקה.")

    if side == "LONG":
        r_now = (current - entry) / base_risk
        best_r = (high_since - entry) / base_risk

        if current <= stop:
            res["exit"] = True
            res["exit_reason"] = "STOP_LOSS"
            actions.append("EXIT_STOP")
            reasons.append("הגענו לסטופ לוס.")

        if (
            bool(rules["exit_if_profitable_trade_turns_red"])
            and age >= float(rules["emergency_exit_after_minutes"])
            and res["max_net_pnl_seen"] >= float(rules["breakeven_after_profit_dollars"])
            and current <= breakeven
        ):
            res["exit"] = True
            res["exit_reason"] = "BREAKEVEN_AFTER_COSTS"
            actions.append("EXIT_BREAKEVEN")
            reasons.append("העסקה הייתה ברווח וחזרה למחיר איזון אחרי עלויות.")

        if age >= min_hold:
            if current_net >= float(rules["breakeven_after_profit_dollars"]):
                new_profit_stop = max(breakeven, current - 0.35 * base_risk)
                if not np.isfinite(res["profit_stop"]) or new_profit_stop > res["profit_stop"]:
                    res["profit_stop"] = new_profit_stop
                    actions.append("SET_BREAKEVEN_PROFIT_STOP")
                    reasons.append("יש רווח אחרי עלויות, סטופ רווח הועלה לפחות לאיזון.")

            if best_r >= float(rules["min_profit_r_for_profit_stop"]):
                new_profit_stop = max(breakeven, high_since - trail_r * base_risk)
                if not np.isfinite(res["profit_stop"]) or new_profit_stop > res["profit_stop"]:
                    res["profit_stop"] = new_profit_stop
                    actions.append("RAISE_PROFIT_STOP")
                    reasons.append("העסקה ברווח, סטופ רווח עלה.")

            if score >= 7 and r_now >= 0.85 and current > ema5 and ema5_slope > 0:
                new_target = max(target, current + 0.80 * base_risk)
                if new_target > target:
                    res["target_reference"] = new_target
                    actions.append("EXTEND_TARGET")
                    reasons.append("ניקוד גבוה ומומנטום חיובי — נותנים לעסקה לרוץ.")

            if current >= target and score < int(rules["exit_on_target_when_score_below"]):
                res["exit"] = True
                res["exit_reason"] = "TARGET_REACHED_SCORE_EXIT"
                actions.append("EXIT_TARGET")
                reasons.append("הגענו ליעד, הניקוד לא מספיק גבוה כדי להמשיך.")

            if np.isfinite(res["profit_stop"]) and current <= res["profit_stop"]:
                res["exit"] = True
                res["exit_reason"] = "PROFIT_STOP"
                actions.append("EXIT_PROFIT_STOP")
                reasons.append("המחיר חזר לסטופ רווח.")

            if res["max_net_pnl_seen"] >= float(rules["lock_profit_after_net_dollars"]) and (red >= 2 or ema5_curv < 0 or macd_slope < 0):
                tightened = max(breakeven, current - 0.18 * base_risk)
                if not np.isfinite(res["profit_stop"]) or tightened > res["profit_stop"]:
                    res["profit_stop"] = tightened
                    actions.append("TIGHTEN_PROFIT_STOP")
                    reasons.append("אחרי רווח יש היחלשות, סטופ רווח הודק כדי לא להחזיר רווח.")

        if age >= float(rules["emergency_exit_after_minutes"]) and r_now < -0.25 and red >= 2 and current < ema5 and ema5_slope < 0:
            res["exit"] = True
            res["exit_reason"] = "EARLY_EXIT_AGAINST_LONG"
            actions.append("EARLY_EXIT")
            reasons.append("לונג הולך חזק נגד הכיוון, יציאה מוקדמת.")

    else:
        r_now = (entry - current) / base_risk
        best_r = (entry - low_since) / base_risk

        if current >= stop:
            res["exit"] = True
            res["exit_reason"] = "STOP_LOSS"
            actions.append("EXIT_STOP")
            reasons.append("הגענו לסטופ לוס.")

        if (
            bool(rules["exit_if_profitable_trade_turns_red"])
            and age >= float(rules["emergency_exit_after_minutes"])
            and res["max_net_pnl_seen"] >= float(rules["breakeven_after_profit_dollars"])
            and current >= breakeven
        ):
            res["exit"] = True
            res["exit_reason"] = "BREAKEVEN_AFTER_COSTS"
            actions.append("EXIT_BREAKEVEN")
            reasons.append("העסקה הייתה ברווח וחזרה למחיר איזון אחרי עלויות.")

        if age >= min_hold:
            if current_net >= float(rules["breakeven_after_profit_dollars"]):
                new_profit_stop = min(breakeven, current + 0.35 * base_risk)
                if not np.isfinite(res["profit_stop"]) or new_profit_stop < res["profit_stop"]:
                    res["profit_stop"] = new_profit_stop
                    actions.append("SET_BREAKEVEN_PROFIT_STOP")
                    reasons.append("יש רווח אחרי עלויות, סטופ רווח ירד לפחות לאיזון.")

            if best_r >= float(rules["min_profit_r_for_profit_stop"]):
                new_profit_stop = min(breakeven, low_since + trail_r * base_risk)
                if not np.isfinite(res["profit_stop"]) or new_profit_stop < res["profit_stop"]:
                    res["profit_stop"] = new_profit_stop
                    actions.append("LOWER_PROFIT_STOP")
                    reasons.append("העסקה ברווח, סטופ רווח ירד.")

            if score >= 7 and r_now >= 0.85 and current < ema5 and ema5_slope < 0:
                new_target = min(target, current - 0.80 * base_risk)
                if new_target < target:
                    res["target_reference"] = new_target
                    actions.append("EXTEND_TARGET")
                    reasons.append("ניקוד גבוה ומומנטום שלילי — נותנים לשורט לרוץ.")

            if current <= target and score < int(rules["exit_on_target_when_score_below"]):
                res["exit"] = True
                res["exit_reason"] = "TARGET_REACHED_SCORE_EXIT"
                actions.append("EXIT_TARGET")
                reasons.append("הגענו ליעד, הניקוד לא מספיק גבוה כדי להמשיך.")

            if np.isfinite(res["profit_stop"]) and current >= res["profit_stop"]:
                res["exit"] = True
                res["exit_reason"] = "PROFIT_STOP"
                actions.append("EXIT_PROFIT_STOP")
                reasons.append("המחיר חזר לסטופ רווח.")

            if res["max_net_pnl_seen"] >= float(rules["lock_profit_after_net_dollars"]) and (green >= 2 or ema5_curv > 0 or macd_slope > 0):
                tightened = min(breakeven, current + 0.18 * base_risk)
                if not np.isfinite(res["profit_stop"]) or tightened < res["profit_stop"]:
                    res["profit_stop"] = tightened
                    actions.append("TIGHTEN_PROFIT_STOP")
                    reasons.append("אחרי רווח יש היחלשות, סטופ רווח הודק כדי לא להחזיר רווח.")

        if age >= float(rules["emergency_exit_after_minutes"]) and r_now < -0.25 and green >= 2 and current > ema5 and ema5_slope > 0:
            res["exit"] = True
            res["exit_reason"] = "EARLY_EXIT_AGAINST_SHORT"
            actions.append("EARLY_EXIT")
            reasons.append("שורט הולך חזק נגד הכיוון, יציאה מוקדמת.")

    if actions:
        res["action"] = " + ".join(sorted(set(actions)))
        res["reason"] = " ".join(reasons)

    return res


def close_trade_at_index(trades, idx, current, reason):
    pnl = pnl_for_trade(trades.loc[idx], current)
    for k, v in pnl.items():
        trades.loc[idx, k] = v

    exit_time = now_ny_iso()
    trades.loc[idx, "current_price"] = current
    trades.loc[idx, "exit_price"] = current
    trades.loc[idx, "status"] = "CLOSED"
    trades.loc[idx, "exit_time"] = exit_time
    trades.loc[idx, "duration_minutes"] = minutes_between(trades.loc[idx, "entry_time"], exit_time)
    trades.loc[idx, "age_minutes"] = trades.loc[idx, "duration_minutes"]
    trades.loc[idx, "exit_reason"] = reason
    trades.loc[idx, "exit_reason_he"] = exit_reason_he(reason)
    return trades, pnl

def current_total_net(trades):
    if trades.empty:
        return 0.0
    return float(pd.to_numeric(trades["net_pnl"], errors="coerce").fillna(0).sum())

def check_cycle_target_and_close():
    trades = load_trades()
    messages = []
    if trades.empty:
        return trades, messages

    account = load_account()
    rules = load_rules()
    target = float(rules["cycle_net_profit_target"])
    locked_profit = float(account.get("locked_profit", 0.0))
    total_net = current_total_net(trades)
    cycle_profit = total_net - locked_profit

    if cycle_profit < target:
        return trades, messages

    open_idx = trades.index[trades["status"].eq("OPEN")].tolist()

    for idx in open_idx:
        ticker = str(trades.loc[idx, "ticker"])
        try:
            df = latest_session(fetch_1m(ticker))
            current = safe_float(df.iloc[-1]["close"]) if not df.empty else safe_float(trades.loc[idx, "current_price"])
        except Exception:
            current = safe_float(trades.loc[idx, "current_price"])

        trades, pnl = close_trade_at_index(trades, idx, current, "CYCLE_TARGET_50")
        trades.loc[idx, "management_action"] = "CYCLE_CLOSE"
        trades.loc[idx, "management_reason"] = f"נסגר כי המחזור הגיע ליעד רווח נטו של ${target:.2f}."

    total_net = current_total_net(trades)
    account["cycles_completed"] = int(account.get("cycles_completed", 0)) + 1
    account["locked_profit"] = float(total_net)
    account["last_cycle_closed_at"] = now_ny_iso()
    account["last_cycle_reason"] = f"המחזור הגיע ליעד רווח נטו של ${target:.2f}."

    save_account(account)
    save_trades(trades)
    messages.append(f"מחזור רווח הושלם: הגעת ל־${target:.2f} נטו מעל המחזור הקודם. כל העסקאות הפתוחות נסגרו.")

    return trades, messages

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
            entry_time = timestamp_to_ny(trades.loc[idx, "entry_time"])
            if entry_time is None:
                after_entry = df.tail(5)
            else:
                after_entry = df[df.index >= entry_time]
                if after_entry.empty:
                    after_entry = df.tail(5)

            decision = manage_trade(trades.loc[idx], after_entry)

            trades.loc[idx, "age_minutes"] = trade_age_minutes(trades.loc[idx])
            trades.loc[idx, "duration_minutes"] = trades.loc[idx, "age_minutes"]
            trades.loc[idx, "current_price"] = current
            trades.loc[idx, "stop_loss"] = decision["stop_loss"]
            trades.loc[idx, "profit_stop"] = decision["profit_stop"]
            trades.loc[idx, "target_reference"] = decision["target_reference"]
            trades.loc[idx, "highest_price"] = decision["highest_price"]
            trades.loc[idx, "lowest_price"] = decision["lowest_price"]
            trades.loc[idx, "max_net_pnl_seen"] = decision["max_net_pnl_seen"]
            trades.loc[idx, "management_action"] = decision["action"]
            trades.loc[idx, "management_reason"] = decision["reason"]

            pnl = pnl_for_trade(trades.loc[idx], current)
            for k, v in pnl.items():
                trades.loc[idx, k] = v

            if decision["exit"]:
                trades, pnl = close_trade_at_index(trades, idx, current, decision["exit_reason"])
                messages.append(f"{ticker}: נסגרה עסקה — {exit_reason_he(decision['exit_reason'])} | נטו ${pnl['net_pnl']:.2f}")

        except Exception as e:
            trades.loc[idx, "management_action"] = "ERROR"
            trades.loc[idx, "management_reason"] = str(e)[:180]

    save_trades(trades)
    trades, cycle_msgs = check_cycle_target_and_close()
    messages.extend(cycle_msgs)
    return trades, messages

def close_trade_manually(trade_id):
    """Manual close should be instant and stable: no yfinance call while clicking."""
    trades = load_trades()
    mask = trades["trade_id"].astype(str).eq(str(trade_id)) & trades["status"].eq("OPEN")
    if trades.empty or not mask.any():
        return False, "העסקה לא נמצאה או כבר סגורה."

    idx = trades.index[mask][0]
    ticker = str(trades.loc[idx, "ticker"])
    current = safe_float(trades.loc[idx, "current_price"], safe_float(trades.loc[idx, "entry_price"]))

    trades, pnl = close_trade_at_index(trades, idx, current, "MANUAL_CLOSE")
    trades.loc[idx, "management_action"] = "MANUAL_CLOSE"
    trades.loc[idx, "management_reason"] = "נסגר ידנית על ידי המשתמש לפי המחיר האחרון הידוע באפליקציה."
    save_trades(trades)
    return True, f"{ticker}: נסגר ידנית במחיר {current:.2f}. נטו ${pnl['net_pnl']:.2f}"


def scan_and_open(tickers, modes, min_score, max_new_override=None, max_open_override=None):
    """Scan candidates, save the best ones as pending, and enter only after a 1-minute re-check."""
    messages = []
    rules = load_rules()
    trades = load_trades()

    max_new = int(max_new_override) if max_new_override is not None else int(rules["max_new_trades_per_scan"])
    max_open = int(max_open_override) if max_open_override is not None else int(rules.get("max_open_trades", 6))

    current_open = 0 if trades.empty else int(trades["status"].eq("OPEN").sum())
    active_pending = load_pending()
    pending_count = 0 if active_pending.empty else int(active_pending["status"].astype(str).eq("PENDING").sum())
    available_slots = max(0, max_open - current_open - pending_count)
    if available_slots <= 0:
        return [f"לא נוספו מועמדות: יש {current_open} עסקאות פתוחות ו־{pending_count} מועמדות. המקסימום הוא {max_open}."]

    max_to_save = min(max_new, available_slots)
    candidates = []

    for ticker in tickers:
        for mode in modes:
            try:
                sig = make_signal(ticker, mode)
                if sig.get("signal") not in ["LONG", "SHORT"]:
                    continue
                if int(sig.get("score", 0)) < int(min_score):
                    continue
                expected_move = abs(float(sig["target"]) - float(sig["entry"]))
                candidates.append((int(sig["score"]), expected_move, sig))
            except Exception as e:
                messages.append(f"{ticker} | {mode}: שגיאה {str(e)[:100]}")
            time.sleep(0.03)

    if not candidates:
        return messages + ["לא נמצאו עסקאות עם ניקוד מספיק גבוה."]

    candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
    saved = 0
    for _, _, sig in candidates:
        if saved >= max_to_save:
            break
        ok, msg = add_pending_signal(sig)
        if ok:
            saved += 1
        messages.append(msg)

    if saved == 0:
        messages.append("נסרקו איתותים, אבל לא נשמרה מועמדת חדשה בגלל עסקאות/מועמדות קיימות.")
    else:
        messages.append(f"נשמרו {saved} מועמדות. הן ייבדקו שוב אחרי דקה ורק אז ייפתחו.")

    return messages


# ============================================================
# Summary + display
# ============================================================

def fmt_price(x):
    return "" if pd.isna(x) else f"{safe_float(x):.2f}"

def fmt_money(x):
    return f"${safe_float(x, 0):,.2f}"

def fmt_minutes(x):
    return f"{safe_float(x, 0):.1f}"

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
    account = load_account()
    balance = float(account.get("starting_balance", 10000.0)) + stats["net_total"]
    cycle_profit = stats["net_total"] - float(account.get("locked_profit", 0.0))
    target = float(load_rules()["cycle_net_profit_target"])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("רווח כולל נטו", fmt_money(stats["net_total"]))
    c2.metric("רווח מהעסקאות ברוטו", fmt_money(stats["gross_total"]))
    c3.metric("עלות כניסה כוללת", fmt_money(stats["entry_cost_total"]))
    c4.metric("סך כל העלויות", fmt_money(stats["cost_total"]))

    d1, d2, d3, d4 = st.columns(4)
    d1.metric("כמות עסקאות שנפתחו", stats["opened_count"])
    d2.metric("עסקאות כעת", stats["open_count"])
    d3.metric("עסקאות סגורות", stats["closed_count"])
    d4.metric("יתרת חשבון דמו", fmt_money(balance))

    e1, e2, e3 = st.columns(3)
    e1.metric("מחזורים שהושלמו", int(account.get("cycles_completed", 0)))
    e2.metric("רווח נעול במחזורים", fmt_money(account.get("locked_profit", 0.0)))
    e3.metric(f"רווח במחזור הנוכחי / יעד {fmt_money(target)}", fmt_money(cycle_profit))


def render_open_trades(open_trades):
    st.markdown("### עסקאות כעת")

    if open_trades.empty:
        st.info("אין עסקאות פתוחות כרגע.")
        return

    head = st.columns([0.55, .75, .8, .65, .8, .8, .8, .8, .9, .9, .75, .7])
    labels = ["סיים", "מניה", "סוג", "כיוון", "כניסה", "נוכחי", "סטופ", "סטופ רווח", "רווח/הפסד", "זמן כניסה", "משך דק׳", "ניקוד"]
    for col, label in zip(head, labels):
        col.markdown(f"**{label}**")

    for _, r in open_trades.iterrows():
        pnl = safe_float(r["net_pnl"], 0)
        klass = "green-row" if pnl >= 0 else "red-row"

        st.markdown(f"<div class='{klass}'>", unsafe_allow_html=True)
        row = st.columns([0.55, .75, .8, .65, .8, .8, .8, .8, .9, .9, .75, .7])

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
        row[10].write(f"{safe_float(r.get('age_minutes', 0), 0):.1f}")
        row[11].write(int(safe_float(r["score"], 0)))

        with st.expander(f"ניהול ושינוי סטופ: {r['ticker']} | {r['mode']} | {str(r['trade_id'])[:8]}"):
            st.write("פעולה אחרונה:", r.get("management_action", ""))
            st.write("סיבה:", r.get("management_reason", ""))
            st.write("למה נכנס:", r.get("signal_reason", ""))
            st.write("מחיר איזון אחרי עלויות:", fmt_price(r.get("breakeven_price", np.nan)))
            st.write("רווח מקסימלי שנראה בעסקה:", fmt_money(r.get("max_net_pnl_seen", 0)))
            st.write("עלות כוללת:", fmt_money(r.get("total_cost", 0)))

            current_stop = safe_float(r.get("stop_loss"), safe_float(r.get("initial_stop_loss"), 0))
            new_stop = st.number_input(
                "שנה סטופ לוס ידנית",
                value=float(current_stop),
                step=0.01,
                format="%.2f",
                key=f"manual_stop_{r['trade_id']}",
            )
            if st.button("💾 עדכן סטופ לעסקה", key=f"manual_stop_btn_{r['trade_id']}"):
                ok, msg = update_trade_stop(str(r["trade_id"]), new_stop)
                if ok:
                    st.success(msg)
                    st.rerun()
                else:
                    st.error(msg)

            show_chart = st.checkbox("📈 הצג גרף חי עם אינדיקטורים", key=f"show_chart_{r['trade_id']}")
            if show_chart:
                with st.spinner("טוען גרף חי..."):
                    fig = make_live_trade_chart(str(r["ticker"]), row=r)
                if fig is not None:
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("לא נמצאו נתונים לגרף כרגע.")

        st.markdown("</div>", unsafe_allow_html=True)


def render_closed_trades(closed_trades):
    st.markdown("### עסקאות שהסתיימו")

    if closed_trades.empty:
        st.info("אין עסקאות סגורות עדיין.")
        return

    d = closed_trades.sort_values("exit_time", ascending=False).copy().reset_index(drop=True)
    d["exit_reason_he"] = d.apply(
        lambda r: r["exit_reason_he"] if isinstance(r.get("exit_reason_he", ""), str) and r.get("exit_reason_he", "") else exit_reason_he(r.get("exit_reason", "")),
        axis=1,
    )

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
        "משך עסקה בדק׳": d["duration_minutes"].map(fmt_minutes),
        "ניקוד": d["score"].fillna(0).astype(int),
        "סיבה ליציאה": d["exit_reason_he"],
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
<h1>🧪 Paper Trading Lab V5.2</h1>
<p>אפליקציית Paper Trading נקייה: יעד מחזור 50$, סטופ ידני, סיבות יציאה, Break-even אחרי עלויות וצמצום הפסדים.</p>
</div>
""",
    unsafe_allow_html=True,
)

tab_paper, tab_costs, tab_units, tab_rules, tab_account, tab_help = st.tabs([
    "🧪 Paper Trading",
    "💸 עלויות",
    "📦 יוניטים לפי ניקוד",
    "⚙️ חוקים חכמים",
    "🏦 חשבון ומחזורים",
    "📘 הסבר",
])


with tab_paper:
    st.markdown("<div class='card warn'><strong>בדמו בלבד:</strong> אין חיבור לברוקר ואין כסף אמיתי.</div>", unsafe_allow_html=True)

    trades, update_msgs = update_open_trades()
    pending_msgs = process_pending_signals(min_score=4)
    update_msgs.extend(pending_msgs)
    trades = load_trades()

    render_summary(trades)

    st.markdown("---")

    clear_pending_btn = False

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
        max_new_now = st.number_input("כמה עסקאות חדשות לפתוח בסריקה", 1, 20, int(load_rules().get("max_new_trades_per_scan", 3)), key="paper_max_new_trades_now")
        max_open_now = st.number_input("מקסימום עסקאות פתוחות במקביל", 1, 30, int(load_rules().get("max_open_trades", 6)), key="paper_max_open_trades_now")

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

    if "clear_pending_btn" in locals() and clear_pending_btn:
        clear_pending()
        st.success("המועמדות בהמתנה נוקו.")
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
                msgs = process_pending_signals(min_score, max_new_override=max_new_now, max_open_override=max_open_now)
                msgs += scan_and_open(selected_tickers, modes, min_score, max_new_override=max_new_now, max_open_override=max_open_now)
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

    pending = load_pending()
    active_pending = pending[pending["status"].astype(str).eq("PENDING")].copy() if not pending.empty else empty_pending()
    if not active_pending.empty:
        st.markdown("### מועמדות בהמתנה לאישור דקה")
        show_cols = ["ticker", "mode", "side", "score", "entry_price", "status", "message", "created_at"]
        st.dataframe(active_pending[show_cols], use_container_width=True, hide_index=True)

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
    st.subheader("⚙️ חוקים חכמים לצמצום הפסדים ולקיחת רווחים")
    rules = load_rules()

    r1, r2 = st.columns(2)
    with r1:
        min_hold_fast = st.number_input("מינימום החזקה לעסקה מהירה, בדקות", 0, 60, int(rules["min_hold_fast_minutes"]))
        min_hold_half = st.number_input("מינימום החזקה לעסקת חצי שעה, בדקות", 0, 120, int(rules["min_hold_half_hour_minutes"]))
        cooldown = st.number_input("Cooldown אחרי סגירה, בדקות", 0, 120, int(rules["cooldown_after_close_minutes"]))
        max_new = st.number_input("מקסימום עסקאות חדשות בכל סריקה", 1, 20, int(rules["max_new_trades_per_scan"]), key="rules_max_new_trades")
        max_open_rule = st.number_input("מקסימום עסקאות פתוחות במקביל", 1, 30, int(rules.get("max_open_trades", 6)), key="rules_max_open_trades")
        cycle_target = st.number_input("יעד רווח נטו למחזור ($)", 1.0, 10000.0, float(rules["cycle_net_profit_target"]), 1.0, key="rules_cycle_target")

    with r2:
        profit_r = st.number_input("כמה רווח R לפני הפעלת סטופ רווח", 0.1, 3.0, float(rules["min_profit_r_for_profit_stop"]), 0.05)
        emergency_minutes = st.number_input("מינימום דקות לפני יציאה מוקדמת נגד הכיוון", 0, 30, int(rules["emergency_exit_after_minutes"]))
        breakeven_after = st.number_input("כמה רווח נטו צריך לפני הגנת איזון ($)", 0.0, 500.0, float(rules["breakeven_after_profit_dollars"]), 1.0)
        lock_profit_after = st.number_input("כמה רווח נטו צריך לפני הידוק אגרסיבי ($)", 0.0, 1000.0, float(rules["lock_profit_after_net_dollars"]), 1.0)
        max_loss = st.number_input("מקסימום הפסד נטו לעסקה ($)", 1.0, 10000.0, float(rules["max_allowed_loss_per_trade_dollars"]), 1.0)
        giveback_pct = st.number_input("יציאה אם ירד X% מהרווח המקסימלי", 1.0, 90.0, float(rules.get("profit_giveback_pct", 10.0)), 1.0)
        min_giveback_profit = st.number_input("מינימום רווח נטו להפעלת ירידת רווח ($)", 0.0, 1000.0, float(rules.get("min_net_profit_for_giveback", 5.0)), 1.0)
        confirm_seconds = st.number_input("כמה שניות לחכות לפני כניסה אחרי זיהוי עסקה", 10, 600, int(rules.get("confirm_before_entry_seconds", 60)))
        pending_expire = st.number_input("אחרי כמה דקות מועמדת פגה", 1, 60, int(rules.get("pending_signal_expire_minutes", 6)))
        exit_score_below = st.slider("לצאת ביעד אם ניקוד נמוך מ־", 1, 9, int(rules["exit_on_target_when_score_below"]))

    exit_turn_red = st.checkbox(
        "אם עסקה הייתה ברווח ואז חוזרת לאיזון אחרי עלויות — לצאת",
        value=bool(rules["exit_if_profitable_trade_turns_red"]),
    )

    st.markdown(
        """
<div class="card">
<strong>מטרת החוקים:</strong><br>
לנסות לצמצם הפסדים, לנעול רווחים קטנים אחרי עלויות, לא להחזיר עסקה מרוויחה להפסד,
ולתת לעסקאות עם ניקוד גבוה לרוץ רק אם המומנטום עדיין תומך.
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
            "max_open_trades": int(max_open_rule),
            "cycle_net_profit_target": float(cycle_target),
            "min_profit_r_for_profit_stop": float(profit_r),
            "emergency_exit_after_minutes": int(emergency_minutes),
            "breakeven_after_profit_dollars": float(breakeven_after),
            "lock_profit_after_net_dollars": float(lock_profit_after),
            "max_allowed_loss_per_trade_dollars": float(max_loss),
            "exit_if_profitable_trade_turns_red": bool(exit_turn_red),
            "exit_on_target_when_score_below": int(exit_score_below),
            "profit_giveback_pct": float(giveback_pct),
            "min_net_profit_for_giveback": float(min_giveback_profit),
            "confirm_before_entry_seconds": int(confirm_seconds),
            "pending_signal_expire_minutes": int(pending_expire),
        })
        st.success("נשמר.")


with tab_account:
    st.subheader("🏦 חשבון דמו ומחזורים")
    account = load_account()
    trades = load_trades()
    stats = summary_stats(trades)
    balance = float(account.get("starting_balance", 10000.0)) + stats["net_total"]

    c1, c2 = st.columns(2)
    with c1:
        starting_balance = st.number_input("יתרת פתיחה דמו ($)", 100.0, 10000000.0, float(account.get("starting_balance", 10000.0)), 100.0)
        st.metric("יתרת חשבון דמו משוערת", fmt_money(balance))
        st.metric("רווח כולל נטו", fmt_money(stats["net_total"]))

    with c2:
        st.metric("מחזורים שהושלמו", int(account.get("cycles_completed", 0)))
        st.metric("רווח נעול במחזורים", fmt_money(account.get("locked_profit", 0.0)))
        st.write("סגירת מחזור אחרונה:", account.get("last_cycle_closed_at", ""))
        st.write("סיבה:", account.get("last_cycle_reason", ""))

    a, b = st.columns(2)
    with a:
        if st.button("💾 שמור יתרת פתיחה", use_container_width=True):
            account["starting_balance"] = float(starting_balance)
            save_account(account)
            st.success("נשמר.")
    with b:
        if st.button("♻️ אפס מחזורים בלבד", use_container_width=True):
            account["cycles_completed"] = 0
            account["locked_profit"] = 0.0
            account["last_cycle_closed_at"] = ""
            account["last_cycle_reason"] = ""
            save_account(account)
            st.success("המחזורים אופסו, העסקאות לא נמחקו.")


with tab_help:
    st.subheader("📘 הסבר")
    st.markdown(
        """
### מה חדש ב־V2/V3?

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

### מה נוסף ב־V5?

**זמן יציאה ומשך עסקה**  
בכל עסקה סגורה מופיע זמן יציאה וגם משך העסקה בדקות.

**סיבה ליציאה בעברית**  
האפליקציה מציינת אם יצאנו בגלל סטופ לוס, סטופ רווח, יעד, יציאה מוקדמת, איזון אחרי עלויות או סגירת מחזור.

**שינוי סטופ ידני**  
בעסקאות פתוחות אפשר לפתוח את אזור הניהול ולשנות ידנית את הסטופ לעסקה מסוימת.

**יעד מחזור 50$**  
ברירת המחדל היא שאם המחזור הנוכחי מגיע ל־50$ רווח נטו, כל העסקאות הפתוחות נסגרות והמחזור נספר.

**הגנת איזון אחרי עלויות**  
אם עסקה הייתה ברווח ואז חוזרת לאיזור שבו אחרי עלויות אין רווח, האפליקציה יכולה לצאת כדי לא להפוך רווח להפסד.

**מטרת האפליקציה**  
לצמצם הפסדים, לקחת רווחים קטנים אחרי עלויות, ולתת לעסקאות חזקות להמשיך רק כאשר האינדיקטורים עדיין תומכים.

"""
    )
