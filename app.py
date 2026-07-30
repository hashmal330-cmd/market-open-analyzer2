
import json
import html
import time
import uuid
import urllib.parse
import urllib.request
import ssl
from collections import Counter, defaultdict
import requests
import certifi
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import yfinance as yf
import plotly.graph_objects as go


# ============================================================
# App setup
# ============================================================

st.set_page_config(page_title="Engineering Pattern Live Monitor V1.8 — Fixed Live Percentages", page_icon="📡", layout="wide")

DATA_DIR = Path("paper_data")
DATA_DIR.mkdir(exist_ok=True)

TRADES_FILE = DATA_DIR / "trades_v3.csv"
TICKERS_FILE = DATA_DIR / "tickers_v3.json"
COSTS_FILE = DATA_DIR / "costs_v3.json"
UNITS_FILE = DATA_DIR / "units_v3.json"
RULES_FILE = DATA_DIR / "rules_v7_2_active_balanced.json"
ACCOUNT_FILE = DATA_DIR / "account_v3.json"
PENDING_FILE = DATA_DIR / "pending_signals_v7_2.csv"
ALERTS_FILE = DATA_DIR / "alerts_v5_8.csv"
ALERT_SETTINGS_FILE = DATA_DIR / "alert_settings_v5_8.json"
ENGINEERING_FILE = DATA_DIR / "engineering_predictions_v7_2.csv"

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
        "1": 0.0, "2": 0.0, "3": 0.0, "4": 0.0,
        "5": 0.0, "6": 0.0, "7": 1.0, "8": 1.25,
        "9": 1.75, "10": 2.5, "11": 3.5, "12": 5.0,
    },
}

DEFAULT_RULES = {
    "min_hold_fast_minutes": 3,
    "min_hold_half_hour_minutes": 10,
    "cooldown_after_close_minutes": 6,
    "max_new_trades_per_scan": 3,
    "max_open_trades": 6,
    "max_same_side_open": 3,
    "max_same_group_open": 2,
    "min_direction_score_gap": 1,
    "min_base_score": 5,
    "min_score_half": 7,
    "min_score_fast": 10,
    "require_5m_alignment": True,
    "min_5m_alignment_score": 2,
    "market_filter_enabled": True,
    "market_reference_ticker": "QQQ",
    "live_data_max_age_minutes": 1,
    "entry_start_time": "09:35",
    "entry_end_time": "15:25",
    "force_flat_time": "15:55",

    # Daily protection
    "daily_loss_limit_dollars": 25.0,
    "max_trades_per_day": 24,
    "max_consecutive_losses": 3,
    "loss_streak_pause_minutes": 15,

    # Entry confirmation
    "confirm_before_entry_seconds": 30,
    "pending_signal_expire_minutes": 10,
    "confirmation_breakout_buffer_pct": 0.0,
    "max_adverse_move_r_before_entry": 0.45,
    "max_target_progress_before_entry_pct": 65.0,
    "min_confirmation_rel_volume": 0.35,
    "confirmation_breakout_tolerance_atr": 0.18,

    # Engineering pattern engine
    "engineering_enabled": True,
    "engineering_require_when_ready": False,
    "engineering_allow_strong_override": True,
    "engineering_window_fast": 24,
    "engineering_window_half": 36,
    "engineering_horizon_fast": 8,
    "engineering_horizon_half": 20,
    "engineering_top_k": 25,
    "engineering_min_samples": 8,
    "engineering_min_confidence": 0.54,
    "engineering_strong_confidence": 0.68,
    "engineering_min_similarity": 0.30,
    "engineering_min_direction_probability": 0.56,
    "engineering_min_expectancy_r": 0.15,
    "engineering_min_expectancy_gap_r": 0.25,
    "engineering_candidate_step": 4,
    "engineering_max_candidates": 320,
    "engineering_time_tolerance_minutes": 180,
    "engineering_backtest_scan_interval": 3,

    # Profit-taking and protection
    "cycle_net_profit_target": 50.0,
    "min_profit_r_for_profit_stop": 0.45,
    "emergency_exit_after_minutes": 2,
    "breakeven_after_profit_dollars": 4.0,
    "lock_profit_after_net_dollars": 8.0,
    "max_allowed_loss_per_trade_dollars": 7.0,
    "exit_if_profitable_trade_turns_red": True,
    "exit_on_target_when_score_below": 13,
    "profit_giveback_pct": 10.0,
    "min_net_profit_for_giveback": 5.0,
    "use_history_after_minutes": 30,
    "history_min_samples": 4,
    "history_max_score_bonus": 1,
    "history_max_score_penalty": 1,
}

DEFAULT_ACCOUNT = {
    "starting_balance": 10000.0,
    "cycles_completed": 0,
    "locked_profit": 0.0,
    "last_cycle_closed_at": "",
    "last_cycle_reason": "",
}

DEFAULT_ALERT_SETTINGS = {
    "alerts_enabled": False,
    "telegram_enabled": False,
    "telegram_bot_token": "",
    "telegram_chat_id": "",
    "send_only_score_at_least": 9,
    "include_reason": True,
}

ALERT_COLUMNS = [
    "alert_id",
    "created_at",
    "trade_id",
    "ticker",
    "mode",
    "side",
    "score",
    "entry_price",
    "stop_loss",
    "target_reference",
    "net_pnl_expected",
    "risk_note",
    "tradingview_url",
    "telegram_sent",
    "telegram_error",
    "message",
]

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
    "entry_price", "stop_loss", "target_reference",
    "signal_high", "signal_low", "signal_bar_time", "atr", "last_rel_vol",
    "long_score", "short_score", "score_gap", "reason",
    "status", "last_checked_at", "message",
]

ENGINEERING_COLUMNS = [
    "prediction_id", "created_at", "bar_time", "ticker", "mode",
    "technical_side", "predicted_side", "decision", "confidence",
    "sample_count", "mean_similarity", "best_similarity", "weakest_similarity",
    "long_probability", "short_probability", "neutral_probability",
    "long_target_rate", "short_target_rate", "long_expectancy_r", "short_expectancy_r",
    "expected_mfe_r", "expected_mae_r", "pattern_state", "function_model",
    "technical_long_score", "technical_short_score", "final_score", "reason",
]

# ============================================================
# Styling
# ============================================================

st.markdown(
    """
<style>
html, body { text-align: right; }
/* Keep the regular Hebrew UI RTL, but do not impose RTL on the dataframe canvas. */
[data-testid="stMarkdownContainer"],
[data-testid="stWidgetLabel"],
[data-testid="stSidebar"],
[data-testid="stTabs"],
[data-testid="stExpander"] {
    direction: rtl;
    text-align: right;
}
[data-testid="stDataFrame"],
[data-testid="stDataFrame"] > div,
[data-testid="stDataFrame"] [role="grid"] {
    direction: ltr !important;
    text-align: left !important;
}
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


def render_mixed_direction_html_table(
    df,
    *,
    height=650,
    row_style_fn=None,
    wrap_columns=None,
    max_rows=None,
):
    """Render a robust HTML table for mixed Hebrew/English content.

    Streamlit's canvas dataframe can clip LTR values when the surrounding app is RTL.
    This isolated HTML component gives every header and cell dir="auto", so Hebrew,
    English tickers, LONG/SHORT and model names are all shown completely.
    """
    if df is None or df.empty:
        st.info("אין נתונים להצגה.")
        return

    table_df = df.copy()
    if max_rows is not None and len(table_df) > int(max_rows):
        table_df = table_df.head(int(max_rows)).copy()

    wrap_columns = set(wrap_columns or [])

    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<style>",
        "html,body{margin:0;padding:0;background:transparent;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Arial,sans-serif;}",
        ".table-wrap{height:100%;overflow:auto;border:1px solid #e5e7eb;border-radius:12px;background:#fff;}",
        "table{border-collapse:separate;border-spacing:0;min-width:100%;width:max-content;table-layout:auto;direction:ltr;}",
        "thead th{position:sticky;top:0;z-index:3;background:#f3f4f6;color:#111827;font-weight:700;border-bottom:1px solid #d1d5db;}",
        "th,td{padding:9px 12px;border-right:1px solid #e5e7eb;border-bottom:1px solid #e5e7eb;vertical-align:top;max-width:none;overflow:visible;text-overflow:clip;unicode-bidi:plaintext;}",
        "th{white-space:nowrap;}",
        "td.nowrap{white-space:nowrap;min-width:max-content;}",
        "td.wrap{white-space:normal;min-width:280px;max-width:560px;line-height:1.35;}",
        "tbody tr:hover td{filter:brightness(0.98);}",
        "</style></head><body><div class='table-wrap'><table><thead><tr>",
    ]

    for col in table_df.columns:
        parts.append(f"<th dir='auto'>{html.escape(str(col))}</th>")
    parts.append("</tr></thead><tbody>")

    for _, row in table_df.iterrows():
        row_style = row_style_fn(row) if callable(row_style_fn) else ""
        parts.append(f"<tr style='{row_style}'>")
        for col in table_df.columns:
            value = row.get(col, "")
            if value is None or (isinstance(value, float) and np.isnan(value)):
                value = ""
            cls = "wrap" if col in wrap_columns else "nowrap"
            parts.append(
                f"<td class='{cls}' dir='auto'>{html.escape(str(value))}</td>"
            )
        parts.append("</tr>")

    parts.append("</tbody></table></div></body></html>")
    components.html("".join(parts), height=int(height), scrolling=False)



LTR_MARK = "\u200e"
BIDI_MARKS = ("\u200e", "\u200f", "\u202a", "\u202b", "\u202c", "\u2066", "\u2067", "\u2069")


def clean_bidi_text(value):
    """Remove invisible direction markers before comparisons and CSV export."""
    text = str(value or "")
    for mark in BIDI_MARKS:
        text = text.replace(mark, "")
    return text


def force_ltr_cell(value):
    """Force pure English/ticker values to render left-to-right in Streamlit's grid."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    text = str(value)
    return f"{LTR_MARK}{text}" if text else ""


def prepare_interactive_table(df, ltr_columns=None):
    """Preserve numeric dtypes for real sorting while fixing LTR text columns."""
    out = df.copy()
    for col in (ltr_columns or []):
        if col in out.columns:
            out[col] = out[col].map(force_ltr_cell)
    return out


def csv_bytes(df):
    """UTF-8 BOM keeps Hebrew readable when opening the CSV in Excel."""
    clean = df.copy()
    for col in clean.select_dtypes(include=["object"]).columns:
        clean[col] = clean[col].map(clean_bidi_text)
    return clean.to_csv(index=False).encode("utf-8-sig")


def prediction_column_config():
    return {
        "מניה": st.column_config.TextColumn("מניה", width="small"),
        "סוג עסקה": st.column_config.TextColumn("סוג עסקה", width="small"),
        "מחיר": st.column_config.NumberColumn("מחיר", format="%.2f", width="small"),
        "LONG %": st.column_config.NumberColumn("LONG %", format="%.1f%%", width="small"),
        "SHORT %": st.column_config.NumberColumn("SHORT %", format="%.1f%%", width="small"),
        "NEUTRAL %": st.column_config.NumberColumn("NEUTRAL %", format="%.1f%%", width="small"),
        "כיוון מוביל": st.column_config.TextColumn("כיוון מוביל", width="small"),
        "יתרון %": st.column_config.NumberColumn("יתרון %", format="%.1f%%", width="small"),
        "ביטחון %": st.column_config.NumberColumn("ביטחון %", format="%.1f%%", width="small"),
        "דמיון %": st.column_config.NumberColumn("דמיון %", format="%.1f%%", width="small"),
        "דמיון מיטבי %": st.column_config.NumberColumn("דמיון מיטבי %", format="%.1f%%", width="small"),
        "דוגמאות": st.column_config.NumberColumn("דוגמאות", format="%d", width="small"),
        "תוחלת LONG R": st.column_config.NumberColumn("תוחלת LONG R", format="%+.2f", width="small"),
        "תוחלת SHORT R": st.column_config.NumberColumn("תוחלת SHORT R", format="%+.2f", width="small"),
        "מצב תבנית": st.column_config.TextColumn("מצב תבנית", width="medium"),
        "מודל": st.column_config.TextColumn("מודל", width="medium"),
        "סטטוס": st.column_config.TextColumn("סטטוס", width="medium"),
        "כניסה": st.column_config.NumberColumn("כניסה", format="%.2f", width="small"),
        "סטופ": st.column_config.NumberColumn("סטופ", format="%.2f", width="small"),
        "יעד": st.column_config.NumberColumn("יעד", format="%.2f", width="small"),
        "גיל נתון (דק׳)": st.column_config.NumberColumn("גיל נתון (דק׳)", format="%.0f", width="small"),
        "רעננות": st.column_config.TextColumn("רעננות", width="small"),
        "עדכון": st.column_config.TextColumn("עדכון", width="medium"),
        "הסבר": st.column_config.TextColumn("הסבר", width="large"),
    }


def trades_column_config():
    return {
        "סטטוס": st.column_config.TextColumn("סטטוס", width="small"),
        "מניה": st.column_config.TextColumn("מניה", width="small"),
        "סוג": st.column_config.TextColumn("סוג", width="small"),
        "כיוון": st.column_config.TextColumn("כיוון", width="small"),
        "LONG % בכניסה": st.column_config.NumberColumn("LONG % בכניסה", format="%.1f%%", width="small"),
        "SHORT % בכניסה": st.column_config.NumberColumn("SHORT % בכניסה", format="%.1f%%", width="small"),
        "ביטחון %": st.column_config.NumberColumn("ביטחון %", format="%.1f%%", width="small"),
        "מחיר כניסה": st.column_config.NumberColumn("מחיר כניסה", format="%.2f", width="small"),
        "מחיר נוכחי": st.column_config.NumberColumn("מחיר נוכחי", format="%.2f", width="small"),
        "סטופ": st.column_config.NumberColumn("סטופ", format="%.2f", width="small"),
        "יעד": st.column_config.NumberColumn("יעד", format="%.2f", width="small"),
        "כמות": st.column_config.NumberColumn("כמות", format="%.4f", width="small"),
        "רווח ברוטו $": st.column_config.NumberColumn("רווח ברוטו $", format="$%.2f", width="small"),
        "עלויות $": st.column_config.NumberColumn("עלויות $", format="$%.2f", width="small"),
        "רווח נטו $": st.column_config.NumberColumn("רווח נטו $", format="$%.2f", width="small"),
        "רווח נטו %": st.column_config.NumberColumn("רווח נטו %", format="%.2f%%", width="small"),
        "שיא רווח $": st.column_config.NumberColumn("שיא רווח $", format="$%.2f", width="small"),
        "זמן כניסה": st.column_config.TextColumn("זמן כניסה", width="medium"),
        "זמן יציאה": st.column_config.TextColumn("זמן יציאה", width="medium"),
        "משך דקות": st.column_config.NumberColumn("משך דקות", format="%.1f", width="small"),
        "סיבת יציאה": st.column_config.TextColumn("סיבת יציאה", width="large"),
        "מצב תבנית": st.column_config.TextColumn("מצב תבנית", width="medium"),
        "מודל": st.column_config.TextColumn("מודל", width="medium"),
    }


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
        "NO_PROGRESS_FAST": "העסקה לא התקדמה אחרי 2–3 נרות",
        "NO_PROGRESS_HALF": "העסקה לא התקדמה מספיק בזמן שהוגדר",
        "END_OF_DAY_SAFETY": "סגירת בטיחות לפני סוף יום המסחר",
        "OVERNIGHT_SAFETY_CLOSE": "העסקה נשארה פתוחה מיום קודם ונסגרה במחיר הסגירה האחרון של אותו יום",
    }
    return mapping.get(str(reason), str(reason or ""))


def empty_trades():
    df = pd.DataFrame(columns=TRADE_COLUMNS)
    for col in TRADE_COLUMNS:
        df[col] = df[col].astype("object")
    return df


def normalize_trade_dtypes(df):
    """
    Newer pandas versions can infer empty/text columns as float64 from CSV.
    Then assigning timestamps like 2026-07-13T... into exit_time crashes.
    This function forces text/date/status columns to object dtype.
    """
    if df is None:
        return empty_trades()

    for col in TRADE_COLUMNS:
        if col not in df.columns:
            df[col] = ""

    df = df[TRADE_COLUMNS].copy()

    text_cols = [
        "trade_id",
        "status",
        "ticker",
        "mode",
        "side",
        "entry_time",
        "exit_time",
        "exit_reason",
        "exit_reason_he",
        "management_action",
        "management_reason",
        "signal_reason",
        "created_settings_snapshot",
    ]

    num_cols = [
        "score",
        "duration_minutes",
        "age_minutes",
        "entry_price",
        "current_price",
        "exit_price",
        "quantity",
        "notional",
        "stop_loss",
        "initial_stop_loss",
        "manual_stop_loss",
        "profit_stop",
        "target_reference",
        "breakeven_price",
        "highest_price",
        "lowest_price",
        "max_net_pnl_seen",
        "entry_cost",
        "exit_cost",
        "total_cost",
        "gross_pnl",
        "net_pnl",
        "net_pnl_pct",
        "cost_pct_per_side",
        "fixed_fee_per_side",
        "min_fee_per_side",
        "max_cost_to_target_pct",
        "base_unit_dollars",
        "unit_multiplier",
    ]

    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].astype("object").where(pd.notna(df[col]), "")

    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df[TRADE_COLUMNS]


def load_trades():
    if not TRADES_FILE.exists() or TRADES_FILE.stat().st_size == 0:
        return empty_trades()
    try:
        df = pd.read_csv(TRADES_FILE)
    except pd.errors.EmptyDataError:
        return empty_trades()
    except Exception:
        return empty_trades()
    return normalize_trade_dtypes(df)


def save_trades(df):
    if df is None or df.empty:
        empty_trades().to_csv(TRADES_FILE, index=False)
        return
    df = normalize_trade_dtypes(df)
    df.to_csv(TRADES_FILE, index=False)

def clear_trades():
    save_trades(empty_trades())


def empty_pending():
    df = pd.DataFrame(columns=PENDING_COLUMNS)
    for col in PENDING_COLUMNS:
        df[col] = df[col].astype("object")
    return df


def load_pending():
    """
    Load pending signals safely.

    Important:
    On newer pandas versions, assigning a string timestamp into a column that
    was inferred as float can raise a TypeError. Therefore we explicitly cast
    text/status/date columns to object/string-friendly dtype.
    """
    if not PENDING_FILE.exists() or PENDING_FILE.stat().st_size == 0:
        return empty_pending()

    try:
        df = pd.read_csv(PENDING_FILE)
    except pd.errors.EmptyDataError:
        return empty_pending()
    except Exception:
        return empty_pending()

    for col in PENDING_COLUMNS:
        if col not in df.columns:
            df[col] = ""

    df = df[PENDING_COLUMNS].copy()

    text_cols = [
        "pending_id",
        "created_at",
        "signal_bar_time",
        "ticker",
        "mode",
        "side",
        "reason",
        "status",
        "last_checked_at",
        "message",
    ]
    num_cols = [
        "score", "entry_price", "stop_loss", "target_reference",
        "signal_high", "signal_low", "atr", "last_rel_vol",
        "long_score", "short_score", "score_gap",
    ]

    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].astype("object").where(pd.notna(df[col]), "")

    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df[PENDING_COLUMNS]


def save_pending(df):
    if df is None or df.empty:
        empty_pending().to_csv(PENDING_FILE, index=False)
        return

    for col in PENDING_COLUMNS:
        if col not in df.columns:
            df[col] = ""

    df = df[PENDING_COLUMNS].copy()

    text_cols = [
        "pending_id",
        "created_at",
        "signal_bar_time",
        "ticker",
        "mode",
        "side",
        "reason",
        "status",
        "last_checked_at",
        "message",
    ]
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].astype("object").where(pd.notna(df[col]), "")

    df.to_csv(PENDING_FILE, index=False)


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
    if has_any_open_trade_for_ticker(trades, ticker):
        return False, f"{ticker}: כבר יש עסקה פתוחה על המניה הזו."
    if has_pending_signal(ticker, mode):
        return False, f"{ticker}: כבר יש מועמדת בהמתנה לבדיקה."
    ok, exposure_msg = exposure_gate(trades, ticker, signal.get("signal", ""), include_pending=True)
    if not ok:
        return False, f"{ticker}: {exposure_msg}"

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
        "signal_high": float(signal.get("signal_high", np.nan)),
        "signal_low": float(signal.get("signal_low", np.nan)),
        "signal_bar_time": str(signal.get("signal_bar_time", "")),
        "atr": float(signal.get("atr", np.nan)),
        "last_rel_vol": float(signal.get("last_rel_vol", np.nan)),
        "long_score": int(signal.get("long_score", 0)),
        "short_score": int(signal.get("short_score", 0)),
        "score_gap": int(signal.get("score_gap", 0)),
        "reason": str(signal.get("reason", "")),
        "status": "PENDING",
        "last_checked_at": "",
        "message": "נמצאה עסקה. מחכים לאישור פריצה חוזר לפני כניסה.",
    }
    pending = pd.concat([pending, pd.DataFrame([row])], ignore_index=True)
    save_pending(pending)
    return True, f"{ticker}: נשמרה מועמדת {signal['signal']} בניקוד {signal.get('score', 0)}; תיבדק שוב אחרי ההמתנה."


def process_pending_signals(min_score, max_new_override=None, max_open_override=None):
    pending = load_pending()
    messages = []
    if pending.empty:
        return messages

    for col in ["last_checked_at", "message", "status"]:
        pending[col] = pending[col].astype("object")

    rules = load_rules()
    trades = load_trades()
    risk_ok, risk_msg = daily_risk_gate(trades)
    if not risk_ok:
        for idx in pending.index[pending["status"].astype(str).eq("PENDING")]:
            pending.loc[idx, "message"] = risk_msg
        save_pending(pending)
        return [risk_msg]

    max_new = int(max_new_override) if max_new_override is not None else int(rules.get("max_new_trades_per_scan", 2))
    max_open = int(max_open_override) if max_open_override is not None else int(rules.get("max_open_trades", 5))
    current_open = int(trades["status"].eq("OPEN").sum()) if not trades.empty else 0
    max_to_open = min(max_new, max(0, max_open - current_open))
    confirm_seconds = float(rules.get("confirm_before_entry_seconds", 45))
    expire_minutes = float(rules.get("pending_signal_expire_minutes", 7))
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
            pending.loc[idx, "message"] = "המועמדת פגה כי לא התקבל נר אישור חדש בזמן."
            messages.append(f"{pending.loc[idx, 'ticker']}: המועמדת פגה.")
            continue
        if age_seconds < confirm_seconds:
            pending.loc[idx, "message"] = f"ממתינים לנר אישור; נשארו כ־{int(confirm_seconds-age_seconds)} שניות."
            continue

        ticker = str(pending.loc[idx, "ticker"])
        mode = str(pending.loc[idx, "mode"])
        try:
            new_signal = make_signal(ticker, mode)
        except Exception as exc:
            pending.loc[idx, "message"] = f"שגיאת בדיקה חוזרת: {str(exc)[:100]}"
            continue

        # Yahoo may not publish a new minute immediately. Keep waiting instead of rejecting.
        original_bar = timestamp_to_ny(pending.loc[idx, "signal_bar_time"])
        new_bar = timestamp_to_ny(new_signal.get("signal_bar_time", ""))
        if new_bar is None or (original_bar is not None and new_bar <= original_bar):
            pending.loc[idx, "message"] = "עדיין אין נר דקה חדש ב־Yahoo; ממשיכים להמתין."
            continue

        if new_signal.get("signal") not in ["LONG", "SHORT"]:
            reason = str(new_signal.get("reason", "האיתות נחלש"))
            pending.loc[idx, "status"] = "REJECTED"
            pending.loc[idx, "message"] = f"נדחה בנר החדש: {reason}"
            messages.append(f"{ticker}: נדחה — {reason}")
            continue

        mode_floor = int(rules.get("min_score_fast", 10)) if str(mode) == "מהירה" else int(rules.get("min_score_half", 7))
        effective_min_score = max(int(min_score), mode_floor)

        confirmed, confirm_msg = signal_confirmed_after_delay(
            original_side=pending.loc[idx, "side"],
            original_score=int(safe_float(pending.loc[idx, "score"], 0)),
            new_signal=new_signal,
            min_score=effective_min_score,
            original_entry=pending.loc[idx, "entry_price"],
            original_stop=pending.loc[idx, "stop_loss"],
            original_target=pending.loc[idx, "target_reference"],
            signal_high=pending.loc[idx, "signal_high"],
            signal_low=pending.loc[idx, "signal_low"],
        )

        if not confirmed:
            pending.loc[idx, "status"] = "REJECTED"
            pending.loc[idx, "message"] = confirm_msg
            messages.append(f"{ticker}: לא נכנסנו — {confirm_msg}")
            continue

        ok, msg = open_trade(new_signal, min_score=max(1, int(effective_min_score) - 1))
        if ok:
            opened += 1
            pending.loc[idx, "status"] = "OPENED"
            pending.loc[idx, "message"] = "נפתחה אחרי נר אישור חדש."
            messages.append(f"{msg} | נפתחה אחרי אישור מאוזן.")
        else:
            pending.loc[idx, "status"] = "REJECTED"
            pending.loc[idx, "message"] = msg
            messages.append(f"{ticker}: לא נפתחה — {msg}")

    save_pending(pending)
    return messages


# ============================================================
# Alerts / Telegram
# ============================================================

def load_alert_settings():
    return read_json(ALERT_SETTINGS_FILE, DEFAULT_ALERT_SETTINGS)


def save_alert_settings(settings):
    safe = dict(DEFAULT_ALERT_SETTINGS)
    safe.update(settings or {})
    write_json(ALERT_SETTINGS_FILE, safe)


def empty_alerts():
    df = pd.DataFrame(columns=ALERT_COLUMNS)
    for col in ALERT_COLUMNS:
        df[col] = df[col].astype("object")
    return df


def load_alerts():
    if not ALERTS_FILE.exists() or ALERTS_FILE.stat().st_size == 0:
        return empty_alerts()
    try:
        df = pd.read_csv(ALERTS_FILE)
    except pd.errors.EmptyDataError:
        return empty_alerts()
    except Exception:
        return empty_alerts()

    for col in ALERT_COLUMNS:
        if col not in df.columns:
            df[col] = ""

    df = df[ALERT_COLUMNS].copy()

    text_cols = [
        "alert_id", "created_at", "trade_id", "ticker", "mode", "side",
        "risk_note", "tradingview_url", "telegram_sent", "telegram_error", "message",
    ]
    num_cols = ["score", "entry_price", "stop_loss", "target_reference", "net_pnl_expected"]

    for col in text_cols:
        df[col] = df[col].astype("object").where(pd.notna(df[col]), "")

    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df[ALERT_COLUMNS]


def save_alerts(df):
    if df is None or df.empty:
        empty_alerts().to_csv(ALERTS_FILE, index=False)
        return

    for col in ALERT_COLUMNS:
        if col not in df.columns:
            df[col] = ""

    df = df[ALERT_COLUMNS].copy()
    df.to_csv(ALERTS_FILE, index=False)


def clear_alerts():
    save_alerts(empty_alerts())


def tradingview_symbol(ticker):
    t = normalize_ticker(ticker)
    return f"NASDAQ:{t}"


def tradingview_chart_url(ticker):
    symbol = urllib.parse.quote(tradingview_symbol(ticker), safe="")
    return f"https://www.tradingview.com/chart/?symbol={symbol}"


def build_alert_message(row, expected_net=None, risk_note="", include_reason=True):
    """
    Simple English-only Telegram alert message.

    Format:
    🟢 LONG 📈
    NVDA
    Stop Loss: 183.70
    Take Profit: 185.10

    or:

    🔴 SHORT 📉
    NVDA
    Stop Loss: 183.70
    Take Profit: 185.10
    """
    ticker = str(row.get("ticker", "")).upper()
    side = str(row.get("side", "")).upper()
    stop = safe_float(row.get("stop_loss"), np.nan)
    target = safe_float(row.get("target_reference"), np.nan)

    if side == "LONG":
        direction = "🟢 LONG 📈"
    elif side == "SHORT":
        direction = "🔴 SHORT 📉"
    else:
        direction = "⚪ SIGNAL"

    return (
        f"{direction}\n"
        f"{ticker}\n"
        f"Stop Loss: {stop:.2f}\n"
        f"Take Profit: {target:.2f}"
    )


def send_telegram_message(bot_token, chat_id, message):
    """
    Send Telegram message using requests + certifi.

    This avoids common SSL errors such as:
    CERTIFICATE_VERIFY_FAILED / self-signed certificate in certificate chain

    We do NOT disable SSL verification. We use certifi's trusted CA bundle.
    """
    if not bot_token or not chat_id:
        return False, "חסר Bot Token או Chat ID."

    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        "chat_id": str(chat_id),
        "text": str(message),
        "disable_web_page_preview": False,
    }

    # Preferred path: requests with certifi CA bundle
    try:
        resp = requests.post(url, data=payload, timeout=15, verify=certifi.where())
        if 200 <= resp.status_code < 300:
            return True, ""
        return False, resp.text[:500]
    except requests.exceptions.SSLError as e:
        # Fallback path: urllib with certifi SSL context
        try:
            data = urllib.parse.urlencode(
                {
                    "chat_id": str(chat_id),
                    "text": str(message),
                    "disable_web_page_preview": "false",
                }
            ).encode("utf-8")

            context = ssl.create_default_context(cafile=certifi.where())
            req = urllib.request.Request(url, data=data, method="POST")
            with urllib.request.urlopen(req, timeout=15, context=context) as resp:
                body = resp.read().decode("utf-8", errors="ignore")
                if 200 <= resp.status < 300:
                    return True, ""
                return False, body[:500]
        except Exception as e2:
            return (
                False,
                "שגיאת SSL גם אחרי certifi. "
                "במחשב Mac נסה להריץ Install Certificates.command או לעדכן certifi. "
                f"פירוט: {str(e2)[:300]}",
            )
    except Exception as e:
        return False, str(e)[:500]


def create_trade_alert(row, expected_net=None, risk_note=""):
    settings = load_alert_settings()
    score = int(safe_float(row.get("score", 0), 0))
    tv_url = tradingview_chart_url(row.get("ticker", ""))
    include_reason = bool(settings.get("include_reason", True))
    message = build_alert_message(
        row=row,
        expected_net=expected_net,
        risk_note=risk_note,
        include_reason=include_reason,
    )

    telegram_sent = False
    telegram_error = ""

    if bool(settings.get("alerts_enabled", False)) and bool(settings.get("telegram_enabled", False)):
        if score >= int(settings.get("send_only_score_at_least", 8)):
            telegram_sent, telegram_error = send_telegram_message(
                bot_token=str(settings.get("telegram_bot_token", "")),
                chat_id=str(settings.get("telegram_chat_id", "")),
                message=message,
            )
        else:
            telegram_error = f"לא נשלח לטלגרם כי הניקוד {score} נמוך מסף ההתראה."
    else:
        telegram_error = "התראות טלגרם כבויות."

    alerts = load_alerts()
    alert_row = {
        "alert_id": str(uuid.uuid4()),
        "created_at": now_ny_iso(),
        "trade_id": str(row.get("trade_id", "")),
        "ticker": str(row.get("ticker", "")),
        "mode": str(row.get("mode", "")),
        "side": str(row.get("side", "")),
        "score": score,
        "entry_price": safe_float(row.get("entry_price"), np.nan),
        "stop_loss": safe_float(row.get("stop_loss"), np.nan),
        "target_reference": safe_float(row.get("target_reference"), np.nan),
        "net_pnl_expected": expected_net if expected_net is not None else np.nan,
        "risk_note": risk_note,
        "tradingview_url": tv_url,
        "telegram_sent": "כן" if telegram_sent else "לא",
        "telegram_error": telegram_error,
        "message": message,
    }

    alerts = pd.concat([alerts, pd.DataFrame([alert_row])], ignore_index=True)
    save_alerts(alerts)

    return telegram_sent, telegram_error


def send_test_telegram_alert():
    settings = load_alert_settings()
    test_msg = (
        "✅ בדיקת Telegram Alerts הצליחה\\n\\n"
        "האפליקציה תוכל לשלוח התראה כאשר עסקת Paper מאושרת אחרי בדיקת הדקה.\\n\\n"
        "בדמו בלבד — לא כסף אמיתי."
    )
    return send_telegram_message(
        bot_token=str(settings.get("telegram_bot_token", "")),
        chat_id=str(settings.get("telegram_chat_id", "")),
        message=test_msg,
    )


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


def parse_hhmm(value, fallback="09:30"):
    try:
        h, m = str(value).split(":", 1)
        return int(h), int(m)
    except Exception:
        h, m = str(fallback).split(":", 1)
        return int(h), int(m)


def minute_of_day(ts):
    t = timestamp_to_ny(ts)
    if t is None:
        return -1
    return int(t.hour) * 60 + int(t.minute)



def minute_data_fresh(bar_time, allowed_lag_minutes=1):
    """Accept only the current NY minute or the immediately previous minute.

    Yahoo labels 1-minute bars by the minute timestamp. Comparing floored minutes
    avoids rejecting a completed previous-minute bar merely because the current
    clock has additional seconds.
    """
    bar = timestamp_to_ny(bar_time)
    if bar is None:
        return False, 999

    now_minute = now_ny().floor("min")
    bar_minute = bar.floor("min")
    gap = int((now_minute - bar_minute).total_seconds() // 60)

    # Reject future timestamps and anything older than the allowed completed bar.
    fresh = 0 <= gap <= int(allowed_lag_minutes)
    return fresh, gap


def live_data_status(df, for_entry=False):
    """Require genuinely minute-fresh intraday data for analysis and trading."""
    rules = load_rules()
    if df is None or df.empty:
        return False, "אין נתוני שוק."

    now = now_ny()
    last_bar = timestamp_to_ny(df.index[-1])
    if last_bar is None:
        return False, "זמן הנר האחרון אינו תקין."
    if now.weekday() >= 5:
        return False, "השוק סגור בסוף שבוע."
    if last_bar.date() != now.date():
        return False, f"הנתון האחרון הוא מ־{last_bar.date()} ולא מהיום."

    fresh, minute_gap = minute_data_fresh(last_bar, allowed_lag_minutes=1)
    if not fresh:
        if minute_gap < 0:
            return False, "זמן הנר האחרון נמצא בעתיד ביחס לשעון ניו־יורק."
        return False, (
            f"הנר האחרון ישן ב־{minute_gap} דקות. "
            "נדרש נר מהדקה הנוכחית או מהדקה הקודמת."
        )

    if for_entry:
        start_h, start_m = parse_hhmm(rules.get("entry_start_time", "09:40"), "09:40")
        end_h, end_m = parse_hhmm(rules.get("entry_end_time", "15:25"), "15:25")
        now_minute_of_day = now.hour * 60 + now.minute
        if now_minute_of_day < start_h * 60 + start_m or now_minute_of_day > end_h * 60 + end_m:
            return False, f"כניסות חדשות מותרות בין {start_h:02d}:{start_m:02d} ל־{end_h:02d}:{end_m:02d} ניו־יורק."

    if minute_gap == 0:
        return True, "הנתון הוא מהדקה הנוכחית."
    return True, "הנתון הוא מהדקה הקודמת שהושלמה."


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
# Advanced context, exposure and daily-risk helpers
# ============================================================

TICKER_GROUPS = {
    "QQQ": "NASDAQ", "TQQQ": "NASDAQ", "SQQQ": "NASDAQ",
    "SPY": "BROAD_INDEX", "IWM": "BROAD_INDEX", "DIA": "BROAD_INDEX",
    "AAPL": "MEGA_TECH", "MSFT": "MEGA_TECH", "NVDA": "SEMIS", "AMD": "SEMIS",
    "AVGO": "SEMIS", "ARM": "SEMIS", "INTC": "SEMIS", "MU": "SEMIS",
    "MRVL": "SEMIS", "SMCI": "SEMIS",
    "META": "MEGA_TECH", "GOOGL": "MEGA_TECH", "AMZN": "MEGA_TECH", "NFLX": "MEGA_TECH",
    "CRM": "SOFTWARE", "ORCL": "SOFTWARE", "ADBE": "SOFTWARE", "SNOW": "SOFTWARE",
    "PLTR": "SOFTWARE", "PANW": "CYBER", "CRWD": "CYBER",
    "MSTR": "CRYPTO", "COIN": "CRYPTO", "HOOD": "FINTECH", "SOFI": "FINTECH",
    "JPM": "BANKS", "BAC": "BANKS", "XOM": "ENERGY", "CVX": "ENERGY",
    "LLY": "HEALTH", "UNH": "HEALTH", "TSLA": "AUTO", "UBER": "MOBILITY",
    "SHOP": "ECOMMERCE", "BABA": "ECOMMERCE",
}


def ticker_group(ticker):
    t = normalize_ticker(ticker)
    return TICKER_GROUPS.get(t, f"OTHER:{t}")


def resample_5m(df):
    if df is None or df.empty:
        return pd.DataFrame()
    d = df.sort_index().copy()
    out = d.resample("5min", label="right", closed="right", offset="30min").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna(subset=["open", "high", "low", "close"])
    return out


def timeframe_alignment_score(df, side):
    d5 = add_indicators(resample_5m(df)).dropna(subset=["close"])
    if len(d5) < 6:
        return 0, "אין מספיק נרות 5 דקות."
    last = d5.iloc[-1]
    prev = d5.iloc[-2]
    close = safe_float(last["close"])
    score = 0
    reasons = []
    if str(side) == "LONG":
        checks = [
            (close > safe_float(last["ema9"]) > safe_float(last["ema21"]), "מחיר מעל EMA9/21 ב־5 דקות"),
            (safe_float(last["ema9_slope"], 0) > 0 and safe_float(last["ema21_slope"], 0) > 0, "שיפוע 5 דקות חיובי"),
            (close > safe_float(last["vwap"], close), "מעל VWAP ב־5 דקות"),
            (close > safe_float(prev["close"], close), "נר 5 דקות מתקדם"),
        ]
    else:
        checks = [
            (close < safe_float(last["ema9"]) < safe_float(last["ema21"]), "מחיר מתחת EMA9/21 ב־5 דקות"),
            (safe_float(last["ema9_slope"], 0) < 0 and safe_float(last["ema21_slope"], 0) < 0, "שיפוע 5 דקות שלילי"),
            (close < safe_float(last["vwap"], close), "מתחת VWAP ב־5 דקות"),
            (close < safe_float(prev["close"], close), "נר 5 דקות נחלש"),
        ]
    for ok, reason in checks:
        if ok:
            score += 1
            reasons.append(reason)
    return int(score), ", ".join(reasons) if reasons else "5 דקות אינן תומכות."


def market_side_for_ticker(ticker, side):
    # SQQQ moves inversely to QQQ.
    if normalize_ticker(ticker) == "SQQQ":
        return "SHORT" if str(side) == "LONG" else "LONG"
    return str(side)



def market_context_check(ticker, side):
    """Use QQQ as a guardrail, not as an absolute blocker in a neutral market."""
    rules = load_rules()
    if not bool(rules.get("market_filter_enabled", True)):
        return True, "פילטר השוק כבוי."

    ref = normalize_ticker(rules.get("market_reference_ticker", "QQQ"))
    try:
        market_df = latest_session(fetch_1m(ref))
    except Exception as exc:
        return False, f"לא ניתן לבדוק את {ref}: {str(exc)[:80]}"

    ok, reason = live_data_status(market_df, for_entry=False)
    if not ok:
        return False, f"נתוני {ref} אינם עדכניים: {reason}"

    required_side = market_side_for_ticker(ticker, side)
    aligned, aligned_reason = timeframe_alignment_score(market_df, required_side)
    opposite = "SHORT" if required_side == "LONG" else "LONG"
    opposite_score, opposite_reason = timeframe_alignment_score(market_df, opposite)
    minimum = int(rules.get("min_5m_alignment_score", 2))

    # Reject only when QQQ clearly supports the opposite direction.
    if opposite_score >= max(3, minimum + 1) and opposite_score >= aligned + 2:
        return False, (
            f"{ref} תומך בבירור בכיוון ההפוך: {opposite_score}/4 מול {aligned}/4. "
            f"{opposite_reason}"
        )

    if aligned >= minimum:
        return True, f"{ref} תומך בכיוון העסקה ({aligned}/4): {aligned_reason}"

    # Neutral QQQ no longer blocks every stock-specific setup.
    return True, f"{ref} ניטרלי ({aligned}/4 מול {opposite_score}/4); העסקה נשענת על המניה עצמה."



def today_trade_mask(trades):
    if trades is None or trades.empty:
        return pd.Series([], dtype=bool)
    today = now_ny().date()
    return trades["entry_time"].apply(lambda x: (timestamp_to_ny(x).date() == today) if timestamp_to_ny(x) is not None else False)


def daily_risk_gate(trades=None):
    rules = load_rules()
    trades = load_trades() if trades is None else normalize_trade_dtypes(trades)
    if trades.empty:
        return True, "הגנת יום תקינה."
    mask = today_trade_mask(trades)
    today_trades = trades[mask].copy() if len(mask) else trades.iloc[0:0].copy()
    if today_trades.empty:
        return True, "הגנת יום תקינה."

    today_net = float(pd.to_numeric(today_trades["net_pnl"], errors="coerce").fillna(0).sum())
    daily_limit = abs(float(rules.get("daily_loss_limit_dollars", 25.0)))
    if today_net <= -daily_limit:
        return False, f"נעצרו כניסות: ההפסד היומי הוא ${today_net:.2f}, מגבלה ${daily_limit:.2f}."

    max_trades = int(rules.get("max_trades_per_day", 18))
    if len(today_trades) >= max_trades:
        return False, f"נעצרו כניסות: נפתחו כבר {len(today_trades)} עסקאות היום (מקסימום {max_trades})."

    closed = today_trades[today_trades["status"].astype(str).eq("CLOSED")].copy()
    if not closed.empty:
        closed["_exit_ts"] = closed["exit_time"].apply(timestamp_to_ny)
        closed = closed.dropna(subset=["_exit_ts"]).sort_values("_exit_ts")
        streak = 0
        for pnl in reversed(pd.to_numeric(closed["net_pnl"], errors="coerce").fillna(0).tolist()):
            if pnl < 0:
                streak += 1
            else:
                break
        needed = int(rules.get("max_consecutive_losses", 3))
        if streak >= needed:
            last_exit = closed.iloc[-1]["_exit_ts"]
            elapsed = (now_ny() - last_exit).total_seconds() / 60.0
            pause = float(rules.get("loss_streak_pause_minutes", 15))
            if elapsed < pause:
                return False, f"הפסקת הגנה: {streak} הפסדים רצופים. נשארו {max(0, pause-elapsed):.1f} דקות."
    return True, f"הגנת יום תקינה | P/L היום ${today_net:.2f}."


def exposure_gate(trades, ticker, side, include_pending=True):
    rules = load_rules()
    trades = normalize_trade_dtypes(trades)
    open_df = trades[trades["status"].astype(str).eq("OPEN")].copy() if not trades.empty else trades
    max_open = int(rules.get("max_open_trades", 5))
    if len(open_df) >= max_open:
        return False, f"כבר יש {len(open_df)} עסקאות פתוחות (מקסימום {max_open})."
    max_side = int(rules.get("max_same_side_open", 3))
    side_count = int(open_df["side"].astype(str).eq(str(side)).sum()) if not open_df.empty else 0
    group = ticker_group(ticker)
    group_count = int(open_df["ticker"].apply(ticker_group).eq(group).sum()) if not open_df.empty else 0

    if include_pending:
        pending = load_pending()
        if not pending.empty:
            active = pending[pending["status"].astype(str).eq("PENDING")]
            side_count += int(active["side"].astype(str).eq(str(side)).sum())
            group_count += int(active["ticker"].apply(ticker_group).eq(group).sum())

    if side_count >= max_side:
        return False, f"מגבלת כיוון: כבר יש {side_count} עסקאות/מועמדות {side}."
    max_group = int(rules.get("max_same_group_open", 2))
    if group_count >= max_group:
        return False, f"מגבלת קבוצה {group}: כבר יש {group_count} עסקאות/מועמדות."
    return True, f"חשיפה תקינה | קבוצה {group}."


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
    score = int(max(1, min(12, int(score))))
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
# Historical pattern filter
# ============================================================

def session_minutes_from_open(ts):
    """
    Minutes from regular US market open 09:30 NY.
    """
    t = timestamp_to_ny(ts)
    if t is None:
        return 0.0
    return (t.hour * 60 + t.minute + t.second / 60.0) - (9 * 60 + 30)


def historical_pattern_adjustment(ticker, mode, side, current_session_df, current_time=None):
    """
    After the first ~30 minutes, compare the current intraday structure of the stock
    with previous days.

    This is not a prediction engine. It is a conservative additional filter:
    - If similar past intraday structures usually continued in the same direction,
      add score.
    - If similar structures usually reversed against the trade direction,
      subtract score.
    - If there are not enough samples, do nothing.

    Features compared:
    - return from open
    - distance from VWAP
    - intraday range %
    - recent slope
    - elapsed minutes from open

    Uses yfinance 1m data, so history is limited to recent days.
    """
    rules = load_rules()
    after_minutes = float(rules.get("use_history_after_minutes", 30))
    min_samples = int(rules.get("history_min_samples", 4))
    max_bonus = int(rules.get("history_max_score_bonus", 1))
    max_penalty = int(rules.get("history_max_score_penalty", 1))

    if current_session_df is None or current_session_df.empty:
        return {"delta": 0, "reason": "אין נתוני היסטוריה להשוואה."}

    d = add_indicators(current_session_df).dropna(subset=["close"]).copy()
    if d.empty:
        return {"delta": 0, "reason": "אין מספיק אינדיקטורים להשוואה היסטורית."}

    if current_time is None:
        current_time = d.index[-1]

    elapsed = session_minutes_from_open(current_time)
    if elapsed < after_minutes:
        return {
            "delta": 0,
            "reason": f"היסטוריה לא הופעלה: עברו {elapsed:.0f} דק׳ מהפתיחה, נדרש {after_minutes:.0f}.",
        }

    # Horizon: what we check after a similar past setup.
    horizon = 10 if str(mode) == "מהירה" else 30

    current_day = d[d.index.date == timestamp_to_ny(current_time).date()].copy()
    if len(current_day) < 10:
        return {"delta": 0, "reason": "אין מספיק נרות ביום הנוכחי להשוואה היסטורית."}

    last = current_day.iloc[-1]
    open_price = safe_float(current_day.iloc[0]["open"], safe_float(last["close"]))
    close_now = safe_float(last["close"])

    current_features = {
        "ret_open": (close_now / open_price - 1) * 100 if open_price else 0,
        "vwap_gap": (close_now / safe_float(last.get("vwap"), close_now) - 1) * 100 if close_now else 0,
        "range_pct": ((safe_float(current_day["high"].max()) - safe_float(current_day["low"].min())) / open_price) * 100 if open_price else 0,
        "slope_pct": (linear_slope_per_bar(current_day["close"], lookback=15) / close_now) * 100 if close_now else 0,
    }

    try:
        all_df = fetch_1m(ticker)
    except Exception:
        return {"delta": 0, "reason": "לא ניתן למשוך היסטוריה מהמניה."}

    if all_df is None or all_df.empty:
        return {"delta": 0, "reason": "אין היסטוריה זמינה מהמניה."}

    all_df = add_indicators(all_df).dropna(subset=["close"]).copy()
    current_date = timestamp_to_ny(current_time).date()

    samples = []
    for day, day_df in all_df.groupby(all_df.index.date):
        if day >= current_date:
            continue

        day_df = day_df.sort_index()
        if len(day_df) < elapsed + horizon + 5:
            # rough guard; elapsed is minutes and data is 1m
            pass

        # Find the candle in that day closest to the same minutes-from-open.
        day_df = day_df.copy()
        day_df["_elapsed"] = [session_minutes_from_open(x) for x in day_df.index]
        past_now = day_df.iloc[(day_df["_elapsed"] - elapsed).abs().argsort()[:1]]
        if past_now.empty:
            continue

        past_idx = past_now.index[0]
        loc = day_df.index.get_loc(past_idx)
        if isinstance(loc, slice) or isinstance(loc, np.ndarray):
            continue

        future_loc = loc + int(horizon)
        if future_loc >= len(day_df):
            continue

        past_slice = day_df.iloc[: loc + 1]
        if len(past_slice) < 10:
            continue

        past_last = day_df.iloc[loc]
        past_open = safe_float(day_df.iloc[0]["open"], safe_float(past_last["close"]))
        past_close = safe_float(past_last["close"])
        future_close = safe_float(day_df.iloc[future_loc]["close"])

        past_features = {
            "ret_open": (past_close / past_open - 1) * 100 if past_open else 0,
            "vwap_gap": (past_close / safe_float(past_last.get("vwap"), past_close) - 1) * 100 if past_close else 0,
            "range_pct": ((safe_float(past_slice["high"].max()) - safe_float(past_slice["low"].min())) / past_open) * 100 if past_open else 0,
            "slope_pct": (linear_slope_per_bar(past_slice["close"], lookback=15) / past_close) * 100 if past_close else 0,
        }

        # Simple normalized distance. Lower is more similar.
        dist = (
            abs(current_features["ret_open"] - past_features["ret_open"]) / 2.0
            + abs(current_features["vwap_gap"] - past_features["vwap_gap"]) / 1.0
            + abs(current_features["range_pct"] - past_features["range_pct"]) / 2.0
            + abs(current_features["slope_pct"] - past_features["slope_pct"]) / 0.05
        )

        future_ret_pct = (future_close / past_close - 1) * 100 if past_close else 0
        if str(side) == "LONG":
            supported = future_ret_pct > 0
        else:
            supported = future_ret_pct < 0

        samples.append(
            {
                "day": str(day),
                "dist": float(dist),
                "future_ret_pct": float(future_ret_pct),
                "supported": bool(supported),
            }
        )

    if len(samples) < min_samples:
        return {
            "delta": 0,
            "reason": f"היסטוריה: נמצאו רק {len(samples)} דוגמאות דומות, לא מספיק להשפעה.",
        }

    samples = sorted(samples, key=lambda x: x["dist"])[: max(min_samples, 5)]
    support_rate = sum(1 for s in samples if s["supported"]) / len(samples)
    avg_future_ret = float(np.mean([s["future_ret_pct"] for s in samples]))

    delta = 0
    if support_rate >= 0.75:
        delta = max_bonus
    elif support_rate >= 0.62:
        delta = 1
    elif support_rate <= 0.25:
        delta = -max_penalty
    elif support_rate <= 0.38:
        delta = -1

    direction_word = "לונג" if str(side) == "LONG" else "שורט"
    reason = (
        f"היסטוריה אחרי {elapsed:.0f} דק׳: מתוך {len(samples)} ימים דומים, "
        f"{support_rate*100:.0f}% תמכו ב־{direction_word}. "
        f"תנועה ממוצעת לאחר {horizon} דק׳: {avg_future_ret:.2f}%. "
        f"שינוי ניקוד: {delta:+d}."
    )

    return {"delta": int(delta), "reason": reason}



# ============================================================
# Engineering pattern engine
# ============================================================

def empty_engineering_predictions():
    df = pd.DataFrame(columns=ENGINEERING_COLUMNS)
    for col in ENGINEERING_COLUMNS:
        df[col] = df[col].astype("object")
    return df


def load_engineering_predictions():
    if not ENGINEERING_FILE.exists() or ENGINEERING_FILE.stat().st_size == 0:
        return empty_engineering_predictions()
    try:
        df = pd.read_csv(ENGINEERING_FILE)
    except Exception:
        return empty_engineering_predictions()
    for col in ENGINEERING_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    return df[ENGINEERING_COLUMNS].copy()


def save_engineering_predictions(df):
    if df is None or df.empty:
        empty_engineering_predictions().to_csv(ENGINEERING_FILE, index=False)
        return
    for col in ENGINEERING_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    df = df[ENGINEERING_COLUMNS].tail(5000).copy()
    df.to_csv(ENGINEERING_FILE, index=False)


def clear_engineering_predictions():
    save_engineering_predictions(empty_engineering_predictions())


def log_engineering_prediction(ticker, mode, bar_time, technical_side, result,
                               long_score=0, short_score=0, final_score=0, decision=""):
    """Store one row per ticker/mode/bar so repeated Streamlit reruns do not duplicate it."""
    try:
        df = load_engineering_predictions()
        bar_text = str(bar_time or "")
        ticker = normalize_ticker(ticker)
        mode = str(mode)
        if not df.empty:
            duplicate = (
                df["ticker"].astype(str).eq(ticker)
                & df["mode"].astype(str).eq(mode)
                & df["bar_time"].astype(str).eq(bar_text)
            )
            if duplicate.any():
                return
        row = {
            "prediction_id": str(uuid.uuid4()),
            "created_at": now_ny_iso(),
            "bar_time": bar_text,
            "ticker": ticker,
            "mode": mode,
            "technical_side": str(technical_side or "WAIT"),
            "predicted_side": str(result.get("predicted_side", "WAIT")),
            "decision": str(decision or result.get("decision", "")),
            "confidence": safe_float(result.get("confidence"), 0.0),
            "sample_count": int(safe_float(result.get("sample_count"), 0)),
            "mean_similarity": safe_float(result.get("mean_similarity"), 0.0),
            "best_similarity": safe_float(result.get("best_similarity"), 0.0),
            "weakest_similarity": safe_float(result.get("weakest_similarity"), 0.0),
            "long_probability": safe_float(result.get("long_probability"), 0.0),
            "short_probability": safe_float(result.get("short_probability"), 0.0),
            "neutral_probability": safe_float(result.get("neutral_probability"), 1.0),
            "long_target_rate": safe_float(result.get("long_target_rate"), 0.0),
            "short_target_rate": safe_float(result.get("short_target_rate"), 0.0),
            "long_expectancy_r": safe_float(result.get("long_expectancy_r"), 0.0),
            "short_expectancy_r": safe_float(result.get("short_expectancy_r"), 0.0),
            "expected_mfe_r": safe_float(result.get("expected_mfe_r"), 0.0),
            "expected_mae_r": safe_float(result.get("expected_mae_r"), 0.0),
            "pattern_state": str(result.get("pattern_state", "UNKNOWN")),
            "function_model": str(result.get("function_model", "UNKNOWN")),
            "technical_long_score": int(long_score),
            "technical_short_score": int(short_score),
            "final_score": int(final_score),
            "reason": str(result.get("reason", "")),
        }
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        save_engineering_predictions(df)
    except Exception:
        pass


def _robust_zscore(values):
    x = np.asarray(values, dtype=float)
    if x.size == 0:
        return x
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    scale = max(1.4826 * mad, np.nanstd(x), 1e-9)
    return np.nan_to_num((x - med) / scale, nan=0.0, posinf=0.0, neginf=0.0)


def _safe_corr(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) != len(b) or len(a) < 3:
        return 0.0
    if np.nanstd(a) < 1e-9 or np.nanstd(b) < 1e-9:
        return 0.0
    value = np.corrcoef(a, b)[0, 1]
    return float(value) if np.isfinite(value) else 0.0


def _dtw_distance(a, b, band=None):
    """Small dependency-free DTW with a Sakoe-Chiba band."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return 99.0
    if band is None:
        band = max(abs(n - m), int(max(n, m) * 0.20))
    inf = float("inf")
    prev = np.full(m + 1, inf, dtype=float)
    prev[0] = 0.0
    for i in range(1, n + 1):
        curr = np.full(m + 1, inf, dtype=float)
        start = max(1, i - band)
        end = min(m, i + band)
        for j in range(start, end + 1):
            cost = abs(a[i - 1] - b[j - 1])
            curr[j] = cost + min(curr[j - 1], prev[j], prev[j - 1])
        prev = curr
    return float(prev[m] / max(n + m, 1))


def _turning_points(values):
    """Count meaningful direction changes after light smoothing and noise rejection."""
    x = np.asarray(values, dtype=float)
    if len(x) < 5:
        return 0
    # Three-point triangular smoothing reduces one-bar zigzags.
    smooth = np.convolve(x, np.asarray([0.25, 0.50, 0.25]), mode="same")
    smooth[0], smooth[-1] = x[0], x[-1]
    diffs = np.diff(smooth)
    threshold = max(float(np.nanstd(diffs)) * 0.35, 0.025)
    signs = np.where(diffs > threshold, 1, np.where(diffs < -threshold, -1, 0))
    # Carry the last meaningful sign through tiny changes.
    cleaned = []
    last = 0
    for sign in signs:
        if sign != 0:
            last = int(sign)
            cleaned.append(last)
        elif last != 0:
            cleaned.append(last)
    if len(cleaned) < 2:
        return 0
    return int(np.sum(np.asarray(cleaned[1:]) != np.asarray(cleaned[:-1])))


def _fit_shape_models(price_shape):
    """Fit several engineering curve families and return the lowest normalized error."""
    y = np.asarray(price_shape, dtype=float)
    if len(y) < 8:
        return {"model": "INSUFFICIENT", "rmse": 1.0}
    x = np.linspace(0.0, 1.0, len(y))
    scale = max(np.std(y), 1e-6)
    models = []
    for degree, name in [(1, "LINEAR"), (2, "QUADRATIC"), (3, "CUBIC")]:
        try:
            coeff = np.polyfit(x, y, degree)
            pred = np.polyval(coeff, x)
            rmse = float(np.sqrt(np.mean((y - pred) ** 2)) / scale)
            models.append((rmse, name))
        except Exception:
            pass

    # Saturating / accelerating exponentials. For each rate, solve a and c linearly.
    for rate in [0.7, 1.2, 2.0, 3.5, 5.0]:
        for name, basis in [
            ("SATURATION", 1.0 - np.exp(-rate * x)),
            ("ACCELERATING", np.exp(rate * x) - 1.0),
            ("DECAY", np.exp(-rate * x)),
        ]:
            try:
                A = np.column_stack([basis, np.ones_like(basis)])
                coeff, *_ = np.linalg.lstsq(A, y, rcond=None)
                pred = A @ coeff
                rmse = float(np.sqrt(np.mean((y - pred) ** 2)) / scale)
                models.append((rmse, name))
            except Exception:
                pass

    # Local oscillation family.
    for cycles in [1.0, 1.5, 2.0, 3.0]:
        try:
            A = np.column_stack([
                np.sin(2 * np.pi * cycles * x),
                np.cos(2 * np.pi * cycles * x),
                x,
                np.ones_like(x),
            ])
            coeff, *_ = np.linalg.lstsq(A, y, rcond=None)
            pred = A @ coeff
            rmse = float(np.sqrt(np.mean((y - pred) ** 2)) / scale)
            models.append((rmse, "OSCILLATORY"))
        except Exception:
            pass

    if not models:
        return {"model": "UNKNOWN", "rmse": 1.0}
    rmse, model = min(models, key=lambda item: item[0])
    return {"model": model, "rmse": float(rmse)}


def _engineering_channels(window_df):
    d = add_indicators(window_df).dropna(subset=["close"]).copy()
    if d.empty:
        return None
    close = d["close"].astype(float).values
    high = d["high"].astype(float).values
    low = d["low"].astype(float).values
    volume = d["volume"].astype(float).values
    atr_series = pd.to_numeric(d.get("atr14", pd.Series(index=d.index, dtype=float)), errors="coerce")
    atr = safe_float(atr_series.dropna().median() if not atr_series.dropna().empty else np.nan, np.nan)
    if not np.isfinite(atr) or atr <= 0:
        atr = max(float(np.nanmedian(high - low)), float(np.nanmedian(close)) * 0.0008, 1e-6)

    price_atr = (close - close[0]) / atr
    velocity = np.diff(price_atr, prepend=price_atr[0])
    acceleration = np.diff(velocity, prepend=velocity[0])
    range_atr = (high - low) / atr
    median_volume = max(float(np.nanmedian(volume)), 1.0)
    volume_rel = np.log1p(np.maximum(volume, 0.0) / median_volume)
    vwap = pd.to_numeric(d.get("vwap", d["close"]), errors="coerce").fillna(d["close"]).values
    vwap_gap = (close - vwap) / atr
    candle_range = np.maximum(high - low, 1e-9)
    close_location = ((close - low) / candle_range) - 0.5

    price_shape = _robust_zscore(price_atr)
    velocity_shape = _robust_zscore(velocity)
    acceleration_shape = _robust_zscore(acceleration)
    volume_shape = _robust_zscore(volume_rel)
    range_shape = _robust_zscore(range_atr)
    vwap_shape = _robust_zscore(vwap_gap)
    location_shape = _robust_zscore(close_location)

    x = np.arange(len(price_atr), dtype=float)
    try:
        slope = float(np.polyfit(x, price_atr, 1)[0])
        curvature = float(np.polyfit(x, price_atr, 2)[0])
    except Exception:
        slope, curvature = 0.0, 0.0
    half = max(3, len(range_atr) // 2)
    early_range = max(float(np.nanmean(range_atr[:half])), 1e-6)
    late_range = float(np.nanmean(range_atr[-half:]))
    compression = late_range / early_range
    model = _fit_shape_models(price_shape)

    return {
        "price": price_shape,
        "velocity": velocity_shape,
        "acceleration": acceleration_shape,
        "volume": volume_shape,
        "range": range_shape,
        "vwap": vwap_shape,
        "location": location_shape,
        "raw_price_atr": price_atr,
        "atr": float(atr),
        "slope": slope,
        "curvature": curvature,
        "compression": float(compression),
        "turning_points": _turning_points(price_shape),
        "model": model["model"],
        "model_rmse": model["rmse"],
    }


def _engineering_state(ch):
    slope = safe_float(ch.get("slope"), 0.0)
    curvature = safe_float(ch.get("curvature"), 0.0)
    compression = safe_float(ch.get("compression"), 1.0)
    turns = int(ch.get("turning_points", 0))
    recent_velocity = float(np.nanmean(ch["velocity"][-5:])) if len(ch["velocity"]) >= 5 else 0.0
    recent_accel = float(np.nanmean(ch["acceleration"][-4:])) if len(ch["acceleration"]) >= 4 else 0.0
    net_move = safe_float(ch["raw_price_atr"][-1] - ch["raw_price_atr"][0], 0.0) if len(ch.get("raw_price_atr", [])) else 0.0

    if compression < 0.72 and abs(slope) < 0.10 and abs(net_move) < 1.0:
        return "COMPRESSION"
    if slope > 0.10 and recent_accel > 0.04:
        return "UP_ACCELERATION"
    if slope < -0.10 and recent_accel < -0.04:
        return "DOWN_ACCELERATION"
    if slope > 0.075 and recent_accel < -0.055:
        return "UP_WEAKENING"
    if slope < -0.075 and recent_accel > 0.055:
        return "DOWN_WEAKENING"
    if slope > 0.055 or recent_velocity > 0.13 or net_move > 1.35:
        return "UP_TREND"
    if slope < -0.055 or recent_velocity < -0.13 or net_move < -1.35:
        return "DOWN_TREND"
    # Oscillation requires repeated meaningful turns AND no directional displacement.
    if turns >= max(4, len(ch["price"]) // 5) and abs(slope) < 0.055 and abs(net_move) < 1.35:
        return "OSCILLATION"
    return "NEUTRAL"


def _pattern_distance(current, candidate, time_penalty=0.0):
    dtw = _dtw_distance(current["price"], candidate["price"])
    vel = (1.0 - _safe_corr(current["velocity"], candidate["velocity"])) / 2.0
    acc = (1.0 - _safe_corr(current["acceleration"], candidate["acceleration"])) / 2.0
    vol = (1.0 - _safe_corr(current["volume"], candidate["volume"])) / 2.0
    rng = (1.0 - _safe_corr(current["range"], candidate["range"])) / 2.0
    vwap = (1.0 - _safe_corr(current["vwap"], candidate["vwap"])) / 2.0
    loc = (1.0 - _safe_corr(current["location"], candidate["location"])) / 2.0
    descriptor = (
        min(abs(current["slope"] - candidate["slope"]) / 0.30, 2.0)
        + min(abs(current["curvature"] - candidate["curvature"]) / 0.06, 2.0)
        + min(abs(current["compression"] - candidate["compression"]), 2.0)
        + min(abs(current["turning_points"] - candidate["turning_points"]) / 6.0, 2.0)
    ) / 4.0
    model_penalty = 0.0 if current["model"] == candidate["model"] else 0.12
    distance = (
        0.38 * dtw + 0.13 * vel + 0.08 * acc + 0.10 * vol
        + 0.08 * rng + 0.09 * vwap + 0.05 * loc + 0.07 * descriptor
        + model_penalty + 0.05 * time_penalty
    )
    return float(max(distance, 0.0))


def _first_touch_outcome(future_df, entry, atr, target_r, stop_r, side):
    if future_df is None or future_df.empty or atr <= 0:
        return "NONE"
    if side == "LONG":
        target = entry + target_r * atr
        stop = entry - stop_r * atr
        for _, bar in future_df.iterrows():
            low = safe_float(bar["low"])
            high = safe_float(bar["high"])
            if low <= stop:
                return "STOP"
            if high >= target:
                return "TARGET"
    else:
        target = entry - target_r * atr
        stop = entry + stop_r * atr
        for _, bar in future_df.iterrows():
            high = safe_float(bar["high"])
            low = safe_float(bar["low"])
            if high >= stop:
                return "STOP"
            if low <= target:
                return "TARGET"
    return "NONE"


def engineering_pattern_analysis(full_df, mode, current_end=None, stop_r=1.0, target_r=1.5):
    """Nearest-pattern predictor using only information available before current_end."""
    rules = load_rules()
    empty = {
        "ready": False, "predicted_side": "WAIT", "decision": "INSUFFICIENT",
        "confidence": 0.0, "sample_count": 0, "mean_similarity": 0.0,
        "long_probability": 0.0, "short_probability": 0.0, "neutral_probability": 1.0,
        "best_similarity": 0.0, "weakest_similarity": 0.0,
        "long_target_rate": 0.0, "short_target_rate": 0.0,
        "long_expectancy_r": 0.0, "short_expectancy_r": 0.0,
        "expected_mfe_r": 0.0, "expected_mae_r": 0.0,
        "pattern_state": "UNKNOWN", "function_model": "UNKNOWN",
        "reason": "אין מספיק נתונים למנוע התבניות.",
    }
    if full_df is None or full_df.empty:
        return empty

    d = full_df.sort_index().copy()
    if current_end is not None:
        try:
            end_ts = timestamp_to_ny(current_end)
            d = d[d.index <= end_ts]
        except Exception:
            pass
    d = add_indicators(d).dropna(subset=["open", "high", "low", "close"]).copy()

    window = int(rules.get("engineering_window_fast", 24) if str(mode) == "מהירה" else rules.get("engineering_window_half", 36))
    horizon = int(rules.get("engineering_horizon_fast", 8) if str(mode) == "מהירה" else rules.get("engineering_horizon_half", 20))
    min_samples = int(rules.get("engineering_min_samples", 10))
    top_k = int(rules.get("engineering_top_k", 25))
    step = max(1, int(rules.get("engineering_candidate_step", 4)))
    max_candidates = max(top_k, int(rules.get("engineering_max_candidates", 320)))
    time_tolerance = float(rules.get("engineering_time_tolerance_minutes", 180))
    min_similarity = float(rules.get("engineering_min_similarity", 0.34))
    similarity_band = float(rules.get("engineering_similarity_band", 0.12))
    neutral_band_r = float(rules.get("engineering_neutral_band_r", 0.10))

    if len(d) < window + horizon + 8:
        empty["reason"] = f"נדרשים לפחות {window + horizon + 8} נרות; קיימים {len(d)}."
        return empty

    current_window = d.iloc[-window:]
    if current_window.index[0].date() != current_window.index[-1].date():
        empty["reason"] = "חלון התבנית חוצה יום מסחר."
        return empty
    current = _engineering_channels(current_window)
    if current is None:
        return empty
    current_state = _engineering_state(current)
    current_model = current.get("model", "UNKNOWN")
    current_tod = session_minutes_from_open(current_window.index[-1])

    last_candidate_end = len(d) - horizon - 2
    first_candidate_end = window - 1
    candidate_ends = list(range(first_candidate_end, last_candidate_end + 1, step))[-max_candidates:]
    prefiltered = []

    for end_idx in candidate_ends:
        start_idx = end_idx - window + 1
        future_end = end_idx + horizon
        cand_window = d.iloc[start_idx:end_idx + 1]
        future = d.iloc[end_idx + 1:future_end + 1]
        if len(cand_window) != window or len(future) != horizon:
            continue
        # No overnight windows or future labels crossing a session boundary.
        day = cand_window.index[-1].date()
        if cand_window.index[0].date() != day or future.index[-1].date() != day:
            continue
        # Avoid overlapping with current pattern.
        if cand_window.index[-1] >= current_window.index[0]:
            continue
        cand_tod = session_minutes_from_open(cand_window.index[-1])
        tod_diff = abs(cand_tod - current_tod)
        if tod_diff > time_tolerance:
            continue
        cand_price = _robust_zscore((cand_window["close"].astype(float).values - float(cand_window["close"].iloc[0])))
        pre_distance = float(np.sqrt(np.mean((current["price"] - cand_price) ** 2)))
        pre_distance += min(tod_diff / max(time_tolerance, 1.0), 1.0) * 0.08
        prefiltered.append((pre_distance, start_idx, end_idx, future_end, tod_diff))

    if not prefiltered:
        empty.update({"pattern_state": current_state, "function_model": current_model})
        empty["reason"] = "לא נמצאו חלונות עבר תקינים להשוואה."
        return empty

    prefiltered.sort(key=lambda x: x[0])
    detailed_pool = prefiltered[: max(top_k * 2, 40)]
    matches = []
    target_r = float(np.clip(target_r, 0.45, 5.0))
    stop_r = float(np.clip(stop_r, 0.30, 5.0))

    for _, start_idx, end_idx, future_end, tod_diff in detailed_pool:
        cand_window = d.iloc[start_idx:end_idx + 1]
        future = d.iloc[end_idx + 1:future_end + 1]
        candidate = _engineering_channels(cand_window)
        if candidate is None:
            continue
        time_penalty = min(tod_diff / max(time_tolerance, 1.0), 1.0)
        distance = _pattern_distance(current, candidate, time_penalty=time_penalty)
        similarity = float(1.0 / (1.0 + distance))
        if similarity < min_similarity:
            continue

        entry = safe_float(cand_window.iloc[-1]["close"])
        atr = max(safe_float(candidate.get("atr"), 0.0), entry * 0.0005, 1e-6)
        final_close = safe_float(future.iloc[-1]["close"])
        future_ret_r = (final_close - entry) / atr
        long_mfe = max(0.0, (safe_float(future["high"].max()) - entry) / atr)
        long_mae = max(0.0, (entry - safe_float(future["low"].min())) / atr)
        short_mfe = max(0.0, (entry - safe_float(future["low"].min())) / atr)
        short_mae = max(0.0, (safe_float(future["high"].max()) - entry) / atr)
        long_outcome = _first_touch_outcome(future, entry, atr, target_r, stop_r, "LONG")
        short_outcome = _first_touch_outcome(future, entry, atr, target_r, stop_r, "SHORT")

        matches.append({
            "similarity": similarity,
            "future_ret_r": float(future_ret_r),
            "long_mfe": float(long_mfe), "long_mae": float(long_mae),
            "short_mfe": float(short_mfe), "short_mae": float(short_mae),
            "long_outcome": long_outcome, "short_outcome": short_outcome,
            "state": _engineering_state(candidate), "model": candidate.get("model", "UNKNOWN"),
            "time": str(cand_window.index[-1]),
        })

    matches.sort(key=lambda x: x["similarity"], reverse=True)
    if matches:
        best_similarity = float(matches[0]["similarity"])
        adaptive_floor = max(min_similarity, best_similarity - max(similarity_band, 0.01))
        matches = [m for m in matches if float(m["similarity"]) >= adaptive_floor][:top_k]
    if not matches:
        empty.update({"pattern_state": current_state, "function_model": current_model})
        empty["reason"] = "החלונות שנמצאו לא עברו את סף הדמיון."
        return empty

    weights = np.asarray([max(m["similarity"], 1e-6) ** 3 for m in matches], dtype=float)
    weights = weights / max(weights.sum(), 1e-9)
    future_r = np.asarray([m["future_ret_r"] for m in matches], dtype=float)
    similarities = np.asarray([m["similarity"] for m in matches], dtype=float)
    mean_similarity = float(np.sum(weights * similarities))
    best_similarity = float(np.nanmax(similarities))
    weakest_similarity = float(np.nanmin(similarities))
    finite_future = np.isfinite(future_r)
    long_mask = finite_future & (future_r > neutral_band_r)
    short_mask = finite_future & (future_r < -neutral_band_r)
    neutral_mask = (~finite_future) | (~long_mask & ~short_mask)
    long_probability = float(np.sum(weights * long_mask.astype(float)))
    short_probability = float(np.sum(weights * short_mask.astype(float)))
    neutral_probability = float(np.sum(weights * neutral_mask.astype(float)))
    probability_total = long_probability + short_probability + neutral_probability
    if probability_total > 0:
        long_probability /= probability_total
        short_probability /= probability_total
        neutral_probability /= probability_total
    long_target_rate = float(np.sum(weights * np.asarray([m["long_outcome"] == "TARGET" for m in matches], dtype=float)))
    short_target_rate = float(np.sum(weights * np.asarray([m["short_outcome"] == "TARGET" for m in matches], dtype=float)))

    def payoff(outcome, terminal_r, side):
        if outcome == "TARGET":
            return target_r
        if outcome == "STOP":
            return -stop_r
        signed = terminal_r if side == "LONG" else -terminal_r
        return float(np.clip(signed, -stop_r, target_r))

    long_payoffs = np.asarray([payoff(m["long_outcome"], m["future_ret_r"], "LONG") for m in matches])
    short_payoffs = np.asarray([payoff(m["short_outcome"], m["future_ret_r"], "SHORT") for m in matches])
    long_expectancy = float(np.sum(weights * long_payoffs))
    short_expectancy = float(np.sum(weights * short_payoffs))
    edge = abs(long_expectancy - short_expectancy)

    min_direction_probability = float(rules.get("engineering_min_direction_probability", 0.60))
    min_expectancy = float(rules.get("engineering_min_expectancy_r", 0.25))
    min_expectancy_gap = float(rules.get("engineering_min_expectancy_gap_r", 0.35))

    if (
        long_expectancy >= min_expectancy
        and long_expectancy > short_expectancy + min_expectancy_gap
        and long_probability >= min_direction_probability
    ):
        predicted_side = "LONG"
        expected_mfe = float(np.sum(weights * np.asarray([m["long_mfe"] for m in matches])))
        expected_mae = float(np.sum(weights * np.asarray([m["long_mae"] for m in matches])))
        direction_prob = long_probability
        target_rate = long_target_rate
    elif (
        short_expectancy >= min_expectancy
        and short_expectancy > long_expectancy + min_expectancy_gap
        and short_probability >= min_direction_probability
    ):
        predicted_side = "SHORT"
        expected_mfe = float(np.sum(weights * np.asarray([m["short_mfe"] for m in matches])))
        expected_mae = float(np.sum(weights * np.asarray([m["short_mae"] for m in matches])))
        direction_prob = short_probability
        target_rate = short_target_rate
    else:
        predicted_side = "WAIT"
        expected_mfe = 0.0
        expected_mae = 0.0
        direction_prob = max(long_probability, short_probability)
        target_rate = max(long_target_rate, short_target_rate)

    reliability = min(1.0, len(matches) / max(min_samples, 1))
    edge_quality = min(1.0, edge / max(target_r + stop_r, 1e-6) * 2.5)
    state_match = float(np.sum(weights * np.asarray([m["state"] == current_state for m in matches], dtype=float)))
    model_match = float(np.sum(weights * np.asarray([m["model"] == current_model for m in matches], dtype=float)))
    confidence = (
        0.28 * direction_prob
        + 0.20 * target_rate
        + 0.20 * mean_similarity
        + 0.14 * reliability
        + 0.10 * edge_quality
        + 0.05 * state_match
        + 0.03 * model_match
    )
    confidence = float(np.clip(confidence, 0.0, 1.0))
    if predicted_side == "WAIT":
        confidence *= 0.70
    ready = len(matches) >= min_samples
    decision = "CONFIRM" if ready and predicted_side in ["LONG", "SHORT"] else ("LEARNING" if not ready else "WAIT")
    reason = (
        f"מנוע הנדסי: {len(matches)} תבניות, דמיון {mean_similarity*100:.0f}%, "
        f"חיזוי {predicted_side}, ביטחון {confidence*100:.0f}%. "
        f"P(LONG) {long_probability*100:.0f}%, P(SHORT) {short_probability*100:.0f}%, "
        f"P(NEUTRAL) {neutral_probability*100:.0f}%. "
        f"תוחלת R לונג {long_expectancy:+.2f}, שורט {short_expectancy:+.2f}. "
        f"מצב {current_state}, מודל {current_model}."
    )
    if not ready:
        reason += f" עדיין בלמידה: נדרשות {min_samples} דוגמאות אמינות."

    return {
        "ready": bool(ready), "predicted_side": predicted_side, "decision": decision,
        "confidence": confidence, "sample_count": int(len(matches)),
        "mean_similarity": mean_similarity, "best_similarity": best_similarity,
        "weakest_similarity": weakest_similarity,
        "long_probability": long_probability, "short_probability": short_probability,
        "neutral_probability": neutral_probability,
        "long_target_rate": long_target_rate, "short_target_rate": short_target_rate,
        "long_expectancy_r": long_expectancy, "short_expectancy_r": short_expectancy,
        "expected_mfe_r": expected_mfe, "expected_mae_r": expected_mae,
        "pattern_state": current_state, "function_model": current_model,
        "reason": reason, "matches": matches[:8],
    }


@st.cache_data(show_spinner=False, ttl=60)
def cached_live_engineering_analysis(ticker, mode, bar_time, stop_r, target_r):
    full = fetch_1m(ticker)
    return engineering_pattern_analysis(
        full_df=full,
        mode=mode,
        current_end=bar_time,
        stop_r=float(stop_r),
        target_r=float(target_r),
    )


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

    return int(max(1, min(10, score))), reasons

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

    return int(max(1, min(10, score))), reasons


def make_signal(ticker, mode):
    ticker = normalize_ticker(ticker)
    full_df = fetch_1m(ticker)
    df = latest_session(full_df)
    if df.empty:
        return {"signal": "WAIT", "ticker": ticker, "mode": mode, "score": 0, "reason": "אין נתונים"}

    live_ok, live_reason = live_data_status(df, for_entry=True)
    if not live_ok:
        return {"signal": "WAIT", "ticker": ticker, "mode": mode, "score": 0, "reason": live_reason}

    d = add_indicators(df).dropna(subset=["close"])
    if d.empty:
        return {"signal": "WAIT", "ticker": ticker, "mode": mode, "score": 0, "reason": "אין אינדיקטורים"}

    if mode == "מהירה":
        ls, lr = score_side_fast(d, "LONG")
        ss, sr = score_side_fast(d, "SHORT")
        atr = safe_float(d.iloc[-1]["atr3"], safe_float(d.iloc[-1]["close"]) * 0.001)
    else:
        ls, lr = score_side_half(d, "LONG")
        ss, sr = score_side_half(d, "SHORT")
        atr = safe_float(d.iloc[-1]["atr14"], safe_float(d.iloc[-1]["close"]) * 0.002)

    rules = load_rules()
    min_gap = int(rules.get("min_direction_score_gap", 2))
    min_base = int(rules.get("min_base_score", 6))
    score_gap = abs(int(ls) - int(ss))

    technical_side = "WAIT"
    technical_score = max(ls, ss)
    technical_reasons = []
    if ls > ss and ls >= min_base and score_gap >= min_gap:
        technical_side, technical_score, technical_reasons = "LONG", ls, lr
    elif ss > ls and ss >= min_base and score_gap >= min_gap:
        technical_side, technical_score, technical_reasons = "SHORT", ss, sr

    # Prepare a provisional engineering analysis even when the classic score is not decisive.
    provisional_side = technical_side
    if provisional_side == "WAIT":
        provisional_side = "LONG" if ls > ss else ("SHORT" if ss > ls else "LONG")
    provisional_plan = chart_based_stop_target(d, provisional_side, mode)
    entry = safe_float(d.iloc[-1]["close"])
    atr = max(float(atr), entry * 0.0008)
    stop_r = abs(entry - safe_float(provisional_plan.get("stop"), entry)) / atr
    target_r = abs(safe_float(provisional_plan.get("target"), entry) - entry) / atr

    engineering = {
        "ready": False, "predicted_side": "WAIT", "confidence": 0.0,
        "sample_count": 0, "reason": "המנוע ההנדסי כבוי.",
    }
    if bool(rules.get("engineering_enabled", True)):
        engineering = cached_live_engineering_analysis(
            ticker=ticker,
            mode=mode,
            bar_time=str(d.index[-1]),
            stop_r=round(stop_r, 2),
            target_r=round(target_r, 2),
        )

    predicted_side = str(engineering.get("predicted_side", "WAIT"))
    confidence = safe_float(engineering.get("confidence"), 0.0)
    min_conf = float(rules.get("engineering_min_confidence", 0.58))
    strong_conf = float(rules.get("engineering_strong_confidence", 0.70))
    ready = bool(engineering.get("ready", False))
    allow_override = bool(rules.get("engineering_allow_strong_override", True))

    side = technical_side
    base_score = technical_score
    reasons = list(technical_reasons)

    # Strong pattern evidence may select a side when technical evidence is close but not contradictory.
    if side == "WAIT" and ready and allow_override and confidence >= strong_conf and predicted_side in ["LONG", "SHORT"]:
        predicted_technical_score = ls if predicted_side == "LONG" else ss
        opposite_score = ss if predicted_side == "LONG" else ls
        if predicted_technical_score >= max(4, min_base - 2) and predicted_technical_score >= opposite_score - 1:
            side = predicted_side
            base_score = int(predicted_technical_score)
            reasons = lr if side == "LONG" else sr
            reasons = list(reasons) + ["המנוע ההנדסי הכריע מצב טכני גבולי."]

    if side == "WAIT":
        log_engineering_prediction(ticker, mode, d.index[-1], technical_side, engineering, ls, ss, max(ls, ss), "WAIT_TECHNICAL")
        return {
            "signal": "WAIT", "ticker": ticker, "mode": mode, "score": max(ls, ss),
            "long_score": int(ls), "short_score": int(ss), "score_gap": int(score_gap),
            "engineering_confidence": confidence,
            "engineering_samples": int(engineering.get("sample_count", 0)),
            "engineering_side": predicted_side,
            "reason": f"לונג {ls}, שורט {ss}; נדרש בסיס {min_base} ופער {min_gap}. | {engineering.get('reason', '')}"
        }

    # When the engine is ready, clear opposition vetoes the setup.
    require_when_ready = bool(rules.get("engineering_require_when_ready", True))
    if ready and require_when_ready:
        if predicted_side == "WAIT":
            log_engineering_prediction(ticker, mode, d.index[-1], technical_side, engineering, ls, ss, base_score, "REJECT_NO_EDGE")
            return {
                "signal": "WAIT", "ticker": ticker, "mode": mode, "score": int(base_score),
                "long_score": int(ls), "short_score": int(ss), "score_gap": int(score_gap),
                "reason": f"המנוע ההנדסי לא מצא יתרון כיוון. | {engineering.get('reason', '')}"
            }
        if predicted_side in ["LONG", "SHORT"] and predicted_side != side and confidence >= min_conf:
            log_engineering_prediction(ticker, mode, d.index[-1], technical_side, engineering, ls, ss, base_score, "REJECT_OPPOSITE")
            return {
                "signal": "WAIT", "ticker": ticker, "mode": mode, "score": int(base_score),
                "long_score": int(ls), "short_score": int(ss), "score_gap": int(score_gap),
                "reason": f"המנוע ההנדסי חזה {predicted_side} בניגוד ל־{side}. | {engineering.get('reason', '')}"
            }
        if predicted_side == side and confidence < min_conf:
            log_engineering_prediction(ticker, mode, d.index[-1], technical_side, engineering, ls, ss, base_score, "REJECT_LOW_CONFIDENCE")
            return {
                "signal": "WAIT", "ticker": ticker, "mode": mode, "score": int(base_score),
                "long_score": int(ls), "short_score": int(ss), "score_gap": int(score_gap),
                "reason": f"התבנית תואמת לכיוון אך הביטחון {confidence*100:.0f}% נמוך מסף {min_conf*100:.0f}%."
            }

    tf5_score, tf5_reason = timeframe_alignment_score(df, side)
    min_tf5 = int(rules.get("min_5m_alignment_score", 2))
    if bool(rules.get("require_5m_alignment", True)) and tf5_score < min_tf5:
        log_engineering_prediction(ticker, mode, d.index[-1], technical_side, engineering, ls, ss, base_score, "REJECT_5M")
        return {
            "signal": "WAIT", "ticker": ticker, "mode": mode, "score": int(base_score),
            "long_score": int(ls), "short_score": int(ss), "score_gap": int(score_gap),
            "tf5_score": int(tf5_score),
            "reason": f"נפסל ב־5 דקות ({tf5_score}/4): {tf5_reason} | {engineering.get('reason', '')}"
        }

    market_ok, market_reason = market_context_check(ticker, side)
    if not market_ok:
        log_engineering_prediction(ticker, mode, d.index[-1], technical_side, engineering, ls, ss, base_score, "REJECT_MARKET")
        return {
            "signal": "WAIT", "ticker": ticker, "mode": mode, "score": int(base_score),
            "long_score": int(ls), "short_score": int(ss), "score_gap": int(score_gap),
            "tf5_score": int(tf5_score),
            "reason": f"נפסל לפי שוק: {market_reason} | {engineering.get('reason', '')}"
        }

    chart_plan = chart_based_stop_target(d, side, mode)
    stop = safe_float(chart_plan["stop"])
    target = safe_float(chart_plan["target"])
    tf_bonus = 1 if tf5_score >= 3 else 0

    engineering_bonus = 0
    if ready and predicted_side == side:
        if confidence >= strong_conf:
            engineering_bonus = 3
        elif confidence >= min_conf + 0.05:
            engineering_bonus = 2
        elif confidence >= min_conf:
            engineering_bonus = 1
    elif not ready:
        engineering_bonus = 0

    final_score = int(max(1, min(12, int(base_score) + tf_bonus + engineering_bonus)))
    last = d.iloc[-1]
    decision = "OPEN_CANDIDATE" if final_score > 0 else "WAIT"
    log_engineering_prediction(ticker, mode, d.index[-1], technical_side, engineering, ls, ss, final_score, decision)

    return {
        "signal": side,
        "ticker": ticker,
        "mode": mode,
        "score": final_score,
        "base_score": int(base_score),
        "long_score": int(ls),
        "short_score": int(ss),
        "score_gap": int(score_gap),
        "tf5_score": int(tf5_score),
        "engineering_confidence": float(confidence),
        "engineering_samples": int(engineering.get("sample_count", 0)),
        "engineering_side": predicted_side,
        "engineering_state": str(engineering.get("pattern_state", "UNKNOWN")),
        "engineering_model": str(engineering.get("function_model", "UNKNOWN")),
        "engineering_expected_mfe_r": safe_float(engineering.get("expected_mfe_r"), 0.0),
        "engineering_expected_mae_r": safe_float(engineering.get("expected_mae_r"), 0.0),
        "entry": float(entry),
        "stop": float(stop),
        "target": float(target),
        "atr": float(atr),
        "signal_high": float(last["high"]),
        "signal_low": float(last["low"]),
        "signal_bar_time": str(d.index[-1]),
        "last_rel_vol": float(safe_float(last.get("rel_vol5"), 0)),
        "reason": " | ".join(
            reasons + [
                live_reason,
                engineering.get("reason", ""),
                f"5 דקות {tf5_score}/4: {tf5_reason}",
                market_reason,
                chart_plan["reason"],
            ]
        ),
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


def has_any_open_trade_for_ticker(trades, ticker):
    """
    Conservative rule:
    Do not allow two open trades on the same ticker, even if one is 'מהירה'
    and the other is 'חצי שעה'. This prevents doubled risk on the same stock.
    """
    if trades.empty:
        return False
    return bool((trades["status"].eq("OPEN") & trades["ticker"].astype(str).eq(str(ticker))).any())


def apply_risk_cap_to_position(side, entry, stop, score_qty, score_notional, max_loss_dollars):
    """
    Cap position size by real dollar risk to stop.
    This prevents score 8 from creating a large loss when the stop is far.
    """
    risk_per_share = abs(float(entry) - float(stop))
    if risk_per_share <= 0:
        return 0.0, 0.0, "מרחק הסטופ לא תקין."

    qty_by_risk = float(max_loss_dollars) / risk_per_share
    qty = min(float(score_qty), float(qty_by_risk))
    notional = abs(qty * float(entry))

    if qty <= 0 or notional <= 0:
        return 0.0, 0.0, "גודל העסקה יצא 0 אחרי הגבלת סיכון."

    risk_dollars = qty * risk_per_share
    return float(qty), float(notional), f"גודל העסקה הוגבל לפי סיכון לסטופ: הפסד מקסימלי משוער ${risk_dollars:.2f}."



def signal_confirmed_after_delay(original_side, original_score, new_signal, min_score,
                                 original_entry=np.nan, original_stop=np.nan, original_target=np.nan,
                                 signal_high=np.nan, signal_low=np.nan):
    """Balanced second-stage confirmation using a new bar's high/low, not only its close."""
    rules = load_rules()
    new_side = str(new_signal.get("signal", "WAIT"))
    new_score = int(new_signal.get("score", 0))
    original_side = str(original_side)

    if new_side != original_side:
        return False, f"הכיוון השתנה מ־{original_side} ל־{new_side}."

    # Allow a small one-minute score fluctuation while keeping the setup strong.
    required_score = max(6, int(min_score) - 1, int(original_score) - 2)
    if new_score < required_score:
        return False, f"הניקוד ירד מ־{original_score} ל־{new_score}; נדרש לפחות {required_score}."

    price = safe_float(new_signal.get("entry"), np.nan)
    entry = safe_float(original_entry, price)
    stop = safe_float(original_stop, np.nan)
    target = safe_float(original_target, np.nan)
    high = safe_float(signal_high, np.nan)
    low = safe_float(signal_low, np.nan)
    atr = safe_float(new_signal.get("atr"), abs(entry - stop) if np.isfinite(entry) and np.isfinite(stop) else 0.0)
    new_high = safe_float(new_signal.get("signal_high"), price)
    new_low = safe_float(new_signal.get("signal_low"), price)

    tolerance_atr = float(rules.get("confirmation_breakout_tolerance_atr", 0.10))
    tolerance = max(abs(atr) * tolerance_atr, abs(entry) * 0.0001 if np.isfinite(entry) else 0.0)
    risk = abs(entry - stop) if np.isfinite(entry) and np.isfinite(stop) else max(abs(atr), 0.0)

    if original_side == "LONG" and np.isfinite(high):
        touched_breakout = np.isfinite(new_high) and new_high >= high - tolerance
        holding_direction = np.isfinite(price) and price >= entry - risk * 0.05
        if not (touched_breakout and holding_direction):
            return False, (
                f"אין אישור לונג: שיא חדש {new_high:.2f} מול רמת {high:.2f}, "
                f"מחיר נוכחי {price:.2f}."
            )

    if original_side == "SHORT" and np.isfinite(low):
        touched_breakdown = np.isfinite(new_low) and new_low <= low + tolerance
        holding_direction = np.isfinite(price) and price <= entry + risk * 0.05
        if not (touched_breakdown and holding_direction):
            return False, (
                f"אין אישור שורט: שפל חדש {new_low:.2f} מול רמת {low:.2f}, "
                f"מחיר נוכחי {price:.2f}."
            )

    max_adverse_r = float(rules.get("max_adverse_move_r_before_entry", 0.35))
    if risk > 0 and np.isfinite(price):
        if original_side == "LONG" and price < entry - risk * max_adverse_r:
            return False, "המחיר זז נגד הלונג בזמן ההמתנה."
        if original_side == "SHORT" and price > entry + risk * max_adverse_r:
            return False, "המחיר זז נגד השורט בזמן ההמתנה."

    if np.isfinite(entry) and np.isfinite(target) and np.isfinite(price):
        full_move = abs(target - entry)
        progress = ((price - entry) if original_side == "LONG" else (entry - price))
        progress_pct = (progress / full_move) * 100 if full_move > 0 else 0
        max_progress = float(rules.get("max_target_progress_before_entry_pct", 55.0))
        if progress_pct > max_progress:
            return False, f"המחיר כבר עבר {progress_pct:.0f}% מהדרך ליעד; לא רודפים אחרי העסקה."

    rel_vol = safe_float(new_signal.get("last_rel_vol"), np.nan)
    min_rel_vol = float(rules.get("min_confirmation_rel_volume", 0.50))
    # Missing/partial volume should not block a very strong setup; clearly weak volume still does.
    if np.isfinite(rel_vol) and rel_vol < min_rel_vol and new_score < max(9, original_score):
        return False, f"ווליום האישור חלש ({rel_vol:.2f}); נדרש {min_rel_vol:.2f} או ניקוד חזק במיוחד."

    return True, "הכיוון, הניקוד והמשך המחיר אושרו בנר חדש."


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

def open_trade(signal, min_score, send_alert=True):
    trades = load_trades()
    costs = load_costs()
    units = load_units()
    rules = load_rules()

    ticker = normalize_ticker(signal["ticker"])
    mode = str(signal["mode"])
    side = str(signal["signal"])
    score = int(signal["score"])

    risk_ok, risk_msg = daily_risk_gate(trades)
    if not risk_ok:
        return False, risk_msg
    if side not in ["LONG", "SHORT"]:
        return False, f"{ticker}: אין איתות."
    if score < int(min_score):
        return False, f"{ticker}: ניקוד {score} נמוך מהמינימום {min_score}."
    if has_any_open_trade_for_ticker(trades, ticker):
        return False, f"{ticker}: כבר יש עסקה פתוחה על המניה."
    exposure_ok, exposure_msg = exposure_gate(trades, ticker, side, include_pending=False)
    if not exposure_ok:
        return False, f"{ticker}: {exposure_msg}"

    cd, msg = in_cooldown(trades, ticker, mode, rules)
    if cd:
        return False, f"{ticker}: {msg}"

    entry = float(signal["entry"])
    stop = float(signal["stop"])
    target = float(signal["target"])
    score_qty, score_notional, unit_mult = position_size(score, entry, units)
    if score_qty <= 0 or score_notional <= 0:
        return False, f"{ticker}: ניקוד {score} אינו מקבל גודל עסקה."

    qty, notional, risk_size_msg = apply_risk_cap_to_position(
        side=side, entry=entry, stop=stop, score_qty=score_qty, score_notional=score_notional,
        max_loss_dollars=float(rules.get("max_allowed_loss_per_trade_dollars", 7.0)),
    )
    if qty <= 0 or notional <= 0:
        return False, f"{ticker}: {risk_size_msg}"

    ok, eg, ec, en, cost_msg = cost_tradeoff(side, entry, target, qty, costs)
    if not ok:
        return False, f"{ticker}: {cost_msg} ברוטו ${eg:.2f}, עלות ${ec:.2f}, נטו ${en:.2f}."

    entry_cost, exit_cost, total_cost_now = estimate_costs(entry, entry, qty, costs)
    row = {
        "trade_id": str(uuid.uuid4()), "status": "OPEN", "ticker": ticker, "mode": mode,
        "side": side, "score": score, "entry_time": now_ny_iso(), "exit_time": "",
        "duration_minutes": 0.0, "age_minutes": 0.0, "entry_price": entry,
        "current_price": entry, "exit_price": np.nan, "quantity": qty, "notional": notional,
        "stop_loss": stop, "initial_stop_loss": stop, "manual_stop_loss": np.nan,
        "profit_stop": np.nan, "target_reference": target, "breakeven_price": np.nan,
        "highest_price": entry, "lowest_price": entry, "max_net_pnl_seen": -total_cost_now,
        "entry_cost": entry_cost, "exit_cost": exit_cost, "total_cost": total_cost_now,
        "gross_pnl": 0.0, "net_pnl": -total_cost_now,
        "net_pnl_pct": (-total_cost_now / notional) * 100 if notional else 0,
        "exit_reason": "", "exit_reason_he": "", "management_action": "OPENED",
        "management_reason": f"נפתחה אחרי דקה + 5 דקות + QQQ. {exposure_msg}",
        "signal_reason": signal.get("reason", ""),
        "cost_pct_per_side": costs["cost_pct_per_side"], "fixed_fee_per_side": costs["fixed_fee_per_side"],
        "min_fee_per_side": costs["min_fee_per_side"], "max_cost_to_target_pct": costs["max_cost_to_target_pct"],
        "base_unit_dollars": units["base_unit_dollars"], "unit_multiplier": unit_mult,
        "created_settings_snapshot": json.dumps({"costs": costs, "units": units, "rules": rules}, ensure_ascii=False),
    }
    row["breakeven_price"] = breakeven_after_costs(row)
    trades = pd.concat([trades, pd.DataFrame([row])], ignore_index=True)
    save_trades(trades)

    if send_alert:
        telegram_sent, telegram_error = create_trade_alert(row, expected_net=en, risk_note=risk_size_msg)
        alert_note = "נשלחה התראה" if telegram_sent else "ההתראה נשמרה"
        if telegram_error and "כבויות" not in telegram_error:
            alert_note += f"; Telegram: {telegram_error}"
    else:
        telegram_sent, telegram_error = False, ""
        alert_note = "Manual Telegram mode"

    return True, f"{ticker}: נפתחה {side} | {mode} | ציון {score}/12 | נטו צפוי ${en:.2f} | {alert_note}."

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
    """Manage one trade using completed 1-minute OHLC bars.

    Execution order is deliberately conservative:
    1) hard stop, 2) an already-active profit stop, 3) fixed target.
    A new trailing stop calculated from the current candle becomes active only
    for the next candle, preventing same-candle look-ahead.
    """
    rules = load_rules()
    side = str(row["side"])
    mode = str(row["mode"])
    score = int(safe_float(row["score"], 1))
    entry = safe_float(row["entry_price"])
    stop = safe_float(row["stop_loss"])
    initial_stop = safe_float(row["initial_stop_loss"], stop)
    target = safe_float(row["target_reference"])
    old_profit_stop = safe_float(row.get("profit_stop"), np.nan)
    age = float(max(0, len(df_after_entry) - 1)) if df_after_entry is not None else 0.0
    min_hold = min_hold_for_mode(mode, rules)
    current_max_net_seen = safe_float(row.get("max_net_pnl_seen"), safe_float(row.get("net_pnl"), 0))

    res = {
        "exit": False, "exit_reason": "", "exit_price": np.nan,
        "stop_loss": stop, "profit_stop": old_profit_stop,
        "target_reference": target, "highest_price": safe_float(row.get("highest_price"), entry),
        "lowest_price": safe_float(row.get("lowest_price"), entry), "max_net_pnl_seen": current_max_net_seen,
        "action": "HOLD", "reason": "מחזיק; אין שינוי.",
    }
    if df_after_entry is None or df_after_entry.empty:
        return res
    d = add_indicators(df_after_entry).dropna(subset=["close"])
    if d.empty:
        return res

    last = d.iloc[-1]
    current = safe_float(last["close"])
    bar_high = safe_float(last["high"], current)
    bar_low = safe_float(last["low"], current)
    high_since = max(res["highest_price"], safe_float(d["high"].max(), current))
    low_since = min(res["lowest_price"], safe_float(d["low"].min(), current))
    res["highest_price"] = high_since
    res["lowest_price"] = low_since
    current_net = pnl_for_trade(row, current)["net_pnl"]
    res["max_net_pnl_seen"] = max(current_max_net_seen, current_net)

    base_risk = abs(entry - initial_stop)
    if base_risk <= 0:
        base_risk = max(entry * 0.001, abs(entry - stop))
    breakeven = safe_float(row.get("breakeven_price"), breakeven_after_costs(row))
    last3 = d.tail(min(3, len(d)))
    green = int((last3["close"] > last3["open"]).sum())
    red = int((last3["close"] < last3["open"]).sum())
    ema5 = safe_float(last["ema5"], current)
    ema5_slope = safe_float(last["ema5_slope"], 0)
    ema5_curv = safe_float(last["ema5_curv"], 0)
    macd_slope = safe_float(last["macd_hist_slope"], 0)

    # Intrabar execution. Hard stop has priority if stop and target both appear
    # inside the same 1-minute candle because the path inside the candle is unknown.
    if side == "LONG" and bar_low <= stop:
        res.update(exit=True, exit_reason="STOP_LOSS", exit_price=stop, action="EXIT_STOP", reason="שפל הנר פגע בסטופ.")
        return res
    if side == "SHORT" and bar_high >= stop:
        res.update(exit=True, exit_reason="STOP_LOSS", exit_price=stop, action="EXIT_STOP", reason="שיא הנר פגע בסטופ.")
        return res

    # Only a profit stop that existed before this candle can execute on it.
    if np.isfinite(old_profit_stop):
        if side == "LONG" and bar_low <= old_profit_stop:
            res.update(exit=True, exit_reason="PROFIT_STOP", exit_price=old_profit_stop, action="EXIT_PROFIT_STOP", reason="שפל הנר פגע בסטופ הרווח הפעיל.")
            return res
        if side == "SHORT" and bar_high >= old_profit_stop:
            res.update(exit=True, exit_reason="PROFIT_STOP", exit_price=old_profit_stop, action="EXIT_PROFIT_STOP", reason="שיא הנר פגע בסטופ הרווח הפעיל.")
            return res

    # Fixed target always closes, even before the minimum holding time.
    if side == "LONG" and bar_high >= target:
        res.update(exit=True, exit_reason="TARGET_REACHED", exit_price=target, action="EXIT_TARGET", reason="שיא הנר הגיע ליעד.")
        return res
    if side == "SHORT" and bar_low <= target:
        res.update(exit=True, exit_reason="TARGET_REACHED", exit_price=target, action="EXIT_TARGET", reason="שפל הנר הגיע ליעד.")
        return res

    if current_net <= -abs(float(rules.get("max_allowed_loss_per_trade_dollars", 7.0))):
        res.update(exit=True, exit_reason="MAX_LOSS_LIMIT", exit_price=current, action="EXIT_MAX_LOSS", reason="הפסד נטו הגיע למגבלה.")
        return res

    peak_profit = float(res["max_net_pnl_seen"])
    giveback_pct = float(rules.get("profit_giveback_pct", 10.0))
    min_giveback = float(rules.get("min_net_profit_for_giveback", 5.0))
    if peak_profit >= min_giveback and current_net <= peak_profit * (1 - giveback_pct / 100.0):
        res.update(exit=True, exit_reason="PROFIT_GIVEBACK", exit_price=current, action="EXIT_PROFIT_GIVEBACK",
                   reason=f"הרווח ירד ביותר מ־{giveback_pct:.0f}% מהשיא ${peak_profit:.2f}.")
        return res

    if bool(rules.get("exit_if_profitable_trade_turns_red", True)) and age >= float(rules.get("emergency_exit_after_minutes", 2)):
        protected_profit = float(rules.get("breakeven_after_profit_dollars", 4.0))
        if peak_profit >= protected_profit:
            if side == "LONG" and current <= breakeven:
                res.update(exit=True, exit_reason="BREAKEVEN_AFTER_COSTS", exit_price=current, action="EXIT_BREAKEVEN", reason="עסקה שהייתה ברווח חזרה לאיזון אחרי עלויות.")
                return res
            if side == "SHORT" and current >= breakeven:
                res.update(exit=True, exit_reason="BREAKEVEN_AFTER_COSTS", exit_price=current, action="EXIT_BREAKEVEN", reason="עסקה שהייתה ברווח חזרה לאיזון אחרי עלויות.")
                return res

    if score >= 10:
        trail_r = 0.70
    elif score >= 9:
        trail_r = 0.55
    elif score >= 8:
        trail_r = 0.40
    else:
        trail_r = 0.30

    if side == "LONG":
        r_now = (current - entry) / base_risk
        best_r = (high_since - entry) / base_risk
        if age >= min_hold:
            new_stop = old_profit_stop
            if current_net >= float(rules.get("breakeven_after_profit_dollars", 4.0)):
                candidate = max(breakeven, current - 0.35 * base_risk)
                new_stop = candidate if not np.isfinite(new_stop) else max(new_stop, candidate)
            if best_r >= float(rules.get("min_profit_r_for_profit_stop", 0.45)):
                candidate = max(breakeven, high_since - trail_r * base_risk)
                new_stop = candidate if not np.isfinite(new_stop) else max(new_stop, candidate)
            if peak_profit >= float(rules.get("lock_profit_after_net_dollars", 8.0)) and (red >= 2 or ema5_curv < 0 or macd_slope < 0):
                candidate = max(breakeven, current - 0.18 * base_risk)
                new_stop = candidate if not np.isfinite(new_stop) else max(new_stop, candidate)
            res["profit_stop"] = new_stop
        if mode == "מהירה" and age >= 3 and current_net <= 0 and red >= 2 and current < entry:
            res.update(exit=True, exit_reason="NO_PROGRESS_FAST", exit_price=current, action="EXIT_NO_PROGRESS", reason="אין התקדמות אחרי 2–3 נרות.")
            return res
        if mode != "מהירה" and age >= 12 and current_net <= 0 and current < ema5 and ema5_slope < 0:
            res.update(exit=True, exit_reason="NO_PROGRESS_HALF", exit_price=current, action="EXIT_NO_PROGRESS", reason="עסקת חצי שעה לא התקדמה והמומנטום נחלש.")
            return res
        if age >= float(rules.get("emergency_exit_after_minutes", 2)) and r_now < -0.25 and red >= 2 and current < ema5 and ema5_slope < 0:
            res.update(exit=True, exit_reason="EARLY_EXIT_AGAINST_LONG", exit_price=current, action="EARLY_EXIT", reason="הלונג נע חזק נגד הכיוון.")
            return res
    else:
        r_now = (entry - current) / base_risk
        best_r = (entry - low_since) / base_risk
        if age >= min_hold:
            new_stop = old_profit_stop
            if current_net >= float(rules.get("breakeven_after_profit_dollars", 4.0)):
                candidate = min(breakeven, current + 0.35 * base_risk)
                new_stop = candidate if not np.isfinite(new_stop) else min(new_stop, candidate)
            if best_r >= float(rules.get("min_profit_r_for_profit_stop", 0.45)):
                candidate = min(breakeven, low_since + trail_r * base_risk)
                new_stop = candidate if not np.isfinite(new_stop) else min(new_stop, candidate)
            if peak_profit >= float(rules.get("lock_profit_after_net_dollars", 8.0)) and (green >= 2 or ema5_curv > 0 or macd_slope > 0):
                candidate = min(breakeven, current + 0.18 * base_risk)
                new_stop = candidate if not np.isfinite(new_stop) else min(new_stop, candidate)
            res["profit_stop"] = new_stop
        if mode == "מהירה" and age >= 3 and current_net <= 0 and green >= 2 and current > entry:
            res.update(exit=True, exit_reason="NO_PROGRESS_FAST", exit_price=current, action="EXIT_NO_PROGRESS", reason="אין התקדמות אחרי 2–3 נרות.")
            return res
        if mode != "מהירה" and age >= 12 and current_net <= 0 and current > ema5 and ema5_slope > 0:
            res.update(exit=True, exit_reason="NO_PROGRESS_HALF", exit_price=current, action="EXIT_NO_PROGRESS", reason="עסקת חצי שעה לא התקדמה והמומנטום נחלש.")
            return res
        if age >= float(rules.get("emergency_exit_after_minutes", 2)) and r_now < -0.25 and green >= 2 and current > ema5 and ema5_slope > 0:
            res.update(exit=True, exit_reason="EARLY_EXIT_AGAINST_SHORT", exit_price=current, action="EARLY_EXIT", reason="השורט נע חזק נגד הכיוון.")
            return res

    if np.isfinite(res["profit_stop"]):
        res["action"] = "UPDATE_PROFIT_STOP"
        res["reason"] = f"סטופ רווח עודכן ל־{res['profit_stop']:.2f}; הוא פעיל מהנר הבא."
    return res

def close_trade_at_index(trades, idx, current, reason, exit_time_override=None):
    trades = normalize_trade_dtypes(trades)

    # Make sure time/text columns can receive strings.
    for _col in ["exit_time", "status", "exit_reason", "exit_reason_he", "management_action", "management_reason"]:
        if _col in trades.columns:
            trades[_col] = trades[_col].astype("object")

    pnl = pnl_for_trade(trades.loc[idx], current)
    for k, v in pnl.items():
        trades.loc[idx, k] = v

    exit_time = str(exit_time_override) if exit_time_override is not None else now_ny_iso()
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
    trades = normalize_trade_dtypes(load_trades())
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
    """Update only from fresh same-day bars; never act on stale Friday data on Monday."""
    trades = normalize_trade_dtypes(load_trades())
    messages = []
    if trades.empty:
        return trades, messages

    rules = load_rules()
    flat_h, flat_m = parse_hhmm(rules.get("force_flat_time", "15:55"), "15:55")
    flat_minute = flat_h * 60 + flat_m

    open_idx = trades.index[trades["status"].eq("OPEN")].tolist()
    for idx in open_idx:
        ticker = str(trades.loc[idx, "ticker"])
        try:
            all_df = fetch_1m(ticker)
            if all_df is None or all_df.empty:
                trades.loc[idx, "management_action"] = "WAIT_DATA"
                trades.loc[idx, "management_reason"] = "אין נתוני 1 דקה; לא בוצעה יציאה."
                continue

            all_df = all_df.sort_index()
            session_df = latest_session(all_df)
            entry_time = timestamp_to_ny(trades.loc[idx, "entry_time"])
            latest_bar_time = timestamp_to_ny(all_df.index[-1])

            if entry_time is not None and latest_bar_time is not None and entry_time.date() < latest_bar_time.date():
                entry_day = all_df[all_df.index.date == entry_time.date()]
                if not entry_day.empty:
                    exit_bar_time = entry_day.index[-1]
                    exit_price = safe_float(entry_day.iloc[-1]["close"], safe_float(trades.loc[idx, "current_price"]))
                    trades, pnl = close_trade_at_index(trades, idx, exit_price, "OVERNIGHT_SAFETY_CLOSE", exit_time_override=exit_bar_time.isoformat())
                    trades.loc[idx, "management_action"] = "OVERNIGHT_SAFETY_CLOSE"
                    trades.loc[idx, "management_reason"] = "עסקת יום נסגרה במחיר הנר האחרון של יום הכניסה."
                    messages.append(f"{ticker}: נסגרה עסקה ישנה במחיר סוף יום הכניסה | נטו ${pnl['net_pnl']:.2f}")
                continue

            live_ok, live_reason = live_data_status(session_df, for_entry=False)
            if not live_ok:
                trades.loc[idx, "management_action"] = "WAIT_FRESH_DATA"
                trades.loc[idx, "management_reason"] = f"לא מנהלים לפי מידע ישן: {live_reason}"
                continue

            current = safe_float(session_df.iloc[-1]["close"])
            if entry_time is None:
                after_entry = session_df.tail(5)
            else:
                after_entry = session_df[session_df.index >= entry_time]
                if after_entry.empty:
                    trades.loc[idx, "management_action"] = "WAIT_ENTRY_BAR"
                    trades.loc[idx, "management_reason"] = "ממתינים לנר עדכני אחרי זמן הכניסה."
                    continue

            if minute_of_day(session_df.index[-1]) >= flat_minute:
                trades, pnl = close_trade_at_index(trades, idx, current, "END_OF_DAY_SAFETY", exit_time_override=session_df.index[-1].isoformat())
                trades.loc[idx, "management_action"] = "END_OF_DAY_SAFETY"
                trades.loc[idx, "management_reason"] = f"סגירת בטיחות בשעה {flat_h:02d}:{flat_m:02d} ניו־יורק."
                messages.append(f"{ticker}: נסגרה בסוף יום | נטו ${pnl['net_pnl']:.2f}")
                continue

            decision = manage_trade(trades.loc[idx], after_entry)
            bar_age = float(max(0, len(after_entry) - 1))
            trades.loc[idx, "age_minutes"] = bar_age
            trades.loc[idx, "duration_minutes"] = bar_age
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
                exit_fill = safe_float(decision.get("exit_price"), current)
                trades, pnl = close_trade_at_index(trades, idx, exit_fill, decision["exit_reason"], exit_time_override=session_df.index[-1].isoformat())
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
    messages = []
    rules = load_rules()
    trades = load_trades()
    risk_ok, risk_msg = daily_risk_gate(trades)
    if not risk_ok:
        return [risk_msg]

    max_new = int(max_new_override) if max_new_override is not None else int(rules.get("max_new_trades_per_scan", 2))
    max_open = int(max_open_override) if max_open_override is not None else int(rules.get("max_open_trades", 5))
    current_open = int(trades["status"].eq("OPEN").sum()) if not trades.empty else 0
    active_pending = load_pending()
    pending_count = int(active_pending["status"].astype(str).eq("PENDING").sum()) if not active_pending.empty else 0
    available_slots = max(0, max_open - current_open - pending_count)
    if available_slots <= 0:
        return [f"אין מקום: {current_open} פתוחות + {pending_count} ממתינות, מקסימום {max_open}."]

    candidates = []
    rejection_counts = Counter()
    rejection_examples = defaultdict(list)

    def rejection_category(reason):
        r = str(reason)
        if "ישן" in r or "לא מהיום" in r or "אין נתוני" in r:
            return "נתונים חסרים/מעוכבים"
        if "כניסות חדשות מותרות" in r or "סוף שבוע" in r:
            return "מחוץ לשעות הכניסה"
        if "נפסל ב־5 דקות" in r or "5 דקות אינן" in r:
            return "אי־התאמה ב־5 דקות"
        if "מנוע ההנדסי" in r or "מנוע הנדסי" in r or "ביטחון הנדסי" in r or "התבנית" in r:
            return "מנוע תבניות הנדסי"
        if "נפסל לפי שוק" in r or "כיוון ההפוך" in r:
            return "QQQ בכיוון מנוגד"
        if "נדרש בסיס" in r or "נדרש" in r and "פער" in r:
            return "ניקוד בסיס/פער כיוונים"
        return "סינון אחר"

    for ticker in tickers:
        for mode in modes:
            try:
                sig = make_signal(ticker, mode)
                if sig.get("signal") not in ["LONG", "SHORT"]:
                    reason = str(sig.get("reason", "אין איתות"))
                    category = rejection_category(reason)
                    rejection_counts[category] += 1
                    if len(rejection_examples[category]) < 4:
                        rejection_examples[category].append(f"{ticker}: {reason}")
                    continue
                mode_floor = int(rules.get("min_score_fast", 10)) if str(mode) == "מהירה" else int(rules.get("min_score_half", 7))
                effective_min_score = max(int(min_score), mode_floor)
                if int(sig.get("score", 0)) < effective_min_score:
                    category = "ציון סופי נמוך"
                    rejection_counts[category] += 1
                    if len(rejection_examples[category]) < 4:
                        rejection_examples[category].append(
                            f"{ticker} ({mode}): ציון {sig.get('score', 0)} מתחת לסף {effective_min_score}"
                        )
                    continue
                sig["required_min_score"] = int(effective_min_score)

                expected_move_pct = abs(float(sig["target"]) - float(sig["entry"])) / float(sig["entry"]) * 100
                candidates.append((
                    int(sig["score"]),
                    float(sig.get("engineering_confidence", 0.0)),
                    int(sig.get("score_gap", 0)),
                    int(sig.get("tf5_score", 0)),
                    expected_move_pct,
                    sig,
                ))
            except Exception as exc:
                category = "שגיאת סריקה"
                rejection_counts[category] += 1
                if len(rejection_examples[category]) < 4:
                    rejection_examples[category].append(f"{ticker}: {str(exc)[:120]}")
            time.sleep(0.03)

    if not candidates:
        messages.append("לא נמצאה כרגע עסקה שעברה את כל הסינון.")
        if rejection_counts:
            summary = " | ".join(f"{name}: {count}" for name, count in rejection_counts.most_common())
            messages.append(f"אבחון פסילות — {summary}")
            for category, _ in rejection_counts.most_common(4):
                for example in rejection_examples[category]:
                    messages.append(f"[{category}] {example}")
        return messages

    candidates.sort(key=lambda x: (x[0], x[1], x[2], x[3], x[4]), reverse=True)
    saved = 0
    for _, _, _, _, _, sig in candidates:
        if saved >= min(max_new, available_slots):
            break
        ok, msg = add_pending_signal(sig)
        messages.append(msg)
        if ok:
            saved += 1

    if saved:
        messages.append(f"נשמרו {saved} מועמדות לאישור בנר הבא מתוך {len(candidates)} שעברו סינון.")
    elif candidates:
        messages.append(f"נמצאו {len(candidates)} מועמדות, אך מגבלות החשיפה/המתנה מנעו שמירה חדשה.")
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

def fmt_order_qty(qty):
    q = safe_float(qty, np.nan)
    if np.isnan(q) or q <= 0:
        return ""
    if abs(q - round(q)) < 1e-6:
        return str(int(round(q)))
    return f"{q:.4f}".rstrip("0").rstrip(".")


def fmt_order_price(x):
    v = safe_float(x, np.nan)
    if np.isnan(v):
        return ""
    return f"{v:.2f}"


def order_ticket_values(row):
    ticker = str(row.get("ticker", "")).upper()
    side = str(row.get("side", "")).upper()
    stop = fmt_order_price(row.get("stop_loss"))
    target = fmt_order_price(row.get("target_reference"))
    qty = fmt_order_qty(row.get("quantity"))
    entry = fmt_order_price(row.get("entry_price"))

    full_ticket = (
        f"Ticker: {ticker}\n"
        f"Side: {side}\n"
        f"Entry: {entry}\n"
        f"Stop Loss: {stop}\n"
        f"Take Profit: {target}\n"
        f"Quantity: {qty}"
    )

    return {
        "ticker": ticker,
        "side": side,
        "entry": entry,
        "stop": stop,
        "target": target,
        "qty": qty,
        "full_ticket": full_ticket,
    }


def copy_button_component(label, value, button_key):
    """
    Clipboard button rendered with a small HTML component.
    Works best on localhost/HTTPS because browsers restrict clipboard access.
    """
    value_json = json.dumps(str(value))
    label_html = html.escape(str(label))
    button_id = f"copy_btn_{button_key}"
    status_id = f"copy_status_{button_key}"

    components.html(
        f"""
        <div style="direction:ltr;text-align:left;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Arial,sans-serif;">
          <button id="{button_id}" onclick='copyValue_{button_key}()'
            style="
              width:100%;
              border:1px solid #d1d5db;
              border-radius:10px;
              padding:9px 10px;
              background:#f9fafb;
              color:#111827;
              cursor:pointer;
              font-weight:600;
            ">
            {label_html}
          </button>
          <div id="{status_id}" style="font-size:11px;color:#16a34a;height:16px;margin-top:3px;"></div>
        </div>
        <script>
          function fallbackCopy_{button_key}(text) {{
            const ta = document.createElement('textarea');
            ta.value = text;
            ta.style.position = 'fixed';
            ta.style.left = '-9999px';
            document.body.appendChild(ta);
            ta.focus();
            ta.select();
            try {{
              document.execCommand('copy');
              document.getElementById('{status_id}').innerText = 'Copied';
            }} catch (err) {{
              document.getElementById('{status_id}').innerText = 'Copy failed';
            }}
            document.body.removeChild(ta);
          }}

          function copyValue_{button_key}() {{
            const text = {value_json};
            if (navigator.clipboard && window.isSecureContext) {{
              navigator.clipboard.writeText(text).then(function() {{
                document.getElementById('{status_id}').innerText = 'Copied';
              }}).catch(function() {{
                fallbackCopy_{button_key}(text);
              }});
            }} else {{
              fallbackCopy_{button_key}(text);
            }}
          }}
        </script>
        """,
        height=62,
    )


def render_order_ticket_card(row, card_idx):
    vals = order_ticket_values(row)

    side = vals["side"]
    direction = "🟢 LONG 📈" if side == "LONG" else "🔴 SHORT 📉" if side == "SHORT" else "⚪ SIGNAL"

    st.markdown(
        f"""
        <div style="
            direction:ltr;
            text-align:left;
            border:1px solid #e5e7eb;
            border-radius:18px;
            padding:16px;
            background:#ffffff;
            box-shadow:0 6px 14px rgba(0,0,0,.05);
            margin:10px 0;
            font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Arial,sans-serif;
        ">
            <div style="font-size:20px;font-weight:800;margin-bottom:8px;">{html.escape(direction)}</div>
            <div style="font-size:17px;line-height:1.7;">
                <strong>Ticker:</strong> {html.escape(vals["ticker"])}<br>
                <strong>Side:</strong> {html.escape(vals["side"])}<br>
                <strong>Stop Loss:</strong> {html.escape(vals["stop"])}<br>
                <strong>Take Profit:</strong> {html.escape(vals["target"])}<br>
                <strong>Quantity:</strong> {html.escape(vals["qty"])}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        copy_button_component("Copy Ticker", vals["ticker"], f"{card_idx}_ticker")
    with c2:
        copy_button_component("Copy Stop Loss", vals["stop"], f"{card_idx}_stop")
    with c3:
        copy_button_component("Copy Take Profit", vals["target"], f"{card_idx}_target")
    with c4:
        copy_button_component("Copy Quantity", vals["qty"], f"{card_idx}_qty")
    with c5:
        copy_button_component("Copy Full Ticket", vals["full_ticket"], f"{card_idx}_full")


def render_order_ticket_helper():
    st.markdown("### 🧾 Order Ticket Helper")
    st.markdown(
        "<div class='card warn'><strong>Manual only:</strong> Copy the values into TradingView yourself. "
        "This app does not send real orders and does not click Buy/Sell automatically.</div>",
        unsafe_allow_html=True,
    )

    trades = load_trades()
    if trades.empty:
        st.info("No trades yet.")
        return

    open_trades = trades[trades["status"].astype(str).eq("OPEN")].copy()
    if open_trades.empty:
        st.info("No open Paper trades right now.")
        st.markdown("#### Recent tickets")
        recent = trades.sort_values("entry_time", ascending=False).head(10).copy()
        if recent.empty:
            return
        for i, (_, row) in enumerate(recent.iterrows()):
            render_order_ticket_card(row, f"recent_{i}")
        return

    st.markdown("#### Open Paper trades")
    open_trades = open_trades.sort_values("entry_time", ascending=False)
    for i, (_, row) in enumerate(open_trades.iterrows()):
        render_order_ticket_card(row, f"open_{i}")


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
        "יעד": d["target_reference"].map(fmt_price),
        "שיא רווח נטו": d["max_net_pnl_seen"].map(fmt_money),
        "רווח ברוטו": d["gross_pnl"].map(fmt_money),
        "עלויות": d["total_cost"].map(fmt_money),
        "רווח/הפסד": d["net_pnl"].map(fmt_money),
        "זמן כניסה": d["entry_time"].astype(str).str.slice(0, 19),
        "זמן יציאה": d["exit_time"].astype(str).str.slice(0, 19),
        "משך עסקה בדק׳": d["duration_minutes"].map(fmt_minutes),
        "ניקוד": d["score"].fillna(0).astype(int),
        "סיבה ליציאה": d["exit_reason_he"],
        "הסבר ניהול": d["management_reason"],
        "הסבר כניסה": d["signal_reason"],
    })

    pnl_values = d["net_pnl"].fillna(0).astype(float).tolist()

    def style_row(row):
        pnl = pnl_values[row.name]
        if pnl >= 0:
            return ["background-color:#dcfce7;color:#064e3b;"] * len(row)
        return ["background-color:#fee2e2;color:#7f1d1d;"] * len(row)

    st.dataframe(display.style.apply(style_row, axis=1), use_container_width=True, hide_index=True)




# ============================================================
# Backtest / historical replay
# ============================================================


def make_signal_from_history(ticker, mode, hist_df, market_hist_df=None, engineering_full_df=None):
    if hist_df is None or hist_df.empty:
        return {"signal": "WAIT", "ticker": ticker, "mode": mode, "score": 0, "reason": "אין נתונים"}
    d = add_indicators(hist_df).dropna(subset=["close"])
    if d.empty:
        return {"signal": "WAIT", "ticker": ticker, "mode": mode, "score": 0, "reason": "אין אינדיקטורים"}

    if mode == "מהירה":
        ls, lr = score_side_fast(d, "LONG")
        ss, sr = score_side_fast(d, "SHORT")
        atr = safe_float(d.iloc[-1]["atr3"], safe_float(d.iloc[-1]["close"]) * 0.001)
    else:
        ls, lr = score_side_half(d, "LONG")
        ss, sr = score_side_half(d, "SHORT")
        atr = safe_float(d.iloc[-1]["atr14"], safe_float(d.iloc[-1]["close"]) * 0.002)

    rules = load_rules()
    gap = abs(ls - ss)
    min_gap = int(rules.get("min_direction_score_gap", 2))
    min_base = int(rules.get("min_base_score", 6))
    technical_side = "WAIT"
    base = max(ls, ss)
    reasons = []
    if ls > ss and ls >= min_base and gap >= min_gap:
        technical_side, base, reasons = "LONG", ls, lr
    elif ss > ls and ss >= min_base and gap >= min_gap:
        technical_side, base, reasons = "SHORT", ss, sr

    provisional_side = technical_side if technical_side != "WAIT" else ("LONG" if ls >= ss else "SHORT")
    provisional_plan = chart_based_stop_target(d, provisional_side, mode)
    entry = safe_float(d.iloc[-1]["close"])
    atr = max(float(atr), entry * 0.0008)
    stop_r = abs(entry - safe_float(provisional_plan.get("stop"), entry)) / atr
    target_r = abs(safe_float(provisional_plan.get("target"), entry) - entry) / atr
    engineering_source = engineering_full_df if engineering_full_df is not None and not engineering_full_df.empty else hist_df
    engineering = engineering_pattern_analysis(engineering_source, mode, current_end=d.index[-1], stop_r=stop_r, target_r=target_r)
    predicted_side = str(engineering.get("predicted_side", "WAIT"))
    confidence = safe_float(engineering.get("confidence"), 0.0)
    ready = bool(engineering.get("ready", False))
    min_conf = float(rules.get("engineering_min_confidence", 0.58))
    strong_conf = float(rules.get("engineering_strong_confidence", 0.70))

    side = technical_side
    if side == "WAIT" and ready and bool(rules.get("engineering_allow_strong_override", True)) and confidence >= strong_conf:
        predicted_score = ls if predicted_side == "LONG" else ss
        opposite_score = ss if predicted_side == "LONG" else ls
        if predicted_side in ["LONG", "SHORT"] and predicted_score >= max(4, min_base - 2) and predicted_score >= opposite_score - 1:
            side = predicted_side
            base = predicted_score
            reasons = lr if side == "LONG" else sr

    if side == "WAIT":
        return {"signal": "WAIT", "ticker": normalize_ticker(ticker), "mode": mode,
                "score": max(ls, ss), "reason": f"אין יתרון כיוון מספיק | {engineering.get('reason', '')}"}

    if ready and bool(rules.get("engineering_require_when_ready", True)):
        if predicted_side == "WAIT":
            return {"signal": "WAIT", "ticker": normalize_ticker(ticker), "mode": mode,
                    "score": base, "reason": f"מנוע הנדסי לא מצא יתרון | {engineering.get('reason', '')}"}
        if predicted_side != side and confidence >= min_conf:
            return {"signal": "WAIT", "ticker": normalize_ticker(ticker), "mode": mode,
                    "score": base, "reason": f"מנוע הנדסי נגד הכיוון | {engineering.get('reason', '')}"}
        if predicted_side == side and confidence < min_conf:
            return {"signal": "WAIT", "ticker": normalize_ticker(ticker), "mode": mode,
                    "score": base, "reason": f"ביטחון הנדסי נמוך {confidence:.2f}"}

    tf5, tf5_reason = timeframe_alignment_score(hist_df, side)
    if bool(rules.get("require_5m_alignment", True)) and tf5 < int(rules.get("min_5m_alignment_score", 2)):
        return {"signal": "WAIT", "ticker": normalize_ticker(ticker), "mode": mode,
                "score": base, "reason": f"5 דקות {tf5}/4"}

    market_reason = "QQQ לא זמין בבקטסט"
    if market_hist_df is not None and not market_hist_df.empty and bool(rules.get("market_filter_enabled", True)):
        required_side = market_side_for_ticker(ticker, side)
        mscore, mreason = timeframe_alignment_score(market_hist_df, required_side)
        opposite, oreason = timeframe_alignment_score(market_hist_df, "SHORT" if required_side == "LONG" else "LONG")
        minimum = int(rules.get("min_5m_alignment_score", 2))
        if opposite >= max(3, minimum + 1) and opposite >= mscore + 2:
            return {"signal": "WAIT", "ticker": normalize_ticker(ticker), "mode": mode,
                    "score": base, "reason": f"QQQ בכיוון מנוגד {opposite}/4 מול {mscore}/4"}
        market_reason = f"QQQ {mscore}/4 מול הפוך {opposite}/4: {mreason or oreason}"

    plan = chart_based_stop_target(d, side, mode)
    tf_bonus = 1 if tf5 >= 3 else 0
    eng_bonus = 3 if ready and predicted_side == side and confidence >= strong_conf else (1 if ready and predicted_side == side and confidence >= min_conf else 0)
    score = int(max(1, min(12, base + tf_bonus + eng_bonus)))
    last = d.iloc[-1]
    return {
        "signal": side,
        "ticker": normalize_ticker(ticker),
        "mode": mode,
        "score": score,
        "long_score": ls,
        "short_score": ss,
        "score_gap": gap,
        "tf5_score": tf5,
        "engineering_confidence": confidence,
        "engineering_samples": int(engineering.get("sample_count", 0)),
        "engineering_side": predicted_side,
        "entry": float(last["close"]),
        "stop": float(plan["stop"]),
        "target": float(plan["target"]),
        "atr": float(atr),
        "signal_high": float(last["high"]),
        "signal_low": float(last["low"]),
        "signal_bar_time": str(d.index[-1]),
        "last_rel_vol": float(safe_float(last.get("rel_vol5"), 0)),
        "reason": " | ".join(reasons + [engineering.get("reason", ""), f"5 דקות {tf5}/4: {tf5_reason}", market_reason, plan.get("reason", "")]),
    }


def backtest_has_open_ticker(open_trades, ticker):
    return any(t["status"] == "OPEN" and str(t["ticker"]) == str(ticker) for t in open_trades)


def backtest_pending_exists(pending, ticker, mode):
    """
    Safe check for pending backtest candidates.
    Older V5.5 candidates did not include 'status', so use .get().
    """
    return any(
        p.get("status", "PENDING") == "PENDING"
        and str(p.get("ticker", "")) == str(ticker)
        and str(p.get("mode", "")) == str(mode)
        for p in pending
    )


def backtest_breakeven_price(trade):
    entry = safe_float(trade["entry_price"])
    qty = safe_float(trade["quantity"])
    if qty <= 0:
        return entry

    costs = {
        "cost_pct_per_side": safe_float(trade["cost_pct_per_side"], DEFAULT_COSTS["cost_pct_per_side"]),
        "fixed_fee_per_side": safe_float(trade["fixed_fee_per_side"], DEFAULT_COSTS["fixed_fee_per_side"]),
        "min_fee_per_side": safe_float(trade["min_fee_per_side"], DEFAULT_COSTS["min_fee_per_side"]),
    }
    _, _, total_cost = estimate_costs(entry, entry, qty, costs)
    buffer_per_share = total_cost / qty

    if str(trade["side"]) == "LONG":
        return entry + buffer_per_share
    return entry - buffer_per_share


def backtest_close_trade(trade, exit_price, exit_time, reason):
    pnl = pnl_for_trade(trade, exit_price)

    trade = dict(trade)
    trade["status"] = "CLOSED"
    trade["exit_price"] = float(exit_price)
    trade["current_price"] = float(exit_price)
    trade["exit_time"] = str(exit_time)
    trade["duration_minutes"] = minutes_between(trade["entry_time"], str(exit_time))
    trade["age_minutes"] = trade["duration_minutes"]
    trade["exit_reason"] = reason
    trade["exit_reason_he"] = exit_reason_he(reason)

    for k, v in pnl.items():
        trade[k] = v

    return trade


def backtest_open_trade_from_signal(signal, entry_time, costs, units, rules, min_score):
    ticker = normalize_ticker(signal["ticker"])
    mode = str(signal["mode"])
    side = str(signal["signal"])
    score = int(signal.get("score", 0))

    if side not in ["LONG", "SHORT"]:
        return None, f"{ticker}: אין איתות."
    if score < int(min_score):
        return None, f"{ticker}: ניקוד {score} נמוך מהמינימום."

    entry = float(signal["entry"])
    stop = float(signal["stop"])
    target = float(signal["target"])

    score_qty, score_notional, unit_mult = position_size(score, entry, units)
    if score_qty <= 0 or score_notional <= 0:
        return None, f"{ticker}: לפי יוניטים, ניקוד {score} לא מקבל כניסה."

    qty, notional, risk_msg = apply_risk_cap_to_position(
        side=side,
        entry=entry,
        stop=stop,
        score_qty=score_qty,
        score_notional=score_notional,
        max_loss_dollars=float(rules.get("max_allowed_loss_per_trade_dollars", 7.0)),
    )

    if qty <= 0 or notional <= 0:
        return None, f"{ticker}: {risk_msg}"

    ok, eg, ec, en, msg = cost_tradeoff(side, entry, target, qty, costs)
    if not ok:
        return None, f"{ticker}: {msg}"

    entry_cost, exit_cost, total_cost_now = estimate_costs(entry, entry, qty, costs)

    trade = {
        "trade_id": str(uuid.uuid4()),
        "status": "OPEN",
        "ticker": ticker,
        "mode": mode,
        "side": side,
        "score": score,
        "entry_time": str(entry_time),
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
        "management_action": "BACKTEST_OPENED",
        "management_reason": "נפתחה בבקטסט אחרי אישור דקה.",
        "signal_reason": signal.get("reason", ""),
        "cost_pct_per_side": costs["cost_pct_per_side"],
        "fixed_fee_per_side": costs["fixed_fee_per_side"],
        "min_fee_per_side": costs["min_fee_per_side"],
        "max_cost_to_target_pct": costs["max_cost_to_target_pct"],
        "base_unit_dollars": units["base_unit_dollars"],
        "unit_multiplier": unit_mult,
        "created_settings_snapshot": json.dumps({"costs": costs, "units": units, "rules": rules}, ensure_ascii=False),
    }
    trade["breakeven_price"] = backtest_breakeven_price(trade)

    return trade, f"{ticker}: נפתחה בבקטסט {side} | {mode} | ניקוד {score} | {risk_msg}"


def backtest_update_trade(trade, hist_df, current_bar, current_time, rules):
    """
    Conservative historical trade management.
    Uses only data up to current_time.
    """
    trade = dict(trade)

    current = safe_float(current_bar["close"])
    high = safe_float(current_bar["high"])
    low = safe_float(current_bar["low"])

    trade["current_price"] = current
    trade["age_minutes"] = minutes_between(trade["entry_time"], str(current_time))
    trade["duration_minutes"] = trade["age_minutes"]

    side = str(trade["side"])
    mode = str(trade["mode"])
    score = int(safe_float(trade["score"], 1))
    entry = safe_float(trade["entry_price"])
    stop = safe_float(trade["stop_loss"])
    target = safe_float(trade["target_reference"])
    breakeven = safe_float(trade.get("breakeven_price"), backtest_breakeven_price(trade))

    trade["highest_price"] = max(safe_float(trade.get("highest_price"), entry), high)
    trade["lowest_price"] = min(safe_float(trade.get("lowest_price"), entry), low)

    pnl = pnl_for_trade(trade, current)
    current_net = pnl["net_pnl"]
    trade["max_net_pnl_seen"] = max(safe_float(trade.get("max_net_pnl_seen"), current_net), current_net)

    for k, v in pnl.items():
        trade[k] = v

    base_risk = abs(entry - safe_float(trade.get("initial_stop_loss"), stop))
    if base_risk <= 0:
        base_risk = max(entry * 0.001, abs(entry - stop))

    d = add_indicators(hist_df).dropna(subset=["close"])
    last = d.iloc[-1] if not d.empty else None
    ema5 = safe_float(last["ema5"], current) if last is not None else current
    ema5_slope = safe_float(last["ema5_slope"], 0) if last is not None else 0

    last3 = d.tail(min(3, len(d))) if not d.empty else pd.DataFrame()
    green = int((last3["close"] > last3["open"]).sum()) if not last3.empty else 0
    red = int((last3["close"] < last3["open"]).sum()) if not last3.empty else 0

    max_loss = abs(float(rules.get("max_allowed_loss_per_trade_dollars", 7.0)))
    giveback_pct = float(rules.get("profit_giveback_pct", 10.0))
    min_giveback_profit = float(rules.get("min_net_profit_for_giveback", 5.0))
    breakeven_after = float(rules.get("breakeven_after_profit_dollars", 4.0))
    lock_profit_after = float(rules.get("lock_profit_after_net_dollars", 8.0))
    age = safe_float(trade["age_minutes"], 0)

    # 1. Hard max loss after costs
    if current_net <= -max_loss:
        return backtest_close_trade(trade, current, current_time, "MAX_LOSS_LIMIT")

    # 2. Hard stop using candle high/low
    if side == "LONG" and low <= stop:
        return backtest_close_trade(trade, stop, current_time, "STOP_LOSS")
    if side == "SHORT" and high >= stop:
        return backtest_close_trade(trade, stop, current_time, "STOP_LOSS")

    # 3. Profit giveback
    peak_profit = safe_float(trade.get("max_net_pnl_seen"), current_net)
    if peak_profit >= min_giveback_profit:
        allowed_drop = peak_profit * (giveback_pct / 100.0)
        if current_net <= peak_profit - allowed_drop:
            return backtest_close_trade(trade, current, current_time, "PROFIT_GIVEBACK")

    # 4. Breakeven after costs if it was profitable
    if bool(rules.get("exit_if_profitable_trade_turns_red", True)):
        if side == "LONG" and peak_profit >= breakeven_after and current <= breakeven:
            return backtest_close_trade(trade, current, current_time, "BREAKEVEN_AFTER_COSTS")
        if side == "SHORT" and peak_profit >= breakeven_after and current >= breakeven:
            return backtest_close_trade(trade, current, current_time, "BREAKEVEN_AFTER_COSTS")

    # 5. No progress
    if side == "LONG":
        if mode == "מהירה" and age >= 3 and current_net <= 0 and red >= 2 and current < entry:
            return backtest_close_trade(trade, current, current_time, "NO_PROGRESS_FAST")
        if mode != "מהירה" and age >= 12 and current_net <= 0 and current < ema5 and ema5_slope < 0:
            return backtest_close_trade(trade, current, current_time, "NO_PROGRESS_HALF")
    else:
        if mode == "מהירה" and age >= 3 and current_net <= 0 and green >= 2 and current > entry:
            return backtest_close_trade(trade, current, current_time, "NO_PROGRESS_FAST")
        if mode != "מהירה" and age >= 12 and current_net <= 0 and current > ema5 and ema5_slope > 0:
            return backtest_close_trade(trade, current, current_time, "NO_PROGRESS_HALF")

    # 6. Profit stop and target behavior
    if side == "LONG":
        if current_net >= breakeven_after:
            new_profit_stop = max(breakeven, current - 0.35 * base_risk)
            if not np.isfinite(safe_float(trade.get("profit_stop"), np.nan)) or new_profit_stop > safe_float(trade.get("profit_stop"), -np.inf):
                trade["profit_stop"] = new_profit_stop

        if peak_profit >= lock_profit_after:
            new_profit_stop = max(breakeven, current - 0.18 * base_risk)
            if not np.isfinite(safe_float(trade.get("profit_stop"), np.nan)) or new_profit_stop > safe_float(trade.get("profit_stop"), -np.inf):
                trade["profit_stop"] = new_profit_stop

        if np.isfinite(safe_float(trade.get("profit_stop"), np.nan)) and low <= safe_float(trade["profit_stop"]):
            return backtest_close_trade(trade, safe_float(trade["profit_stop"]), current_time, "PROFIT_STOP")

        if high >= target and score < int(rules.get("exit_on_target_when_score_below", 7)):
            return backtest_close_trade(trade, target, current_time, "TARGET_REACHED_SCORE_EXIT")

        if high >= target and score >= int(rules.get("exit_on_target_when_score_below", 7)) and current > ema5 and ema5_slope > 0:
            trade["target_reference"] = max(target, current + 0.80 * base_risk)
        elif high >= target:
            return backtest_close_trade(trade, target, current_time, "TARGET_REACHED")

    else:
        if current_net >= breakeven_after:
            new_profit_stop = min(breakeven, current + 0.35 * base_risk)
            if not np.isfinite(safe_float(trade.get("profit_stop"), np.nan)) or new_profit_stop < safe_float(trade.get("profit_stop"), np.inf):
                trade["profit_stop"] = new_profit_stop

        if peak_profit >= lock_profit_after:
            new_profit_stop = min(breakeven, current + 0.18 * base_risk)
            if not np.isfinite(safe_float(trade.get("profit_stop"), np.nan)) or new_profit_stop < safe_float(trade.get("profit_stop"), np.inf):
                trade["profit_stop"] = new_profit_stop

        if np.isfinite(safe_float(trade.get("profit_stop"), np.nan)) and high >= safe_float(trade["profit_stop"]):
            return backtest_close_trade(trade, safe_float(trade["profit_stop"]), current_time, "PROFIT_STOP")

        if low <= target and score < int(rules.get("exit_on_target_when_score_below", 7)):
            return backtest_close_trade(trade, target, current_time, "TARGET_REACHED_SCORE_EXIT")

        if low <= target and score >= int(rules.get("exit_on_target_when_score_below", 7)) and current < ema5 and ema5_slope < 0:
            trade["target_reference"] = min(target, current - 0.80 * base_risk)
        elif low <= target:
            return backtest_close_trade(trade, target, current_time, "TARGET_REACHED")

    return trade


@st.cache_data(show_spinner=False, ttl=120)
def load_backtest_data_for_date(tickers_tuple, date_str):
    """
    Load the selected session plus all earlier 1m bars available from yfinance.
    The selected-day frame drives the replay timeline; full_history lets the
    engineering engine compare against prior sessions without future leakage.
    """
    selected_date = pd.to_datetime(date_str).date()
    day_data = {}
    full_history = {}
    missing = []

    for ticker in tickers_tuple:
        try:
            df = fetch_1m(ticker)
            if df is None or df.empty:
                missing.append(ticker)
                continue
            df = df.sort_index()
            day_df = df[df.index.date == selected_date].copy()
            if day_df.empty:
                missing.append(ticker)
                continue
            day_data[ticker] = day_df
            full_history[ticker] = df[df.index.date <= selected_date].copy()
        except Exception:
            missing.append(ticker)

    return day_data, full_history, missing



def backtest_entry_window_ok(current_time, rules):
    start_h, start_m = parse_hhmm(rules.get("entry_start_time", "09:45"), "09:45")
    end_h, end_m = parse_hhmm(rules.get("entry_end_time", "15:25"), "15:25")
    minute = current_time.hour * 60 + current_time.minute
    return start_h * 60 + start_m <= minute <= end_h * 60 + end_m


def backtest_risk_gate(open_trades, closed_trades, current_time, total_opened, rules):
    if total_opened >= int(rules.get("max_trades_per_day", 18)):
        return False, "מקסימום עסקאות יומי"
    total_net = sum(safe_float(t.get("net_pnl"), 0) for t in closed_trades)
    total_net += sum(safe_float(t.get("net_pnl"), 0) for t in open_trades)
    if total_net <= -abs(float(rules.get("daily_loss_limit_dollars", 25.0))):
        return False, "הפסד יומי הגיע למגבלה"
    streak = 0
    for trade in reversed(closed_trades):
        if safe_float(trade.get("net_pnl"), 0) < 0:
            streak += 1
        else:
            break
    needed = int(rules.get("max_consecutive_losses", 3))
    if streak >= needed and closed_trades:
        last_exit = timestamp_to_ny(closed_trades[-1].get("exit_time"))
        if last_exit is not None:
            elapsed = (timestamp_to_ny(current_time) - last_exit).total_seconds() / 60.0
            if elapsed < float(rules.get("loss_streak_pause_minutes", 15)):
                return False, "הפסקה אחרי רצף הפסדים"
    return True, "תקין"


def backtest_exposure_ok(open_trades, pending, ticker, side, rules, include_pending=True):
    max_side = int(rules.get("max_same_side_open", 3))
    max_group = int(rules.get("max_same_group_open", 2))
    side_count = sum(1 for t in open_trades if str(t.get("side")) == str(side))
    group = ticker_group(ticker)
    group_count = sum(1 for t in open_trades if ticker_group(t.get("ticker")) == group)
    if include_pending:
        side_count += sum(1 for x in pending if str(x.get("side")) == str(side))
        group_count += sum(1 for x in pending if ticker_group(x.get("ticker")) == group)
    return side_count < max_side and group_count < max_group

def run_day_backtest(tickers, date_value, modes, min_score, max_open, max_trades_total):
    """
    Replay a single historical day minute-by-minute.
    This is a paper/backtest simulation only.
    """
    costs = load_costs()
    units = load_units()
    rules = load_rules()

    date_str = str(pd.to_datetime(date_value).date())
    requested_tickers = list(dict.fromkeys([normalize_ticker(t) for t in tickers]))
    load_tickers_bt = list(dict.fromkeys(requested_tickers + ["QQQ"]))
    data, history_data, missing_all = load_backtest_data_for_date(tuple(load_tickers_bt), date_str)
    missing = [t for t in missing_all if t in requested_tickers]

    if not data:
        return {
            "trades": pd.DataFrame(),
            "summary": {},
            "equity": pd.DataFrame(),
            "messages": [f"לא נמצאו נתוני 1 דקה לתאריך {date_str}. ב־yfinance בדרך כלל צריך לבחור יום מהימים האחרונים."],
            "missing": missing,
        }

    # Build unified timeline
    all_times = sorted(set().union(*[set(df.index) for df in data.values()]))
    if not all_times:
        return {
            "trades": pd.DataFrame(),
            "summary": {},
            "equity": pd.DataFrame(),
            "messages": ["לא נמצאו נרות למסחר."],
            "missing": missing,
        }

    open_trades = []
    closed_trades = []
    pending = []
    messages = []
    equity_points = []
    total_opened = 0

    confirm_seconds = float(rules.get("confirm_before_entry_seconds", 60))
    pending_expire_seconds = float(rules.get("pending_signal_expire_minutes", 5)) * 60

    for current_time in all_times:
        # Update open trades
        updated_open = []
        for trade in open_trades:
            ticker = trade["ticker"]
            if ticker not in data:
                updated_open.append(trade)
                continue

            hist = data[ticker][data[ticker].index <= current_time]
            if hist.empty:
                updated_open.append(trade)
                continue

            current_bar = hist.iloc[-1]
            updated = backtest_update_trade(trade, hist, current_bar, current_time, rules)

            if updated["status"] == "CLOSED":
                closed_trades.append(updated)
            else:
                updated_open.append(updated)

        open_trades = updated_open
        entry_window_ok = backtest_entry_window_ok(current_time, rules)
        risk_gate_ok, risk_gate_reason = backtest_risk_gate(open_trades, closed_trades, current_time, total_opened, rules)

        # Process pending candidates
        new_pending = []
        for p in pending:
            if p.get("status", "PENDING") != "PENDING":
                continue

            age_seconds = (current_time - p["created_at"]).total_seconds()
            ticker = p["ticker"]
            mode = p["mode"]

            if age_seconds > pending_expire_seconds:
                continue

            if age_seconds < confirm_seconds:
                new_pending.append(p)
                continue

            if not entry_window_ok or not risk_gate_ok:
                continue

            if total_opened >= int(max_trades_total):
                continue

            if len(open_trades) >= int(max_open):
                new_pending.append(p)
                continue

            if backtest_has_open_ticker(open_trades, ticker):
                continue
            if not backtest_exposure_ok(open_trades, new_pending, ticker, p.get("side", ""), rules, include_pending=False):
                continue

            hist = data.get(ticker, pd.DataFrame())
            hist = hist[hist.index <= current_time]
            if hist.empty:
                new_pending.append(p)
                continue

            market_hist = data.get("QQQ", pd.DataFrame())
            market_hist = market_hist[market_hist.index <= current_time] if not market_hist.empty else market_hist
            engineering_hist = history_data.get(ticker, hist)
            engineering_hist = engineering_hist[engineering_hist.index <= current_time]
            new_signal = make_signal_from_history(
                ticker, mode, hist, market_hist_df=market_hist, engineering_full_df=engineering_hist
            )
            mode_floor = int(rules.get("min_score_fast", 10)) if str(mode) == "מהירה" else int(rules.get("min_score_half", 7))
            effective_min_score = max(int(min_score), mode_floor)

            confirmed, confirm_msg = signal_confirmed_after_delay(
                original_side=p["side"],
                original_score=p["score"],
                new_signal=new_signal,
                min_score=effective_min_score,
                original_entry=p.get("entry_price", np.nan),
                original_stop=p.get("stop_loss", np.nan),
                original_target=p.get("target_reference", np.nan),
                signal_high=p.get("signal_high", np.nan),
                signal_low=p.get("signal_low", np.nan),
            )

            if not confirmed:
                continue

            trade, msg = backtest_open_trade_from_signal(
                new_signal,
                entry_time=current_time,
                costs=costs,
                units=units,
                rules=rules,
                min_score=max(1, effective_min_score - 1),
            )

            if trade is not None:
                open_trades.append(trade)
                total_opened += 1
                messages.append(msg)

        pending = new_pending

        # Create new pending candidates if there is room. The engineering engine is evaluated at a controlled cadence.
        bt_scan_interval = max(1, int(rules.get("engineering_backtest_scan_interval", 3)))
        engineering_scan_bar = (int(current_time.minute) % bt_scan_interval == 0)
        if engineering_scan_bar and entry_window_ok and risk_gate_ok and total_opened < min(int(max_trades_total), int(rules.get("max_trades_per_day", 18))) and len(open_trades) < int(max_open):
            for ticker in requested_tickers:
                df = data.get(ticker, pd.DataFrame())
                if total_opened >= int(max_trades_total):
                    break
                if len(open_trades) >= int(max_open):
                    break
                if backtest_has_open_ticker(open_trades, ticker):
                    continue

                hist = df[df.index <= current_time]
                if len(hist) < 15:
                    continue

                for mode in modes:
                    if backtest_pending_exists(pending, ticker, mode):
                        continue

                    market_hist = data.get("QQQ", pd.DataFrame())
                    market_hist = market_hist[market_hist.index <= current_time] if not market_hist.empty else market_hist
                    engineering_hist = history_data.get(ticker, hist)
                    engineering_hist = engineering_hist[engineering_hist.index <= current_time]
                    sig = make_signal_from_history(
                        ticker, mode, hist, market_hist_df=market_hist, engineering_full_df=engineering_hist
                    )
                    if sig.get("signal") not in ["LONG", "SHORT"]:
                        continue
                    mode_floor = int(rules.get("min_score_fast", 10)) if str(mode) == "מהירה" else int(rules.get("min_score_half", 7))
                    effective_min_score = max(int(min_score), mode_floor)
                    if int(sig.get("score", 0)) < effective_min_score:
                        continue
                    if not backtest_exposure_ok(open_trades, pending, ticker, sig.get("signal", ""), rules, include_pending=True):
                        continue

                    pending.append(
                        {
                            "pending_id": str(uuid.uuid4()),
                            "created_at": current_time,
                            "ticker": ticker,
                            "mode": mode,
                            "side": sig["signal"],
                            "score": int(sig["score"]),
                            "entry_price": float(sig.get("entry", np.nan)),
                            "stop_loss": float(sig.get("stop", np.nan)),
                            "target_reference": float(sig.get("target", np.nan)),
                            "signal_high": float(sig.get("signal_high", np.nan)),
                            "signal_low": float(sig.get("signal_low", np.nan)),
                            "status": "PENDING",
                            "message": "מועמדת בבקטסט מחכה לאישור חוזר.",
                        }
                    )

        # Equity snapshot
        closed_net = sum(safe_float(t.get("net_pnl"), 0) for t in closed_trades)
        open_net = 0.0
        for trade in open_trades:
            try:
                ticker = trade["ticker"]
                hist = data[ticker][data[ticker].index <= current_time]
                if not hist.empty:
                    current_price = safe_float(hist.iloc[-1]["close"])
                    open_net += pnl_for_trade(trade, current_price)["net_pnl"]
            except Exception:
                pass

        equity_points.append(
            {
                "time": current_time,
                "closed_net": closed_net,
                "open_net": open_net,
                "total_net": closed_net + open_net,
                "open_trades": len(open_trades),
                "closed_trades": len(closed_trades),
            }
        )

    # Close remaining open trades at last available price
    for trade in open_trades:
        ticker = trade["ticker"]
        df = data.get(ticker, pd.DataFrame())
        if df.empty:
            continue
        last_bar = df.iloc[-1]
        closed_trades.append(backtest_close_trade(trade, safe_float(last_bar["close"]), df.index[-1], "END_OF_DAY"))

    trades_df = pd.DataFrame(closed_trades)
    if not trades_df.empty:
        for col in TRADE_COLUMNS:
            if col not in trades_df.columns:
                trades_df[col] = np.nan

    equity_df = pd.DataFrame(equity_points)

    if trades_df.empty:
        summary = {
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "net": 0.0,
            "gross": 0.0,
            "costs": 0.0,
        }
    else:
        net_series = pd.to_numeric(trades_df["net_pnl"], errors="coerce").fillna(0)
        summary = {
            "trades": int(len(trades_df)),
            "wins": int((net_series > 0).sum()),
            "losses": int((net_series < 0).sum()),
            "win_rate": float((net_series > 0).mean() * 100),
            "net": float(net_series.sum()),
            "gross": float(pd.to_numeric(trades_df["gross_pnl"], errors="coerce").fillna(0).sum()),
            "costs": float(pd.to_numeric(trades_df["total_cost"], errors="coerce").fillna(0).sum()),
        }

    return {
        "trades": trades_df,
        "summary": summary,
        "equity": equity_df,
        "messages": messages,
        "missing": missing,
    }


def render_backtest_trades_table(trades_df):
    if trades_df.empty:
        st.info("הבקטסט לא פתח עסקאות ביום הזה.")
        return

    d = trades_df.copy().reset_index(drop=True)
    d["exit_reason_he"] = d.apply(
        lambda r: r["exit_reason_he"] if isinstance(r.get("exit_reason_he", ""), str) and r.get("exit_reason_he", "") else exit_reason_he(r.get("exit_reason", "")),
        axis=1,
    )

    display = pd.DataFrame({
        "מניה": d["ticker"],
        "סוג": d["mode"],
        "כיוון": d["side"],
        "ניקוד": d["score"].fillna(0).astype(int),
        "כניסה": d["entry_price"].map(fmt_price),
        "יציאה": d["exit_price"].map(fmt_price),
        "סטופ": d["stop_loss"].map(fmt_price),
        "TP": d["target_reference"].map(fmt_price),
        "רווח/הפסד נטו": d["net_pnl"].map(fmt_money),
        "עלות": d["total_cost"].map(fmt_money),
        "זמן כניסה": d["entry_time"].astype(str).str.slice(0, 19),
        "זמן יציאה": d["exit_time"].astype(str).str.slice(0, 19),
        "משך דק׳": d["duration_minutes"].map(fmt_minutes),
        "סיבת יציאה": d["exit_reason_he"],
    })

    pnl_values = pd.to_numeric(d["net_pnl"], errors="coerce").fillna(0).tolist()

    def style_row(row):
        pnl = pnl_values[row.name]
        if pnl >= 0:
            return ["background-color:#dcfce7;color:#064e3b;"] * len(row)
        return ["background-color:#fee2e2;color:#7f1d1d;"] * len(row)

    st.dataframe(display.style.apply(style_row, axis=1), use_container_width=True, hide_index=True)




# ============================================================
# Standalone live engineering monitor
# ============================================================

from concurrent.futures import ThreadPoolExecutor, as_completed

# This app is intentionally isolated from V7.2 trading files so both apps can run
# side by side without overwriting each other's trades or predictions.
TRADES_FILE = DATA_DIR / "pattern_monitor_trades_v1_7.csv"
PENDING_FILE = DATA_DIR / "pattern_monitor_pending_v1_7.csv"
RULES_FILE = DATA_DIR / "pattern_monitor_rules_v1_7.json"
ACCOUNT_FILE = DATA_DIR / "pattern_monitor_account_v1_7.json"
ALERTS_FILE = DATA_DIR / "pattern_monitor_alerts_v1_7.csv"
ALERT_SETTINGS_FILE = DATA_DIR / "pattern_monitor_alert_settings_v1_7.json"
ENGINEERING_FILE = DATA_DIR / "pattern_monitor_engineering_predictions_v1_7.csv"
MONITOR_SETTINGS_FILE = DATA_DIR / "pattern_monitor_settings_v1_7.json"
MONITOR_SNAPSHOTS_FILE = DATA_DIR / "pattern_monitor_snapshots_v1_7.csv"
MONITOR_METADATA_FILE = DATA_DIR / "pattern_monitor_trade_metadata_v1_7.csv"
MONITOR_EVALUATIONS_FILE = DATA_DIR / "pattern_monitor_prediction_evaluations_v1_7.csv"
MANUAL_TELEGRAM_FILE = DATA_DIR / "pattern_monitor_manual_telegram_trades_v1_9.csv"

MONITOR_DEFAULT_RULES = dict(DEFAULT_RULES)
MONITOR_DEFAULT_RULES.update({
    "cooldown_after_close_minutes": 6,
    "max_open_trades": 8,
    "max_same_side_open": 5,
    "max_same_group_open": 3,
    "daily_loss_limit_dollars": 35.0,
    "max_trades_per_day": 30,
    "max_consecutive_losses": 4,
    "loss_streak_pause_minutes": 10,
    "entry_start_time": "09:35",
    "entry_end_time": "15:30",
    "force_flat_time": "15:55",
    "max_allowed_loss_per_trade_dollars": 7.0,
    "cycle_net_profit_target": 999999.0,
    "min_score_fast": 7,
    "min_score_half": 7,
    "engineering_top_k": 25,
    "engineering_min_samples": 6,
    "engineering_candidate_step": 6,
    "engineering_max_candidates": 260,
    "engineering_min_similarity": 0.34,
    "engineering_similarity_band": 0.12,
    "engineering_neutral_band_r": 0.10,
})
DEFAULT_RULES.clear()
DEFAULT_RULES.update(MONITOR_DEFAULT_RULES)

MONITOR_DEFAULT_SETTINGS = {
    "auto_trade_enabled": False,
    "modes": ["מהירה", "חצי שעה"],
    "refresh_seconds": 60,
    "max_workers": 6,
    "max_new_trades_per_scan": 3,
    "max_open_trades": 8,
    "min_samples": 6,
    "min_similarity": 0.34,
    "min_best_similarity": 0.44,
    "max_neutral_probability": 0.45,
    "min_expectancy_gap_r": 0.10,
    "min_confidence_fast": 0.50,
    "min_confidence_half": 0.48,
    "min_probability_fast": 0.60,
    "min_probability_half": 0.56,
    "min_probability_gap_fast": 0.10,
    "min_probability_gap_half": 0.06,
    "min_expectancy_r": 0.10,
    "require_positive_expectancy": True,
    "consecutive_scans_required": 1,
    "trend_min_technical_fast": 5,
    "trend_min_technical_half": 6,
    "trend_min_5m_fast": 1,
    "trend_min_5m_half": 2,

    # V1.8 display fallback: prevents the table from showing meaningless 0% when
    # the engineering pattern engine is still learning or has no valid matches.
    # This fallback is for visibility/diagnostics only by default.
    "display_technical_fallback": True,
    "allow_technical_fallback_trades": False,

    "entry_start_time": "09:35",
    "entry_end_time_fast": "15:55",
    "entry_end_time_half": "15:30",
}

SNAPSHOT_COLUMNS = [
    "scan_id", "scan_time", "bar_time", "data_age_minutes", "data_fresh", "market_window_ok", "ticker", "mode", "last_price",
    "long_probability", "short_probability", "neutral_probability", "probability_leader", "engine_side",
    "dominant_side", "dominant_probability", "probability_gap", "confidence",
    "best_similarity", "weakest_similarity",
    "technical_long_score", "technical_short_score", "trend_side",
    "alignment_5m", "alignment_5m_opposite", "trend_confirmed",
    "sample_count", "mean_similarity",
    "long_target_rate", "short_target_rate", "long_expectancy_r", "short_expectancy_r",
    "expected_mfe_r", "expected_mae_r", "pattern_state", "function_model",
    "ready", "eligible", "trade_allowed", "status", "reason",
    "entry", "stop", "target", "score",
]

EVALUATION_COLUMNS = [
    "evaluation_id", "prediction_key", "evaluated_at", "scan_time", "bar_time",
    "ticker", "mode", "side", "horizon_minutes", "entry", "stop", "target",
    "future_price", "first_touch", "direction_result", "correct", "realized_r",
    "max_favorable_r", "max_adverse_r", "long_probability", "short_probability",
    "neutral_probability", "confidence", "mean_similarity", "status", "note",
]


MANUAL_TELEGRAM_COLUMNS = [
    "manual_id", "trade_id", "created_at", "exit_at",
    "ticker", "mode", "side",
    "entry_price", "exit_price", "stop_loss", "target_reference",
    "quantity", "unit_multiplier",
    "status",
    "entry_telegram_sent", "entry_telegram_error",
    "exit_telegram_sent", "exit_telegram_error",
    "source_scan_id",
]

METADATA_COLUMNS = [
    "trade_id", "created_at", "ticker", "mode", "side",
    "long_probability", "short_probability", "neutral_probability", "dominant_probability", "probability_gap",
    "confidence", "sample_count", "mean_similarity", "long_expectancy_r",
    "short_expectancy_r", "pattern_state", "function_model", "scan_id",
]


def load_monitor_settings():
    return read_json(MONITOR_SETTINGS_FILE, MONITOR_DEFAULT_SETTINGS)


def save_monitor_settings(settings):
    safe = dict(MONITOR_DEFAULT_SETTINGS)
    safe.update(settings or {})
    write_json(MONITOR_SETTINGS_FILE, safe)


def ensure_monitor_files():
    if not RULES_FILE.exists():
        save_rules(MONITOR_DEFAULT_RULES)
    if not ACCOUNT_FILE.exists():
        save_account(DEFAULT_ACCOUNT)
    if not TRADES_FILE.exists():
        save_trades(empty_trades())
    if not PENDING_FILE.exists():
        save_pending(empty_pending())
    if not ALERTS_FILE.exists():
        save_alerts(empty_alerts())
    if not ALERT_SETTINGS_FILE.exists():
        save_alert_settings(DEFAULT_ALERT_SETTINGS)
    if not ENGINEERING_FILE.exists():
        save_engineering_predictions(empty_engineering_predictions())
    if not MONITOR_SETTINGS_FILE.exists():
        save_monitor_settings(MONITOR_DEFAULT_SETTINGS)
    if not MONITOR_METADATA_FILE.exists():
        pd.DataFrame(columns=METADATA_COLUMNS).to_csv(MONITOR_METADATA_FILE, index=False)
    if not MONITOR_SNAPSHOTS_FILE.exists():
        pd.DataFrame(columns=SNAPSHOT_COLUMNS).to_csv(MONITOR_SNAPSHOTS_FILE, index=False)
    if not MONITOR_EVALUATIONS_FILE.exists():
        pd.DataFrame(columns=EVALUATION_COLUMNS).to_csv(MONITOR_EVALUATIONS_FILE, index=False)
    if not MANUAL_TELEGRAM_FILE.exists():
        pd.DataFrame(columns=MANUAL_TELEGRAM_COLUMNS).to_csv(MANUAL_TELEGRAM_FILE, index=False)


def load_monitor_metadata():
    if not MONITOR_METADATA_FILE.exists() or MONITOR_METADATA_FILE.stat().st_size == 0:
        return pd.DataFrame(columns=METADATA_COLUMNS)
    try:
        df = pd.read_csv(MONITOR_METADATA_FILE)
    except Exception:
        return pd.DataFrame(columns=METADATA_COLUMNS)
    for col in METADATA_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    return df[METADATA_COLUMNS].copy()


def save_monitor_metadata(df):
    for col in METADATA_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    df[METADATA_COLUMNS].tail(5000).to_csv(MONITOR_METADATA_FILE, index=False)


def append_monitor_metadata(row):
    df = load_monitor_metadata()
    trade_id = str(row.get("trade_id", ""))
    if trade_id and not df.empty and df["trade_id"].astype(str).eq(trade_id).any():
        return
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    save_monitor_metadata(df)


def empty_manual_telegram_trades():
    df = pd.DataFrame(columns=MANUAL_TELEGRAM_COLUMNS)
    for col in MANUAL_TELEGRAM_COLUMNS:
        df[col] = df[col].astype("object")
    return df


def load_manual_telegram_trades():
    if not MANUAL_TELEGRAM_FILE.exists() or MANUAL_TELEGRAM_FILE.stat().st_size == 0:
        return empty_manual_telegram_trades()
    try:
        df = pd.read_csv(MANUAL_TELEGRAM_FILE, dtype="object")
    except Exception:
        return empty_manual_telegram_trades()
    for col in MANUAL_TELEGRAM_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    return df[MANUAL_TELEGRAM_COLUMNS].copy()


def save_manual_telegram_trades(df):
    if df is None or df.empty:
        empty_manual_telegram_trades().to_csv(MANUAL_TELEGRAM_FILE, index=False)
        return
    out = df.copy()
    for col in MANUAL_TELEGRAM_COLUMNS:
        if col not in out.columns:
            out[col] = ""
    out = out[MANUAL_TELEGRAM_COLUMNS].tail(5000)
    out.to_csv(MANUAL_TELEGRAM_FILE, index=False)


def fmt_manual_qty(qty):
    q = safe_float(qty, np.nan)
    if np.isnan(q):
        return ""
    if abs(q - round(q)) < 1e-6:
        return str(int(round(q)))
    return f"{q:.4f}".rstrip("0").rstrip(".")


def fmt_signed_money(value):
    v = safe_float(value, 0.0)
    sign = "+" if v >= 0 else "-"
    return f"{sign}${abs(v):.2f}"


def manual_telegram_enabled_and_ready():
    settings = load_alert_settings()
    if not bool(settings.get("telegram_enabled", False)):
        return False, "Telegram כבוי בלשונית ההגדרות."
    token = str(settings.get("telegram_bot_token", "")).strip()
    chat_id = str(settings.get("telegram_chat_id", "")).strip()
    if not token or not chat_id:
        return False, "חסר Bot Token או Chat ID."
    return True, ""


def send_manual_telegram(message):
    ready, problem = manual_telegram_enabled_and_ready()
    if not ready:
        return False, problem
    settings = load_alert_settings()
    return send_telegram_message(
        bot_token=str(settings.get("telegram_bot_token", "")).strip(),
        chat_id=str(settings.get("telegram_chat_id", "")).strip(),
        message=str(message),
    )


def manual_entry_message(trade_row):
    ticker = str(trade_row.get("ticker", "")).upper()
    side = str(trade_row.get("side", "")).upper()
    units = safe_float(trade_row.get("unit_multiplier"), np.nan)
    qty = safe_float(trade_row.get("quantity"), np.nan)
    stop = safe_float(trade_row.get("stop_loss"), np.nan)
    target = safe_float(trade_row.get("target_reference"), np.nan)

    direction = "🟢 ENTRY LONG 📈" if side == "LONG" else "🔴 ENTRY SHORT 📉" if side == "SHORT" else "⚪ ENTRY"
    units_text = "" if np.isnan(units) else f"{units:.2f}".rstrip("0").rstrip(".")
    qty_text = fmt_manual_qty(qty)

    return (
        f"{direction}\n"
        f"Ticker: {ticker}\n"
        f"Side: {side}\n"
        f"Units: {units_text}\n"
        f"Quantity: {qty_text}\n"
        f"Stop Loss: {stop:.2f}\n"
        f"Take Profit: {target:.2f}"
    )


def manual_exit_message(trade_row):
    ticker = str(trade_row.get("ticker", "")).upper()
    side = str(trade_row.get("side", "")).upper()
    entry = safe_float(trade_row.get("entry_price"), np.nan)
    exit_price = safe_float(trade_row.get("exit_price"), safe_float(trade_row.get("current_price"), np.nan))
    pnl = safe_float(trade_row.get("net_pnl"), 0.0)

    return (
        f"🔴 EXIT\n"
        f"Ticker: {ticker}\n"
        f"Side: {side}\n"
        f"Entry: {entry:.2f}\n"
        f"Exit: {exit_price:.2f}\n"
        f"Approx P/L: {fmt_signed_money(pnl)}"
    )


def append_manual_telegram_trade(trade_row, source_row, telegram_sent=False, telegram_error=""):
    df = load_manual_telegram_trades()
    trade_id = str(trade_row.get("trade_id", ""))
    if trade_id and not df.empty and df["trade_id"].astype(str).eq(trade_id).any():
        return

    row = {
        "manual_id": str(uuid.uuid4()),
        "trade_id": trade_id,
        "created_at": now_ny_iso(),
        "exit_at": "",
        "ticker": str(trade_row.get("ticker", "")),
        "mode": str(trade_row.get("mode", "")),
        "side": str(trade_row.get("side", "")),
        "entry_price": safe_float(trade_row.get("entry_price"), np.nan),
        "exit_price": "",
        "stop_loss": safe_float(trade_row.get("stop_loss"), np.nan),
        "target_reference": safe_float(trade_row.get("target_reference"), np.nan),
        "quantity": safe_float(trade_row.get("quantity"), np.nan),
        "unit_multiplier": safe_float(trade_row.get("unit_multiplier"), np.nan),
        "status": "OPEN",
        "entry_telegram_sent": "כן" if telegram_sent else "לא",
        "entry_telegram_error": telegram_error,
        "exit_telegram_sent": "",
        "exit_telegram_error": "",
        "source_scan_id": str(source_row.get("scan_id", "")) if isinstance(source_row, dict) else "",
    }
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    save_manual_telegram_trades(df)


def update_manual_telegram_after_exit(trade_row, telegram_sent=False, telegram_error=""):
    df = load_manual_telegram_trades()
    if df.empty:
        return
    trade_id = str(trade_row.get("trade_id", ""))
    mask = df["trade_id"].astype(str).eq(trade_id)
    if not mask.any():
        return
    idx = df.index[mask][0]
    df.loc[idx, "status"] = "CLOSED"
    df.loc[idx, "exit_at"] = now_ny_iso()
    df.loc[idx, "exit_price"] = safe_float(trade_row.get("exit_price"), safe_float(trade_row.get("current_price"), np.nan))
    df.loc[idx, "exit_telegram_sent"] = "כן" if telegram_sent else "לא"
    df.loc[idx, "exit_telegram_error"] = telegram_error
    save_manual_telegram_trades(df)


def append_monitor_metadata_from_prediction(new_trade, row):
    append_monitor_metadata({
        "trade_id": str(new_trade.get("trade_id", "")),
        "created_at": now_ny_iso(),
        "ticker": str(row.get("ticker", "")),
        "mode": str(row.get("mode", "")),
        "side": str(row.get("dominant_side", "")),
        "long_probability": float(safe_float(row.get("long_probability"), 0.0)),
        "short_probability": float(safe_float(row.get("short_probability"), 0.0)),
        "neutral_probability": float(safe_float(row.get("neutral_probability"), 0.0)),
        "dominant_probability": float(safe_float(row.get("dominant_probability"), 0.0)),
        "probability_gap": float(safe_float(row.get("probability_gap"), 0.0)),
        "confidence": float(safe_float(row.get("confidence"), 0.0)),
        "sample_count": int(safe_float(row.get("sample_count"), 0)),
        "mean_similarity": float(safe_float(row.get("mean_similarity"), 0.0)),
        "long_expectancy_r": float(safe_float(row.get("long_expectancy_r"), 0.0)),
        "short_expectancy_r": float(safe_float(row.get("short_expectancy_r"), 0.0)),
        "pattern_state": str(row.get("pattern_state", "")),
        "function_model": str(row.get("function_model", "")),
        "scan_id": str(row.get("scan_id", "")),
    })


def manual_signal_from_prediction(row):
    signal = build_monitor_signal(row)
    # Manual click should have a practical unit recommendation.
    # Scores below 7 get 1 unit according to this monitor's sizing table.
    signal["score"] = max(7, int(safe_float(signal.get("score"), 0)))
    return signal


def manual_enter_from_prediction(row):
    ticker = normalize_ticker(row.get("ticker", ""))
    if str(row.get("dominant_side", "")).upper() not in ["LONG", "SHORT"]:
        return False, f"{ticker}: אין כיוון LONG/SHORT לפתיחה ידנית."

    before = load_trades()
    before_ids = set(before["trade_id"].astype(str)) if not before.empty else set()

    ok, message = open_trade(manual_signal_from_prediction(row), min_score=1, send_alert=False)
    if not ok:
        return False, message

    after = load_trades()
    new_rows = after[~after["trade_id"].astype(str).isin(before_ids)].copy()
    if new_rows.empty:
        return False, f"{ticker}: העסקה נפתחה אך לא נמצאה ברשימה."

    new_trade = new_rows.sort_values("entry_time").iloc[-1].to_dict()
    append_monitor_metadata_from_prediction(new_trade, row)

    telegram_sent, telegram_error = send_manual_telegram(manual_entry_message(new_trade))
    append_manual_telegram_trade(new_trade, row, telegram_sent=telegram_sent, telegram_error=telegram_error)

    if telegram_sent:
        return True, f"{ticker}: נשלחה הודעת כניסה לטלגרם ונפתחה עסקת מעקב Paper."
    return True, f"{ticker}: נפתחה עסקת מעקב Paper, אבל Telegram לא נשלח: {telegram_error}"


def manual_exit_trade_and_send(trade_id):
    # Refresh prices first so the exit P/L is as close as possible to live.
    try:
        update_open_trades()
    except Exception:
        pass

    trades = load_trades()
    mask = trades["trade_id"].astype(str).eq(str(trade_id))
    if trades.empty or not mask.any():
        return False, "העסקה לא נמצאה."

    row_before = trades[mask].iloc[0]
    if str(row_before.get("status", "")) != "OPEN":
        return False, "העסקה כבר סגורה."

    ok, close_msg = close_trade_manually(trade_id)
    trades_after = load_trades()
    closed = trades_after[trades_after["trade_id"].astype(str).eq(str(trade_id))]
    if closed.empty:
        return False, close_msg

    closed_row = closed.iloc[0].to_dict()
    telegram_sent, telegram_error = send_manual_telegram(manual_exit_message(closed_row))
    update_manual_telegram_after_exit(closed_row, telegram_sent=telegram_sent, telegram_error=telegram_error)

    if telegram_sent:
        return True, close_msg + " | נשלחה הודעת יציאה לטלגרם."
    return True, close_msg + f" | Telegram לא נשלח: {telegram_error}"


def manual_tracked_trades_live_df():
    manual = load_manual_telegram_trades()
    if manual.empty:
        return pd.DataFrame()

    trades = load_trades()
    if trades.empty:
        return manual

    merged = manual.merge(
        trades,
        on="trade_id",
        how="left",
        suffixes=("_manual", ""),
    )

    # Prefer live values from trades table.
    for col in ["ticker", "mode", "side", "entry_price", "stop_loss", "target_reference", "quantity", "unit_multiplier", "status"]:
        live_col = col
        manual_col = f"{col}_manual"
        if live_col in merged.columns:
            merged[col] = merged[live_col]
        elif manual_col in merged.columns:
            merged[col] = merged[manual_col]

    return merged


def render_manual_tracked_trades_panel(compact=False, key_prefix='manual'):
    st.markdown("### 📋 העסקאות שנכנסת אליהן")
    st.caption("טבלת מעקב חי לעסקאות שנפתחו דרך כפתור הכניסה. כפתור יציאה אדום שולח Telegram עם רווח/הפסד משוער.")

    try:
        update_open_trades()
    except Exception:
        pass

    df = manual_tracked_trades_live_df()
    if df.empty:
        st.info("עדיין לא לחצת כניסה על שום מניה.")
        return

    df["_status_order"] = df["status"].astype(str).map({"OPEN": 0, "CLOSED": 1}).fillna(2)
    df = df.sort_values(["_status_order", "created_at"], ascending=[True, False]).drop(columns=["_status_order"])

    header_cols = st.columns([0.75, 0.8, 0.75, 0.8, 0.8, 0.75, 0.75, 0.75, 0.85, 0.95])
    headers = ["מניה", "כיוון", "סוג", "כניסה", "נוכחי", "SL", "TP", "Units", "רווח/הפסד", "פעולה"]
    for c, h in zip(header_cols, headers):
        c.markdown(f"**{h}**")

    st.markdown(
        "<div style='height:1px;background:#e5e7eb;margin:2px 0 6px 0;'></div>",
        unsafe_allow_html=True,
    )

    for i, row in df.head(60).iterrows():
        status = str(row.get("status", ""))
        ticker = str(row.get("ticker", "")).upper()
        side = str(row.get("side", "")).upper()
        pnl = safe_float(row.get("net_pnl"), 0.0)
        pnl_bg = "#dcfce7" if pnl >= 0 else "#fee2e2"
        pnl_color = "#064e3b" if pnl >= 0 else "#7f1d1d"
        row_bg = "#ffffff" if status == "OPEN" else "#f9fafb"

        with st.container():
            st.markdown(
                f"<div style='background:{row_bg};border:1px solid #e5e7eb;border-radius:12px;padding:4px 8px;margin:3px 0;'>",
                unsafe_allow_html=True,
            )

            c1, c2, c3, c4, c5, c6, c7, c8, c9, c10 = st.columns(
                [0.75, 0.8, 0.75, 0.8, 0.8, 0.75, 0.75, 0.75, 0.85, 0.95]
            )

            c1.markdown(f"**{ticker}**")
            c2.write("🟢 LONG" if side == "LONG" else "🔴 SHORT" if side == "SHORT" else side)
            c3.write(str(row.get("mode", "")))
            c4.write(fmt_price(row.get("entry_price")))
            c5.write(fmt_price(row.get("current_price")))
            c6.write(fmt_price(row.get("stop_loss")))
            c7.write(fmt_price(row.get("target_reference")))
            c8.write(f"{safe_float(row.get('unit_multiplier'), 0):.2f}".rstrip("0").rstrip("."))

            c9.markdown(
                f"<div style='background:{pnl_bg};color:{pnl_color};border-radius:10px;padding:6px;text-align:center;'>"
                f"<strong>{fmt_signed_money(pnl)}</strong></div>",
                unsafe_allow_html=True,
            )

            if status == "OPEN":
                if c10.button("🔴 יציאה", key=f"{key_prefix}_inline_manual_exit_{row.get('trade_id')}_{i}", use_container_width=True):
                    ok, msg = manual_exit_trade_and_send(str(row.get("trade_id", "")))
                    if ok:
                        st.success(msg)
                    else:
                        st.error(msg)
                    st.rerun()
            else:
                c10.button("סגור", key=f"{key_prefix}_inline_closed_{row.get('trade_id')}_{i}", disabled=True, use_container_width=True)

            st.markdown("</div>", unsafe_allow_html=True)

    table = pd.DataFrame({
        "סטטוס": df["status"],
        "מניה": df["ticker"],
        "כיוון": df["side"],
        "כניסה": pd.to_numeric(df["entry_price"], errors="coerce"),
        "נוכחי": pd.to_numeric(df.get("current_price"), errors="coerce"),
        "סטופ": pd.to_numeric(df["stop_loss"], errors="coerce"),
        "יעד": pd.to_numeric(df["target_reference"], errors="coerce"),
        "כמות": pd.to_numeric(df["quantity"], errors="coerce"),
        "יוניטים": pd.to_numeric(df["unit_multiplier"], errors="coerce"),
        "רווח נטו $": pd.to_numeric(df.get("net_pnl"), errors="coerce"),
        "זמן כניסה": df["created_at"],
    })

    with st.expander("טבלת מעקב מלאה", expanded=not compact):
        st.dataframe(table, use_container_width=True, hide_index=True)



def render_prediction_entry_buttons(rows, key_prefix='predictions'):
    st.markdown("### 📡 תחזיות בזמן אמת — כניסה מתוך הטבלה")
    st.caption("הכפתור הירוק פותח עסקת מעקב Paper ושולח Telegram. אין ביצוע קנייה אמיתית.")

    if not rows:
        st.info("אין תחזיות להצגה.")
        return

    df = pd.DataFrame(rows).copy()

    for col in [
        "ticker", "mode", "dominant_side", "decision", "engine_decision",
        "score", "entry", "stop", "target", "confidence",
        "long_probability", "short_probability", "neutral_probability",
        "dominant_probability", "probability_gap", "trade_allowed",
        "freshness", "data_age_minutes",
    ]:
        if col not in df.columns:
            df[col] = np.nan if col not in ["ticker", "mode", "dominant_side", "decision", "engine_decision", "trade_allowed", "freshness"] else ""

    for col in [
        "score", "entry", "stop", "target", "confidence",
        "long_probability", "short_probability", "neutral_probability",
        "dominant_probability", "probability_gap", "data_age_minutes",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df["ticker"] = df["ticker"].map(normalize_ticker)
    df = df[df["ticker"].astype(str).str.len() > 0].copy()

    if df.empty:
        st.info("אין כרגע שורות להצגה.")
        return

    # Put stronger rows first, same feeling as the main monitor table.
    df["_side_ok"] = df["dominant_side"].astype(str).isin(["LONG", "SHORT"])
    df = df.sort_values(
        ["_side_ok", "trade_allowed", "confidence", "dominant_probability", "probability_gap"],
        ascending=[False, False, False, False, False],
    ).drop(columns=["_side_ok"])

    open_trades = load_trades()
    open_tickers = set(
        open_trades.loc[open_trades["status"].astype(str).eq("OPEN"), "ticker"].astype(str)
    ) if not open_trades.empty else set()

    units = load_units()

    header_cols = st.columns([0.75, 0.85, 0.75, 0.75, 0.75, 0.75, 0.9, 0.9, 0.75, 0.75, 0.95])
    headers = [
        "מניה", "סוג עסקה", "מחיר", "LONG %", "SHORT %", "NEUTRAL %",
        "כיוון מוביל", "החלטת מנוע", "Units", "SL / TP", "פעולה"
    ]
    for c, h in zip(header_cols, headers):
        c.markdown(f"**{h}**")

    st.markdown(
        "<div style='height:1px;background:#e5e7eb;margin:2px 0 6px 0;'></div>",
        unsafe_allow_html=True,
    )

    for i, row in df.head(80).iterrows():
        ticker = normalize_ticker(row.get("ticker", ""))
        side = str(row.get("dominant_side", "")).upper()
        score = max(7, int(safe_float(row.get("score"), 0)))
        unit_mult = units_for_score(score, units)
        already_open = ticker in open_tickers

        can_enter = side in ["LONG", "SHORT"] and not already_open
        long_pct = safe_float(row.get("long_probability"), 0.0) * 100 if safe_float(row.get("long_probability"), 0.0) <= 1.5 else safe_float(row.get("long_probability"), 0.0)
        short_pct = safe_float(row.get("short_probability"), 0.0) * 100 if safe_float(row.get("short_probability"), 0.0) <= 1.5 else safe_float(row.get("short_probability"), 0.0)
        neutral_pct = safe_float(row.get("neutral_probability"), 0.0) * 100 if safe_float(row.get("neutral_probability"), 0.0) <= 1.5 else safe_float(row.get("neutral_probability"), 0.0)

        row_bg = "#f8fafc"
        if already_open:
            row_bg = "#ecfdf5"
        elif side == "LONG":
            row_bg = "#f0fdf4"
        elif side == "SHORT":
            row_bg = "#fff1f2"

        with st.container():
            st.markdown(
                f"<div style='background:{row_bg};border:1px solid #e5e7eb;border-radius:12px;padding:4px 8px;margin:3px 0;'>",
                unsafe_allow_html=True,
            )
            c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11 = st.columns(
                [0.75, 0.85, 0.75, 0.75, 0.75, 0.75, 0.9, 0.9, 0.75, 0.75, 0.95]
            )

            c1.markdown(f"**{ticker}**")
            c2.write(str(row.get("mode", "")))
            c3.write(fmt_price(row.get("entry")))
            c4.write(f"{long_pct:.1f}%")
            c5.write(f"{short_pct:.1f}%")
            c6.write(f"{neutral_pct:.1f}%")
            c7.write(side if side else "WAIT")
            c8.write(str(row.get("decision", row.get("engine_decision", ""))) or "WAIT")
            c9.write(f"{unit_mult:.2f}".rstrip("0").rstrip("."))
            c10.write(f"{fmt_price(row.get('stop'))} / {fmt_price(row.get('target'))}")

            if already_open:
                c11.button("פתוח", key=f"{key_prefix}_inline_open_{ticker}_{row.get('mode','')}_{i}", disabled=True, use_container_width=True)
            elif can_enter:
                if c11.button("🟢 כניסה", key=f"{key_prefix}_inline_entry_{ticker}_{row.get('mode','')}_{i}", use_container_width=True):
                    ok, msg = manual_enter_from_prediction(row.to_dict())
                    if ok:
                        st.success(msg)
                    else:
                        st.error(msg)
                    st.rerun()
            else:
                c11.button("אין כניסה", key=f"{key_prefix}_inline_noentry_{ticker}_{row.get('mode','')}_{i}", disabled=True, use_container_width=True)

            st.markdown("</div>", unsafe_allow_html=True)



def render_manual_telegram_settings_tab(rows=None):
    st.subheader("📨 טלגרם — כניסה ויציאה ידנית")

    st.markdown(
        """
<div class="card warn">
<strong>בדמו בלבד:</strong> הכפתורים רק שולחים הודעה לטלגרם ופותחים/סוגרים עסקת מעקב Paper בתוך התוכנה.
אין ביצוע קנייה או מכירה אמיתית.
</div>
""",
        unsafe_allow_html=True,
    )

    settings = load_alert_settings()
    c1, c2 = st.columns(2)
    with c1:
        telegram_enabled = st.checkbox("שלח הודעות Telegram", value=bool(settings.get("telegram_enabled", False)), key="manual_tg_enabled")
        bot_token = st.text_input("Telegram Bot Token", value=str(settings.get("telegram_bot_token", "")), type="password", key="manual_tg_token")
    with c2:
        chat_id = st.text_input("Telegram Chat ID", value=str(settings.get("telegram_chat_id", "")), key="manual_tg_chat")
        alerts_enabled = st.checkbox("שמור גם ב־Alerts log", value=bool(settings.get("alerts_enabled", True)), key="manual_alerts_enabled")

    a, b = st.columns(2)
    if a.button("💾 שמור הגדרות Telegram", use_container_width=True):
        safe = dict(DEFAULT_ALERT_SETTINGS)
        safe.update(settings or {})
        safe["telegram_enabled"] = bool(telegram_enabled)
        safe["alerts_enabled"] = bool(alerts_enabled)
        safe["telegram_bot_token"] = str(bot_token).strip()
        safe["telegram_chat_id"] = str(chat_id).strip()
        # Manual buttons ignore score threshold, but keep this low for regular alerts too.
        safe["send_only_score_at_least"] = min(int(safe.get("send_only_score_at_least", 9)), 1)
        save_alert_settings(safe)
        st.success("הגדרות Telegram נשמרו.")

    if b.button("📨 שלח בדיקת Telegram", use_container_width=True):
        test_msg = "✅ Telegram manual entry/exit test succeeded"
        ok, err = send_manual_telegram(test_msg)
        if ok:
            st.success("נשלחה הודעת בדיקה.")
        else:
            st.error(err)

    st.markdown("---")
    render_manual_tracked_trades_panel(compact=False, key_prefix='telegram')

    if rows:
        st.markdown("---")
        render_prediction_entry_buttons(rows, key_prefix='telegram')


def append_monitor_snapshots(rows):
    if not rows:
        return
    new_df = pd.DataFrame(rows)
    for col in SNAPSHOT_COLUMNS:
        if col not in new_df.columns:
            new_df[col] = ""
    new_df = new_df[SNAPSHOT_COLUMNS]
    if MONITOR_SNAPSHOTS_FILE.exists() and MONITOR_SNAPSHOTS_FILE.stat().st_size > 0:
        try:
            old = pd.read_csv(MONITOR_SNAPSHOTS_FILE)
        except Exception:
            old = pd.DataFrame(columns=SNAPSHOT_COLUMNS)
    else:
        old = pd.DataFrame(columns=SNAPSHOT_COLUMNS)
    merged = pd.concat([old, new_df], ignore_index=True)
    merged = merged.drop_duplicates(subset=["ticker", "mode", "bar_time"], keep="last").tail(25000)
    merged.to_csv(MONITOR_SNAPSHOTS_FILE, index=False)


def load_monitor_snapshots():
    if not MONITOR_SNAPSHOTS_FILE.exists() or MONITOR_SNAPSHOTS_FILE.stat().st_size == 0:
        return pd.DataFrame(columns=SNAPSHOT_COLUMNS)
    try:
        df = pd.read_csv(MONITOR_SNAPSHOTS_FILE)
    except Exception:
        return pd.DataFrame(columns=SNAPSHOT_COLUMNS)
    for col in SNAPSHOT_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    return df[SNAPSHOT_COLUMNS].copy()


def load_prediction_evaluations():
    if not MONITOR_EVALUATIONS_FILE.exists() or MONITOR_EVALUATIONS_FILE.stat().st_size == 0:
        return pd.DataFrame(columns=EVALUATION_COLUMNS)
    try:
        df = pd.read_csv(MONITOR_EVALUATIONS_FILE)
    except Exception:
        return pd.DataFrame(columns=EVALUATION_COLUMNS)
    for col in EVALUATION_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    return df[EVALUATION_COLUMNS].copy()


def save_prediction_evaluations(df):
    if df is None:
        df = pd.DataFrame(columns=EVALUATION_COLUMNS)
    for col in EVALUATION_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    df[EVALUATION_COLUMNS].tail(30000).to_csv(MONITOR_EVALUATIONS_FILE, index=False)


def _prediction_key(row):
    return f"{normalize_ticker(row.get('ticker'))}|{row.get('mode')}|{row.get('bar_time')}"


def _evaluate_prediction_row(row, full_df):
    side = str(row.get("engine_side", "WAIT"))
    if side not in ["LONG", "SHORT"]:
        return None
    bar_time = timestamp_to_ny(row.get("bar_time"))
    if bar_time is None or full_df is None or full_df.empty:
        return None
    horizon = 5 if str(row.get("mode")) == "מהירה" else 30
    future = full_df[(full_df.index > bar_time) & (full_df.index <= bar_time + pd.Timedelta(minutes=horizon))].copy()
    # Evaluate only after the full requested horizon is available.
    if future.empty or future.index[-1] < bar_time + pd.Timedelta(minutes=horizon - 1):
        return None
    future = future[future.index.date == bar_time.date()]
    if len(future) < max(3, horizon - 2):
        return None

    entry = safe_float(row.get("entry"), np.nan)
    stop = safe_float(row.get("stop"), np.nan)
    target = safe_float(row.get("target"), np.nan)
    if not np.isfinite(entry) or not np.isfinite(stop) or not np.isfinite(target):
        return None
    risk = abs(entry - stop)
    if risk <= 0:
        return None

    first_touch = "NONE"
    for _, bar in future.iterrows():
        high = safe_float(bar.get("high"), np.nan)
        low = safe_float(bar.get("low"), np.nan)
        if side == "LONG":
            hit_stop = np.isfinite(low) and low <= stop
            hit_target = np.isfinite(high) and high >= target
        else:
            hit_stop = np.isfinite(high) and high >= stop
            hit_target = np.isfinite(low) and low <= target
        # If both happened in one minute, use the conservative assumption.
        if hit_stop:
            first_touch = "STOP"
            break
        if hit_target:
            first_touch = "TARGET"
            break

    future_price = safe_float(future.iloc[-1]["close"], entry)
    signed_move = (future_price - entry) if side == "LONG" else (entry - future_price)
    terminal_r = signed_move / risk
    if first_touch == "TARGET":
        realized_r = abs(target - entry) / risk
        direction_result = "CORRECT"
        correct = True
    elif first_touch == "STOP":
        realized_r = -1.0
        direction_result = "WRONG"
        correct = False
    else:
        realized_r = float(terminal_r)
        if terminal_r > 0.10:
            direction_result, correct = "CORRECT", True
        elif terminal_r < -0.10:
            direction_result, correct = "WRONG", False
        else:
            direction_result, correct = "NEUTRAL", False

    if side == "LONG":
        mfe = max(0.0, (safe_float(future["high"].max(), entry) - entry) / risk)
        mae = max(0.0, (entry - safe_float(future["low"].min(), entry)) / risk)
    else:
        mfe = max(0.0, (entry - safe_float(future["low"].min(), entry)) / risk)
        mae = max(0.0, (safe_float(future["high"].max(), entry) - entry) / risk)

    return {
        "evaluation_id": str(uuid.uuid4()),
        "prediction_key": _prediction_key(row),
        "evaluated_at": now_ny_iso(),
        "scan_time": str(row.get("scan_time", "")), "bar_time": str(row.get("bar_time", "")),
        "ticker": normalize_ticker(row.get("ticker")), "mode": str(row.get("mode", "")),
        "side": side, "horizon_minutes": horizon,
        "entry": entry, "stop": stop, "target": target, "future_price": future_price,
        "first_touch": first_touch, "direction_result": direction_result,
        "correct": bool(correct), "realized_r": float(realized_r),
        "max_favorable_r": float(mfe), "max_adverse_r": float(mae),
        "long_probability": safe_float(row.get("long_probability"), 0.0),
        "short_probability": safe_float(row.get("short_probability"), 0.0),
        "neutral_probability": safe_float(row.get("neutral_probability"), 0.0),
        "confidence": safe_float(row.get("confidence"), 0.0),
        "mean_similarity": safe_float(row.get("mean_similarity"), 0.0),
        "status": str(row.get("status", "")),
        "note": f"נבדק לאחר {horizon} דקות; first touch={first_touch}",
    }


def evaluate_pending_predictions(max_predictions=300):
    snapshots = load_monitor_snapshots()
    if snapshots.empty:
        return 0
    evaluated = load_prediction_evaluations()
    existing = set(evaluated["prediction_key"].astype(str)) if not evaluated.empty else set()
    candidates = snapshots[
        snapshots["engine_side"].astype(str).isin(["LONG", "SHORT"])
        & snapshots["data_fresh"].astype(str).str.lower().isin(["true", "1"])
        & snapshots["market_window_ok"].astype(str).str.lower().isin(["true", "1"])
    ].copy()
    if candidates.empty:
        return 0
    candidates["prediction_key"] = candidates.apply(_prediction_key, axis=1)
    candidates = candidates[~candidates["prediction_key"].astype(str).isin(existing)]
    candidates = candidates.sort_values("bar_time").head(max_predictions)
    if candidates.empty:
        return 0

    new_rows = []
    for ticker, group in candidates.groupby(candidates["ticker"].astype(str)):
        try:
            full = monitor_fetch_1m(ticker)
        except Exception:
            continue
        for _, row in group.iterrows():
            result = _evaluate_prediction_row(row, full)
            if result is not None:
                new_rows.append(result)
    if new_rows:
        merged = pd.concat([evaluated, pd.DataFrame(new_rows)], ignore_index=True)
        merged = merged.drop_duplicates(subset=["prediction_key"], keep="last")
        save_prediction_evaluations(merged)
    return len(new_rows)


def evaluation_display_df():
    df = load_prediction_evaluations()
    if df.empty:
        return pd.DataFrame()
    for col in ["horizon_minutes", "entry", "stop", "target", "future_price", "realized_r",
                "max_favorable_r", "max_adverse_r", "long_probability", "short_probability",
                "neutral_probability", "confidence", "mean_similarity"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return pd.DataFrame({
        "זמן תחזית": df["bar_time"], "מניה": df["ticker"], "סוג": df["mode"],
        "כיוון": df["side"], "אופק דקות": df["horizon_minutes"],
        "LONG %": df["long_probability"] * 100, "SHORT %": df["short_probability"] * 100,
        "NEUTRAL %": df["neutral_probability"] * 100, "ביטחון %": df["confidence"] * 100,
        "דמיון %": df["mean_similarity"] * 100, "סטטוס תחזית": df["status"],
        "תוצאה": df["direction_result"], "פגיעה ראשונה": df["first_touch"], "R בפועל": df["realized_r"],
        "MFE R": df["max_favorable_r"], "MAE R": df["max_adverse_r"],
        "כניסה": df["entry"], "מחיר לאחר האופק": df["future_price"],
        "נבדק": df["evaluated_at"],
    }).sort_values("זמן תחזית", ascending=False)


@st.cache_data(show_spinner=False, ttl=20)
def monitor_fetch_1m(ticker, days=7):
    ticker = normalize_ticker(ticker)
    try:
        df = yf.download(
            ticker,
            period=f"{min(int(days), 7)}d",
            interval="1m",
            progress=False,
            auto_adjust=True,
            prepost=False,
            threads=False,
        )
    except Exception:
        return pd.DataFrame()
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [str(c[0]).lower() for c in df.columns]
    else:
        df.columns = [str(c).lower() for c in df.columns]
    required = ["open", "high", "low", "close", "volume"]
    if not all(col in df.columns for col in required):
        return pd.DataFrame()
    df = df[required].dropna().copy()
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC").tz_convert(NY_TZ)
    else:
        df.index = df.index.tz_convert(NY_TZ)
    return df.between_time("09:30", "16:00")


def monitor_market_time_ok(settings, mode):
    now = now_ny()
    if now.weekday() >= 5:
        return False, "סוף שבוע — מוצגות תחזיות, אך לא נפתחות עסקאות."
    start_h, start_m = parse_hhmm(settings.get("entry_start_time", "09:35"), "09:35")
    if str(mode) == "מהירה":
        end_text = settings.get("entry_end_time_fast", "15:55")
        default_end = "15:55"
    else:
        end_text = settings.get("entry_end_time_half", "15:30")
        default_end = "15:30"
    end_h, end_m = parse_hhmm(end_text, default_end)
    current = now.hour * 60 + now.minute
    start = start_h * 60 + start_m
    end = end_h * 60 + end_m
    if current < start or current > end:
        return False, f"מחוץ לחלון {str(mode)}: {start_h:02d}:{start_m:02d}–{end_h:02d}:{end_m:02d} ניו־יורק."
    return True, "שעות הכניסה פעילות."


def probability_score(probability, confidence):
    # 56% maps near 8/12; stronger probabilities gradually increase size.
    raw = 7.0 + max(0.0, float(probability) - 0.50) * 20.0 + max(0.0, float(confidence) - 0.50) * 4.0
    return int(np.clip(round(raw), 7, 12))


def mode_thresholds(mode, settings):
    if str(mode) == "מהירה":
        return (
            float(settings.get("min_probability_fast", 0.60)),
            float(settings.get("min_probability_gap_fast", 0.10)),
            float(settings.get("min_confidence_fast", 0.50)),
        )
    return (
        float(settings.get("min_probability_half", 0.56)),
        float(settings.get("min_probability_gap_half", 0.06)),
        float(settings.get("min_confidence_half", 0.48)),
    )



def technical_fallback_from_trend(trend, mode):
    """Create diagnostic probabilities from live technical scores.

    This is NOT the historical engineering model. It is used only so the monitor
    does not show 0%/0% while the pattern engine is still learning, has too few
    examples, or cannot find similar historical windows.
    """
    long_score = max(0.0, safe_float(trend.get("long_score", 0), 0))
    short_score = max(0.0, safe_float(trend.get("short_score", 0), 0))
    align_long = max(0.0, safe_float(trend.get("alignment_long", 0), 0))
    align_short = max(0.0, safe_float(trend.get("alignment_short", 0), 0))

    long_raw = long_score + 0.75 * align_long
    short_raw = short_score + 0.75 * align_short
    total = long_raw + short_raw

    if total <= 0:
        return {
            "side": "WAIT", "long_probability": 0.0, "short_probability": 0.0,
            "neutral_probability": 1.0, "confidence": 0.0,
            "reason": "fallback טכני: אין מספיק ניקוד חי/5 דקות."
        }

    raw_long = long_raw / total
    raw_short = short_raw / total
    gap = abs(raw_long - raw_short)
    neutral = float(np.clip(0.45 - gap * 0.55, 0.10, 0.65))
    non_neutral = 1.0 - neutral
    long_p = raw_long * non_neutral
    short_p = raw_short * non_neutral

    side = "LONG" if long_p > short_p else "SHORT" if short_p > long_p else "WAIT"
    confidence = float(np.clip(0.35 + gap * 0.45 + max(align_long, align_short) / 4.0 * 0.20, 0.0, 0.82))

    return {
        "side": side,
        "long_probability": float(long_p),
        "short_probability": float(short_p),
        "neutral_probability": float(neutral),
        "confidence": confidence,
        "reason": (
            f"fallback טכני לתצוגה בלבד: LONG score {long_score:.0f}, SHORT score {short_score:.0f}, "
            f"5m LONG {align_long:.0f}, 5m SHORT {align_short:.0f}."
        ),
    }


def monitor_placeholder_row(ticker, mode, status, reason, scan_id="", scan_time=None, bar_time=""):
    """Return a complete row so the table never breaks or hides the real reason."""
    return {
        "scan_id": scan_id, "scan_time": scan_time or now_ny_iso(), "bar_time": str(bar_time or ""),
        "data_age_minutes": np.nan, "data_fresh": False, "market_window_ok": False,
        "ticker": normalize_ticker(ticker), "mode": mode, "last_price": np.nan,
        "long_probability": 0.0, "short_probability": 0.0, "neutral_probability": 1.0,
        "probability_leader": "WAIT", "engine_side": "WAIT", "dominant_side": "WAIT",
        "dominant_probability": 0.0, "probability_gap": 0.0, "confidence": 0.0,
        "technical_long_score": 0, "technical_short_score": 0,
        "trend_side": "WAIT", "alignment_5m": 0, "alignment_5m_opposite": 0,
        "trend_confirmed": False, "sample_count": 0, "mean_similarity": 0.0,
        "best_similarity": 0.0, "weakest_similarity": 0.0,
        "long_target_rate": 0.0, "short_target_rate": 0.0,
        "long_expectancy_r": 0.0, "short_expectancy_r": 0.0,
        "expected_mfe_r": 0.0, "expected_mae_r": 0.0,
        "pattern_state": "UNKNOWN", "function_model": "UNKNOWN",
        "ready": False, "eligible": False, "trade_allowed": False,
        "status": status, "reason": reason,
        "entry": np.nan, "stop": np.nan, "target": np.nan, "score": 0,
    }


def realtime_trend_confirmation(indicators, full_df, mode, settings):
    """Classify the current live direction. Counter-trend entries are blocked.

    The pattern engine answers: what tended to happen after similar historical shapes?
    This function answers: what direction is the market actually moving now?
    A trade is allowed only when both answers agree.
    """
    if indicators is None or indicators.empty:
        return {
            "side": "WAIT", "long_score": 0, "short_score": 0,
            "alignment_long": 0, "alignment_short": 0,
            "slope": 0.0, "reason": "אין מספיק אינדיקטורים למגמה בזמן אמת.",
        }

    d = indicators.dropna(subset=["close"]).copy()
    last = d.iloc[-1]
    close = safe_float(last.get("close"), np.nan)
    vwap = safe_float(last.get("vwap"), close)

    if str(mode) == "מהירה":
        long_score, long_reasons = score_side_fast(d, "LONG")
        short_score, short_reasons = score_side_fast(d, "SHORT")
        slope = linear_slope_per_bar(d["close"], lookback=6)
        min_technical = int(settings.get("trend_min_technical_fast", 5))
        min_5m = int(settings.get("trend_min_5m_fast", 1))
        long_structure = (
            close > safe_float(last.get("ema5"), close)
            and safe_float(last.get("ema3"), close) >= safe_float(last.get("ema5"), close)
            and slope > 0
            and close >= vwap
        )
        short_structure = (
            close < safe_float(last.get("ema5"), close)
            and safe_float(last.get("ema3"), close) <= safe_float(last.get("ema5"), close)
            and slope < 0
            and close <= vwap
        )
    else:
        long_score, long_reasons = score_side_half(d, "LONG")
        short_score, short_reasons = score_side_half(d, "SHORT")
        slope = linear_slope_per_bar(d["close"], lookback=12)
        min_technical = int(settings.get("trend_min_technical_half", 6))
        min_5m = int(settings.get("trend_min_5m_half", 2))
        long_structure = (
            close > safe_float(last.get("ema9"), close) > safe_float(last.get("ema21"), close)
            and slope > 0
            and close >= vwap
        )
        short_structure = (
            close < safe_float(last.get("ema9"), close) < safe_float(last.get("ema21"), close)
            and slope < 0
            and close <= vwap
        )

    alignment_long, alignment_long_reason = timeframe_alignment_score(full_df, "LONG")
    alignment_short, alignment_short_reason = timeframe_alignment_score(full_df, "SHORT")

    long_ok = bool(
        long_structure
        and long_score >= min_technical
        and long_score >= short_score + 1
        and alignment_long >= min_5m
    )
    short_ok = bool(
        short_structure
        and short_score >= min_technical
        and short_score >= long_score + 1
        and alignment_short >= min_5m
    )

    if long_ok and not short_ok:
        side = "LONG"
        reason = (
            f"מגמה חיה LONG: טכני {long_score}/{short_score}, "
            f"שיפוע {slope:+.4f}, 5 דקות {alignment_long}/4. "
            f"{', '.join(long_reasons[:4])}"
        )
    elif short_ok and not long_ok:
        side = "SHORT"
        reason = (
            f"מגמה חיה SHORT: טכני {short_score}/{long_score}, "
            f"שיפוע {slope:+.4f}, 5 דקות {alignment_short}/4. "
            f"{', '.join(short_reasons[:4])}"
        )
    else:
        side = "WAIT"
        reason = (
            f"אין מגמה חיה מאושרת: LONG טכני {long_score}, 5ד׳ {alignment_long}/4; "
            f"SHORT טכני {short_score}, 5ד׳ {alignment_short}/4; שיפוע {slope:+.4f}."
        )

    return {
        "side": side,
        "long_score": int(long_score),
        "short_score": int(short_score),
        "alignment_long": int(alignment_long),
        "alignment_short": int(alignment_short),
        "slope": float(slope),
        "reason": reason,
        "long_alignment_reason": alignment_long_reason,
        "short_alignment_reason": alignment_short_reason,
    }

def analyze_monitor_ticker(ticker, modes, settings, scan_id):
    ticker = normalize_ticker(ticker)
    full = monitor_fetch_1m(ticker)
    scan_time = now_ny_iso()
    if full.empty:
        return [monitor_placeholder_row(ticker, mode, "NO_DATA", "אין נתוני Yahoo / yfinance לא החזיר נתונים.", scan_id, scan_time) for mode in modes]

    session = latest_session(full)
    if session.empty:
        return [monitor_placeholder_row(ticker, mode, "NO_SESSION", "אין נתוני מסחר רגיל להיום.", scan_id, scan_time, str(full.index[-1])) for mode in modes]

    indicators = add_indicators(session).dropna(subset=["close"])
    if indicators.empty:
        return []
    last = indicators.iloc[-1]
    bar_time = indicators.index[-1]
    last_price = safe_float(last["close"])
    fresh, minute_gap = minute_data_fresh(bar_time, allowed_lag_minutes=1)
    data_age = float(minute_gap) if minute_gap >= 0 else np.nan
    results = []
    for mode in modes:
        market_ok, market_reason = monitor_market_time_ok(settings, mode)
        stop_r, target_r = ((0.90, 1.20) if str(mode) == "מהירה" else (1.00, 1.50))
        try:
            eng = engineering_pattern_analysis(full, mode, current_end=bar_time, stop_r=stop_r, target_r=target_r)
        except Exception as exc:
            eng = {
                "ready": False, "predicted_side": "WAIT", "confidence": 0.0,
                "sample_count": 0, "mean_similarity": 0.0,
                "best_similarity": 0.0, "weakest_similarity": 0.0,
                "long_probability": 0.0, "short_probability": 0.0, "neutral_probability": 1.0,
                "long_target_rate": 0.0, "short_target_rate": 0.0,
                "long_expectancy_r": 0.0, "short_expectancy_r": 0.0,
                "expected_mfe_r": 0.0, "expected_mae_r": 0.0,
                "pattern_state": "ERROR", "function_model": "ERROR",
                "reason": f"שגיאת מנוע: {str(exc)[:160]}",
            }

        long_p = float(eng.get("long_probability", 0.0))
        short_p = float(eng.get("short_probability", 0.0))
        neutral_p = float(eng.get("neutral_probability", max(0.0, 1.0 - long_p - short_p)))
        probability_leader = "LONG" if long_p >= short_p else "SHORT"
        engine_side = str(eng.get("predicted_side", "WAIT"))
        if engine_side not in ["LONG", "SHORT"]:
            engine_side = "WAIT"

        # The displayed/ tradable direction is the engine decision, not merely the larger raw percentage.
        dominant_side = engine_side
        dominant_probability = (
            long_p if dominant_side == "LONG" else short_p if dominant_side == "SHORT" else max(long_p, short_p)
        )
        probability_gap = abs(long_p - short_p)
        confidence = float(eng.get("confidence", 0.0))
        samples = int(eng.get("sample_count", 0))
        similarity = float(eng.get("mean_similarity", 0.0))
        best_similarity = float(eng.get("best_similarity", similarity))
        weakest_similarity = float(eng.get("weakest_similarity", similarity))
        min_prob, min_gap, min_conf = mode_thresholds(mode, settings)

        trend = realtime_trend_confirmation(indicators, full, mode, settings)
        trend_side = str(trend.get("side", "WAIT"))
        trend_confirmed = bool(engine_side in ["LONG", "SHORT"] and trend_side == engine_side)
        alignment_5m = int(
            trend.get("alignment_long", 0) if engine_side == "LONG"
            else trend.get("alignment_short", 0) if engine_side == "SHORT" else 0
        )
        alignment_5m_opposite = int(
            trend.get("alignment_short", 0) if engine_side == "LONG"
            else trend.get("alignment_long", 0) if engine_side == "SHORT" else 0
        )
        technical_long_score = int(trend.get("long_score", 0))
        technical_short_score = int(trend.get("short_score", 0))

        fallback_used = False
        fallback_reason = ""
        raw_engine_ready = bool(eng.get("ready", False))
        raw_prob_zero = (abs(long_p) < 1e-9 and abs(short_p) < 1e-9)
        if bool(settings.get("display_technical_fallback", True)) and (not raw_engine_ready or raw_prob_zero):
            fb = technical_fallback_from_trend(trend, mode)
            if fb.get("side") in ["LONG", "SHORT"]:
                fallback_used = True
                fallback_reason = fb.get("reason", "")
                long_p = float(fb.get("long_probability", 0.0))
                short_p = float(fb.get("short_probability", 0.0))
                neutral_p = float(fb.get("neutral_probability", 0.0))
                probability_leader = "LONG" if long_p >= short_p else "SHORT"
                engine_side = str(fb.get("side", "WAIT"))
                dominant_side = engine_side
                dominant_probability = long_p if dominant_side == "LONG" else short_p
                probability_gap = abs(long_p - short_p)
                confidence = max(confidence, float(fb.get("confidence", 0.0)))

        expectancy = float(
            eng.get("long_expectancy_r", 0.0) if engine_side == "LONG"
            else eng.get("short_expectancy_r", 0.0) if engine_side == "SHORT" else 0.0
        )
        opposite_expectancy = float(
            eng.get("short_expectancy_r", 0.0) if engine_side == "LONG"
            else eng.get("long_expectancy_r", 0.0) if engine_side == "SHORT" else 0.0
        )
        min_expectancy = float(settings.get("min_expectancy_r", 0.10))
        min_expectancy_gap = float(settings.get("min_expectancy_gap_r", 0.10))
        max_neutral_probability = float(settings.get("max_neutral_probability", 0.45))
        min_best_similarity = float(settings.get("min_best_similarity", 0.44))

        engine_ready_for_trade = bool(eng.get("ready", False)) or (fallback_used and bool(settings.get("allow_technical_fallback_trades", False)))
        eligible = bool(
            engine_ready_for_trade
            and engine_side in ["LONG", "SHORT"]
            and samples >= int(settings.get("min_samples", 6))
            and similarity >= float(settings.get("min_similarity", 0.34))
            and best_similarity >= min_best_similarity
            and neutral_p <= max_neutral_probability
            and dominant_probability >= min_prob
            and probability_gap >= min_gap
            and confidence >= min_conf
            and expectancy >= min_expectancy
            and expectancy >= opposite_expectancy + min_expectancy_gap
            and trend_confirmed
        )
        trade_allowed = bool(eligible and market_ok and fresh)

        if fallback_used and not bool(eng.get("ready", False)):
            status = "TECH_FALLBACK"
        elif not eng.get("ready", False):
            status = "LEARNING"
        elif engine_side == "WAIT":
            status = "ENGINE_WAIT"
        elif not trend_confirmed:
            status = "COUNTER_TREND"
        elif neutral_p > max_neutral_probability:
            status = "TOO_NEUTRAL"
        elif best_similarity < min_best_similarity or similarity < float(settings.get("min_similarity", 0.34)):
            status = "WEAK_MATCH"
        elif expectancy < min_expectancy or expectancy < opposite_expectancy + min_expectancy_gap:
            status = "NO_EXPECTANCY"
        elif not market_ok:
            status = "OUT_OF_WINDOW"
        elif not fresh:
            status = "STALE_DATA"
        elif eligible and trade_allowed:
            status = "ENTRY_CANDIDATE"
        elif eligible:
            status = "SIGNAL_ONLY"
        else:
            status = "WATCH"

        plan_side = engine_side if engine_side in ["LONG", "SHORT"] else probability_leader
        plan = chart_based_stop_target(indicators, plan_side, mode)
        score = probability_score(dominant_probability, confidence) if engine_side in ["LONG", "SHORT"] else 0
        reason = (
            f"אחוזים: LONG {long_p*100:.1f}% | SHORT {short_p*100:.1f}% | NEUTRAL {neutral_p*100:.1f}% | "
            f"מוביל באחוזים {probability_leader}; החלטת מנוע {engine_side}. "
            f"יתרון {probability_gap*100:.1f}% | ביטחון {confidence*100:.1f}% | "
            f"דמיון ממוצע {similarity*100:.1f}% ומיטבי {best_similarity*100:.1f}% | "
            f"תוחלת הכיוון {expectancy:+.2f}R מול {opposite_expectancy:+.2f}R | "
            f"מגמה בזמן אמת {trend_side}. {trend.get('reason', '')} | {eng.get('reason', '')}"
        )
        if fallback_used:
            reason += " | " + fallback_reason + " | חשוב: זה fallback לתצוגה/אבחון; לא מנוע תבניות היסטורי מלא."
        if engine_side == "WAIT":
            reason += " | נחסם: אחוז גבוה לבדו אינו איתות; מנוע התבניות לא אישר עסקה."
        elif not trend_confirmed:
            reason += f" | נחסם נגד מגמה: המנוע חזה {engine_side}, אך הגרף החי מסווג {trend_side}."
        if not fresh:
            if np.isfinite(data_age):
                reason += (
                    f" | הנתון האחרון ישן ב־{data_age:.0f} דקות; "
                    "לא תיפתח עסקה ללא נר מהדקה הנוכחית או הקודמת."
                )
            else:
                reason += " | זמן הנר אינו תקין; המסחר נחסם."
        if not market_ok:
            reason += f" | {market_reason}"

        results.append({
            "scan_id": scan_id, "scan_time": scan_time, "bar_time": str(bar_time),
            "data_age_minutes": data_age, "data_fresh": bool(fresh), "market_window_ok": bool(market_ok),
            "ticker": ticker, "mode": mode, "last_price": last_price,
            "long_probability": long_p, "short_probability": short_p,
            "neutral_probability": neutral_p,
            "probability_leader": probability_leader, "engine_side": engine_side,
            "dominant_side": dominant_side, "dominant_probability": dominant_probability,
            "probability_gap": probability_gap, "confidence": confidence,
            "technical_long_score": technical_long_score,
            "technical_short_score": technical_short_score,
            "trend_side": trend_side, "alignment_5m": alignment_5m,
            "alignment_5m_opposite": alignment_5m_opposite,
            "trend_confirmed": trend_confirmed,
            "sample_count": samples, "mean_similarity": similarity,
            "best_similarity": best_similarity, "weakest_similarity": weakest_similarity,
            "long_target_rate": float(eng.get("long_target_rate", 0.0)),
            "short_target_rate": float(eng.get("short_target_rate", 0.0)),
            "long_expectancy_r": float(eng.get("long_expectancy_r", 0.0)),
            "short_expectancy_r": float(eng.get("short_expectancy_r", 0.0)),
            "expected_mfe_r": float(eng.get("expected_mfe_r", 0.0)),
            "expected_mae_r": float(eng.get("expected_mae_r", 0.0)),
            "pattern_state": str(eng.get("pattern_state", "UNKNOWN")),
            "function_model": str(eng.get("function_model", "UNKNOWN")),
            "ready": bool(eng.get("ready", False)), "eligible": eligible,
            "trade_allowed": trade_allowed, "status": status, "reason": reason,
            "entry": last_price, "stop": safe_float(plan.get("stop")),
            "target": safe_float(plan.get("target")), "score": score,
        })
    return results


def scan_all_tickers_concurrently(tickers, modes, settings):
    scan_id = str(uuid.uuid4())
    rows = []
    tickers = [normalize_ticker(t) for t in tickers if normalize_ticker(t)]
    max_workers = max(1, min(int(settings.get("max_workers", 6)), 10, len(tickers) or 1))
    progress = st.progress(0.0, text="סורק את כל המניות במקביל...")
    completed = 0
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(analyze_monitor_ticker, ticker, modes, settings, scan_id): ticker
            for ticker in tickers
        }
        for future in as_completed(futures):
            ticker = futures[future]
            try:
                rows.extend(future.result())
            except Exception as exc:
                for mode in modes:
                    rows.append({
                        "scan_id": scan_id, "scan_time": now_ny_iso(), "bar_time": "",
                        "data_age_minutes": np.nan, "data_fresh": False, "market_window_ok": False,
                        "ticker": ticker, "mode": mode, "status": "ERROR",
                        "reason": str(exc)[:180], "ready": False, "eligible": False,
                        "trade_allowed": False,
                    })
            completed += 1
            progress.progress(completed / max(len(tickers), 1), text=f"נסרקו {completed}/{len(tickers)} מניות")
    progress.empty()
    return rows, scan_id


def signal_streak(row, previous_snapshots):
    required_key = (str(row.get("ticker")), str(row.get("mode")), str(row.get("dominant_side")))
    if previous_snapshots is None or previous_snapshots.empty:
        return 1
    subset = previous_snapshots[
        previous_snapshots["ticker"].astype(str).eq(required_key[0])
        & previous_snapshots["mode"].astype(str).eq(required_key[1])
    ].copy()
    if subset.empty:
        return 1
    subset = subset.sort_values("scan_time").tail(5)
    streak = 1
    for side in reversed(subset["dominant_side"].astype(str).tolist()):
        if side == required_key[2]:
            streak += 1
        else:
            break
    return streak


def build_monitor_signal(row):
    return {
        "signal": str(row["dominant_side"]),
        "ticker": normalize_ticker(row["ticker"]),
        "mode": str(row["mode"]),
        "score": int(row["score"]),
        "entry": float(row["entry"]),
        "stop": float(row["stop"]),
        "target": float(row["target"]),
        "reason": str(row["reason"]),
    }


def open_monitor_candidates(rows, settings, previous_snapshots):
    if not bool(settings.get("auto_trade_enabled", False)):
        return ["מסחר Paper אוטומטי כבוי — התחזיות בלבד מתעדכנות."]
    if not rows:
        return ["אין תוצאות סריקה."]

    rules = load_rules()
    rules["max_open_trades"] = int(settings.get("max_open_trades", 8))
    save_rules(rules)
    trades = load_trades()
    open_tickers = set(trades.loc[trades["status"].astype(str).eq("OPEN"), "ticker"].astype(str)) if not trades.empty else set()
    required_streak = max(1, int(settings.get("consecutive_scans_required", 1)))

    candidates = []
    for row in rows:
        if not bool(row.get("trade_allowed", False)):
            continue
        if normalize_ticker(row.get("ticker")) in open_tickers:
            continue
        streak = signal_streak(row, previous_snapshots)
        if streak < required_streak:
            continue
        candidates.append((
            float(row.get("dominant_probability", 0.0)),
            float(row.get("probability_gap", 0.0)),
            float(row.get("confidence", 0.0)),
            float(row.get("mean_similarity", 0.0)),
            row,
        ))

    # Only one mode per ticker: keep the stronger prediction.
    best_by_ticker = {}
    for item in sorted(candidates, reverse=True, key=lambda x: x[:4]):
        ticker = normalize_ticker(item[4].get("ticker"))
        if ticker not in best_by_ticker:
            best_by_ticker[ticker] = item
    candidates = sorted(best_by_ticker.values(), reverse=True, key=lambda x: x[:4])

    messages = []
    max_new = max(0, int(settings.get("max_new_trades_per_scan", 3)))
    opened = 0
    for _, _, _, _, row in candidates:
        if opened >= max_new:
            break
        before = load_trades()
        before_ids = set(before["trade_id"].astype(str)) if not before.empty else set()
        ok, message = open_trade(build_monitor_signal(row), min_score=7)
        messages.append(message)
        if not ok:
            continue
        opened += 1
        after = load_trades()
        new_rows = after[~after["trade_id"].astype(str).isin(before_ids)].copy()
        if new_rows.empty:
            continue
        new_trade = new_rows.sort_values("entry_time").iloc[-1]
        append_monitor_metadata({
            "trade_id": str(new_trade["trade_id"]), "created_at": now_ny_iso(),
            "ticker": str(row["ticker"]), "mode": str(row["mode"]),
            "side": str(row["dominant_side"]),
            "long_probability": float(row["long_probability"]),
            "short_probability": float(row["short_probability"]),
            "neutral_probability": float(row.get("neutral_probability", 0.0)),
            "dominant_probability": float(row["dominant_probability"]),
            "probability_gap": float(row["probability_gap"]),
            "confidence": float(row["confidence"]),
            "sample_count": int(row["sample_count"]),
            "mean_similarity": float(row["mean_similarity"]),
            "long_expectancy_r": float(row["long_expectancy_r"]),
            "short_expectancy_r": float(row["short_expectancy_r"]),
            "pattern_state": str(row["pattern_state"]),
            "function_model": str(row["function_model"]),
            "scan_id": str(row["scan_id"]),
        })
    if opened == 0 and not messages:
        messages.append("לא הייתה תחזית שעברה את ספי הכניסה בסריקה זו.")
    elif opened:
        messages.append(f"נפתחו {opened} עסקאות Paper לפי אחוזי מנוע התבניות.")
    return messages


def prediction_display_df(rows):
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).copy()
    defaults = {
        "ticker": "", "mode": "", "last_price": np.nan,
        "long_probability": 0.0, "short_probability": 0.0, "neutral_probability": 1.0,
        "dominant_probability": 0.0, "probability_gap": 0.0, "confidence": 0.0,
        "mean_similarity": 0.0, "best_similarity": 0.0, "weakest_similarity": 0.0,
        "long_target_rate": 0.0, "short_target_rate": 0.0,
        "long_expectancy_r": 0.0, "short_expectancy_r": 0.0,
        "expected_mfe_r": 0.0, "expected_mae_r": 0.0,
        "entry": np.nan, "stop": np.nan, "target": np.nan, "score": 0,
        "technical_long_score": 0, "technical_short_score": 0,
        "alignment_5m": 0, "alignment_5m_opposite": 0, "data_age_minutes": np.nan,
        "probability_leader": "WAIT", "engine_side": "WAIT", "trend_side": "WAIT",
        "trend_confirmed": False, "dominant_side": "WAIT",
        "sample_count": 0, "pattern_state": "UNKNOWN", "function_model": "UNKNOWN",
        "status": "UNKNOWN", "bar_time": "", "reason": "",
    }
    for col, value in defaults.items():
        if col not in df.columns:
            df[col] = value
    numeric_cols = [
        "last_price", "long_probability", "short_probability", "neutral_probability", "dominant_probability",
        "probability_gap", "confidence", "mean_similarity", "best_similarity", "weakest_similarity", "long_target_rate",
        "short_target_rate", "long_expectancy_r", "short_expectancy_r",
        "expected_mfe_r", "expected_mae_r", "entry", "stop", "target", "score",
        "technical_long_score", "technical_short_score", "alignment_5m",
        "alignment_5m_opposite", "data_age_minutes",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    freshness_series = (
        df["data_fresh"] if "data_fresh" in df.columns
        else pd.Series(False, index=df.index)
    )
    age_series = (
        pd.to_numeric(df["data_age_minutes"], errors="coerce")
        if "data_age_minutes" in df.columns
        else pd.Series(np.nan, index=df.index)
    )
    probability_leader_series = df["probability_leader"] if "probability_leader" in df.columns else pd.Series("", index=df.index)
    engine_side_series = df["engine_side"] if "engine_side" in df.columns else pd.Series("WAIT", index=df.index)
    trend_side_series = df["trend_side"] if "trend_side" in df.columns else pd.Series("WAIT", index=df.index)
    trend_confirmed_series = df["trend_confirmed"] if "trend_confirmed" in df.columns else pd.Series(False, index=df.index)
    technical_long_series = pd.to_numeric(df["technical_long_score"], errors="coerce") if "technical_long_score" in df.columns else pd.Series(0, index=df.index)
    technical_short_series = pd.to_numeric(df["technical_short_score"], errors="coerce") if "technical_short_score" in df.columns else pd.Series(0, index=df.index)
    alignment_5m_series = pd.to_numeric(df["alignment_5m"], errors="coerce") if "alignment_5m" in df.columns else pd.Series(0, index=df.index)

    out = pd.DataFrame({
        "מניה": df["ticker"],
        "סוג עסקה": df["mode"],
        "מחיר": df["last_price"],
        "LONG %": df["long_probability"] * 100,
        "SHORT %": df["short_probability"] * 100,
        "NEUTRAL %": pd.to_numeric(df.get("neutral_probability"), errors="coerce").fillna(0) * 100,
        "מוביל באחוזים": probability_leader_series,
        "החלטת מנוע": engine_side_series,
        "מגמה בזמן אמת": trend_side_series,
        "כיוון מוביל": df["dominant_side"] if "dominant_side" in df.columns else engine_side_series,
        "אישור מגמה": trend_confirmed_series.map(lambda x: "כן" if bool(x) else "לא"),
        "ניקוד טכני LONG": technical_long_series,
        "ניקוד טכני SHORT": technical_short_series,
        "התאמת 5 דקות": alignment_5m_series,
        "יתרון %": df["probability_gap"] * 100,
        "ביטחון %": df["confidence"] * 100,
        "דמיון %": df["mean_similarity"] * 100,
        "דמיון מיטבי %": pd.to_numeric(df.get("best_similarity"), errors="coerce").fillna(0) * 100,
        "דוגמאות": df["sample_count"],
        "תוחלת LONG R": df["long_expectancy_r"],
        "תוחלת SHORT R": df["short_expectancy_r"],
        "מצב תבנית": df["pattern_state"],
        "מודל": df["function_model"],
        "סטטוס": df["status"],
        "כניסה": df["entry"],
        "סטופ": df["stop"],
        "יעד": df["target"],
        "גיל נתון (דק׳)": age_series,
        "רעננות": freshness_series.map(lambda x: "LIVE" if bool(x) else "STALE"),
        "עדכון": df["bar_time"],
        "הסבר": df["reason"],
    })
    return out.sort_values(["ביטחון %", "יתרון %"], ascending=False)


def style_prediction_rows(row):
    side = clean_bidi_text(row.get("כיוון מוביל", ""))
    status = clean_bidi_text(row.get("סטטוס", ""))
    if status == "ENTRY_CANDIDATE" and side == "LONG":
        return ["background-color: #dcfce7; color: #064e3b"] * len(row)
    if status == "ENTRY_CANDIDATE" and side == "SHORT":
        return ["background-color: #fee2e2; color: #7f1d1d"] * len(row)
    return [""] * len(row)


def trades_display_df():
    trades = load_trades()
    if trades.empty:
        return pd.DataFrame()
    meta = load_monitor_metadata()
    df = trades.merge(meta, on="trade_id", how="left", suffixes=("", "_meta"))
    out = pd.DataFrame({
        "סטטוס": df["status"], "מניה": df["ticker"], "סוג": df["mode"],
        "כיוון": df["side"], "LONG % בכניסה": pd.to_numeric(df.get("long_probability"), errors="coerce") * 100,
        "SHORT % בכניסה": pd.to_numeric(df.get("short_probability"), errors="coerce") * 100,
        "ביטחון %": pd.to_numeric(df.get("confidence"), errors="coerce") * 100,
        "מחיר כניסה": pd.to_numeric(df["entry_price"], errors="coerce"),
        "מחיר נוכחי": pd.to_numeric(df["current_price"], errors="coerce"),
        "סטופ": pd.to_numeric(df["stop_loss"], errors="coerce"),
        "יעד": pd.to_numeric(df["target_reference"], errors="coerce"),
        "כמות": pd.to_numeric(df["quantity"], errors="coerce"),
        "רווח ברוטו $": pd.to_numeric(df["gross_pnl"], errors="coerce"),
        "עלויות $": pd.to_numeric(df["total_cost"], errors="coerce"),
        "רווח נטו $": pd.to_numeric(df["net_pnl"], errors="coerce"),
        "רווח נטו %": pd.to_numeric(df["net_pnl_pct"], errors="coerce"),
        "שיא רווח $": pd.to_numeric(df["max_net_pnl_seen"], errors="coerce"),
        "זמן כניסה": df["entry_time"], "זמן יציאה": df["exit_time"],
        "משך דקות": pd.to_numeric(df["duration_minutes"], errors="coerce"),
        "סיבת יציאה": df["exit_reason_he"],
        "מצב תבנית": df.get("pattern_state", ""), "מודל": df.get("function_model", ""),
    })
    status_order = out["סטטוס"].astype(str).map({"OPEN": 0, "CLOSED": 1}).fillna(2)
    out = out.assign(_order=status_order).sort_values(["_order", "זמן כניסה"], ascending=[True, False]).drop(columns="_order")
    return out


def style_trade_rows(row):
    pnl = safe_float(row.get("רווח נטו $"), 0.0)
    if pnl > 0:
        return ["background-color: #dcfce7; color: #064e3b"] * len(row)
    if pnl < 0:
        return ["background-color: #fee2e2; color: #7f1d1d"] * len(row)
    return ["background-color: #f9fafb"] * len(row)


def render_monitor_metrics(rows):
    trades = load_trades()
    open_df = trades[trades["status"].astype(str).eq("OPEN")] if not trades.empty else trades
    closed_df = trades[trades["status"].astype(str).eq("CLOSED")] if not trades.empty else trades
    total_net = float(pd.to_numeric(trades["net_pnl"], errors="coerce").fillna(0).sum()) if not trades.empty else 0.0
    eligible = sum(bool(r.get("trade_allowed", False)) for r in rows)
    long_leads = sum(str(r.get("dominant_side")) == "LONG" for r in rows)
    short_leads = sum(str(r.get("dominant_side")) == "SHORT" for r in rows)
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("תחזיות פעילות", len(rows))
    c2.metric("מועמדות לכניסה", eligible)
    c3.metric("LONG מוביל", long_leads)
    c4.metric("SHORT מוביל", short_leads)
    c5.metric("עסקאות פתוחות", len(open_df))
    c6.metric("P/L נטו", f"${total_net:.2f}")
    if rows:
        status_counts = Counter(str(r.get("status", "UNKNOWN")) for r in rows)
        st.caption("סטטוסים בסריקה: " + " | ".join(f"{k}: {v}" for k, v in status_counts.most_common()))


ensure_monitor_files()

st.markdown(
    """
<div class="title-box">
  <h1>📡 מנוע תבניות — ניטור והשקעה בזמן אמת</h1>
  <p>סורק במקביל את כל המניות, מחשב אחוז LONG ואחוז SHORT בשני אופקי זמן, ויכול לפתוח עסקאות Paper לפי האחוז הגבוה יותר.</p>
</div>
""",
    unsafe_allow_html=True,
)

st.warning("האפליקציה פותחת עסקאות Paper בלבד. אין בה חיבור לברוקר ואין להשתמש בה לכסף אמיתי לפני בדיקות ממושכות.")
st.info(
    "⏱️ מצב רעננות מחמיר פעיל: עסקה תותר רק כאשר הנר האחרון הוא מהדקה הנוכחית או הקודמת. "
    "ב־V1.8 הטבלה עדיין תציג fallback טכני כדי לא לקבל 0% סתם, אבל fallback לא פותח עסקאות אלא אם הפעלת זאת ידנית."
)

settings = load_monitor_settings()
all_tickers = load_tickers()

with st.sidebar:
    st.header("הגדרות ניטור")
    selected_tickers = st.multiselect(
        "מניות לסריקה במקביל",
        options=all_tickers,
        default=all_tickers,
    )
    col_mode1, col_mode2 = st.columns(2)
    use_fast = col_mode1.checkbox("מהירה", value="מהירה" in settings.get("modes", []))
    use_half = col_mode2.checkbox("חצי שעה", value="חצי שעה" in settings.get("modes", []))
    modes = (["מהירה"] if use_fast else []) + (["חצי שעה"] if use_half else [])
    auto_trade = st.toggle("פתח עסקאות Paper אוטומטית", value=bool(settings.get("auto_trade_enabled", False)))
    auto_refresh = st.toggle("רענון אוטומטי", value=True)
    refresh_seconds = st.select_slider("רענון כל", options=[30, 60, 90, 120], value=int(settings.get("refresh_seconds", 60)), format_func=lambda x: f"{x} שניות")
    max_workers = st.slider("כמות סריקות מקבילות", 2, 10, int(settings.get("max_workers", 6)))
    max_new = st.slider("עסקאות חדשות בסריקה", 1, 6, int(settings.get("max_new_trades_per_scan", 3)))
    max_open = st.slider("מקסימום עסקאות פתוחות", 2, 15, int(settings.get("max_open_trades", 8)))
    consecutive = st.slider("מספר סריקות רצופות לאישור", 1, 3, int(settings.get("consecutive_scans_required", 1)))

    with st.expander("ספי אחוזים — עסקה מהירה", expanded=False):
        fast_prob = st.slider("אחוז כיוון מינימלי", 50, 80, int(float(settings.get("min_probability_fast", 0.60)) * 100), key="fast_prob") / 100
        fast_gap = st.slider("פער מינימלי בין LONG ל־SHORT", 0, 30, int(float(settings.get("min_probability_gap_fast", 0.10)) * 100), key="fast_gap") / 100
        fast_conf = st.slider("ביטחון מינימלי", 30, 80, int(float(settings.get("min_confidence_fast", 0.50)) * 100), key="fast_conf") / 100

    with st.expander("ספי אחוזים — חצי שעה", expanded=False):
        half_prob = st.slider("אחוז כיוון מינימלי", 50, 80, int(float(settings.get("min_probability_half", 0.56)) * 100), key="half_prob") / 100
        half_gap = st.slider("פער מינימלי בין LONG ל־SHORT", 0, 30, int(float(settings.get("min_probability_gap_half", 0.06)) * 100), key="half_gap") / 100
        half_conf = st.slider("ביטחון מינימלי", 30, 80, int(float(settings.get("min_confidence_half", 0.48)) * 100), key="half_conf") / 100

    min_samples = st.slider("מינימום תבניות דומות", 3, 25, int(settings.get("min_samples", 6)))
    display_fallback = st.checkbox("הצג fallback טכני אם מנוע התבניות בלמידה", value=bool(settings.get("display_technical_fallback", True)))
    allow_fallback_trades = st.checkbox("לאפשר עסקאות על fallback טכני", value=bool(settings.get("allow_technical_fallback_trades", False)))
    st.caption("מומלץ להשאיר עסקאות fallback כבוי. זה נועד בעיקר כדי שלא תראה 0% כשאין מספיק תבניות.")
    min_similarity = st.slider("דמיון ממוצע מינימלי", 10, 80, int(float(settings.get("min_similarity", 0.34)) * 100)) / 100
    min_best_similarity = st.slider("דמיון של התבנית הקרובה ביותר", 20, 90, int(float(settings.get("min_best_similarity", 0.44)) * 100)) / 100
    max_neutral_probability = st.slider("NEUTRAL מקסימלי", 10, 80, int(float(settings.get("max_neutral_probability", 0.45)) * 100)) / 100
    st.info("ב־V2.1 תוקן StreamlitDuplicateElementKey: לכל כפתור כניסה/יציאה יש key ייחודי לפי הלשונית.")
    require_expectancy = True
    st.checkbox("חובה תוחלת R חיובית", value=True, disabled=True)
    min_expectancy = st.slider("תוחלת R מינימלית", 0.00, 1.00, float(settings.get("min_expectancy_r", 0.10)), step=0.05)
    min_expectancy_gap = st.slider("יתרון תוחלת מול הכיוון ההפוך", 0.00, 0.75, float(settings.get("min_expectancy_gap_r", 0.10)), step=0.05)
    entry_end_fast = st.text_input("סיום כניסות מהירות (ניו־יורק)", value=str(settings.get("entry_end_time_fast", "15:55")))
    entry_end_half = st.text_input("סיום כניסות חצי שעה (ניו־יורק)", value=str(settings.get("entry_end_time_half", "15:30")))

    new_settings = {
        "auto_trade_enabled": auto_trade, "modes": modes,
        "refresh_seconds": refresh_seconds, "max_workers": max_workers,
        "max_new_trades_per_scan": max_new, "max_open_trades": max_open,
        "min_samples": min_samples, "min_similarity": min_similarity,
        "min_best_similarity": min_best_similarity,
        "max_neutral_probability": max_neutral_probability,
        "min_confidence_fast": fast_conf, "min_confidence_half": half_conf,
        "min_probability_fast": fast_prob, "min_probability_half": half_prob,
        "min_probability_gap_fast": fast_gap, "min_probability_gap_half": half_gap,
        "min_expectancy_r": min_expectancy, "min_expectancy_gap_r": min_expectancy_gap,
        "require_positive_expectancy": require_expectancy,
        "display_technical_fallback": display_fallback,
        "allow_technical_fallback_trades": allow_fallback_trades,
        "consecutive_scans_required": consecutive,
        "entry_start_time": settings.get("entry_start_time", "09:35"),
        "entry_end_time_fast": entry_end_fast,
        "entry_end_time_half": entry_end_half,
    }
    if st.button("שמור הגדרות", use_container_width=True):
        save_monitor_settings(new_settings)
        st.success("ההגדרות נשמרו.")
    settings = new_settings

    st.divider()
    if st.button("נקה עסקאות של המוניטור", type="secondary", use_container_width=True):
        clear_trades()
        save_monitor_metadata(pd.DataFrame(columns=METADATA_COLUMNS))
        st.success("עסקאות המוניטור נוקו.")
    if st.button("נקה היסטוריית תחזיות", type="secondary", use_container_width=True):
        pd.DataFrame(columns=SNAPSHOT_COLUMNS).to_csv(MONITOR_SNAPSHOTS_FILE, index=False)
        st.success("היסטוריית התחזיות נוקתה.")

if not modes:
    st.error("יש לבחור לפחות סוג עסקה אחד.")
    st.stop()
if not selected_tickers:
    st.error("יש לבחור לפחות מניה אחת.")
    st.stop()

refresh_arg = f"{refresh_seconds}s" if auto_refresh else None

@st.fragment(run_every=refresh_arg)
def live_monitor_fragment():
    tab_live, tab_telegram, tab_trades, tab_validation, tab_history, tab_details = st.tabs([
        "📡 תחזיות בזמן אמת", "📨 טלגרם כניסה/יציאה", "💼 עסקאות ורווח בזמן אמת", "✅ אימות תחזיות",
        "🗂️ היסטוריית תחזיות", "ℹ️ הסבר"
    ])

    with tab_live:
        scan_clicked = st.button("🔄 סרוק עכשיו את כל המניות", type="primary")
        should_scan = auto_refresh or scan_clicked or "monitor_last_rows" not in st.session_state
        if should_scan:
            with st.spinner("מעדכן עסקאות פתוחות ומריץ את מנוע התבניות על כל המניות..."):
                try:
                    _updated_trades, update_messages = update_open_trades()
                    update_messages = list(update_messages or [])
                except Exception as exc:
                    update_messages = [f"שגיאה בעדכון עסקאות: {str(exc)[:180]}"]
                previous_snapshots = load_monitor_snapshots()
                rows, scan_id = scan_all_tickers_concurrently(selected_tickers, modes, settings)
                trade_messages = list(open_monitor_candidates(rows, settings, previous_snapshots) or [])
                append_monitor_snapshots(rows)
                evaluated_now = evaluate_pending_predictions()
                if evaluated_now:
                    trade_messages.append(f"אומתו {evaluated_now} תחזיות שהגיעו לאופק הזמן שלהן.")
                st.session_state["monitor_last_rows"] = rows
                st.session_state["monitor_last_scan_id"] = scan_id
                st.session_state["monitor_last_messages"] = update_messages + trade_messages
                st.session_state["monitor_last_scan_time"] = now_ny_iso()
        rows = st.session_state.get("monitor_last_rows", [])
        last_scan = st.session_state.get("monitor_last_scan_time", "טרם בוצעה סריקה")
        st.caption(f"סריקה אחרונה: {last_scan} | מניות: {len(selected_tickers)} | מצבים: {', '.join(modes)}")
        render_monitor_metrics(rows)
        display = prediction_display_df(rows)
        if display.empty:
            st.info("אין עדיין תוצאות.")
        else:
            interactive = prepare_interactive_table(
                display,
                ltr_columns={"מניה", "כיוון מוביל", "מצב תבנית", "מודל", "סטטוס", "רעננות"},
            )
            st.download_button(
                "⬇️ הורד תחזיות נוכחיות CSV",
                data=csv_bytes(display),
                file_name="pattern_monitor_current_predictions.csv",
                mime="text/csv",
                key="download_current_predictions",
            )
            st.caption("אפשר ללחוץ על כל כותרת עמודה כדי למיין מהנמוך לגבוה או להפך.")
            st.dataframe(
                interactive.style.apply(style_prediction_rows, axis=1),
                use_container_width=True,
                hide_index=True,
                height=650,
                column_config=prediction_column_config(),
            )
        messages = st.session_state.get("monitor_last_messages", [])
        if messages:
            with st.expander("פעולות שבוצעו בסריקה האחרונה", expanded=False):
                for message in messages:
                    st.write("•", message)

        st.markdown("---")
        render_prediction_entry_buttons(rows, key_prefix='live')
        st.markdown("---")
        render_manual_tracked_trades_panel(compact=True, key_prefix='live')

    with tab_telegram:
        rows = st.session_state.get("monitor_last_rows", [])
        render_manual_telegram_settings_tab(rows)

    with tab_trades:
        st.subheader("עסקאות Paper — רווח והפסד בזמן אמת")
        trade_df = trades_display_df()
        if trade_df.empty:
            st.info("עדיין אין עסקאות של המוניטור.")
        else:
            interactive_trades = prepare_interactive_table(
                trade_df,
                ltr_columns={"סטטוס", "מניה", "כיוון", "מצב תבנית", "מודל"},
            )
            st.download_button(
                "⬇️ הורד עסקאות CSV",
                data=csv_bytes(trade_df),
                file_name="pattern_monitor_trades.csv",
                mime="text/csv",
                key="download_monitor_trades",
            )
            st.caption("אפשר למיין לפי רווח, אחוזים, מניה, זמן או כל עמודה אחרת בלחיצה על הכותרת.")
            st.dataframe(
                interactive_trades.style.apply(style_trade_rows, axis=1),
                use_container_width=True,
                hide_index=True,
                height=650,
                column_config=trades_column_config(),
            )

    with tab_validation:
        st.subheader("אימות אוטומטי של התחזיות")
        st.caption("עסקה מהירה נבדקת לאחר 5 דקות; חצי שעה לאחר 30 דקות. אם יעד או סטופ נפגעו קודם, הפגיעה הראשונה קובעת.")
        evaluation_df = evaluation_display_df()
        if evaluation_df.empty:
            st.info("עדיין אין תחזיות שהגיעו לאופק הזמן הנדרש לבדיקה.")
        else:
            correct_count = int((evaluation_df["תוצאה"].astype(str) == "CORRECT").sum())
            wrong_count = int((evaluation_df["תוצאה"].astype(str) == "WRONG").sum())
            neutral_count = int((evaluation_df["תוצאה"].astype(str) == "NEUTRAL").sum())
            decisive = correct_count + wrong_count
            accuracy = (correct_count / decisive * 100) if decisive else 0.0
            tradable = evaluation_df[evaluation_df["סטטוס תחזית"].astype(str).eq("ENTRY_CANDIDATE")]
            tradable_correct = int((tradable["תוצאה"].astype(str) == "CORRECT").sum()) if not tradable.empty else 0
            tradable_wrong = int((tradable["תוצאה"].astype(str) == "WRONG").sum()) if not tradable.empty else 0
            tradable_decisive = tradable_correct + tradable_wrong
            tradable_accuracy = (tradable_correct / tradable_decisive * 100) if tradable_decisive else 0.0
            e1, e2, e3, e4, e5, e6 = st.columns(6)
            e1.metric("תחזיות שנבדקו", len(evaluation_df))
            e2.metric("נכונות", correct_count)
            e3.metric("שגויות", wrong_count)
            e4.metric("ניטרליות", neutral_count)
            e5.metric("דיוק מנוע", f"{accuracy:.1f}%")
            e6.metric("דיוק מועמדות", f"{tradable_accuracy:.1f}%")
            st.download_button(
                "⬇️ הורד אימות תחזיות CSV", data=csv_bytes(evaluation_df),
                file_name="pattern_monitor_prediction_evaluations.csv", mime="text/csv",
                key="download_prediction_evaluations",
            )
            st.dataframe(
                prepare_interactive_table(evaluation_df, ltr_columns={"מניה", "כיוון", "סטטוס תחזית", "תוצאה", "פגיעה ראשונה"}),
                use_container_width=True, hide_index=True, height=620,
                column_config={
                    "LONG %": st.column_config.NumberColumn("LONG %", format="%.1f%%"),
                    "SHORT %": st.column_config.NumberColumn("SHORT %", format="%.1f%%"),
                    "NEUTRAL %": st.column_config.NumberColumn("NEUTRAL %", format="%.1f%%"),
                    "ביטחון %": st.column_config.NumberColumn("ביטחון %", format="%.1f%%"),
                    "דמיון %": st.column_config.NumberColumn("דמיון %", format="%.1f%%"),
                    "R בפועל": st.column_config.NumberColumn("R בפועל", format="%+.2f"),
                    "MFE R": st.column_config.NumberColumn("MFE R", format="%.2f"),
                    "MAE R": st.column_config.NumberColumn("MAE R", format="%.2f"),
                },
            )

    with tab_history:
        history = load_monitor_snapshots()
        if history.empty:
            st.info("אין עדיין היסטוריית תחזיות.")
        else:
            history["scan_time"] = history["scan_time"].astype(str)
            latest_history = history.sort_values("scan_time", ascending=False).head(3000)
            st.download_button(
                "⬇️ הורד היסטוריית תחזיות CSV",
                data=csv_bytes(history),
                file_name="pattern_monitor_predictions.csv",
                mime="text/csv",
                key="download_prediction_history",
            )
            history_display = latest_history.head(500).copy()
            history_display = prepare_interactive_table(
                history_display,
                ltr_columns={"ticker", "probability_leader", "engine_side", "dominant_side", "trend_side", "pattern_state", "function_model", "status"},
            )
            st.caption("אפשר למיין את ההיסטוריה בלחיצה על כותרות העמודות.")
            st.dataframe(
                history_display,
                use_container_width=True,
                hide_index=True,
                height=620,
                column_config={
                    "ticker": st.column_config.TextColumn("ticker", width="small"),
                    "dominant_side": st.column_config.TextColumn("dominant_side", width="small"),
                    "pattern_state": st.column_config.TextColumn("pattern_state", width="medium"),
                    "function_model": st.column_config.TextColumn("function_model", width="medium"),
                    "status": st.column_config.TextColumn("status", width="medium"),
                    "reason": st.column_config.TextColumn("reason", width="large"),
                },
            )
            if len(latest_history) > 500:
                st.caption("מוצגות 500 השורות האחרונות על המסך; קובץ ה-CSV כולל את כל ההיסטוריה.")

    with tab_details:
        st.markdown("""
### איך מתקבלת החלטה

לכל מניה ולכל מצב — **מהירה** ו־**חצי שעה** — המנוע מחפש תבניות עבר דומות ומחשב בנפרד:

- אחוז LONG.
- אחוז SHORT.
- אחוז NEUTRAL — מקרים שלא הראו כיוון משמעותי.
- פער האחוזים בין הכיוונים.
- ביטחון משוקלל.
- מספר התבניות הדומות.
- דמיון ממוצע.
- תוחלת רווח ביחידות R.
- מצב הנדסי וסוג הפונקציה המתאימה.

הכיוון בעל האחוז הגבוה יותר הוא הכיוון המוביל. עסקת Paper נפתחת רק כאשר האחוז, הפער, הביטחון וכמות הדוגמאות עוברים את הספים שהוגדרו בסרגל הצד. אם שני המצבים מספקים איתות באותה מניה, נבחר המצב בעל ההסתברות הגבוהה יותר.

### צבעי טבלת העסקאות

- שורה ירוקה: רווח נטו חיובי בזמן העדכון.
- שורה אדומה: הפסד נטו בזמן העדכון.
- שורה אפורה: קרוב לאיזון.

האפליקציה מנהלת קבצים נפרדים מ־V7.2, ולכן אפשר להפעיל את שתי התוכנות במקביל על פורטים שונים.
""")

live_monitor_fragment()
