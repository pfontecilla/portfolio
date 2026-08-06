"""
Real-Time Retirement Portfolio Manager & Risk-Adjusted Monitoring Tool
========================================================================
Built on: retirement-portfolio-plan.md (8-fund ETF model, ~80/20 Mid-High
risk baseline) — extended with a risk-level slider, Bitcoin (IBIT), and
single-stock satellite controls (NVDA, TSLA, GOOGL, MSTR, SPY, QQQ).

Run:
    pip install streamlit yfinance pandas numpy plotly
    streamlit run portfolio_manager.py

Educational tool only. Not investment, tax, or legal advice.
"""

import json
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

try:
    import yfinance as yf
except ImportError:
    yf = None

# ----------------------------------------------------------------------
# 1. MODEL DATA — sourced from retirement-portfolio-plan.md
# ----------------------------------------------------------------------

RISK_LEVELS = ["Low", "Mid-Low", "Mid", "Mid-High (plan default)", "High", "Maximum Growth"]

CORE_TICKERS = ["VTI", "VXUS", "AVUV", "AVDV", "VNQ", "GLDM", "BND", "VTIP"]
SAT_TICKERS = ["IBIT", "MSTR", "BSOL", "NVDA", "TSLA", "GOOGL", "SPCX", "SPY", "QQQ"]

# Core target weights (%) by risk level, BEFORE satellite carve-out.
# Mid-High column (index 3) reproduces the plan's base 80/20 allocation exactly.
CORE_WEIGHTS = {
    # Columns: [Low, Mid-Low, Mid, Mid-High, High]
    # Mid-High (index 3) reproduces the original plan's 80/20 model EXACTLY — do not edit.
    # High (index 4) was made more aggressive in the production upgrade: 28% small-cap
    # value (AVUV+AVDV) and only 6% defensive, so a no-crypto High allocation maximizes
    # expected return within traditional assets. This raises expected vol/drawdown too.
    # Maximum Growth (index 5): 100% equity, 55% small-cap value, zero defensive sleeve.
    # This is the honest ceiling for expected return using only traditional, liquid,
    # low-cost assets. It is NOT "High plus a bit" — see the docstring of
    # compute_metrics(): the extra arithmetic return is partly eaten by variance drag,
    # so geometric (compounded) return rises far less than the headline number, while
    # expected max drawdown deepens to roughly -48%. Suitable only for money that will
    # genuinely not be touched for 10+ years.
    "VTI":  [22, 28, 38, 42, 40, 25],
    "VXUS": [10, 13, 18, 20, 18, 15],
    "AVUV": [2,  4,  5,  8,  16, 32],
    "AVDV": [0,  2,  0,  5,  12, 23],
    "VNQ":  [3,  4,  4,  4,  4,  5],
    "GLDM": [6,  6,  5,  5,  4,  0],
    "BND":  [35, 27, 18, 10, 3,  0],
    "VTIP": [22, 16, 12, 6,  3,  0],
}
SAT_BUDGET = [0, 5, 8, 10, 15, 20]  # max total satellite % by risk level
CRYPTO_CAP = [0, 3, 5, 7, 10, 12]   # max crypto-class % by risk level

# cls: correlation class · r: est. nominal return % · v: est. annual vol % · er: expense ratio %
ASSET_META = {
    "VTI":   dict(name="Vanguard Total US Market",        cls="eq",     r=7.5,  v=15.5, er=0.03),
    "VXUS":  dict(name="Vanguard Total International",    cls="eq",     r=7.5,  v=16.5, er=0.05),
    "AVUV":  dict(name="Avantis US Small Cap Value",       cls="fac",    r=8.5,  v=20.0, er=0.25),
    "AVDV":  dict(name="Avantis Intl Small Cap Value",     cls="fac",    r=8.5,  v=19.0, er=0.36),
    "VNQ":   dict(name="Vanguard Real Estate",             cls="reit",   r=6.5,  v=17.0, er=0.13),
    "GLDM":  dict(name="SPDR Gold MiniShares",             cls="gold",   r=4.0,  v=15.0, er=0.10),
    "BND":   dict(name="Vanguard Total Bond Market",       cls="bond",   r=4.3,  v=5.5,  er=0.03),
    "VTIP":  dict(name="Vanguard Short-Term TIPS",         cls="tips",   r=3.8,  v=3.0,  er=0.04),
    "IBIT":  dict(name="iShares Bitcoin Trust",            cls="crypto", r=12.0, v=60.0, er=0.25),
    "MSTR":  dict(name="Strategy (BTC proxy equity)",      cls="crypto", r=13.0, v=90.0, er=0.00),
    "BSOL":  dict(name="Bitwise Solana Staking ETF",       cls="crypto", r=13.0, v=75.0, er=0.20),
    "NVDA":  dict(name="NVIDIA",                           cls="tech",   r=9.0,  v=45.0, er=0.00),
    "TSLA":  dict(name="Tesla",                            cls="tech",   r=9.0,  v=55.0, er=0.00),
    "GOOGL": dict(name="Alphabet",                         cls="tech",   r=8.5,  v=30.0, er=0.00),
    "SPCX":  dict(name="SpaceX (Space Exploration Tech.)", cls="tech",   r=9.0,  v=65.0, er=0.00),
    "SPY":   dict(name="SPDR S&P 500",                     cls="eq",     r=7.5,  v=15.5, er=0.09),
    "QQQ":   dict(name="Invesco Nasdaq-100",                cls="tech",   r=8.0,  v=19.0, er=0.20),
}

# Snapshot of each built-in asset's MODERATE baseline volatility, captured before
# any assumption set can modify it. apply_return_assumptions() always scales from
# this snapshot, which keeps volatility adjustments idempotent across reruns.
BASELINE_VOLS = {_t: _m["v"] for _t, _m in ASSET_META.items()}

# Approximate long-run correlation matrix between asset CLASSES
CORR = {
    "eq":     {"eq": 1.00, "fac": 0.88, "reit": 0.70, "gold": 0.05, "bond": 0.10, "tips": 0.05, "crypto": 0.35, "tech": 0.85},
    "fac":    {"eq": 0.88, "fac": 1.00, "reit": 0.65, "gold": 0.05, "bond": 0.10, "tips": 0.05, "crypto": 0.30, "tech": 0.72},
    "reit":   {"eq": 0.70, "fac": 0.65, "reit": 1.00, "gold": 0.10, "bond": 0.30, "tips": 0.25, "crypto": 0.20, "tech": 0.55},
    "gold":   {"eq": 0.05, "fac": 0.05, "reit": 0.10, "gold": 1.00, "bond": 0.15, "tips": 0.30, "crypto": 0.25, "tech": 0.05},
    "bond":   {"eq": 0.10, "fac": 0.10, "reit": 0.30, "gold": 0.15, "bond": 1.00, "tips": 0.60, "crypto": 0.05, "tech": 0.05},
    "tips":   {"eq": 0.05, "fac": 0.05, "reit": 0.25, "gold": 0.30, "bond": 0.60, "tips": 1.00, "crypto": 0.05, "tech": 0.05},
    "crypto": {"eq": 0.35, "fac": 0.30, "reit": 0.20, "gold": 0.25, "bond": 0.05, "tips": 0.05, "crypto": 1.00, "tech": 0.50},
    "tech":   {"eq": 0.85, "fac": 0.72, "reit": 0.55, "gold": 0.05, "bond": 0.05, "tips": 0.05, "crypto": 0.50, "tech": 1.00},
}
GROWTH_CLASSES = {"eq", "fac", "reit", "tech", "crypto"}

# Tax-loss-harvest partner funds, straight from plan §6
TLH_PARTNERS = {
    "VTI": "ITOT / SCHB", "VXUS": "IXUS", "AVUV": "VBR / ISCV", "AVDV": "DISV",
    "VNQ": "USRT / SCHH", "BND": "AGG / IUSB", "VTIP": "STIP", "GLDM": "IAUM",
}

# ----------------------------------------------------------------------
# 1b. PRO-ANALYTICS DATA — look-through overlap + stress scenarios
# ----------------------------------------------------------------------
# Approximate weight (%) of each single stock INSIDE broad ETFs (rough
# mid-2026 estimates; index weights drift constantly — verify and edit).
# Used only by the Effective Exposure calculator (Feature 1).
OVERLAP_ESTIMATES = {
    "NVDA":  {"VTI": 6.5, "SPY": 7.5, "QQQ": 9.5},
    "TSLA":  {"VTI": 1.5, "SPY": 1.8, "QQQ": 3.0},
    "GOOGL": {"VTI": 3.5, "SPY": 4.0, "QQQ": 5.5},
    "MSTR":  {"VTI": 0.2, "SPY": 0.0, "QQQ": 0.5},
    "SPCX":  {"VTI": 0.6, "SPY": 0.0, "QQQ": 2.5},
}

# Class-level return shocks (%) for scenario stress testing (Feature 4).
# Calibrated loosely to historical episodes; illustrative, not predictive.
STRESS_SCENARIOS = {
    "2008-style equity crash":  {"eq": -50, "fac": -55, "reit": -60, "gold": 5,  "bond": 7,   "tips": 0,  "crypto": -70, "tech": -50},
    "2022-style stagflation":   {"eq": -20, "fac": -12, "reit": -26, "gold": 0,  "bond": -13, "tips": -3, "crypto": -64, "tech": -33},
    "Sharp rate-hike shock":    {"eq": -12, "fac": -10, "reit": -20, "gold": -5, "bond": -10, "tips": -2, "crypto": -25, "tech": -18},
    "Crypto / tech meltdown":   {"eq": -8,  "fac": -6,  "reit": -5,  "gold": 3,  "bond": 2,   "tips": 1,  "crypto": -75, "tech": -35},
}

# Holdings with too little trading history for their model assumptions to mean
# much. Shown wherever the ticker is selected or held.
NEW_LISTING_WARNING = {
    "SPCX": (
        "⚠️ **SPCX (SpaceX)** listed on Nasdaq on 12 Jun 2026 — only weeks of price history exists. "
        "It has already traded from a $225 high down to roughly $110, and its return/volatility "
        "assumptions here are rough guesses, not estimates. Treat every SPCX figure in this app as "
        "illustrative and size the position accordingly."
    ),
    "BSOL": (
        "⚠️ **BSOL (Solana ETF)** is a recently launched crypto product. Its staking mechanics affect "
        "tax treatment, and SOL's volatility history is shorter and wilder than Bitcoin's."
    ),
}

# ----------------------------------------------------------------------
# 1c. PRODUCTION — RETURN ASSUMPTION SETS + OPERATING MODES
# ----------------------------------------------------------------------
# Four capital-market assumption sets, by asset CLASS.
#
# ==== THESE ARE ARITHMETIC MEANS, NOT CAGRs ====
# compute_metrics() treats these as the arithmetic mean and derives the geometric
# CAGR by subtracting variance drag, so a documented historical GEOMETRIC figure
# must never be typed in directly (that would subtract the drag twice).
#
# ==== RETURNS AND VOLATILITY ARE PAIRED PER SET ====
# The two historical sets use the documented historical ARITHMETIC mean together
# with the documented historical VOLATILITY (see VOL_SCALARS below). Because both
# halves of the pair come from the same historical record, the geometric CAGR the
# model derives lands on the historical geometric figure on its own:
#
#   class  set          arith    vol      -> derived geo    historical geo
#   eq     Optimistic   12.0%    19.8%       10.29%         ~10.2%  (S&P 500 1926-2024)
#   fac    Optimistic   16.2%    30.0%       12.51%         ~12.5%  (small-cap value)
#   reit   Optimistic   11.0%    19.6%        9.31%          ~9.3%  (NAREIT 1972-)
#   bond   Optimistic    5.5%     5.5%        5.35%          ~5.4%  (US IG long-run)
#   eq     Hist.Upper   12.4%    19.8%       10.66%         upper end
#   fac    Hist.Upper   17.0%    31.0%       13.10%         ~13.1%  (FF small value)
#
# Earlier versions of this file kept low (forward-looking) volatilities in the
# historical sets, which understated risk AND inflated the derived CAGR by roughly
# half a point. Pairing each set's returns with its own volatility fixes both.
#
# Sources for the historical figures:
# - eq: SBBI / Ibbotson large-cap US 1926-2024 — arithmetic ~11.9-12.5%,
#   volatility ~19.8%, geometric ~10.0-10.5% nominal (NYU Stern dataset ~10.2%).
# - fac: SBBI small-cap 1926-2024 — arithmetic 16.2% at 31.6% volatility, implying
#   ~12.1% geometric. Fama-French / Dimensional small-VALUE series run modestly
#   higher; AVUV-style profitability screening trims volatility slightly, hence
#   30-31% rather than 31.6%.
# - International equity has historically trailed the US over the same period, so
#   holding it in the same "eq" class is mildly generous to ex-US markets.
# - crypto: NO long history exists. Speculative placeholder only, and excluded
#   entirely in Bank-Compliant Mode.
RETURN_ASSUMPTIONS = {
    # Current institutional forward-looking CMAs (Vanguard, BlackRock et al.,
    # 2025-26): US equity 10-yr expected returns mostly 4.0-6.5% on elevated
    # valuations. The set a professional would defend for a projection today.
    "Conservative": {"eq": 6.0,  "fac": 7.0,  "reit": 5.0,  "gold": 3.0, "bond": 4.0, "tips": 3.5, "crypto": 8.0,  "tech": 6.5},
    # Balanced view: blends forward CMAs with long-run history.
    "Base":         {"eq": 7.5,  "fac": 8.5,  "reit": 6.5,  "gold": 4.0, "bond": 4.3, "tips": 3.8, "crypto": 12.0, "tech": 8.5},
    # Documented long-run historical ARITHMETIC means (paired with historical vol).
    "Optimistic":   {"eq": 12.0, "fac": 16.2, "reit": 11.0, "gold": 6.0, "bond": 5.5, "tips": 4.2, "crypto": 27.0, "tech": 13.0},
    # Upper end of documented historical experience. PAST != FUTURE.
    "Historical Upper Bound": {"eq": 12.4, "fac": 17.0, "reit": 11.4, "gold": 6.4, "bond": 5.7, "tips": 4.4, "crypto": 30.0, "tech": 14.0},
}

# Per-set VOLATILITY multipliers applied to each asset's baseline volatility in
# ASSET_META, by asset class. Conservative and Base deliberately have NO
# adjustment: they keep the model's moderate, forward-looking volatilities.
# The two historical sets scale volatility up to long-run realized levels so that
# risk and return come from the same historical record.
#
#   class  baseline -> Optimistic / Hist.Upper       justification
#   eq      15.5%   ->  19.8%  (x1.277)             SBBI large-cap realized vol
#   fac     20.0%   ->  30.0% / 31.0% (x1.50/1.55)  SBBI small-cap 31.6%, trimmed
#                                                    slightly for profitability screen
#   reit    17.0%   ->  19.6%  (x1.153)             NAREIT realized vol
#   gold    15.0%   ->  18.8%  (x1.253)             gold post-1971 realized vol
#   tech    19.0%   ->  21.0%  (x1.105)             modest; class vols already high
#   bond/tips       ->  ~unchanged                  historical vol close to baseline
VOL_SCALARS = {
    "Conservative": {},   # baseline volatilities — no adjustment
    "Base": {},           # baseline volatilities — no adjustment
    "Optimistic":              {"eq": 1.277, "fac": 1.50, "reit": 1.153, "gold": 1.253, "tips": 1.10, "tech": 1.105},
    "Historical Upper Bound":  {"eq": 1.277, "fac": 1.55, "reit": 1.153, "gold": 1.253, "tips": 1.10, "tech": 1.105},
}

# Shown in the UI wherever an assumption set is selected.
ASSUMPTION_NOTES = {
    "Conservative": "Current institutional forward-looking CMAs (Vanguard/BlackRock-style, 2025-26), with the "
                    "model's moderate baseline volatilities. The most defensible set for a projection today.",
    "Base": "Balanced view — blends forward-looking CMAs with long-run history, at baseline volatilities.",
    "Optimistic": "Long-run historical averages, **paired with long-run historical volatilities** "
                  "(equities 19.8%, small-cap value 30%). Risk and return both come from the same historical "
                  "record, so the derived CAGR lands on the historical figure (~10.2% equities, ~12.5% small value). "
                  "Assumes the future resembles the past — an assumption, not a forecast.",
    "Historical Upper Bound": "⚠️ Upper end of documented long-run historical experience, weighted toward the "
                              "small-cap value premium, **at historical volatilities (up to 31%)**. "
                              "**Based on past performance — NOT a forecast.** Note the deeper drawdown estimate: "
                              "this is what earning those returns historically felt like. Use to bound the "
                              "optimistic case, never as a planning baseline.",
}

MODES = ("Corporate Treasury", "Personal Retirement")
MODE_TEXT = {
    "Corporate Treasury": {
        "horizon_label": "Investment horizon (years)",
        "tagline": "Corporate treasury allocation monitor — excess USD cash, capital preservation + growth.",
        "ips_title": "TREASURY INVESTMENT POLICY STATEMENT — SUMMARY",
        "horizon_line": "Investment horizon:",
        "glide": [
            ("Deployment & growth", "8+ yrs to need", "Hold targets; rebalance by the 5/25 bands; keep operating liquidity outside this portfolio."),
            ("Progressive de-risking", "4–7 yrs to need", "Step growth down ~2 pts/yr via inflows and distributions — trim factor sleeves and satellites first."),
            ("Liquidity build", "≤3 yrs to need", "Reach ~60–65% growth; hold 1–2 years of expected cash needs in T-bills/money market."),
        ],
    },
    "Personal Retirement": {
        "horizon_label": "Years to retirement",
        "tagline": "Live-monitoring companion to the retirement plan — the 8-fund ETF core with optional satellites.",
        "ips_title": "INVESTMENT POLICY STATEMENT — SUMMARY",
        "horizon_line": "Horizon:",
        "glide": [
            ("Accumulation", "8+ yrs out", "Hold targets; rebalance by the 5/25 bands."),
            ("Early de-risking", "4–7 yrs out", "Step growth down ~2 pts/yr via dividends and new cash — trim AVDV and satellites first."),
            ("Pre-retirement", "≤3 yrs out", "Reach ~60–65% growth; build 1–2 years of withdrawals in cash/T-bills."),
        ],
    },
}

FALLBACK_PRICES = {  # used only if live fetch fails / yfinance unavailable
    "VTI": 330, "VXUS": 72, "AVUV": 105, "AVDV": 82, "VNQ": 95, "GLDM": 68,
    "BND": 74, "VTIP": 50, "IBIT": 65, "MSTR": 380, "BSOL": 10, "NVDA": 175, "TSLA": 320,
    "GOOGL": 200, "SPCX": 120, "SPY": 640, "QQQ": 570,
}

ALL_TICKERS = CORE_TICKERS + SAT_TICKERS


def corr(a, b):
    return CORR[a][b]


# ----------------------------------------------------------------------
# 2. STREAMLIT SETUP
# ----------------------------------------------------------------------

st.set_page_config(page_title="Retirement Portfolio Manager", layout="wide", page_icon="📒")

PINE = "#1D6B54"
LOSS = "#A8432E"
AMBER = "#A97F24"
MUTED = "#6E7B73"

st.markdown(
    """
    <style>
    /* The ledger aesthetic is a fixed light theme. We force BOTH background and
       text colors: forcing only the background made every label unreadable for
       visitors whose browser/OS is in dark mode (Streamlit sets light text). */
    .stApp, .stApp p, .stApp li, .stApp label, .stApp span, .stApp div,
    .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6,
    .stApp td, .stApp th, .stApp caption {
        color: #16241E;
    }
    .stApp { background-color: #F3F5F2; }
    section[data-testid="stSidebar"] { background-color: #EAEEE9; }
    /* Keep our own dark cards (mandate summary, bank banner, health score)
       light-on-dark — they set their own inline colors, so re-assert them. */
    .ledger-invert, .ledger-invert * { color: #F3F5F2 !important; }
    /* Padding-top was too small: the masthead was clipped under the toolbar. */
    .block-container { padding-top: 3.2rem; max-width: 1150px; }
    .ledger-header { font-family: Georgia, 'Times New Roman', serif; }
    div[data-testid="stMetricValue"] { font-family: ui-monospace, monospace; }
    div[data-testid="stMetricLabel"], div[data-testid="stMetricValue"] { color: #16241E; }
    /* Tab labels + expander headers need explicit color in dark mode too */
    button[data-baseweb="tab"] div, div[data-testid="stExpander"] summary { color: #16241E; }
    /* --- Mobile responsiveness --- */
    @media (max-width: 640px) {
        .block-container { padding-left: 0.7rem; padding-right: 0.7rem; padding-top: 2.6rem; }
        .ledger-header { font-size: 1.4rem !important; }
        div[data-testid="stMetricValue"] { font-size: 1.05rem; }
        button[kind], .stButton button, .stDownloadButton button { min-height: 44px; }
        div[data-baseweb="tab-list"] { overflow-x: auto; flex-wrap: nowrap; }
        div[data-baseweb="tab"] { min-height: 44px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if "shares" not in st.session_state:
    st.session_state.shares = {t: 0.0 for t in ALL_TICKERS}
if "prices" not in st.session_state:
    st.session_state.prices = dict(FALLBACK_PRICES)
if "prices_live" not in st.session_state:
    st.session_state.prices_live = False
if "last_fetch" not in st.session_state:
    st.session_state.last_fetch = None
if "custom_meta" not in st.session_state:
    st.session_state.custom_meta = {}  # user-added tickers: {ticker: dict(name, cls, r, v, er)}

# Widget defaults live in session state (keys, not `value=` args) so that a
# loaded config can programmatically set them without Streamlit conflicts.
_WIDGET_DEFAULTS = {
    "w_risk": 3, "w_basis": 250_000, "w_monthly": 0, "w_years": 15,
    "w_ibit": False, "w_bsol": False, "w_percap": 5, "w_sats": [],
    "w_ibit_pct": 0.0, "w_bsol_pct": 0.0,
    "w_mode": "Corporate Treasury", "w_bank": True, "w_assume": "Base",
}
for _k, _v in _WIDGET_DEFAULTS.items():
    st.session_state.setdefault(_k, _v)

# Apply a config loaded last run (Save/Load, sidebar §⑤). Must happen BEFORE
# any widget is instantiated — Streamlit forbids setting widget state after.
if "pending_config" in st.session_state:
    _cfg = st.session_state.pop("pending_config")
    try:
        for _k, _v in _cfg.get("widgets", {}).items():
            st.session_state[_k] = _v
        st.session_state.custom_meta = dict(_cfg.get("custom_meta", {}))
        for _t, _s in _cfg.get("shares", {}).items():
            st.session_state.shares[_t] = float(_s)
        for _t, _p in _cfg.get("prices", {}).items():
            st.session_state.prices[_t] = float(_p)
        st.session_state["_cfg_loaded"] = True
    except Exception as _e:  # malformed file — keep running with current state
        st.session_state["_cfg_error"] = str(_e)

# Mode flags are read from session state here (widgets render later in the
# sidebar, but keyed session values are already current on rerun).
_mode = st.session_state.get("w_mode", "Corporate Treasury")
_bank = bool(st.session_state.get("w_bank", True))
_title = "Treasury Portfolio Manager" if _mode == "Corporate Treasury" else "Retirement Portfolio Manager"

st.markdown(
    f'<div class="ledger-header" style="font-size:1.9rem;">📒 {_title}</div>',
    unsafe_allow_html=True,
)
st.caption(MODE_TEXT[_mode]["tagline"] + " Educational tool — not investment advice.")

if _bank:
    st.markdown(
        "<div class='ledger-invert' style='background:#16241E;color:#F3F5F2;border-radius:8px;padding:8px 14px;"
        "margin-bottom:0.5rem;font-size:0.85rem;'>🏦 <b>BANK-COMPLIANT MODE</b> — crypto assets "
        "(IBIT, BSOL, MSTR and crypto-class custom tickers) are excluded per bank policy. "
        "Traditional assets only: equities, bonds, gold, REITs, permitted single stocks.</div>",
        unsafe_allow_html=True,
    )

with st.expander("📄 Important disclosures"):
    st.markdown(
        """
**Educational tool only.** This application is provided for educational and analytical purposes.
It does not constitute investment advice, a recommendation, an offer, or a solicitation to buy or
sell any security, and it has not been reviewed or approved by any bank, regulator, or licensed
adviser. **All figures are model estimates.** Expected returns, volatilities, correlations,
drawdowns, projections, and health scores derive from simplified assumptions that the user can
change; they are not forecasts, and different assumption sets produce materially different
results. **Past performance does not guarantee future results.** Investing involves risk,
including possible loss of principal. Compliance with any institution's investment policies,
banking regulations, or tax law remains solely the responsibility of the user. Consult licensed
financial, tax, and legal professionals before acting on anything shown here.
        """
    )

# Global drift-alert banner (visible on every tab). Filled after the drift
# engine runs — see the block right after compute_portfolio() below.
alert_slot = st.container()
if st.session_state.pop("_cfg_loaded", False):
    st.success("✅ Configuration loaded — risk level, satellites, custom tickers, and holdings restored.")
if "_cfg_error" in st.session_state:
    st.error(f"Config file could not be applied: {st.session_state.pop('_cfg_error')}")

# ----------------------------------------------------------------------
# 3. SIDEBAR — MANDATE + SATELLITE CONTROLS
# ----------------------------------------------------------------------

with st.sidebar:
    # Pinned summary — filled in after compute_satellites() runs (see §3c).
    summary_slot = st.container()
    st.divider()

    with st.expander("① Mandate", expanded=True):
        mode = st.selectbox("Operating mode", list(MODES), key="w_mode")
        bank_mode = st.toggle("🏦 Bank-Compliant Mode (no crypto)", key="w_bank",
                              help="Excludes IBIT, BSOL, MSTR and any crypto-class custom tickers, per bank policy.")
        assume = st.selectbox("Return assumptions", list(RETURN_ASSUMPTIONS), key="w_assume",
                              help="Conservative = current institutional forward CMAs · Base = balanced blend · "
                                   "Optimistic = long-run historical averages · Historical Upper Bound = upper end "
                                   "of past experience (past ≠ future). Values are ARITHMETIC means; the app derives "
                                   "the compounded CAGR from them. The two historical sets also raise VOLATILITY to "
                                   "historical levels (small-cap value 30-31%), so risk and return stay paired.")
        if assume == "Historical Upper Bound":
            st.warning(ASSUMPTION_NOTES[assume])
        else:
            st.caption(ASSUMPTION_NOTES[assume])
        if VOL_SCALARS.get(assume):
            st.caption(f"📊 Volatility: **historical levels** — equities "
                       f"{BASELINE_VOLS['VTI'] * VOL_SCALARS[assume]['eq']:.1f}%, small-cap value "
                       f"{BASELINE_VOLS['AVUV'] * VOL_SCALARS[assume]['fac']:.1f}%.")
        else:
            st.caption(f"📊 Volatility: **baseline (moderate)** — equities {BASELINE_VOLS['VTI']:.1f}%, "
                       f"small-cap value {BASELINE_VOLS['AVUV']:.1f}%.")
        risk_idx = st.select_slider(
            "Risk level", options=list(range(len(RISK_LEVELS))), key="w_risk", format_func=lambda i: RISK_LEVELS[i]
        )
        basis = st.number_input("Target basis ($)", min_value=0, step=5_000, key="w_basis")
        monthly = st.number_input("Monthly contribution ($)", min_value=0, step=100, key="w_monthly")
        years = st.slider(MODE_TEXT[mode]["horizon_label"], 1, 30, key="w_years")

    with st.expander("② Satellites & Bitcoin", expanded=False):
        if bank_mode:
            # Sanitize any previously selected crypto satellites before widgets render
            st.session_state["w_sats"] = [s for s in st.session_state.get("w_sats", [])
                                           if ASSET_META.get(s, {}).get("cls") != "crypto"]
            st.caption("🏦 Crypto satellites (IBIT, BSOL, MSTR) are disabled in Bank-Compliant Mode.")
            ibit_on, bsol_on = False, False
        else:
            ibit_on = st.toggle("Enable IBIT (Bitcoin)", key="w_ibit")
            bsol_on = st.toggle("Enable BSOL (Solana, staked)", key="w_bsol")
        per_cap = st.slider("Per-name cap (%)", 1, 10, key="w_percap")

        stock_sats = [x for x in SAT_TICKERS if x not in ("IBIT", "BSOL")
                      and not (bank_mode and ASSET_META[x]["cls"] == "crypto")]
        chosen_sats = st.multiselect(
            "Single-stock satellites to include",
            stock_sats,
            key="w_sats",
            help="Only names you pick here get an exposure input. Includes SPCX (SpaceX) and MSTR (Bitcoin proxy equity).",
        )

        sat_raw = {t: 0.0 for t in SAT_TICKERS}
        sat_raw["IBIT"] = st.slider("IBIT %", 0.0, float(per_cap), step=0.5, key="w_ibit_pct") if ibit_on else 0.0
        sat_raw["BSOL"] = st.slider("BSOL %", 0.0, float(per_cap), step=0.5, key="w_bsol_pct") if bsol_on else 0.0
        if chosen_sats:
            cols = st.columns(2)
            for i, t in enumerate(chosen_sats):
                with cols[i % 2]:
                    sat_raw[t] = st.number_input(f"{t} %", min_value=0.0, max_value=float(per_cap), value=0.0, step=0.5, key=f"sat_{t}")
        if "SPCX" in chosen_sats:
            st.warning(NEW_LISTING_WARNING["SPCX"])
        if not chosen_sats and not ibit_on and not bsol_on:
            st.caption("No satellites active — you're running the pure 8-fund core model from the plan.")

    with st.expander("③ Custom Tickers", expanded=False):
        st.caption("Not in the built-in list? Add any ticker — it's treated as a satellite, subject to the same caps.")
        new_t = st.text_input("Ticker symbol", value="", placeholder="e.g. AMZN").strip().upper()
        new_cls = st.selectbox(
            "Asset class (drives its risk/correlation assumptions)",
            [c for c in ["tech", "eq", "fac", "reit", "gold", "bond", "tips", "crypto"]
             if not (bank_mode and c == "crypto")],
            index=0,
        )
        cc1, cc2, cc3 = st.columns(3)
        new_r = cc1.number_input("Est. return %", value=8.0, step=0.5, key="new_r")
        new_v = cc2.number_input("Est. vol %", value=30.0, step=1.0, key="new_v")
        new_er = cc3.number_input("Expense ratio %", value=0.0, step=0.01, key="new_er")
        if st.button("Add ticker", width='stretch'):
            if not new_t:
                st.warning("Enter a ticker symbol first.")
            elif new_t in ASSET_META or new_t in st.session_state.custom_meta:
                st.warning(f"{new_t} is already in the model.")
            else:
                st.session_state.custom_meta[new_t] = dict(name=new_t, cls=new_cls, r=new_r, v=new_v, er=new_er)
                px = None
                if yf is not None:
                    try:
                        fast = yf.Ticker(new_t).fast_info
                        px = fast.get("lastPrice") or fast.get("last_price")
                    except Exception:
                        px = None
                st.session_state.prices[new_t] = round(float(px), 2) if px else 100.0
                st.session_state.shares.setdefault(new_t, 0.0)
                st.success(f"Added {new_t} — set its target % below.")
                st.rerun()

        custom_raw = {}
        if st.session_state.custom_meta:
            st.caption("Your custom tickers:")
            for t in list(st.session_state.custom_meta.keys()):
                meta = st.session_state.custom_meta[t]
                rc1, rc2 = st.columns([2, 1])
                with rc1:
                    custom_raw[t] = st.number_input(
                        f"{t} % ({meta['cls']})", min_value=0.0, max_value=float(per_cap), value=0.0, step=0.5, key=f"custom_pct_{t}"
                    )
                with rc2:
                    st.write("")
                    if st.button("✕ Remove", key=f"rm_{t}"):
                        st.session_state.custom_meta.pop(t, None)
                        st.session_state.shares.pop(t, None)
                        st.session_state.prices.pop(t, None)
                        st.rerun()
        else:
            st.caption("None added yet.")

    with st.expander("④ Live Prices", expanded=True):
        if yf is None:
            st.warning("`yfinance` not installed — run `pip install yfinance` for live quotes.")
        fetch_clicked = st.button("🔄 Fetch live prices (yfinance)", width='stretch')
        if st.session_state.prices_live:
            st.success(f"🟢 LIVE prices · fetched {st.session_state.last_fetch}")
        else:
            st.error("🔴 PLACEHOLDER prices — drift %, orders, and $ values are illustrative only until you fetch live quotes.")

    with st.expander("⑤ Save / Load & Export", expanded=False):
        # --- Save: everything needed to reconstruct this exact setup ---
        _cfg_out = {
            "version": 1,
            "saved": time.strftime("%Y-%m-%d %H:%M:%S"),
            "widgets": {
                "w_risk": int(risk_idx), "w_basis": int(basis), "w_monthly": int(monthly),
                "w_years": int(years), "w_ibit": bool(ibit_on), "w_bsol": bool(bsol_on),
                "w_percap": int(per_cap), "w_sats": list(chosen_sats),
                "w_mode": mode, "w_bank": bool(bank_mode), "w_assume": assume,
                "w_ibit_pct": float(sat_raw["IBIT"]), "w_bsol_pct": float(sat_raw["BSOL"]),
                **{f"sat_{t}": float(sat_raw.get(t, 0.0)) for t in chosen_sats},
                **{f"custom_pct_{t}": float(v) for t, v in custom_raw.items()},
            },
            "custom_meta": st.session_state.custom_meta,
            "shares": {t: s for t, s in st.session_state.shares.items() if s > 0},
            "prices": st.session_state.prices,
        }
        st.download_button("💾 Save config (.json)", json.dumps(_cfg_out, indent=2),
                           "portfolio_config.json", "application/json", width='stretch')

        # --- Load: stash → rerun → applied at top of script before widgets ---
        _up = st.file_uploader("Load config (.json)", type="json", key="cfg_upload")
        if _up is not None and st.button("Apply loaded config", width='stretch'):
            try:
                st.session_state["pending_config"] = json.load(_up)
                st.rerun()
            except Exception as e:
                st.error(f"Not a valid config file: {e}")
        st.caption("CSV exports and the IPS summary live on the **Dashboard** tab.")

# ----------------------------------------------------------------------
# 3b. FOLD CUSTOM TICKERS INTO THE MODEL FOR THIS RUN
# ----------------------------------------------------------------------
sat_raw.update(custom_raw)
ASSET_META.update(st.session_state.custom_meta)
SAT_TICKERS = SAT_TICKERS + [t for t in st.session_state.custom_meta if t not in SAT_TICKERS]
ALL_TICKERS = CORE_TICKERS + SAT_TICKERS

# ----------------------------------------------------------------------
# 4. SATELLITE ENGINE — caps enforced exactly like the plan / in-chat tool
# ----------------------------------------------------------------------

def compute_satellites(sat_raw, risk_idx, per_cap, bank=False):
    warns = []
    eff = {t: max(0.0, sat_raw.get(t, 0.0)) for t in SAT_TICKERS}
    if bank:
        _zeroed = [t for t in eff if eff[t] > 0 and ASSET_META.get(t, {}).get("cls") == "crypto"]
        for t in _zeroed:
            eff[t] = 0.0
        if _zeroed:
            warns.append(f"Bank-Compliant Mode: crypto satellites disabled per bank policy ({', '.join(_zeroed)} set to 0).")
    for t in SAT_TICKERS:
        if eff[t] > per_cap:
            warns.append(f"{t} trimmed to the {per_cap}% per-name cap.")
            eff[t] = per_cap

    cap = CRYPTO_CAP[risk_idx]
    crypto_names = [t for t in eff if ASSET_META.get(t, {}).get("cls") == "crypto"]
    crypto = sum(eff[t] for t in crypto_names)
    if crypto > cap and crypto > 0:
        k = cap / crypto
        for t in crypto_names:
            eff[t] *= k
        warns.append(f"Crypto exposure ({' + '.join(crypto_names)}) scaled down to the {cap}% cap for {RISK_LEVELS[risk_idx]} risk.")

    budget = SAT_BUDGET[risk_idx]
    total = sum(eff.values())
    if total > budget:
        if budget == 0:
            warns.append(f"Satellites are disabled at {RISK_LEVELS[risk_idx]} risk — all weights set to 0.")
            eff = {t: 0.0 for t in eff}
        else:
            k = budget / total
            eff = {t: v * k for t, v in eff.items()}
            warns.append(f"Total satellites scaled down to the {budget}% budget for {RISK_LEVELS[risk_idx]} risk.")
        total = sum(eff.values())

    return eff, total, sum(eff[t] for t in eff if ASSET_META.get(t, {}).get("cls") == "crypto"), warns, cap, budget


def apply_return_assumptions(assume_name: str) -> None:
    """Apply the selected assumption set's expected returns AND volatilities.

    Returns come from RETURN_ASSUMPTIONS[set][class]; volatilities are the
    asset's BASELINE_VOLS value scaled by VOL_SCALARS[set][class] (default 1.0,
    i.e. Conservative and Base keep the model's moderate baseline volatilities).

    Scaling always starts from BASELINE_VOLS rather than the current value, so
    calling this repeatedly in one session never compounds the adjustment.

    Custom (user-added) tickers keep the return and volatility the user entered —
    the app has no historical basis on which to override their inputs.
    """
    rets = RETURN_ASSUMPTIONS[assume_name]
    vols = VOL_SCALARS.get(assume_name, {})
    for _t, _meta in ASSET_META.items():
        if _t in st.session_state.custom_meta:
            continue
        _cls = _meta["cls"]
        _meta["r"] = rets[_cls]
        _meta["v"] = BASELINE_VOLS[_t] * vols.get(_cls, 1.0)


apply_return_assumptions(assume)

sat_eff, sat_total, sat_crypto, sat_warns, crypto_cap_val, sat_budget_val = compute_satellites(
    sat_raw, risk_idx, per_cap, bank_mode
)

# ----------------------------------------------------------------------
# 3c. PINNED SIDEBAR SUMMARY (rendered into the slot reserved above)
# ----------------------------------------------------------------------
with summary_slot:
    st.markdown(
        f"<div class='ledger-invert' style='background:#16241E;color:#F3F5F2;border-radius:8px;padding:10px 12px;'>"
        f"<div style='font-size:0.7rem;letter-spacing:.08em;opacity:.65;'>MANDATE</div>"
        f"<div style='font-family:monospace;font-size:0.95rem;font-weight:600;'>{RISK_LEVELS[risk_idx]}</div>"
        f"<div style='font-family:monospace;font-size:0.78rem;opacity:.85;margin-top:4px;'>"
        f"${basis:,.0f} · {years} yrs · sat {sat_total:.1f}% · crypto {sat_crypto:.1f}%</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

# ----------------------------------------------------------------------
# 5. TARGET WEIGHTS — core rescaled around satellite carve-out
# ----------------------------------------------------------------------

def compute_targets(risk_idx, sat_eff, sat_total):
    scale = (100 - sat_total) / 100
    targets = {t: CORE_WEIGHTS[t][risk_idx] * scale for t in CORE_TICKERS}
    for t in SAT_TICKERS:
        if sat_eff[t] > 0.001:
            targets[t] = sat_eff[t]
    return targets


targets = compute_targets(risk_idx, sat_eff, sat_total)

# ----------------------------------------------------------------------
# 6. LIVE PRICE FETCH (yfinance)
# ----------------------------------------------------------------------

if fetch_clicked:
    if yf is None:
        st.sidebar.error("yfinance is not installed in this environment.")
    else:
        active = [t for t in ALL_TICKERS if targets.get(t, 0) > 0.001 or st.session_state.shares.get(t, 0) > 0]
        if not active:
            active = ALL_TICKERS
        with st.spinner(f"Fetching {len(active)} live quotes…"):
            ok, failed = 0, []
            for t in active:
                px = None
                try:
                    fast = yf.Ticker(t).fast_info
                    px = fast.get("lastPrice") or fast.get("last_price")
                except Exception:
                    px = None
                if not px or px <= 0:
                    # Fallback: daily history close, more robust than fast_info for thin/odd tickers
                    try:
                        hist = yf.Ticker(t).history(period="1d", auto_adjust=True)
                        if not hist.empty:
                            px = float(hist["Close"].iloc[-1])
                    except Exception:
                        px = None
                if px and px > 0:
                    st.session_state.prices[t] = round(float(px), 2)
                    ok += 1
                else:
                    failed.append(t)
            st.session_state.prices_live = ok > 0
            st.session_state.last_fetch = time.strftime("%Y-%m-%d %H:%M:%S")
        if failed:
            st.sidebar.warning(f"Updated {ok}/{len(active)}. Failed: {', '.join(failed)} — edit manually.")
        else:
            st.sidebar.success(f"Updated all {ok} prices.")

# ----------------------------------------------------------------------
# 7. MAIN LAYOUT
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
# 9. METRICS + PROJECTIONS
# ----------------------------------------------------------------------

def compute_metrics(targets):
    """Portfolio risk/return metrics.

    IMPORTANT — two different return figures, both reported:
    * `ret` (arithmetic): weighted average of holdings' expected returns. This
      is the standard input to mean-variance math, but it is NOT what your
      money actually compounds at.
    * `cagr` (geometric): the compounded annual growth rate, approximated as
      ln(1+mu) - sigma^2 / (2(1+mu)^2), i.e. the arithmetic mean minus
      "variance drag". Terminal wealth follows THIS number, and the gap widens
      with volatility (~0.2pp at 6% vol, ~1.6pp at 18% vol). Projections and
      the IPS use `cagr`; Sharpe uses `ret` by convention.
    """
    w = {t: p / 100 for t, p in targets.items() if p > 0}
    ret = sum(w[t] * ASSET_META[t]["r"] for t in w)
    var = 0.0
    for i, wi in w.items():
        for j, wj in w.items():
            var += wi * wj * (ASSET_META[i]["v"] / 100) * (ASSET_META[j]["v"] / 100) * corr(ASSET_META[i]["cls"], ASSET_META[j]["cls"])
    vol = (var ** 0.5) * 100
    growth = sum(v for t, v in w.items() if ASSET_META[t]["cls"] in GROWTH_CLASSES) * 100
    er = sum(w[t] * ASSET_META[t]["er"] for t in w)
    sharpe = (ret - 3.5) / vol if vol else 0
    dd = -min(60, vol * 2.8)
    # Geometric (compounded) return — exact lognormal conversion.
    # Calibrate a lognormal to the arithmetic mean `ret` and volatility `vol`:
    #   s_log^2 = ln(1 + sigma^2 / (1+mu)^2)
    #   m_log   = ln(1+mu) - s_log^2 / 2
    #   CAGR    = exp(m_log) - 1
    # The final exp()-1 converts the log return back to a simple return.
    # Verified against Monte Carlo simulation to 3 decimal places.
    _m, _s = ret / 100.0, vol / 100.0
    _s_log2 = np.log(1.0 + (_s ** 2) / ((1.0 + _m) ** 2))
    _m_log = np.log(1.0 + _m) - _s_log2 / 2.0
    cagr = (np.exp(_m_log) - 1.0) * 100.0
    return dict(ret=ret, cagr=cagr, drag=ret - cagr, vol=vol, growth=growth,
                er=er, sharpe=sharpe, dd=dd)


metrics = compute_metrics(targets)


tab_dash, tab_targets, tab_holdings, tab_drift, tab_outlook, tab_analytics, tab_guide = st.tabs(
    ["🩺 Dashboard", "🎯 Targets", "✏️ Holdings", "📊 Drift & Orders", "📈 Outlook", "🔬 Analytics", "📋 How to Use"]
)

# ---- Tab: Targets --------------------------------------------------
with tab_targets:
    left, right = st.columns([1, 1.3])

    with left:
        st.subheader(f"Model at {RISK_LEVELS[risk_idx]} risk")
        st.metric("Satellite budget used", f"{sat_total:.1f}% / {sat_budget_val}%")
        st.metric("Crypto exposure (BTC/SOL/proxies)", f"{sat_crypto:.1f}% / {crypto_cap_val}%")
        for w in sat_warns:
            st.warning(w)
        if st.button("📥 Load model at these targets", width='stretch'):
            for t in ALL_TICKERS:
                tgt = targets.get(t, 0)
                px = st.session_state.prices.get(t, FALLBACK_PRICES.get(t, 100.0))
                st.session_state.shares[t] = round((tgt / 100 * basis) / px, 3) if tgt > 0 else 0.0
            st.success("Holdings set to model targets at current basis and prices.")

    with right:
        tdf = pd.DataFrame(
            [
                {
                    "Ticker": t,
                    "Name": ASSET_META[t]["name"],
                    "Target %": round(p, 1),
                    "$ Amount": round(p / 100 * basis),
                    "Est. return": f"{ASSET_META[t]['r']:.1f}%",
                    f"Est. $ in {years} yrs": round((p / 100 * basis) * (1 + ASSET_META[t]["r"] / 100) ** years),
                }
                for t, p in sorted(targets.items(), key=lambda x: -x[1])
                if p > 0.05
            ]
        )
        fig = go.Figure(
            go.Pie(labels=tdf["Ticker"], values=tdf["Target %"], hole=0.5,
                   marker=dict(colors=["#1D6B54", "#2F8A6E", "#5AA98C", "#8AC3AC", "#B98A2B",
                                        "#D9B15F", "#3F5D52", "#7A8B82", "#A8432E", "#C4745F",
                                        "#8A5A44", "#5B4A6B", "#3C6E8F"]))
        )
        fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=320,
                           paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, width='stretch', key="targets_pie_chart")
        st.dataframe(tdf, hide_index=True, width='stretch')
        no_rebal_total = tdf[f"Est. $ in {years} yrs"].sum()
        blended_cagr = sum((p / 100) * ASSET_META[t]["r"] for t, p in targets.items() if p > 0)
        st.caption(
            f"**Assumption set: {assume}.** "
            f"Est. return = the model's long-run *arithmetic* annual assumption per holding. "
            f"Projected values compound each sleeve independently for {years} years with no rebalancing "
            f"(sum ≈ ${no_rebal_total:,.0f}) and ignore volatility drag, so they overstate realistic outcomes. "
            f"Blended arithmetic mean ≈ {blended_cagr:.1f}%/yr, but the portfolio's compounded CAGR is "
            f"{metrics['cagr']:.1f}%/yr — the Outlook tab projects with the compounded figure."
        )

# ---- Tab: Holdings --------------------------------------------------
with tab_holdings:
    st.subheader("Current holdings")
    st.caption("Enter your actual share counts. Prices are editable — fetch live quotes in the sidebar or type your own.")

    active_tickers = [t for t in ALL_TICKERS if targets.get(t, 0) > 0.001 or st.session_state.shares.get(t, 0) > 0]
    if not active_tickers:
        active_tickers = CORE_TICKERS

    edit_df = pd.DataFrame(
        {
            "Ticker": active_tickers,
            "Shares": [st.session_state.shares.get(t, 0.0) for t in active_tickers],
            "Price ($)": [st.session_state.prices.get(t, FALLBACK_PRICES.get(t, 100.0)) for t in active_tickers],
        }
    )
    edited = st.data_editor(
        edit_df,
        column_config={
            "Ticker": st.column_config.TextColumn(disabled=True),
            "Shares": st.column_config.NumberColumn(min_value=0.0, step=0.01, format="%.3f"),
            "Price ($)": st.column_config.NumberColumn(min_value=0.0, step=0.01, format="%.2f"),
        },
        hide_index=True,
        width='stretch',
        key="holdings_editor",
    )
    for _, row in edited.iterrows():
        st.session_state.shares[row["Ticker"]] = row["Shares"]
        st.session_state.prices[row["Ticker"]] = row["Price ($)"]

    total_value = sum(st.session_state.shares.get(t, 0) * st.session_state.prices.get(t, 0) for t in active_tickers)
    st.metric("Total portfolio value", f"${total_value:,.0f}")

# ----------------------------------------------------------------------
# 8. PORTFOLIO / DRIFT CALC (shared)
# ----------------------------------------------------------------------

def compute_portfolio(shares, prices, targets):
    tickers = [t for t in ALL_TICKERS if targets.get(t, 0) > 0.001 or shares.get(t, 0) > 0]
    total = sum(shares.get(t, 0) * prices.get(t, 0) for t in tickers)
    rows = []
    for t in tickers:
        val = shares.get(t, 0) * prices.get(t, 0)
        cur = (val / total * 100) if total > 0 else 0.0
        tgt = targets.get(t, 0.0)
        band = 5.0 if tgt >= 15 else max(tgt * 0.25, 0.6)
        drift = cur - tgt
        ratio = abs(drift) / band if band else 0
        status = "—" if total == 0 else ("REBALANCE" if ratio > 1 else ("WATCH" if ratio > 0.6 else "OK"))
        rows.append(dict(Ticker=t, Value=val, Current=cur, Target=tgt, Band=band, Drift=drift, Status=status))
    return pd.DataFrame(rows), total


port_df, port_total = compute_portfolio(st.session_state.shares, st.session_state.prices, targets)

# ---- Phase 3: global drift alerts (banner on every tab + one-time toast) ----
with alert_slot:
    if port_total > 0:
        _re = port_df[port_df["Status"] == "REBALANCE"]["Ticker"].tolist()
        _wa = port_df[port_df["Status"] == "WATCH"]["Ticker"].tolist()
        if _re:
            st.error(f"🔴 **Drift alert:** {len(_re)} holding(s) outside their 5/25 bands — "
                     f"{', '.join(_re)}. See **Drift & Orders** for tax-aware fixes.")
        elif _wa:
            st.warning(f"🟡 Approaching bands (>60% of tolerance used): {', '.join(_wa)}. "
                       f"No action needed yet — steer new cash toward the laggards.")
    _cur_flags = frozenset(port_df[port_df["Status"] == "REBALANCE"]["Ticker"]) if port_total > 0 else frozenset()
    if _cur_flags and _cur_flags != st.session_state.get("_last_flags"):
        st.toast(f"Drift alert: {', '.join(sorted(_cur_flags))} outside bands", icon="🔴")
    st.session_state["_last_flags"] = _cur_flags

# ---- Tab: Drift & Orders --------------------------------------------
with tab_drift:
    st.subheader("Drift vs. target (5/25 rule)")
    st.caption("±5 pts band for holdings targeted ≥15%; ±25% relative band for smaller sleeves — per plan §4.")

    if port_total == 0:
        st.info("No holdings yet — load the model or enter shares in the Holdings tab.")
    else:
        def status_color(s):
            return {"OK": PINE, "WATCH": AMBER, "REBALANCE": LOSS, "—": MUTED}.get(s, MUTED)

        for _, r in port_df.sort_values("Target", ascending=False).iterrows():
            c1, c2, c3 = st.columns([1, 5, 1])
            with c1:
                st.markdown(f"**{r['Ticker']}**")
            with c2:
                lo, hi = max(0, r["Target"] - r["Band"]), r["Target"] + r["Band"]
                maxscale = max(hi, r["Current"]) * 1.3 + 1
                fig = go.Figure()
                fig.add_shape(type="rect", x0=lo, x1=hi, y0=0, y1=1, fillcolor="#E3EEE9", line_width=0)
                fig.add_shape(type="line", x0=r["Target"], x1=r["Target"], y0=0, y1=1, line=dict(color=PINE, width=2))
                fig.add_trace(go.Scatter(x=[r["Current"]], y=[0.5], mode="markers",
                                          marker=dict(size=14, color=status_color(r["Status"]))))
                fig.update_xaxes(range=[0, maxscale], showgrid=False, zeroline=False)
                fig.update_yaxes(visible=False, range=[0, 1])
                fig.update_layout(height=55, margin=dict(t=0, b=0, l=0, r=0),
                                   paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", showlegend=False)
                st.plotly_chart(fig, width='stretch', config={"displayModeBar": False},
                                 key=f"drift_chart_{r['Ticker']}")
            with c3:
                badge = {"OK": "🟢", "WATCH": "🟡", "REBALANCE": "🔴", "—": "⚪"}[r["Status"]]
                st.markdown(f"{badge} **{r['Status']}**")
                st.caption(f"{r['Current']:.1f}% vs {r['Target']:.1f}%")

        st.divider()
        st.subheader("Suggested orders (tax-aware ordering)")
        flagged = port_df[port_df["Status"] == "REBALANCE"].copy()
        if flagged.empty:
            st.success("All holdings inside their bands. Direct new dividends and contributions to the lowest holdings per plan §4.")
        else:
            flagged["Trade $"] = (flagged["Target"] - flagged["Current"]) / 100 * port_total
            flagged = flagged.sort_values("Trade $", ascending=False)
            for _, r in flagged.iterrows():
                action = "BUY" if r["Trade $"] >= 0 else "SELL"
                color = PINE if action == "BUY" else LOSS
                partner = TLH_PARTNERS.get(r["Ticker"], "—")
                st.markdown(
                    f"<div style='background:{'#E3EEE9' if action=='BUY' else '#F4E4DF'}; "
                    f"padding:8px 12px; border-radius:6px; margin-bottom:6px; display:flex; justify-content:space-between;'>"
                    f"<span><b>{action} {r['Ticker']}</b></span>"
                    f"<span style='font-family:monospace;'>${abs(r['Trade $']):,.0f}</span></div>",
                    unsafe_allow_html=True,
                )
                if action == "SELL":
                    st.caption(f"↳ If harvesting a loss on {r['Ticker']}, its TLH partner per the plan is **{partner}**.")
            st.caption(
                "Order of operations per plan §4: (1) direct dividends to underweights, "
                "(2) direct new contributions to underweights, (3) harvest losses, "
                "(4) only then sell overweight winners — long-term lots via specific-lot ID."
            )

    # Look-through concentration notes are rendered here, but computed later in
    # the script (they need the Analytics tab's concentration limit).
    st.divider()
    lookthrough_slot = st.container()

# ----------------------------------------------------------------------
# 10. PRO ANALYTICS — Features 1–4
# All functions are pure (inputs → outputs) and reuse existing structures:
# the `targets` dict, ASSET_META classes, the CORR class matrix, and the
# 5/25 band rule. Nothing here mutates core model state.
# ----------------------------------------------------------------------

def compute_effective_exposure(targets: dict, min_implicit: float = 0.25) -> pd.DataFrame:
    """FEATURE 1 — Look-through exposure.
    Effective weight of a single name = its explicit satellite %
    + sum over broad ETFs of (ETF target % × that stock's weight inside it).
    Uses OVERLAP_ESTIMATES (editable, approximate).

    A name is listed when it is actually held explicitly, or when its implicit
    weight alone reaches `min_implicit`. Without that floor the table lists every
    name that appears anywhere inside a broad index fund — including ones the user
    has switched off, which reads as a bug. Residual sub-threshold exposure is
    still reported, via residual_lookthrough() below.
    """
    rows = []
    for stock, holders in OVERLAP_ESTIMATES.items():
        implicit = sum(targets.get(h, 0.0) * frac / 100.0 for h, frac in holders.items())
        explicit = targets.get(stock, 0.0)
        if explicit > 0 or implicit >= min_implicit:
            rows.append({
                "Stock": stock,
                "Implicit % (inside ETFs)": round(implicit, 2),
                "Explicit satellite %": round(explicit, 2),
                "Effective total %": round(implicit + explicit, 2),
            })
    df = pd.DataFrame(rows)
    return df.sort_values("Effective total %", ascending=False) if not df.empty else df


def residual_lookthrough(targets: dict, cls: str, min_implicit: float = 0.25) -> dict:
    """Names of a given asset class that the portfolio holds ONLY indirectly, via
    broad index funds, below the display threshold.

    This matters for Bank-Compliant Mode: excluding crypto satellites does not
    reach zero crypto exposure, because broad market funds hold names like MSTR.
    Reported so the residual is disclosed rather than silently dropped.
    """
    out = {}
    for stock, holders in OVERLAP_ESTIMATES.items():
        if ASSET_META.get(stock, {}).get("cls") != cls:
            continue
        if targets.get(stock, 0.0) > 0:
            continue  # held explicitly — it appears in the main table instead
        implicit = sum(targets.get(h, 0.0) * frac / 100.0 for h, frac in holders.items())
        if 0 < implicit < min_implicit:
            out[stock] = implicit
    return out


def compute_diversification(targets: dict, port_vol: float) -> dict:
    """FEATURE 2 — Diversification Ratio (weighted-avg vol ÷ portfolio vol,
    Choueifaty–Coignard) and effective independent bets (≈ DR²), plus a
    simple 1/HHI effective-holdings count."""
    w = {t: p / 100.0 for t, p in targets.items() if p > 0}
    wavg_vol = sum(w[t] * ASSET_META[t]["v"] for t in w)
    dr = (wavg_vol / port_vol) if port_vol else 1.0
    return {
        "wavg_vol": wavg_vol,
        "dr": dr,
        "enb": dr ** 2,
        "eff_holdings": (1.0 / sum(v * v for v in w.values())) if w else 0.0,
    }


def class_corr_matrix(targets: dict) -> pd.DataFrame:
    """Correlation matrix restricted to the asset classes actually held."""
    classes = sorted({ASSET_META[t]["cls"] for t, p in targets.items() if p > 0})
    return pd.DataFrame(
        [[corr(a, b) for b in classes] for a in classes], index=classes, columns=classes
    )


@st.cache_data(show_spinner=False)
def simulate_glide(targets: dict, years: int, end_growth: float,
                   start_val: float, monthly_c: float, assume_key: str = "Base") -> pd.DataFrame:
    """FEATURE 3 — Year-by-year de-risking per plan §5.
    `assume_key` exists purely as a cache key: results depend on the active
    assumption set (which mutates ASSET_META), so it must invalidate the cache.
    hold current growth % while >7 yrs remain, then shift linearly to
    `end_growth` by retirement. Freed weight moves to the defensive sleeve
    (BND/VTIP/etc.) pro-rata; growth holdings (incl. satellites/AVDV — the
    'trim first' names in practice) shrink proportionally. Value compounds
    at each year's blended CAGR + monthly contributions."""
    g0 = sum(p for t, p in targets.items() if ASSET_META[t]["cls"] in GROWTH_CLASSES)
    defensive = {t: p for t, p in targets.items() if ASSET_META[t]["cls"] not in GROWTH_CLASSES}
    def_total = sum(defensive.values())
    if def_total <= 0:
        # Maximum Growth starts with NO defensive sleeve, so there is nothing to
        # scale up pro-rata. De-risking has to build one from scratch — seed a
        # conventional core-bond / short-TIPS split and grow that instead.
        defensive = {"BND": 60.0, "VTIP": 40.0}
        def_total = 100.0
    rows, val = [], start_val
    for y in range(0, years + 1):
        remaining = years - y
        g = g0 if remaining > 7 else end_growth + (g0 - end_growth) * max(remaining, 0) / 7.0
        g = min(g, g0)
        scale = (g / g0) if g0 else 0.0
        yr_t = {}
        for t, p in targets.items():
            if ASSET_META[t]["cls"] in GROWTH_CLASSES:
                yr_t[t] = p * scale
        # Defensive sleeve is rebuilt from `defensive` (which may be the seeded
        # BND/VTIP mix when the starting allocation held none), so the freed
        # growth weight always lands somewhere and each year still sums to 100.
        for t, p in defensive.items():
            yr_t[t] = yr_t.get(t, 0.0) + p + (g0 - g) * (p / def_total)
        ret = sum((p / 100.0) * ASSET_META[t]["r"] for t, p in yr_t.items())
        sat = sum(yr_t.get(t, 0.0) for t in SAT_TICKERS)
        cry = sum(p for t, p in yr_t.items() if ASSET_META[t]["cls"] == "crypto")
        rows.append({"Year": y, "Growth %": round(g, 1), "Defensive %": round(100 - g, 1),
                     "Satellite %": round(sat, 1), "Crypto %": round(cry, 1),
                     "Blended CAGR %": round(ret, 2), "Projected value": round(val)})
        val = val * (1 + ret / 100.0) + monthly_c * 12
    return pd.DataFrame(rows)


def run_scenario(targets: dict, shocks: dict, blended_ret: float) -> dict:
    """FEATURE 4 — Apply class-level shocks to target weights.
    Returns portfolio loss, per-holding loss contributions, which holdings
    the 5/25 bands would flag post-shock, and log-math recovery years at the
    blended CAGR."""
    w = {t: p / 100.0 for t, p in targets.items() if p > 0}
    contrib = {t: w[t] * shocks[ASSET_META[t]["cls"]] for t in w}
    loss = sum(contrib.values())
    post = {t: w[t] * (1 + shocks[ASSET_META[t]["cls"]] / 100.0) for t in w}
    post_total = sum(post.values()) or 1.0
    flags = []
    for t in w:
        cur = post[t] / post_total * 100.0
        tgt = targets[t]
        band = 5.0 if tgt >= 15 else max(tgt * 0.25, 0.6)
        drift = cur - tgt
        if drift > band:
            flags.append({"Ticker": t, "Action": "SELL (overweight after shock)", "Drift pp": round(drift, 1)})
        elif -drift > band:
            flags.append({"Ticker": t, "Action": "BUY (fell through band)", "Drift pp": round(drift, 1)})
    recovery = 0.0
    if loss < 0 and blended_ret > 0:
        recovery = float(np.log(1.0 / (1.0 + loss / 100.0)) / np.log(1.0 + blended_ret / 100.0))
    return {"loss": loss, "contrib": contrib, "flags": flags, "recovery": recovery}


# ----------------------------------------------------------------------
# 11. PHASE 1 — HEALTH SCORE + EXPORTS
# ----------------------------------------------------------------------

def risk_contributions(targets: dict) -> dict:
    """Decompose portfolio variance into each holding's share of total risk.

    Uses the same pairwise covariance engine as compute_metrics(). Returns each
    holding's risk-contribution share (summing to 1), the largest single share,
    and the Effective Number of Bets (1 / sum of squared risk shares) — the
    diversification measure used by the health score.
    """
    ts = [t for t, p in targets.items() if p > 0]
    if not ts:
        return {"rc": {}, "top": 0.0, "enb": 0.0}
    w = np.array([targets[t] / 100.0 for t in ts])
    vol = np.array([ASSET_META[t]["v"] / 100.0 for t in ts])
    C = np.array([[corr(ASSET_META[a]["cls"], ASSET_META[b]["cls"]) for b in ts] for a in ts])
    S = np.outer(vol, vol) * C
    pvar = float(w @ S @ w)
    if pvar <= 0:
        return {"rc": {}, "top": 0.0, "enb": 0.0}
    rc = w * (S @ w) / pvar
    return {"rc": dict(zip(ts, rc)), "top": float(rc.max()),
            "enb": 1.0 / float((rc ** 2).sum())}


def compute_health(targets: dict, port_df: pd.DataFrame, div: dict,
                   exp_df: pd.DataFrame, conc_limit: float = 9.0,
                   ref_vol: float | None = None, port_vol: float | None = None) -> dict:
    """Portfolio Health Score (0–100) across five equally weighted pillars.

    1. Diversification    — Effective Number of Bets (risk-contribution based);
                            4.0 independent bets scores full marks.
    2. Concentration      — worst effective single-name weight vs `conc_limit`.
    3. Drift discipline   — share of holdings inside their 5/25 bands.
    4. Speculation        — combined crypto + tech weight vs a 20% soft ceiling.
    5. Risk vs mandate    — portfolio volatility vs the volatility of the PURE
                            core model at the same risk level (`ref_vol`).
                            Penalizes risk added *beyond* the chosen mandate.

    Why not the Diversification Ratio for pillar 1? DR (and ENB, and top risk
    contribution) all *improve* when you bolt on high-volatility satellites —
    mathematically correct, since more assets across more classes genuinely is
    more diversified. Scoring on DR alone therefore rewards speculation. Pillar
    5 exists to close that gap: it asks whether total risk still matches the
    mandate you selected, not merely whether risk is spread around. DR is still
    reported in the UI as an informative metric.
    """
    # Pillar 1 — diversification via effective number of independent bets
    rcs = risk_contributions(targets)
    p_div = max(0.0, min(100.0, (rcs["enb"] - 1.0) / 3.0 * 100.0))

    # Pillar 2 — worst effective single-name concentration (look-through)
    max_conc = float(exp_df["Effective total %"].max()) if not exp_df.empty else 0.0
    p_conc = 100.0 if max_conc <= conc_limit else max(0.0, 100.0 - (max_conc - conc_limit) * 12.0)

    # Pillar 3 — drift discipline (neutral 100 when no holdings entered yet)
    if port_df.empty or port_df["Status"].eq("—").all():
        p_drift, out_of_band = 100.0, 0
    else:
        out_of_band = int((port_df["Status"] == "REBALANCE").sum())
        p_drift = max(0.0, 100.0 - (out_of_band / max(len(port_df), 1)) * 200.0)

    # Pillar 4 — speculative cluster (crypto + tech)
    spec = sum(p for t, p in targets.items() if ASSET_META[t]["cls"] in ("crypto", "tech"))
    p_spec = 100.0 if spec <= 20.0 else max(0.0, 100.0 - (spec - 20.0) * 5.0)

    # Pillar 5 — risk taken beyond the mandate's own core model
    overshoot = 0.0
    if ref_vol and port_vol and ref_vol > 0:
        overshoot = max(0.0, (port_vol - ref_vol) / ref_vol * 100.0)
    p_vol = max(0.0, 100.0 - overshoot * 3.0)

    score = (p_div + p_conc + p_drift + p_spec + p_vol) / 5.0
    band = "Strong" if score >= 80 else ("Adequate" if score >= 60 else ("Needs attention" if score >= 40 else "Fragile"))
    return {"score": score, "band": band, "p_div": p_div, "p_conc": p_conc,
            "p_drift": p_drift, "p_spec": p_spec, "p_vol": p_vol,
            "max_conc": max_conc, "spec": spec, "out_of_band": out_of_band,
            "enb": rcs["enb"], "top_rc": rcs["top"], "overshoot": overshoot}


def build_export_frames(targets: dict, port_df: pd.DataFrame, basis: float,
                        port_total: float) -> dict:
    """Assemble the three CSV export tables: targets, holdings, and drift/orders."""
    tgt = pd.DataFrame([
        {"Ticker": t, "Name": ASSET_META[t]["name"], "Class": ASSET_META[t]["cls"],
         "Target %": round(p, 2), "Target $": round(p / 100 * basis, 2),
         "Est CAGR %": ASSET_META[t]["r"], "Expense ratio %": ASSET_META[t]["er"]}
        for t, p in sorted(targets.items(), key=lambda x: -x[1]) if p > 0.05
    ])
    hold = pd.DataFrame([
        {"Ticker": t, "Shares": st.session_state.shares.get(t, 0.0),
         "Price": st.session_state.prices.get(t, 0.0),
         "Market value": round(st.session_state.shares.get(t, 0.0) * st.session_state.prices.get(t, 0.0), 2)}
        for t in ALL_TICKERS if st.session_state.shares.get(t, 0.0) > 0
    ])
    orders = port_df.copy()
    if not orders.empty and port_total > 0:
        orders["Trade $"] = ((orders["Target"] - orders["Current"]) / 100 * port_total).round(2)
        orders["TLH partner"] = orders["Ticker"].map(lambda t: TLH_PARTNERS.get(t, ""))
        orders = orders[["Ticker", "Current", "Target", "Band", "Drift", "Status", "Trade $", "TLH partner"]].round(2)
    return {"targets": tgt, "holdings": hold, "orders": orders}


def build_ips_text(targets: dict, risk_label: str, basis: float, years: int,
                   metrics: dict, health: dict, mode_name: str = "Personal Retirement",
                   assume_name: str = "Base", bank: bool = False) -> str:
    """Plain-text Investment Policy Statement summary, ready to paste into a doc."""
    mt = MODE_TEXT[mode_name]
    lines = [
        mt["ips_title"],
        f"Generated: {time.strftime('%Y-%m-%d')}",
        "",
        f"Operating mode:      {mode_name}" + ("  ·  BANK-COMPLIANT (no crypto)" if bank else ""),
        f"Return assumptions:  {assume_name}",
        f"  basis:             {'long-run historical averages (past != future)' if 'Optimistic' in assume_name or 'Historical' in assume_name else 'forward-looking capital market assumptions'}",
        f"  volatility basis:  {'historical realized levels (equities ~19.8%, small-cap value ~30%)' if VOL_SCALARS.get(assume_name) else 'baseline moderate (equities 15.5%, small-cap value 20%)'}",
        f"Risk mandate:        {risk_label}",
        f"Portfolio basis:     ${basis:,.0f}",
        f"{mt['horizon_line']:<21}{years} years",
        f"Growth exposure:     {metrics['growth']:.0f}%",
        f"Est. return (arith): {metrics['ret']:.1f}%/yr",
        f"Est. CAGR (compnd):  {metrics['cagr']:.1f}%/yr   <- terminal wealth follows this",
        f"Est. volatility:     {metrics['vol']:.1f}%",
        f"Est. max drawdown:   {metrics['dd']:.0f}%",
        f"Blended expense:     {metrics['er']:.2f}%",
        f"Health score:        {health['score']:.0f}/100 ({health['band']})",
        "",
        "TARGET ALLOCATION",
    ]
    for t, p in sorted(targets.items(), key=lambda x: -x[1]):
        if p > 0.05:
            lines.append(f"  {t:<6} {p:>5.1f}%   ${p / 100 * basis:>12,.0f}   {ASSET_META[t]['name']}")
    lines += [
        "",
        "REBALANCING POLICY",
        "  Bands: +/-5 percentage points for holdings targeted >=15%;",
        "         +/-25% relative for smaller sleeves (the 5/25 rule).",
        "  Review: quarterly drift check; full review each December.",
        "  Order of operations: (1) direct dividends to underweights,",
        "    (2) direct new contributions to underweights, (3) harvest losses,",
        "    (4) only then sell overweight winners, long-term lots via specific-lot ID.",
        "",
        "GLIDE PATH",
        "  Hold target growth % until ~7 years from retirement, then step down",
        "  to ~60-65% growth, funding the shift with cash flows rather than sales.",
        "  Build 1-2 years of planned withdrawals in cash/T-bills in the final 3 years.",
        "",
        "This summary is educational and not investment, tax, or legal advice.",
    ]
    return "\n".join(lines)


# ----------------------------------------------------------------------
# 12. PHASE 2 — LOOK-THROUGH-AWARE ORDERS, RISK COMPARISON,
#     MONTE CARLO, REBALANCING COST
# ----------------------------------------------------------------------

def lookthrough_flags(targets: dict, conc_limit: float) -> dict:
    """Map ticker -> warning string when trading it worsens an already
    concentrated look-through exposure. Used to annotate BUY suggestions."""
    exp = compute_effective_exposure(targets)
    out = {}
    if exp.empty:
        return out
    for _, r in exp.iterrows():
        if r["Effective total %"] > conc_limit:
            stock = r["Stock"]
            out[stock] = (f"{stock} effective exposure is already {r['Effective total %']:.1f}% "
                          f"(vs {conc_limit:.0f}% limit) once ETF holdings are counted.")
            for etf in OVERLAP_ESTIMATES.get(stock, {}):
                if targets.get(etf, 0) > 0:
                    out.setdefault(etf, "")
                    out[etf] += (f" Buying {etf} also adds {stock} exposure "
                                 f"({OVERLAP_ESTIMATES[stock][etf]:.1f}% of {etf}).")
    return {k: v.strip() for k, v in out.items() if v.strip()}


@st.cache_data(show_spinner=False)
def compare_risk_levels(idx_a: int, idx_b: int, sat_raw: dict, per_cap: int,
                        basis: float, years: int, bank: bool = False,
                        assume_key: str = "Base") -> pd.DataFrame:
    """Side-by-side comparison of two risk levels using the SAME satellite
    inputs, cap system, bank restriction, and assumption set (assume_key also
    invalidates the cache when the return set changes)."""
    rows = []
    for idx in (idx_a, idx_b):
        eff, tot, _cry, _w, _cap, _bud = compute_satellites(sat_raw, idx, per_cap, bank)
        tg = compute_targets(idx, eff, tot)
        m = compute_metrics(tg)
        d = compute_diversification(tg, m["vol"])
        rows.append({
            "Risk level": RISK_LEVELS[idx],
            "Growth %": round(m["growth"], 1),
            "Est. CAGR %": round(m["ret"], 2),
            "Est. vol %": round(m["vol"], 1),
            "Sharpe": round(m["sharpe"], 2),
            "Est. max DD %": round(m["dd"], 0),
            "Diversification ratio": round(d["dr"], 2),
            "Blended ER %": round(m["er"], 3),
            f"Est. value in {years}y": round(basis * (1 + m["ret"] / 100) ** years),
        })
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def monte_carlo(start_val: float, monthly_c: float, years: int, mu: float,
                sigma: float, n_paths: int = 2000, seed: int = 42) -> dict:
    """Lognormal Monte Carlo on annual steps.

    Draws annual returns from a lognormal distribution calibrated to the
    portfolio's estimated arithmetic mean (`mu`) and volatility (`sigma`),
    adding contributions at each year end. Returns percentile bands and the
    full terminal distribution.

    Caveat by construction: this assumes returns are independent and
    identically distributed year to year, which understates real-world
    fat tails, volatility clustering, and sequence risk.
    """
    rng = np.random.default_rng(seed)
    m, s = mu / 100.0, sigma / 100.0
    # Convert arithmetic moments to lognormal parameters
    sig2 = np.log(1.0 + (s ** 2) / ((1.0 + m) ** 2))
    mean_log = np.log(1.0 + m) - 0.5 * sig2
    draws = rng.lognormal(mean_log, np.sqrt(sig2), size=(n_paths, years))
    paths = np.empty((n_paths, years + 1))
    paths[:, 0] = start_val
    for y in range(years):
        paths[:, y + 1] = paths[:, y] * draws[:, y] + monthly_c * 12
    pct = {p: np.percentile(paths, p, axis=0) for p in (5, 25, 50, 75, 95)}
    terminal = paths[:, -1]
    contributed = start_val + monthly_c * 12 * years
    return {"pct": pct, "terminal": terminal,
            "p_loss": float((terminal < contributed).mean() * 100.0),
            "contributed": contributed}


def rebalance_cost(port_df: pd.DataFrame, port_total: float, ltcg_rate: float,
                   unrealized_pct: float, spread_bps: float) -> dict:
    """Approximate the cost of acting on the current REBALANCE flags.

    Tax drag assumes sells realize `unrealized_pct` of proceeds as long-term
    gains taxed at `ltcg_rate`. Transaction cost applies a bid/ask spread to
    all traded notional (commissions are $0 at major brokers).
    """
    flagged = port_df[port_df["Status"] == "REBALANCE"] if not port_df.empty else port_df
    if flagged.empty or port_total <= 0:
        return {"sell_notional": 0.0, "buy_notional": 0.0, "tax": 0.0,
                "spread": 0.0, "total": 0.0, "pct": 0.0}
    trades = (flagged["Target"] - flagged["Current"]) / 100 * port_total
    sell_notional = float(-trades[trades < 0].sum())
    buy_notional = float(trades[trades > 0].sum())
    tax = sell_notional * (unrealized_pct / 100.0) * (ltcg_rate / 100.0)
    spread = (sell_notional + buy_notional) * (spread_bps / 10_000.0)
    total = tax + spread
    return {"sell_notional": sell_notional, "buy_notional": buy_notional,
            "tax": tax, "spread": spread, "total": total,
            "pct": total / port_total * 100.0}


# ----------------------------------------------------------------------
# 13. PHASE 3 — ASSET LOCATION + HISTORICAL BACKTEST
# ----------------------------------------------------------------------

def asset_location_table(targets: dict) -> pd.DataFrame:
    """PHASE 3 — Asset location per plan §6 logic.

    Rules, in the order they're checked:
    - VXUS       → taxable (the foreign tax credit is only claimable there)
    - bond/tips/reit classes → traditional 401(k)/IRA (ordinary-income yield)
    - fac class  → Roth if available (highest expected return, tax-free growth)
    - gold       → tax-advantaged if space allows (28% collectibles rate in taxable)
    - BSOL       → tax-advantaged preferred (staking distributions = ordinary income)
    - other crypto → either (ETF wrapper is reasonably tax-efficient)
    - everything else (equity ETFs / single stocks) → taxable
    """
    rows = []
    for t, p in sorted(targets.items(), key=lambda x: -x[1]):
        if p <= 0.05:
            continue
        cls = ASSET_META[t]["cls"]
        if t == "VXUS":
            rec, why = "Taxable", "Foreign tax credit is only claimable in a taxable account"
        elif cls in ("bond", "tips", "reit"):
            rec, why = "Traditional 401(k)/IRA", "Distributions taxed as ordinary income — shelter them"
        elif cls == "fac":
            rec, why = "Roth IRA (else taxable)", "Highest expected return — maximize tax-free compounding"
        elif cls == "gold":
            rec, why = "Tax-advantaged if space allows", "Gold gains face the collectibles rate (max 28%) in taxable"
        elif t == "BSOL":
            rec, why = "Tax-advantaged preferred", "Staking distributions are taxed as ordinary income"
        elif cls == "crypto":
            rec, why = "Either (taxable OK)", "ETF wrapper is fairly tax-efficient; Roth amplifies wins AND losses"
        else:
            rec, why = "Taxable", "Tax-efficient equity — qualified dividends, gain timing under your control"
        rows.append({"Ticker": t, "Target %": round(p, 1),
                     "Recommended account": rec, "Why": why})
    return pd.DataFrame(rows)


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_history(tickers: tuple, period: str) -> pd.DataFrame:
    """PHASE 3 — Download monthly adjusted closes via yfinance.

    Returns an empty DataFrame on any failure (no yfinance, no network,
    delisted tickers) so callers can degrade gracefully. Cached 1 hour.
    """
    if yf is None:
        return pd.DataFrame()
    try:
        data = yf.download(list(tickers), period=period, interval="1mo",
                           auto_adjust=True, progress=False)
        if data is None or len(data) == 0:
            return pd.DataFrame()
        close = data["Close"] if "Close" in data else data
        if isinstance(close, pd.Series):
            close = close.to_frame(name=tickers[0])
        return close.dropna(how="all")
    except Exception:
        return pd.DataFrame()


def run_backtest(close: pd.DataFrame, weights: dict, min_months: int = 24) -> dict:
    """PHASE 3 — Backtest target weights on monthly closes, monthly-rebalanced.

    Tickers with fewer than `min_months` of data (e.g. SPCX, BSOL) are
    excluded and reported in `dropped`; the remaining weights are
    renormalized to 100%. Pure function — testable without yfinance.
    """
    cols = [t for t in weights if t in close.columns and close[t].notna().sum() >= min_months]
    dropped = [t for t in weights if t not in cols]
    if len(cols) < 2:
        return {"ok": False, "dropped": dropped}
    px = close[cols].dropna()
    if len(px) < min_months:
        return {"ok": False, "dropped": dropped}
    rets = px.pct_change().dropna()
    w = np.array([weights[t] for t in cols], dtype=float)
    w = w / w.sum()
    series = pd.Series(rets.values @ w, index=rets.index)
    cum = (1.0 + series).cumprod()
    n_years = len(series) / 12.0
    cagr = cum.iloc[-1] ** (1.0 / n_years) - 1.0
    vol = float(series.std()) * np.sqrt(12.0)
    maxdd = float((cum / cum.cummax() - 1.0).min())
    yearly = (1.0 + series).groupby(series.index.year).prod() - 1.0
    return {"ok": True, "dropped": dropped, "used": cols,
            "renorm": dict(zip(cols, (w * 100).round(1))),
            "cum": cum, "cagr": cagr * 100, "vol": vol * 100,
            "maxdd": maxdd * 100, "years": n_years, "yearly": yearly}


with tab_outlook:
    st.subheader("Risk & return estimates")
    st.caption(f"**Assumption set: {assume}.** {ASSUMPTION_NOTES[assume]}")

    # --- Both return figures, side by side and explained ---
    r1, r2, r3 = st.columns([1, 1, 1.4])
    r1.metric("Arithmetic expected return", f"{metrics['ret']:.2f}%",
              help="Weighted average of each holding's expected return. This is the standard "
                   "mean-variance input — but no portfolio actually compounds at this rate.")
    r2.metric("Geometric CAGR (compounded)", f"{metrics['cagr']:.2f}%",
              delta=f"-{metrics['drag']:.2f}pp variance drag", delta_color="inverse",
              help="What wealth actually compounds at, after variance drag. Terminal value follows THIS number.")
    r3.info(
        f"**Why they differ:** volatility itself destroys compounded return. At {metrics['vol']:.1f}% "
        f"volatility the gap is **{metrics['drag']:.2f} percentage points per year**. A portfolio that "
        f"gains 50% then loses 33% has an arithmetic mean of +8.5% but ends exactly flat. "
        f"**Use {metrics['cagr']:.2f}% for any projection of terminal wealth.**"
    )
    if VOL_SCALARS.get(assume):
        st.caption(
            f"ℹ️ **{assume}** pairs historical returns with **historical volatilities** "
            f"(equities {BASELINE_VOLS['VTI'] * VOL_SCALARS[assume]['eq']:.1f}%, small-cap value "
            f"{BASELINE_VOLS['AVUV'] * VOL_SCALARS[assume]['fac']:.1f}%), so the drag above is larger — and the "
            f"drawdown estimate deeper — than under Conservative/Base. That is deliberate: earning historical "
            f"returns meant living through historical volatility."
        )

    st.divider()
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Growth exposure", f"{metrics['growth']:.0f}%")
    m2.metric("Est. volatility", f"{metrics['vol']:.1f}%")
    m3.metric("Sharpe (rf≈3.5%)", f"{metrics['sharpe']:.2f}")
    m4.metric("Est. max drawdown", f"{metrics['dd']:.0f}%")
    m5.metric("Blended ER", f"{metrics['er']:.2f}%")
    m6.metric("Variance drag", f"{metrics['drag']:.2f}pp")

    st.divider()
    st.subheader(f"{years}-year projection")

    base_val = port_total if port_total > 0 else basis

    def project(rate_pct):
        g = 1 + rate_pct / 100
        lump = base_val * g ** years
        contrib = monthly * 12 * years if rate_pct == 0 else monthly * 12 * ((g ** years - 1) / (rate_pct / 100))
        return lump + contrib

    scenarios = pd.DataFrame(
        {
            "Scenario": ["Conservative", "Base", "Optimistic"],
            "Return": [max(0.5, metrics["cagr"] - 2), metrics["cagr"], metrics["cagr"] + 2],
        }
    )
    scenarios["Projected value"] = scenarios["Return"].apply(project)
    scenarios_display = scenarios.copy()
    scenarios_display["Return"] = scenarios_display["Return"].map(lambda x: f"{x:.1f}%")
    scenarios_display["Projected value"] = scenarios_display["Projected value"].map(lambda x: f"${x:,.0f}")
    st.dataframe(scenarios_display, hide_index=True, width='stretch')

    years_range = np.arange(0, years + 1)
    fig2 = go.Figure()
    for label, rate, color in [
        ("Conservative", max(0.5, metrics["cagr"] - 2), "#8AC3AC"),
        ("Base", metrics["cagr"], PINE),
        ("Optimistic", metrics["cagr"] + 2, "#0F4A38"),
    ]:
        g = 1 + rate / 100
        path = base_val * g ** years_range + (monthly * 12 * ((g ** years_range - 1) / (rate / 100)) if rate > 0 else monthly * 12 * years_range)
        fig2.add_trace(go.Scatter(x=years_range, y=path, name=label, line=dict(color=color, width=2)))
    fig2.update_layout(height=320, margin=dict(t=20, b=20, l=20, r=20), paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)", xaxis_title="Years", yaxis_title="Portfolio value ($)")
    st.plotly_chart(fig2, width='stretch', key="projection_chart")

    # ============ PHASE 2: MONTE CARLO ============
    st.divider()
    st.subheader("🎲 Monte Carlo simulation")
    st.caption("The chart above is deterministic — one smooth line per assumption. This runs thousands of random return paths instead.")
    mc_on = st.toggle("Run Monte Carlo", value=False, key="mc_toggle")
    if mc_on:
        n_paths = st.select_slider("Paths", options=[500, 1000, 2000, 5000], value=2000)
        mc = monte_carlo(float(base_val), float(monthly), int(years),
                         float(metrics["ret"]), float(metrics["vol"]), int(n_paths))
        q1, q2, q3, q4 = st.columns(4)
        q1.metric("5th percentile", f"${mc['pct'][5][-1]:,.0f}", help="Bad-luck outcome: 95% of paths ended above this.")
        q2.metric("Median", f"${mc['pct'][50][-1]:,.0f}")
        q3.metric("95th percentile", f"${mc['pct'][95][-1]:,.0f}")
        q4.metric("Odds of ending below contributions", f"{mc['p_loss']:.1f}%",
                  help=f"Total contributed over the period ≈ ${mc['contributed']:,.0f}.")

        yrs_ax = np.arange(0, years + 1)
        mfig = go.Figure()
        for lo, hi, shade in ((5, 95, "rgba(29,107,84,0.12)"), (25, 75, "rgba(29,107,84,0.25)")):
            mfig.add_trace(go.Scatter(x=np.concatenate([yrs_ax, yrs_ax[::-1]]),
                                       y=np.concatenate([mc["pct"][hi], mc["pct"][lo][::-1]]),
                                       fill="toself", fillcolor=shade, line=dict(width=0),
                                       name=f"{lo}–{hi}th percentile", hoverinfo="skip"))
        mfig.add_trace(go.Scatter(x=yrs_ax, y=mc["pct"][50], name="Median",
                                   line=dict(color=PINE, width=3)))
        mfig.update_layout(height=340, margin=dict(t=20, b=20, l=20, r=20),
                            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                            xaxis_title="Years", yaxis_title="Portfolio value ($)",
                            legend=dict(orientation="h", y=1.14))
        st.plotly_chart(mfig, width='stretch', key="monte_carlo_chart")
        st.caption(
            "💡 **Master practitioner tip:** plan around the 25th percentile, not the median — half of all outcomes land "
            "below the median, and 'I hit my number in the average case' is not a plan. The gap between the 5th and 95th "
            "percentile is the honest answer to 'how much do I actually know about my ending wealth?' Note the model draws "
            "returns independently each year, so it understates fat tails, volatility clustering, and sequence-of-returns "
            "risk — real-world bad outcomes cluster worse than this."
        )

    st.divider()
    st.subheader("Glide path (plan §5)")
    phase_idx = 0 if years > 7 else (1 if years > 3 else 2)
    phases = MODE_TEXT[mode]["glide"]
    for i, (label, when, note) in enumerate(phases):
        marker = " · **YOU ARE HERE**" if i == phase_idx else ""
        with st.container(border=True):
            st.markdown(f"**{label}** — *{when}*{marker}")
            st.caption(note)


# ---- Tab: Analytics --------------------------------------------------
with tab_analytics:
    sub_exp, sub_div, sub_glide, sub_stress, sub_cmp, sub_cost, sub_loc, sub_bt = st.tabs(
        ["Effective Exposure", "Diversification & Correlations", "Glide Path",
         "Stress Testing", "Risk Comparison", "Rebalancing Cost",
         "Asset Location", "Backtest"]
    )

with sub_exp:

    # ============ FEATURE 1: EFFECTIVE EXPOSURE & CONCENTRATION ============
    st.subheader("🔍 Effective Exposure & Concentration")
    st.caption(
        "Satellites you buy explicitly are ALSO held inside VTI/SPY/QQQ. This look-through view "
        "shows each name's true total weight, so a '5% NVDA satellite' doesn't quietly become 12% real exposure."
    )
    conc_limit = st.slider("Concentration alert limit (effective % per single name)", 5, 15, 9,
                            help="Best practice: 8–10% max effective single-name exposure for a retirement portfolio.")
    exp_df = compute_effective_exposure(targets)
    if exp_df.empty:
        st.info("No overlapping single names detected in the current targets.")
    else:
        st.dataframe(exp_df, hide_index=True, width='stretch')
        for _, r in exp_df.iterrows():
            eff = r["Effective total %"]
            if eff > conc_limit:
                st.error(f"⚠️ {r['Stock']}: effective exposure {eff:.1f}% exceeds your {conc_limit}% concentration limit — consider trimming the satellite.")
            elif r["Explicit satellite %"] > 0 and eff > per_cap:
                st.warning(f"{r['Stock']}: effective exposure {eff:.1f}% exceeds the {per_cap}% per-name cap once ETF holdings are counted (explicit position alone is within cap).")
    st.caption(
        "💡 **Master practitioner's use:** before adding any single-stock satellite, check its implicit weight here first. "
        "If VTI+QQQ already give you 8% NVDA, a 5% satellite is a 13% concentrated bet — size the satellite off the "
        "*effective* number, not the sticker number. Overlap %s are rough estimates — edit OVERLAP_ESTIMATES with current index weights."
    )

    # Residual crypto held only inside broad index funds — policy-relevant in bank mode.
    _resid = residual_lookthrough(targets, "crypto")
    if _resid and bank_mode:
        _tot = sum(_resid.values())
        st.info(
            "🏦 **Bank-Compliant Mode note — residual indirect exposure.** Excluding crypto satellites removes "
            "every *direct* position, but broad index funds still hold some crypto-linked equities: "
            + ", ".join(f"**{k}** {v:.2f}%" for k, v in sorted(_resid.items(), key=lambda x: -x[1]))
            + f" (total ≈ **{_tot:.2f}%** of the portfolio, held via VTI/QQQ, not purchased directly). "
            "These are below the table's display threshold and are not positions you control. Flagging them because "
            "a strict no-crypto policy cannot reach exactly zero while holding total-market index funds — worth "
            "confirming how your institution treats indirect index exposure."
        )
    elif _resid:
        st.caption(
            "Held only indirectly, inside broad index funds (below display threshold): "
            + ", ".join(f"{k} {v:.2f}%" for k, v in sorted(_resid.items(), key=lambda x: -x[1]))
        )


with sub_div:

    # ============ FEATURE 2: DIVERSIFICATION & CORRELATIONS ============
    st.subheader("🧩 Diversification & Correlations")
    div = compute_diversification(targets, metrics["vol"])
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Diversification ratio", f"{div['dr']:.2f}",
              help="Weighted-avg holding vol ÷ portfolio vol. >1.25 = solid; 1.0 = no diversification benefit.")
    d2.metric("Effective independent bets", f"{div['enb']:.1f}",
              help="≈ DR². How many truly uncorrelated return sources you effectively own.")
    d3.metric("Effective # of holdings", f"{div['eff_holdings']:.1f}",
              help="1 ÷ sum of squared weights (concentration-adjusted holding count).")
    d4.metric("Wtd-avg holding vol", f"{div['wavg_vol']:.1f}%")

    cmat = class_corr_matrix(targets)
    heat = go.Figure(go.Heatmap(
        z=cmat.values, x=list(cmat.columns), y=list(cmat.index),
        colorscale="RdYlGn_r", zmin=-0.2, zmax=1.0,
        text=cmat.round(2).values, texttemplate="%{text}", showscale=True,
    ))
    heat.update_layout(height=340, margin=dict(t=20, b=20, l=20, r=20),
                        paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(heat, width='stretch', key="corr_heatmap")
    st.caption(
        "💡 **Master practitioner's use:** red cells are where diversification dies. Note the crypto↔tech (0.50) and "
        "eq↔tech (0.85) cluster — stacking IBIT + BSOL + NVDA + QQQ adds *volatility*, not diversification. Toggle a "
        "satellite on/off and watch the Diversification Ratio move: if adding a holding lowers DR, it concentrated you."
    )


with sub_glide:

    # ============ FEATURE 3: GLIDE PATH SIMULATOR ============
    st.subheader("🛬 Glide Path & Future Evolution")
    end_growth = st.slider("Target growth % at retirement", 40, 80, 60,
                            help="Plan §5 recommends landing at ~60–65% growth by retirement.")
    glide_df = simulate_glide(targets, int(years), float(end_growth),
                               float(port_total if port_total > 0 else basis), float(monthly),
                               assume_key=assume)
    gfig = go.Figure()
    gfig.add_trace(go.Scatter(x=glide_df["Year"], y=glide_df["Projected value"],
                               name="Projected value ($)", line=dict(color=PINE, width=3), yaxis="y1"))
    gfig.add_trace(go.Scatter(x=glide_df["Year"], y=glide_df["Growth %"],
                               name="Growth sleeve %", line=dict(color=AMBER, width=2, dash="dot"), yaxis="y2"))
    gfig.add_trace(go.Scatter(x=glide_df["Year"], y=glide_df["Satellite %"],
                               name="Satellite %", line=dict(color=LOSS, width=2, dash="dot"), yaxis="y2"))
    gfig.update_layout(
        height=360, margin=dict(t=20, b=20, l=20, r=20), paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", xaxis_title="Years from today",
        yaxis=dict(title="Portfolio value ($)"),
        yaxis2=dict(title="Sleeve %", overlaying="y", side="right", range=[0, 100]),
        legend=dict(orientation="h", y=1.12),
    )
    st.plotly_chart(gfig, width='stretch', key="glide_sim_chart")
    with st.expander("Year-by-year glide table"):
        gd = glide_df.copy()
        gd["Projected value"] = gd["Projected value"].map(lambda x: f"${x:,.0f}")
        st.dataframe(gd, hide_index=True, width='stretch')
    st.caption(
        "💡 **Master practitioner's use:** the de-risking starts at 7 years out (flat before that), and satellites/crypto "
        "shrink proportionally with the growth sleeve — in live execution you'd trim them (and AVDV) *first* using dividends "
        "and new contributions rather than sales, per plan §5. If the ending value at 60% growth is enough for your income "
        "needs, taking more risk than that is optional, not required."
    )


with sub_stress:

    # ============ FEATURE 4: SCENARIO STRESS TESTING ============
    st.subheader("⛈️ Scenario Stress Testing")
    scen_name = st.selectbox("Scenario", list(STRESS_SCENARIOS.keys()) + ["Custom scenario"])
    if scen_name == "Custom scenario":
        st.caption("Set a return shock (%) for each asset class:")
        sc1, sc2, sc3, sc4 = st.columns(4)
        shocks = {
            "eq":     sc1.slider("Equity core", -80, 40, -30, key="shk_eq"),
            "fac":    sc1.slider("Small value", -80, 40, -35, key="shk_fac"),
            "reit":   sc2.slider("REITs", -80, 40, -35, key="shk_reit"),
            "gold":   sc2.slider("Gold", -50, 50, 0, key="shk_gold"),
            "bond":   sc3.slider("Bonds", -30, 30, 0, key="shk_bond"),
            "tips":   sc3.slider("TIPS", -30, 30, 0, key="shk_tips"),
            "crypto": sc4.slider("Crypto", -95, 100, -60, key="shk_crypto"),
            "tech":   sc4.slider("Tech stocks", -80, 40, -40, key="shk_tech"),
        }
    else:
        shocks = STRESS_SCENARIOS[scen_name]
        st.caption("Assumed class shocks: " + " · ".join(f"{k} {v:+d}%" for k, v in shocks.items()))

    res = run_scenario(targets, shocks, metrics["cagr"])
    s1, s2, s3 = st.columns(3)
    s1.metric("Est. portfolio return in scenario", f"{res['loss']:+.1f}%")
    s2.metric("$ impact on current value", f"${(res['loss'] / 100) * (port_total if port_total > 0 else basis):,.0f}")
    s3.metric("Est. recovery time @ blended CAGR", f"{res['recovery']:.1f} yrs" if res["recovery"] else "—")

    contrib_df = (pd.DataFrame([{"Ticker": t, "Loss contribution (pp)": round(v, 2)} for t, v in res["contrib"].items()])
                  .sort_values("Loss contribution (pp)"))
    cfig = go.Figure(go.Bar(
        x=contrib_df["Loss contribution (pp)"], y=contrib_df["Ticker"], orientation="h",
        marker_color=[LOSS if v < 0 else PINE for v in contrib_df["Loss contribution (pp)"]],
    ))
    cfig.update_layout(height=max(240, 26 * len(contrib_df)), margin=dict(t=10, b=10, l=10, r=10),
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        xaxis_title="Contribution to portfolio return (pp)")
    st.plotly_chart(cfig, width='stretch', key="stress_contrib_chart")

    if res["flags"]:
        st.markdown("**How the 5/25 rebalancing bands would react post-shock:**")
        st.dataframe(pd.DataFrame(res["flags"]), hide_index=True, width='stretch')
        st.caption("BUY flags are the discipline paying off — the bands force you to buy what just fell, at lower prices.")
    else:
        st.info("No holdings would breach their 5/25 bands under this shock — the portfolio self-heals within tolerance.")
    st.caption(
        "💡 **Master practitioner's use:** run the 2008 scenario and decide *today* whether you could stomach that dollar "
        "loss without selling — if not, drop a risk level now, in calm markets. Then run the crypto/tech meltdown with your "
        "satellites on vs. off to see exactly what those positions add to tail risk. Shocks are illustrative class-level "
        "estimates, not predictions; correlations in real crises are usually worse than models assume."
    )

with sub_cmp:

    # ============ PHASE 2: SIDE-BY-SIDE RISK LEVEL COMPARISON ============
    st.subheader("⚖️ Risk Level Comparison")
    st.caption("Same satellite inputs, same cap logic — only the core weight table changes.")
    other_idx = st.selectbox(
        "Compare current mandate against:",
        [i for i in range(len(RISK_LEVELS)) if i != risk_idx],
        format_func=lambda i: RISK_LEVELS[i],
    )
    cmp_df = compare_risk_levels(risk_idx, int(other_idx), sat_raw, int(per_cap),
                                 float(basis), int(years), bool(bank_mode), assume)
    st.dataframe(cmp_df.set_index("Risk level").T, width='stretch')

    d_ret = cmp_df.iloc[1]["Est. CAGR %"] - cmp_df.iloc[0]["Est. CAGR %"]
    d_vol = cmp_df.iloc[1]["Est. vol %"] - cmp_df.iloc[0]["Est. vol %"]
    if d_vol != 0:
        st.metric("Extra return per extra unit of volatility",
                  f"{d_ret / d_vol:.2f}",
                  help="Return difference ÷ volatility difference. Below ~0.3 means you're paying a lot of risk for a little return.")
    st.caption(
        "💡 **Master practitioner tip:** don't compare CAGRs in isolation — compare the *ratio* above. "
        "Moving up a risk level that buys 0.4% more return for 3% more volatility is usually a bad trade at "
        "a 15-year horizon; the same 0.4% for 0.8% more vol is worth taking. Also check whether the higher "
        "level's Diversification Ratio falls: if it does, you're being paid less for more concentrated risk."
    )

with sub_cost:

    # ============ PHASE 2: REBALANCING COST ESTIMATOR ============
    st.subheader("🧾 Rebalancing Cost Estimator")
    st.caption("Approximate friction from acting on today's REBALANCE flags in a taxable account.")
    k1, k2, k3 = st.columns(3)
    ltcg = k1.slider("Long-term cap gains rate (%)", 0, 40, 15,
                     help="Federal LTCG (0/15/20%) plus any state tax and the 3.8% NIIT if applicable.")
    unreal = k2.slider("Assumed unrealized gain in sold lots (%)", 0, 100, 30,
                       help="Portion of sale proceeds that is taxable gain. Specific-lot ID lets you pick lower-gain lots.")
    spread = k3.slider("Round-trip spread (bps)", 0, 50, 3,
                       help="1 bp = 0.01%. Broad ETFs run ~1–3 bps; thin single names and new listings are wider.")

    cost = rebalance_cost(port_df, port_total, float(ltcg), float(unreal), float(spread))
    if cost["total"] == 0:
        st.success("No rebalancing needed right now — zero friction to incur.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Sell notional", f"${cost['sell_notional']:,.0f}")
        c2.metric("Est. tax drag", f"${cost['tax']:,.0f}")
        c3.metric("Est. spread cost", f"${cost['spread']:,.0f}")
        c4.metric("Total friction", f"${cost['total']:,.0f}", f"{cost['pct']:.2f}% of portfolio")
        st.info(
            f"Acting on every flag today would cost roughly **${cost['total']:,.0f}** "
            f"({cost['pct']:.2f}% of the portfolio). The plan's cash-flow-first hierarchy exists to avoid most of this: "
            f"dividends and new contributions rebalance at **zero** tax cost."
        )
    st.caption(
        "💡 **Master practitioner tip:** compare this friction against the drift you'd be correcting. If a holding is "
        "1.5pp past its band and fixing it costs 0.4% of the portfolio in tax, wait and steer cash flows instead — "
        "the band is a trigger to *consider* rebalancing, not an obligation to trade the same day. In tax-advantaged "
        "accounts, set the tax rate to 0 and rebalance freely."
    )

with sub_loc:

    # ============ PHASE 3: ASSET LOCATION OPTIMIZER ============
    st.subheader("🏦 Asset Location Optimizer")
    st.caption(
        "Same dollars, same funds, different accounts — asset location alone is worth an estimated "
        "0.1–0.3%/yr after tax. Recommendations follow plan §6."
    )
    loc_df = asset_location_table(targets)
    st.dataframe(loc_df, hide_index=True, width='stretch')
    st.info(
        "**Fill order:** max any employer match first, then Roth IRA with the highest-return sleeves, "
        "then traditional 401(k)/IRA with the income-heavy sleeves (BND/VTIP/VNQ), and only then place "
        "the remainder in taxable — equities and VXUS first."
    )
    st.caption(
        "💡 **Master practitioner tip:** asset location only matters if you actually have tax-advantaged "
        "space — this table is a priority list, not a mandate. If everything must sit in taxable, swap BND "
        "for a municipal fund (VTEB) when your bracket is 32%+, and keep VNQ small. Never let location "
        "override allocation: the right portfolio in the wrong accounts beats the wrong portfolio in the "
        "right ones. This is educational, not tax advice — brackets and state rules change the math."
    )

with sub_bt:

    # ============ PHASE 3: HISTORICAL BACKTEST ============
    st.subheader("📜 Historical Backtest")
    st.caption(
        "Runs TODAY's target weights against actual monthly history (monthly-rebalanced). "
        "Note the built-in hindsight: you're testing the survivors you already chose."
    )
    bt_period = st.selectbox("Lookback window", ["5y", "10y"], index=1, key="bt_period")
    if yf is None:
        st.warning("`yfinance` is required for backtesting — install it and restart.")
    elif st.button("Run backtest", key="bt_run", width='stretch'):
        bt_weights = {t: p for t, p in targets.items() if p > 0.05}
        with st.spinner("Downloading monthly history…"):
            hist = fetch_history(tuple(sorted(bt_weights)), bt_period)
        if hist.empty:
            st.error("Could not download price history (network or yfinance issue) — try again in a minute.")
        else:
            bt = run_backtest(hist, bt_weights)
            if not bt["ok"]:
                st.error("Not enough overlapping history across these holdings to backtest.")
                if bt["dropped"]:
                    st.caption(f"Insufficient history: {', '.join(bt['dropped'])}")
            else:
                if bt["dropped"]:
                    st.warning(
                        f"Excluded for insufficient history: **{', '.join(bt['dropped'])}** — new listings "
                        f"like SPCX and BSOL simply don't have enough data. Remaining weights renormalized: "
                        + ", ".join(f"{t} {w}%" for t, w in bt["renorm"].items())
                    )
                hb1, hb2, hb3, hb4 = st.columns(4)
                hb1.metric("Realized CAGR", f"{bt['cagr']:.1f}%")
                hb2.metric("Realized volatility", f"{bt['vol']:.1f}%")
                hb3.metric("Max drawdown", f"{bt['maxdd']:.1f}%")
                hb4.metric("Window", f"{bt['years']:.1f} yrs")

                bfig = go.Figure(go.Scatter(x=bt["cum"].index, y=bt["cum"].values * 10_000,
                                             line=dict(color=PINE, width=2.5), name="Growth of $10k"))
                bfig.update_layout(height=300, margin=dict(t=20, b=20, l=20, r=20),
                                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                    yaxis_title="Growth of $10,000")
                st.plotly_chart(bfig, width='stretch', key="bt_growth_chart")

                yfig = go.Figure(go.Bar(x=[str(y) for y in bt["yearly"].index],
                                         y=(bt["yearly"].values * 100),
                                         marker_color=[PINE if v >= 0 else LOSS for v in bt["yearly"].values]))
                yfig.update_layout(height=240, margin=dict(t=10, b=10, l=10, r=10),
                                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                    yaxis_title="Annual return (%)")
                st.plotly_chart(yfig, width='stretch', key="bt_yearly_chart")
                st.caption(
                    "💡 **Master practitioner tip:** compare the realized CAGR/vol/drawdown here against the "
                    "*assumptions* on the Outlook tab — if realized vol was 14% and the model assumes 12.8%, the "
                    "model is flattering you. The most useful number is the max drawdown: that's what this exact "
                    "mix actually did, not a stylized scenario. Past performance does not guarantee future "
                    "results — a 5–10 year window is one regime, not a law of nature."
                )

# ---- Phase 2: look-through-aware order notes, rendered into the Drift tab ----
with lookthrough_slot:
    st.markdown("**Look-through concentration check**")
    lt = lookthrough_flags(targets, float(conc_limit))
    if not lt:
        st.caption("No single-name concentration issues detected in the current targets.")
    else:
        for tkr, msg in lt.items():
            st.warning(f"**{tkr}** — {msg}")
        st.caption(
            "💡 **Master practitioner tip:** rebalancing back to target can quietly *increase* a concentration problem — "
            "buying more VTI or QQQ to fix an underweight also buys more of whatever single name is already too large. "
            "When a name is flagged here, fix the drift by buying a *different* underweight sleeve (international, bonds, "
            "gold) rather than the ETF that compounds the concentration."
        )

# ---- Tab: Dashboard (rendered last; reads metrics computed above) -----
with tab_dash:
    st.subheader("🩺 Portfolio Health")

    # Reference: volatility of the PURE core model at this same risk level,
    # i.e. the risk the mandate implies before any satellites were added.
    _core_only = compute_targets(risk_idx, {t: 0.0 for t in SAT_TICKERS}, 0.0)
    ref_vol = compute_metrics(_core_only)["vol"]

    health = compute_health(targets, port_df, div, exp_df, float(conc_limit),
                            ref_vol=ref_vol, port_vol=metrics["vol"])
    hcolor = PINE if health["score"] >= 80 else (AMBER if health["score"] >= 60 else LOSS)

    top = st.columns([1.2, 1, 1, 1, 1])
    top[0].markdown(
        f"<div class='ledger-invert' style='background:{hcolor};color:#fff;border-radius:10px;padding:14px;text-align:center;'>"
        f"<div style='font-size:0.7rem;letter-spacing:.08em;opacity:.85;'>HEALTH SCORE</div>"
        f"<div style='font-family:monospace;font-size:2rem;font-weight:700;line-height:1.1;'>{health['score']:.0f}</div>"
        f"<div style='font-size:0.8rem;'>{health['band']}</div></div>",
        unsafe_allow_html=True,
    )
    top[1].metric("Diversification ratio", f"{div['dr']:.2f}")
    top[2].metric("Max effective single name", f"{health['max_conc']:.1f}%")
    top[3].metric("Holdings out of band", f"{health['out_of_band']}")
    top[4].metric("Crypto + tech weight", f"{health['spec']:.1f}%")

    st.divider()
    st.markdown("**Score breakdown** — each pillar is worth 20 points:")
    pillars = [
        ("Diversification", health["p_div"],
         f"{health['enb']:.1f} effective independent bets (4.0 = full marks) · largest single risk contributor {health['top_rc']:.0%}"),
        ("Concentration", health["p_conc"],
         f"worst effective single name {health['max_conc']:.1f}% vs {conc_limit:.0f}% limit"),
        ("Drift discipline", health["p_drift"],
         f"{health['out_of_band']} holding(s) outside 5/25 bands"),
        ("Speculation control", health["p_spec"],
         f"crypto + tech = {health['spec']:.1f}% (20% soft ceiling)"),
        ("Risk vs mandate", health["p_vol"],
         f"vol {metrics['vol']:.1f}% vs {ref_vol:.1f}% for the pure {RISK_LEVELS[risk_idx]} core "
         f"({'+' if health['overshoot'] else ''}{health['overshoot']:.0f}% overshoot from satellites)"),
    ]
    for label, val, note in pillars:
        icon = "🟢" if val >= 80 else ("🟡" if val >= 50 else "🔴")
        pc1, pc2 = st.columns([1, 3])
        pc1.markdown(f"{icon} **{label}**")
        with pc2:
            st.progress(min(1.0, val / 100.0), text=note)

    # Actionable alerts
    st.divider()
    alerts = []
    if health["out_of_band"] > 0:
        alerts.append(("error", f"{health['out_of_band']} holding(s) are outside their 5/25 bands — see the Drift & Orders tab."))
    if health["max_conc"] > conc_limit:
        alerts.append(("error", f"Effective single-name concentration hits {health['max_conc']:.1f}%, above your {conc_limit:.0f}% limit."))
    if div["dr"] < 1.15:
        alerts.append(("warning", f"Diversification ratio is only {div['dr']:.2f} — holdings are moving together more than they should."))
    if health["spec"] > 25:
        alerts.append(("warning", f"Crypto + tech is {health['spec']:.1f}% of the portfolio — a single correlated cluster driving most of your risk."))
    if not st.session_state.prices_live:
        alerts.append(("warning", "Prices are placeholders — fetch live quotes before acting on any number here."))
    for t in ("SPCX", "BSOL"):
        if targets.get(t, 0) > 0:
            alerts.append(("warning", NEW_LISTING_WARNING[t]))
    if port_total == 0:
        alerts.append(("info", "No holdings entered yet. Load the model in the Targets tab, or type real share counts in Holdings, to activate drift monitoring."))

    if alerts:
        for kind, msg in alerts:
            getattr(st, kind)(msg)
    else:
        st.success("No alerts — allocation is on target, diversified, and within all concentration limits.")

    # ---- Exports ----
    st.divider()
    st.markdown("**Export**")
    frames = build_export_frames(targets, port_df, float(basis), float(port_total))
    e1, e2, e3, e4 = st.columns(4)
    e1.download_button("⬇️ Targets CSV", frames["targets"].to_csv(index=False),
                       "targets.csv", "text/csv", width='stretch')
    e2.download_button("⬇️ Holdings CSV",
                       frames["holdings"].to_csv(index=False) if not frames["holdings"].empty else "no holdings\n",
                       "holdings.csv", "text/csv", width='stretch',
                       disabled=frames["holdings"].empty)
    e3.download_button("⬇️ Drift & orders CSV",
                       frames["orders"].to_csv(index=False) if not frames["orders"].empty else "no holdings\n",
                       "drift_orders.csv", "text/csv", width='stretch',
                       disabled=frames["orders"].empty)
    ips = build_ips_text(targets, RISK_LEVELS[risk_idx], float(basis), int(years), metrics, health,
                         mode_name=mode, assume_name=assume, bank=bool(bank_mode))
    e4.download_button("⬇️ IPS summary (.txt)", ips, "investment_policy_statement.txt",
                       "text/plain", width='stretch')
    with st.expander("Preview IPS summary"):
        st.code(ips, language=None)

    st.caption(
        "💡 **Master practitioner tip:** the Health Score is a conversation starter, not a verdict — a deliberately "
        "concentrated bet you fully understand can be the right call even though it scores badly. What matters is that "
        "every red pillar is a *choice you made on purpose*, not a drift you never noticed. Export the IPS text and "
        "keep it dated: comparing this quarter's score to last quarter's tells you whether the portfolio is drifting "
        "away from your intentions over time."
    )

with tab_guide:
    st.subheader("Quick start")
    st.markdown(
        """
1. **Set your mandate** in the sidebar — risk level (Mid-High matches the plan's default), basis, contributions, and years to retirement.
2. **Optionally add satellites** — pick single stocks from the multiselect and/or enable IBIT, then set each one's target %. Caps (per-name, total crypto, total satellite budget) are enforced automatically with warnings if you exceed them.
3. **Targets tab** — review the resulting pie chart and $ allocation at your chosen risk level.
4. **Holdings tab** — click *Load model at these targets* to auto-fill shares at current prices, or type in your real share counts.
5. **Fetch live prices** in the sidebar (needs internet + `yfinance`) before trusting any drift or order numbers.
6. **Drift & Orders tab** — see exactly which holdings are outside their 5/25 band and the tax-aware buy/sell amounts to fix it.
7. **Outlook tab** — check estimated return/vol/Sharpe/drawdown, run the {y}-year projection, and see where you sit on the glide path.
        """.format(y=years)
    )

    st.subheader("Recommended workflow")
    st.markdown(
        """
- **Quarterly (15 min):** check the Drift tab; direct any accumulated dividends and new contributions to underweight holdings first — that alone resolves most drift with zero tax cost.
- **Annually (December):** full review — rebalance anything still flagged, harvest losses using the TLH partner funds shown next to each SELL suggestion, and revisit your risk level and glide-path phase.
- **When life changes:** update the risk level slider any time; every target, band, and projection recalculates instantly.
- **Crypto/single stocks:** treat as high-volatility satellites only — the caps exist to stop concentration risk from quietly creeping up.
        """
    )

    st.subheader("Next steps outside this tool")
    st.markdown(
        """
1. Answer the account-structure questions from the plan (§9): tax bracket, other retirement accounts, emergency fund, income needs.
2. Open a brokerage account (Fidelity/Schwab/Vanguard), set cost basis to **Specific Lot Identification**, and turn off automatic dividend reinvestment.
3. Deploy new capital per the plan's §6 approach — roughly half as a lump sum, the rest dollar-cost-averaged over ~5–6 months.
4. Write a one-page Investment Policy Statement recording your chosen risk level, targets, and rebalancing rules — the document you write calmly is what keeps you disciplined in a selloff.
        """
    )
    st.info("Questions or a bug? Paste this script plus the error into your AI assistant for help customizing it.")

    st.divider()
    st.subheader("🚀 Deploy & share")
    st.markdown(
        """
**Streamlit Community Cloud (free, gives you a shareable link):**
1. Create a GitHub repository containing two files: `portfolio_manager.py` and `requirements.txt`.
2. Go to **share.streamlit.io**, sign in with GitHub, click **New app**, and point it at the repo
   (main file: `portfolio_manager.py`).
3. It builds automatically and gives you a permanent `https://<yourapp>.streamlit.app` link that
   works on any phone or laptop.
4. Under the app's settings you can make it **private** (viewers must be invited by email) — recommended
   before sharing anything with real holdings in it.

**Hugging Face Spaces (alternative):** create a Space with the *Streamlit* SDK, upload the same two
files, done.

**Notes:** the app has no API keys or secrets, so no extra configuration is needed. Live prices come
from Yahoo via `yfinance`, which occasionally rate-limits shared cloud servers — if a fetch fails,
try again in a minute or enter prices manually. Anyone with the link sees a fresh session (your
holdings are NOT shared — each visitor starts blank; use Save/Load configs to move your setup
between devices).
        """
    )

st.divider()
st.caption(
    "⚠️ Educational tool only — not investment, tax, or legal advice. Returns, volatilities, "
    "correlations, drawdowns, and projections are rough estimates from historical asset-class "
    "behavior; actual results will differ, and crypto/single-stock estimates are especially "
    "uncertain. Live prices depend on yfinance/Yahoo Finance data and may be delayed or unavailable "
    "— verify with your broker before trading. Past performance does not guarantee future results. "
    "Consult a licensed fiduciary financial advisor and a qualified tax professional before acting."
)
