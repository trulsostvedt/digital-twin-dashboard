# -*- coding: utf-8 -*-
"""
Digital Twin — Sustainability Dashboard (Lean & Polished)
- Focused UX: KPIs → Monthly Leaderboard → Latest Submissions → Class Detail
- Dark/Light theme aware (CSS + Plotly)
- Robust timestamp parsing; flexible scoring from form answers
- Natural class sort (1A, 1B, …, 10A)
- Embedded Google Form in a separate tab
"""

import re
from io import StringIO
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import streamlit as st
from streamlit_autorefresh import st_autorefresh
import streamlit.components.v1 as components


# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Digital Twin – Sustainability Dashboard",
    page_icon="🌿",
    layout="wide",
)

# Branding (optional)
APP_TITLE = "Digital Twin — Sustainability Dashboard"
APP_SUBTITLE = "Daily sustainability log and monthly class leaderboard"

# Fixed Google Sheets CSV (Answer Log tab)
SHEET_CSV_URL = (
    "https://docs.google.com/spreadsheets/d/"
    "1I9o3wvPS73huWO5_lLenhylSmjfMzDfnuR3kr4GcK34/"
    "gviz/tq?tqx=out:csv&sheet=Answer%20Log"
)

# Embedded Google Form (same form your teachers use)
GOOGLE_FORM_URL = (
    "https://docs.google.com/forms/d/e/1FAIpQLSefFkxKJE8sYn0Zsn_cxZ-fesYpCEPrLClbnz22pkWuT4MZ4g/viewform?usp=sf_link"
)

# ----------------------------------------------------------------------
# THEME-AWARE CSS (keeps it sleek in dark & light)
# ----------------------------------------------------------------------
THEME_CSS = """
<style>
:root { --radius: 14px; }
.block-container { padding-top: 0.75rem; padding-bottom: 0.75rem; }

/* Header */
.header-card {
  background: var(--secondary-background-color);
  border: 1px solid var(--background-color);
  border-radius: var(--radius);
  padding: 16px 18px;
  margin-bottom: 6px;
}

/* Metrics */
.kpi-row [data-testid="stMetric"]{
  background: var(--secondary-background-color);
  border: 1px solid var(--background-color);
  border-radius: var(--radius);
  padding: 10px 12px;
  box-shadow: 0 1px 8px rgba(0,0,0,0.06);
}

/* Section cards */
.section-card {
  background: var(--secondary-background-color);
  border: 1px solid var(--background-color);
  border-radius: var(--radius);
  padding: 14px;
  box-shadow: 0 1px 8px rgba(0,0,0,0.04);
  margin-bottom: 10px;
}

/* DataFrames */
div[data-testid="stDataFrame"] {
  border: 1px solid var(--secondary-background-color);
  border-radius: var(--radius);
  box-shadow: 0 1px 8px rgba(0,0,0,0.04);
}

/* Typography */
h1,h2,h3 { letter-spacing: .2px; }
hr, .st-emotion-cache-12w0qpk { border-color: var(--secondary-background-color)!important; }
</style>
"""
st.markdown(THEME_CSS, unsafe_allow_html=True)

def plotly_template() -> str:
    base = st.get_option("theme.base")
    return "plotly_dark" if str(base).lower() == "dark" else "plotly_white"


# ----------------------------------------------------------------------
# AUTO-REFRESH
# ----------------------------------------------------------------------
with st.sidebar:
    st.title("Refresh")
    refresh_sec = st.slider("Auto-refresh every (seconds)", 15, 300, 60)
_ = st_autorefresh(interval=refresh_sec*1000, key="auto")


# ----------------------------------------------------------------------
# HTTP (retry) + CACHE
# ----------------------------------------------------------------------
def _requests_session() -> requests.Session:
    s = requests.Session()
    retries = Retry(
        total=3, backoff_factor=0.4,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"])
    )
    s.mount("https://", HTTPAdapter(max_retries=retries))
    s.mount("http://", HTTPAdapter(max_retries=retries))
    return s

@st.cache_data(ttl=20, show_spinner=False)
def fetch_csv(url: str) -> pd.DataFrame:
    r = _requests_session().get(url, timeout=10)
    r.raise_for_status()
    return pd.read_csv(StringIO(r.text))


# ----------------------------------------------------------------------
# DATA CLEANING & SCORING
# ----------------------------------------------------------------------
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.strip() for c in df.columns]
    ts_col = next((c for c in df.columns if "timestamp" in c.lower()), None)

    if ts_col:
        s = df[ts_col].astype(str).str.strip()

        # Prefer ISO (e.g. 2025-10-18 19:22:05)
        dt = pd.to_datetime(s, errors="coerce", utc=False)

        # Fallback for "16.10.2025 kl. 12.13.18"
        mask = dt.isna()
        if mask.any():
            s2 = s[mask].str.replace(r"\s*kl\.?\s*", " ", regex=True, case=False)
            s2 = s2.str.replace(r"(\d{1,2})\.(\d{2})\.(\d{2})(?!\d)", r"\1:\2:\3", regex=True)
            dt2 = pd.to_datetime(s2, errors="coerce", dayfirst=True)
            dt.loc[mask] = dt2

        df["Timestamp_dt"] = dt
        df["Date"] = pd.to_datetime(df["Timestamp_dt"], errors="coerce").dt.date
        df["YearMonth"] = pd.to_datetime(df["Timestamp_dt"], errors="coerce").dt.to_period("M").astype(str)
    else:
        df["Timestamp_dt"] = pd.NaT
        df["Date"] = pd.NaT
        df["YearMonth"] = np.nan

    return df

def infer_class_col(df: pd.DataFrame) -> str:
    for p in ["What is your class?", "Which class?", "Class"]:
        if p in df.columns:
            return p
    opts = [c for c in df.columns if "class" in c.lower()]
    return opts[0] if opts else df.columns[0]

def count_yeses(text: str) -> int:
    return text.count("Yes") if isinstance(text, str) else 0

def score_heater(val: str) -> int:
    if not isinstance(val, str): return 0
    s = val.lower()
    if "did not use the heater" in s: return 1
    if "windows and the door" in s:   return 2
    if "closed the windows" in s:     return 1
    if "closed the door" in s:        return 1
    if "did not close anything" in s: return 0
    return 0

def _num(v) -> float:
    try: return float(str(v).replace(",", "."))
    except: return 0.0

def score_plastic(g) -> float:
    return min(round(_num(g)/100, 1), 5)

def score_paper(g) -> float:
    return min(round(_num(g)/100, 1), 5)

def score_garden(water, collect, plant) -> int:
    pts = 0
    if isinstance(water, str) and water.strip() == "Yes": pts += 1
    if isinstance(collect, str) and "Yes, we collected some plants" in collect: pts += 1
    if isinstance(plant, str) and "Yes, we planted new things!" in plant: pts += 1
    return pts

def ensure_points(df: pd.DataFrame) -> pd.DataFrame:
    cols = df.columns
    if all(c in cols for c in ["Lights pts","Heater pts","Plastic pts","Paper pts","Garden pts"]) and "Total pts" in cols:
        return df

    C_LIGHTS  = next((c for c in cols if "turn off the lights" in c.lower()), None)
    C_HEATER  = next((c for c in cols if "using the heater" in c.lower()), None)
    C_PLAST   = next((c for c in cols if "grams of plastic" in c.lower()), None)
    C_PAPER   = next((c for c in cols if "grams of paper" in c.lower() or "carton" in c.lower()), None)
    C_WATER   = next((c for c in cols if "water the plants" in c.lower()), None)
    C_COLLECT = next((c for c in cols if "collect any plants" in c.lower()), None)
    C_PLANT   = next((c for c in cols if "plant any seeds" in c.lower() or "planted new things" in c.lower()), None)

    df["Lights pts"]  = df[C_LIGHTS].apply(count_yeses) if C_LIGHTS in cols else 0
    df["Heater pts"]  = df[C_HEATER].apply(score_heater) if C_HEATER in cols else 0
    df["Plastic pts"] = df[C_PLAST].apply(score_plastic) if C_PLAST in cols else 0
    df["Paper pts"]   = df[C_PAPER].apply(score_paper)   if C_PAPER in cols else 0

    df["Garden pts"]  = [
        score_garden(w, c, p)
        for w,c,p in zip(df[C_WATER] if C_WATER in cols else [None]*len(df),
                         df[C_COLLECT] if C_COLLECT in cols else [None]*len(df),
                         df[C_PLANT] if C_PLANT in cols else [None]*len(df))
    ]
    df["Total pts"]   = df[["Lights pts","Heater pts","Plastic pts","Paper pts","Garden pts"]].sum(axis=1)
    return df

def load_data() -> Tuple[pd.DataFrame, str]:
    df = fetch_csv(SHEET_CSV_URL)
    df = normalize_columns(df)
    # drop helper "Result..." columns from Forms
    drop_like = [c for c in df.columns if c.strip().lower().startswith("result")]
    if drop_like:
        df = df.drop(columns=drop_like)
    class_col = infer_class_col(df)
    df = ensure_points(df)
    return df, class_col

# Natural class sorting key: 1A, 1B, …, 10A, 10B
def class_sort_key(x: str):
    if not isinstance(x, str): return (9999, "")
    s = x.strip()
    m = re.search(r"(\d+)\s*([A-Za-z]+)", s)
    if m:
        return (int(m.group(1)), m.group(2).lower())
    return (9998, s.lower())


# ----------------------------------------------------------------------
# SIMPLE PLOTTING
# ----------------------------------------------------------------------
def bar_series(s: pd.Series, height: int=360):
    s = s.copy()
    if s.index.dtype == object:
        s = s.reindex(sorted(s.index, key=class_sort_key))
    fig = px.bar(
        x=s.index.astype(str), y=s.values,
        labels={"x":"", "y":""},
        template=plotly_template()
    )
    fig.update_layout(margin=dict(l=0,r=0,t=10,b=0), height=height)
    st.plotly_chart(fig, use_container_width=True, theme=None)


# ----------------------------------------------------------------------
# APP LAYOUT
# ----------------------------------------------------------------------
tabs = st.tabs(["Dashboard", "Submit log"])

# ===== TAB 2: SUBMIT =====
with tabs[1]:
    st.title("Submit log")
    st.write("Register points via the form below. Submissions appear in the dashboard automatically.")
    components.iframe(GOOGLE_FORM_URL, height=1200)

# ===== TAB 1: DASHBOARD =====
with tabs[0]:
    # Header
    st.markdown('<div class="header-card">', unsafe_allow_html=True)
    st.title(APP_TITLE)
    st.caption(APP_SUBTITLE)
    st.markdown('</div>', unsafe_allow_html=True)

    try:
        df, CLASS = load_data()
    except Exception as e:
        st.error(
            "Could not load data from Google Sheets. "
            "Ensure the spreadsheet is shared as 'Anyone with the link → Viewer'.\n\nDetails: {}".format(e)
        )
        st.stop()

    if df.empty:
        st.warning("No data yet. Submit a response in the form.")
        st.stop()

    # Sidebar filters
    with st.sidebar:
        st.title("Filters")
        classes_all = sorted([c for c in df[CLASS].dropna().unique().tolist()], key=class_sort_key)
        picked = st.multiselect("Class", classes_all, default=classes_all)

        if df["Date"].notna().any():
            dmin, dmax = pd.to_datetime(df["Date"]).min(), pd.to_datetime(df["Date"]).max()
            date_range = st.date_input("Date range", value=(dmin, dmax))
        else:
            date_range = (None, None)

        st.markdown("---")
        st.subheader("Leaderboard month")
        months_all = sorted([m for m in df["YearMonth"].dropna().unique()])
        current_month = pd.Timestamp.now().strftime("%Y-%m")
        default_month = months_all.index(current_month) if current_month in months_all else (len(months_all)-1 if months_all else 0)
        sel_month = st.selectbox("Month", months_all if months_all else ["–"], index=max(default_month, 0))
        metric = st.radio("Ranking by", ["Total points", "Average points"], index=0)

    # Apply filters
    view = df.copy()
    if picked:
        view = view[view[CLASS].isin(picked)]
    if date_range[0] and date_range[1]:
        view = view[(pd.to_datetime(view["Date"]) >= pd.to_datetime(date_range[0]))
                    & (pd.to_datetime(view["Date"]) <= pd.to_datetime(date_range[1]))]

    view["Day"] = pd.to_datetime(view["Timestamp_dt"], errors="coerce").dt.floor("D")
    month_view = view[view["YearMonth"] == sel_month]

    # KPIs
    st.markdown('<div class="kpi-row">', unsafe_allow_html=True)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Submissions", f"{len(view):,}")
    c2.metric("Avg points / submission", f"{pd.to_numeric(view['Total pts'], errors='coerce').mean():.2f}" if len(view) else "–")
    today = pd.Timestamp.now().floor("D")
    today_pts = pd.to_numeric(view.loc[view["Day"]==today, "Total pts"], errors="coerce").sum()
    c3.metric("Today total", int(today_pts) if pd.notna(today_pts) else 0)
    best = pd.to_numeric(view["Total pts"], errors="coerce").max()
    c4.metric("Highest single score", 0 if pd.isna(best) else int(best))

    # Monthly top class KPI
    if not month_view.empty:
        s_month = (month_view.groupby(CLASS)["Total pts"]
                   .apply(lambda x: pd.to_numeric(x, errors="coerce").sum()
                          if metric=="Total points"
                          else pd.to_numeric(x, errors="coerce").mean()))
        s_month = s_month.reindex(sorted(s_month.index, key=class_sort_key))
        if not s_month.empty:
            c5.metric("Top class (monthly)", f"{s_month.idxmax()} ({s_month.max():.2f})")
        else:
            c5.metric("Top class (monthly)", "–")
    else:
        c5.metric("Top class (monthly)", "–")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("")

    # Monthly leaderboard (single hero chart)
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader(f"Monthly leaderboard — {sel_month} ({metric})")
    if not month_view.empty:
        s = (month_view.groupby(CLASS)["Total pts"]
             .apply(lambda x: pd.to_numeric(x, errors="coerce").sum()
                    if metric=="Total points"
                    else pd.to_numeric(x, errors="coerce").mean()))
        bar_series(s)
    else:
        st.info("No submissions for the selected month with the current filters.")
    st.markdown('</div>', unsafe_allow_html=True)

    # Latest submissions (wide)
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Latest submissions")
    base_cols = ["Timestamp_dt", CLASS, "Total pts", "Lights pts", "Heater pts", "Plastic pts", "Paper pts", "Garden pts"]
    show_cols = [c for c in base_cols if c in view.columns]
    latest = (view.sort_values("Timestamp_dt", ascending=False)[show_cols]
              .rename(columns={"Timestamp_dt":"Timestamp"}).head(50))
    st.dataframe(latest, use_container_width=True, height=420)
    st.download_button(
        label="Download filtered data (CSV)",
        data=view.to_csv(index=False).encode("utf-8"),
        file_name="digital-twin_filtered.csv",
        mime="text/csv",
        use_container_width=True
    )
    st.markdown('</div>', unsafe_allow_html=True)

    # Class detail
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Class detail")
    classes_sorted = sorted([c for c in df[CLASS].dropna().unique().tolist()], key=class_sort_key)
    d1, d2 = st.columns([1,1])
    pick_one = d1.selectbox("Select class", classes_sorted if classes_sorted else ["—"])
    sub = view[view[CLASS]==pick_one].copy()

    if not sub.empty:
        d1.write(f"Average points: {pd.to_numeric(sub['Total pts'], errors='coerce').mean():.2f}  |  Submissions: {len(sub)}")
        cat_cols = [c for c in ["Lights pts","Heater pts","Plastic pts","Paper pts","Garden pts"] if c in sub.columns]
        if cat_cols:
            s_cat = sub[cat_cols].apply(pd.to_numeric, errors="coerce").mean().sort_values(ascending=True)
            # Use same minimalist bar
            bar_series(s_cat, height=300)
        recent = (sub.sort_values("Timestamp_dt", ascending=False)[["Timestamp_dt","Total pts"] + [c for c in cat_cols]]
                  .rename(columns={"Timestamp_dt":"Timestamp"}).head(20))
        d2.dataframe(recent, use_container_width=True, height=340)
    else:
        st.info("No rows for this class with current filters.")
    st.markdown('</div>', unsafe_allow_html=True)
