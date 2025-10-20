# -*- coding: utf-8 -*-
"""
Digital Twin — Sustainability Dashboard
- Dark/Light theme aware
- Robust timestamp parsing (ISO + legacy 'kl.' formats)
- Monthly leaderboard (with current leader + last-month winner callouts)
- Category breakdown, overall leaderboard (avg), points-over-time
- Latest submissions (full width), class detail
- Natural class sorting (1A, 1B, …, 10A, 10B)
- Embedded Google Form
- Retry + cache + graceful error messages
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

# Google Sheets CSV (fixed “Answer Log” tab)
SHEET_CSV_URL = (
    "https://docs.google.com/spreadsheets/d/"
    "1I9o3wvPS73huWO5_lLenhylSmjfMzDfnuR3kr4GcK34/"
    "gviz/tq?tqx=out:csv&sheet=Answer%20Log"
)

# Embedded Google Form (for submission inside the app)
GOOGLE_FORM_URL = (
    "https://docs.google.com/forms/d/e/1FAIpQLSefFkxKJE8sYn0Zsn_cxZ-fesYpCEPrLClbnz22pkWuT4MZ4g/viewform?usp=sf_link"
)

# ----------------------------------------------------------------------
# THEME-AWARE CSS (works for dark & light)
# ----------------------------------------------------------------------
THEME_CSS = """
<style>
:root { --radius: 14px; }
[data-testid="stMetric"] {
  background: var(--secondary-background-color);
  border: 1px solid var(--background-color);
  border-radius: var(--radius);
  padding: 12px 12px;
  box-shadow: 0 1px 8px rgba(0,0,0,0.06);
}
div[data-testid="stDataFrame"] {
  border: 1px solid var(--secondary-background-color);
  border-radius: var(--radius);
  box-shadow: 0 1px 8px rgba(0,0,0,0.04);
}
hr, .st-emotion-cache-12w0qpk {
  border-color: var(--secondary-background-color) !important;
}
h1, h2, h3 { letter-spacing: 0.2px; }
.block-container { padding-top: 1.0rem; padding-bottom: 1.0rem; }
</style>
"""
st.markdown(THEME_CSS, unsafe_allow_html=True)

# ----------------------------------------------------------------------
# AUTO-REFRESH
# ----------------------------------------------------------------------
with st.sidebar:
    st.title("Auto-refresh")
    refresh_sec = st.slider("Refresh every (seconds)", 15, 300, 60)
_ = st_autorefresh(interval=refresh_sec * 1000, key="auto")

# ----------------------------------------------------------------------
# HTTP (retry) + CACHE
# ----------------------------------------------------------------------
def _requests_session() -> requests.Session:
    session = requests.Session()
    retries = Retry(
        total=3, backoff_factor=0.4,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"])
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session

@st.cache_data(ttl=10, show_spinner=False)
def fetch_csv(url: str) -> pd.DataFrame:
    s = _requests_session()
    r = s.get(url, timeout=10)
    r.raise_for_status()
    return pd.read_csv(StringIO(r.text))

# ----------------------------------------------------------------------
# DATA CLEANING & SCORING
# ----------------------------------------------------------------------
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.strip() for c in df.columns]

    # Timestamp column autodetect
    ts_col = next((c for c in df.columns if "timestamp" in c.lower()), None)

    if ts_col:
        s = df[ts_col].astype(str).str.strip()

        # 1) Try ISO & typical formats first (handles "2025-10-18 19:22:05")
        dt = pd.to_datetime(s, errors="coerce", utc=False)

        # 2) Fallback for old “kl.” format like "16.10.2025 kl. 12.13.18"
        mask = dt.isna()
        if mask.any():
            s2 = s[mask].str.replace(r"\s*kl\.?\s*", " ", regex=True, case=False)
            s2 = s2.str.replace(r"(\d{1,2})\.(\d{2})\.(\d{2})(?!\d)", r"\1:\2:\3", regex=True)
            dt2 = pd.to_datetime(s2, errors="coerce", dayfirst=True)
            dt.loc[mask] = dt2

        df["Timestamp_dt"] = dt
        # Derive Date/YearMonth robustly
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

def score_plastic(g) -> float:
    try: v = float(str(g).replace(",", "."))
    except: return 0.0
    return min(round(v/100, 1), 5)

def score_paper(g) -> float:
    try: v = float(str(g).replace(",", "."))
    except: return 0.0
    return min(round(v/100, 1), 5)

def score_garden(water, collect, plant) -> int:
    pts = 0
    if isinstance(water, str) and water.strip() == "Yes": pts += 1
    if isinstance(collect, str) and "Yes, we collected some plants" in collect: pts += 1
    if isinstance(plant, str) and "Yes, we planted new things!" in plant: pts += 1
    return pts

def ensure_points(df: pd.DataFrame) -> pd.DataFrame:
    cols = df.columns
    has_points = all(c in cols for c in
                     ["Lights pts","Heater pts","Plastic pts","Paper pts","Garden pts"]) and ("Total pts" in cols)
    if has_points: 
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

    # Garden pts depends on three fields; handle missing gracefully
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

    # Remove any "Result …" columns that slipped through
    drop_like = [c for c in df.columns if c.strip().lower().startswith("result")]
    if drop_like:
        df = df.drop(columns=drop_like)

    class_col = infer_class_col(df)
    df = ensure_points(df)
    return df, class_col

# Natural class sorting key: 1A, 1B, …, 10A, 10B; others last
def class_sort_key(x: str):
    if not isinstance(x, str): return (9999, "")
    s = x.strip()
    m = re.search(r"(\d+)\s*([A-Za-z]+)", s)
    if m:
        return (int(m.group(1)), m.group(2).lower())
    return (9998, s.lower())

# ----------------------------------------------------------------------
# PLOTTING HELPERS
# ----------------------------------------------------------------------
def _plot_template() -> str:
    base = st.get_option("theme.base")
    return "plotly_dark" if str(base).lower() == "dark" else "plotly_white"

def bar_chart_series(series: pd.Series, title: Optional[str] = None):
    s = series.copy()
    # Ensure natural order
    if s.index.dtype == object:
        s = s.reindex(sorted(s.index, key=class_sort_key))
    fig = px.bar(
        x=s.index.astype(str), y=s.values,
        labels={"x": "", "y": ""},
        title=title or None,
        template=_plot_template(),
    )
    fig.update_layout(margin=dict(l=0, r=0, t=30, b=0), height=360)
    st.plotly_chart(fig, use_container_width=True, theme=None)

def line_chart_df(df: pd.DataFrame, title: Optional[str] = None):
    # df index should be datetime or date; column must be numeric
    fig = px.line(
        df.reset_index(), x=df.index.name or "index", y=df.columns[0],
        labels={"x": "", "y": ""},
        markers=True,
        title=title or None,
        template=_plot_template(),
    )
    fig.update_layout(margin=dict(l=0, r=0, t=30, b=0), height=360)
    st.plotly_chart(fig, use_container_width=True, theme=None)

# ----------------------------------------------------------------------
# APP
# ----------------------------------------------------------------------
tabs = st.tabs(["Dashboard", "Submit log"])

# ===== TAB 2: SUBMIT =====
with tabs[1]:
    st.title("Submit log")
    st.write("Register points directly through the form below. Submissions update the dashboard automatically.")
    components.iframe(GOOGLE_FORM_URL, height=1200)

# ===== TAB 1: DASHBOARD =====
with tabs[0]:
    st.title("Digital Twin — Sustainability Dashboard")

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
        st.subheader("Monthly leaderboard")
        months_all = sorted([m for m in df["YearMonth"].dropna().unique()])
        current_month = pd.Timestamp.now().strftime("%Y-%m")
        default_month = months_all.index(current_month) if current_month in months_all else (len(months_all)-1 if months_all else 0)
        sel_month = st.selectbox("Month", months_all if months_all else ["–"], index=max(default_month, 0))
        metric = st.radio("Metric", ["Total points", "Average points"], index=0)

    # Global filtered view
    view = df.copy()
    if picked:
        view = view[view[CLASS].isin(picked)]
    if date_range[0] and date_range[1]:
        view = view[(pd.to_datetime(view["Date"]) >= pd.to_datetime(date_range[0]))
                    & (pd.to_datetime(view["Date"]) <= pd.to_datetime(date_range[1]))]

    # Convenience columns
    view["Day"] = pd.to_datetime(view["Timestamp_dt"], errors="coerce").dt.floor("D")

    # ----- Monthly winners (helpers) -----
    def monthly_leader(df_month: pd.DataFrame) -> Tuple[Optional[str], Optional[float]]:
        if df_month.empty:
            return None, None
        if metric == "Total points":
            s = df_month.groupby(CLASS)["Total pts"].apply(lambda x: pd.to_numeric(x, errors="coerce").sum())
        else:
            s = df_month.groupby(CLASS)["Total pts"].apply(lambda x: pd.to_numeric(x, errors="coerce").mean())
        if s.empty:
            return None, None
        s = s.reindex(sorted(s.index, key=class_sort_key))
        return s.idxmax(), float(s.max())

    # Find last-month id and current
    this_month = current_month
    last_month = None
    if months_all:
        if this_month in months_all:
            idx = months_all.index(this_month)
            if idx > 0:
                last_month = months_all[idx - 1]
        else:
            last_month = months_all[-1] if len(months_all) >= 1 else None

    # ----- KPI row (Top class = monthly) -----
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Submissions", f"{len(view):,}")
    col2.metric("Avg points / submission", f"{pd.to_numeric(view['Total pts'], errors='coerce').mean():.2f}" if len(view) else "–")

    today = pd.Timestamp.now().floor("D")
    today_pts = pd.to_numeric(view.loc[view["Day"] == today, "Total pts"], errors="coerce").sum()
    col3.metric("Today total", int(today_pts) if pd.notna(today_pts) else 0)

    best = pd.to_numeric(view["Total pts"], errors="coerce").max()
    col4.metric("Highest single score", 0 if pd.isna(best) else int(best))

    mv = view[view["YearMonth"] == sel_month]
    m_winner, m_value = monthly_leader(mv)
    if m_winner:
        suffix = "total" if metric == "Total points" else "avg"
        col5.metric("Top class (monthly)", f"{m_winner} ({m_value:.2f} {suffix})")
    else:
        col5.metric("Top class (monthly)", "–")

    st.markdown("---")

    # ===== Row 1: Monthly leaderboard (left) + Category breakdown (right) =====
    left, right = st.columns([1.05, 1])

    with left:
        st.subheader(f"Monthly leaderboard — {sel_month} ({metric})")

        # Winner callouts (above chart)
        if last_month:
            lm = view[view["YearMonth"] == last_month]
            lm_winner, lm_value = monthly_leader(lm)
            if lm_winner:
                st.info(f"Last month winner: Class {lm_winner} — {lm_value:.2f} points")

        if this_month and sel_month == this_month:
            tm = view[view["YearMonth"] == this_month]
            tm_winner, tm_value = monthly_leader(tm)
            if tm_winner:
                st.success(f"Current month leader: Class {tm_winner} — {tm_value:.2f}")

        # Selected month chart
        if not mv.empty:
            if metric == "Total points":
                leader_m = mv.groupby(CLASS)["Total pts"].apply(lambda s: pd.to_numeric(s, errors="coerce").sum())
            else:
                leader_m = mv.groupby(CLASS)["Total pts"].apply(lambda s: pd.to_numeric(s, errors="coerce").mean())
            bar_chart_series(leader_m)
        else:
            st.write("No submissions for the selected month (with current filters).")

    with right:
        st.subheader("Category breakdown (average per submission)")
        cat_cols = [c for c in ["Lights pts", "Heater pts", "Plastic pts", "Paper pts", "Garden pts"] if c in view.columns]
        if cat_cols:
            cats = view[cat_cols].apply(pd.to_numeric, errors="coerce").mean()
            cats = cats.sort_values(ascending=True)
            fig = px.bar(
                x=cats.index, y=cats.values,
                labels={"x": "", "y": ""},
                template=_plot_template()
            )
            fig.update_layout(margin=dict(l=0, r=0, t=30, b=0), height=360)
            st.plotly_chart(fig, use_container_width=True, theme=None)
        else:
            st.write("No category columns available.")

    # ===== Row 2: Overall leaderboard (avg) + Points over time =====
    left2, right2 = st.columns([1.05, 1])

    with left2:
        st.subheader("Leaderboard (average points by class)")
        leader_all = (view.groupby(CLASS)["Total pts"]
                      .apply(lambda s: pd.to_numeric(s, errors="coerce").mean()))
        if not leader_all.empty:
            bar_chart_series(leader_all.sort_values(ascending=True))
        else:
            st.write("No data in current filters.")

    with right2:
        st.subheader("Points over time")
        # Ensure Day exists, drop NaT, aggregate
        valid = view.dropna(subset=["Day"])
        if not valid.empty:
            trend = (valid.groupby("Day")[["Total pts"]]
                     .sum(numeric_only=True)
                     .sort_index())
            if not trend.empty and trend["Total pts"].sum() > 0:
                # Plotly line with markers
                line_chart_df(trend, None)
                st.caption(f"Showing {len(trend)} days of activity.")
            else:
                st.info("No non-zero totals in the selected period.")
        else:
            st.warning("No valid timestamps found — check the Timestamp column in the sheet.")

    # ===== Row 3: Latest submissions (full width) =====
    st.subheader("Latest submissions")
    show_cols = ["Timestamp_dt", CLASS, "Total pts"] + \
                [c for c in ["Lights pts", "Heater pts", "Plastic pts", "Paper pts", "Garden pts"] if c in view.columns]
    latest = (view.sort_values("Timestamp_dt", ascending=False)[show_cols]
              .rename(columns={"Timestamp_dt": "Timestamp"}).head(50))
    st.dataframe(latest, use_container_width=True, height=420)
    # Export filtered view as CSV
    st.download_button(
        label="Download filtered data (CSV)",
        data=view.to_csv(index=False).encode("utf-8"),
        file_name="digital-twin_filtered.csv",
        mime="text/csv",
        use_container_width=True
    )

    st.markdown("---")

    # ===== Row 4: Class detail =====
    st.subheader("Class detail")
    classes_sorted = sorted([c for c in df[CLASS].dropna().unique().tolist()], key=class_sort_key)
    c1, c2 = st.columns([1, 1])
    pick_one = c1.selectbox("Select class", classes_sorted if classes_sorted else ["—"])
    sub = view[view[CLASS] == pick_one].copy()
    if not sub.empty:
        c1.write(f"Average points: {pd.to_numeric(sub['Total pts'], errors='coerce').mean():.2f}  |  Submissions: {len(sub)}")
        if sub["Day"].notna().any():
            c2.write("Points over time (selected class)")
            t2 = (sub.dropna(subset=["Day"]).groupby("Day")[["Total pts"]]
                  .sum(numeric_only=True).sort_index())
            if not t2.empty:
                line_chart_df(t2)
        c1.write("Category averages (selected class)")
        cat_cols_cd = [c for c in ["Lights pts", "Heater pts", "Plastic pts", "Paper pts", "Garden pts"] if c in sub.columns]
        if cat_cols_cd:
            cavg = sub[cat_cols_cd].apply(pd.to_numeric, errors="coerce").mean().sort_values(ascending=True)
            bar_chart_series(cavg, None)
    else:
        st.info("No rows for this class with current filters.")
