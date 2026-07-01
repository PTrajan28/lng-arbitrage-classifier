# ============================================================
# Singapore LNG Market Intelligence Engine — Streamlit App
# Run:  streamlit run app.py
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import joblib
import os
from datetime import datetime

# ── Page Config (must be first Streamlit call) ───────────────
st.set_page_config(
    page_title="LNG Arbitrage Regime Classifier",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constants ────────────────────────────────────────────────
LABEL_MAP = {0: "Europe Premium", 1: "Neutral", 2: "Asia Premium"}

COLORS = {
    "Europe Premium": "#c0392b",
    "Neutral":        "#e67e22",
    "Asia Premium":   "#2980b9",
}

FEATURES = [
    "ttf_price", "jkm_lag1", "jkm_lag3", "jkm_rolling_3m",
    "henry_price", "jkm_ttf_spread", "jkm_hh_spread", "lng_share",
]

FREIGHT     = 1.50          # USD/MMBtu Asia–Europe shipping cost assumption
SPLIT_DATE  = "2023-01-01"
MODEL_PATH  = "models/best_model.pkl"
SCALER_PATH = "models/scaler.pkl"
DATA_PATH   = "data/lng_arbitrage_clean.csv"

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
/* ---------- global ---------- */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ---------- sidebar ---------- */
section[data-testid="stSidebar"] { background-color: #0f1923; }
section[data-testid="stSidebar"] * { color: #e8eaed !important; }
section[data-testid="stSidebar"] .stSlider label { font-size: 0.82rem; }

/* ---------- metric cards ---------- */
div[data-testid="stMetric"] {
    background: #f7f9fc;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 14px 18px;
}
div[data-testid="stMetricValue"] { font-size: 1.6rem !important; font-weight: 700; }

/* ---------- regime pills ---------- */
.pill {
    display: inline-block;
    padding: 8px 24px;
    border-radius: 50px;
    font-size: 1.3rem;
    font-weight: 700;
    letter-spacing: 0.5px;
    margin-bottom: 6px;
}
.pill-asia    { background:#dbeafe; color:#1e40af; border:2px solid #2980b9; }
.pill-neutral { background:#fef3c7; color:#92400e; border:2px solid #e67e22; }
.pill-europe  { background:#fee2e2; color:#991b1b; border:2px solid #c0392b; }

/* ---------- info box ---------- */
.info-box {
    background: #f0f9ff;
    border-left: 4px solid #2980b9;
    border-radius: 6px;
    padding: 12px 16px;
    margin: 10px 0;
    font-size: 0.9rem;
    line-height: 1.6;
}

/* ---------- section header ---------- */
.section-header {
    font-size: 1.05rem;
    font-weight: 600;
    color: #1e293b;
    margin: 18px 0 8px;
    padding-bottom: 4px;
    border-bottom: 2px solid #e2e8f0;
}

/* ---------- footer ---------- */
footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Cached loaders ────────────────────────────────────────────
@st.cache_data(show_spinner="Loading dataset…")
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index)
    return df


@st.cache_resource(show_spinner="Loading model…")
def load_model(model_path: str, scaler_path: str):
    model  = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler


# ── Prediction helper ─────────────────────────────────────────
def predict(model, scaler, feature_values: dict):
    X_raw    = pd.DataFrame([feature_values])[FEATURES]
    X_scaled = scaler.transform(X_raw)
    label_int = int(model.predict(X_scaled)[0])
    label_str = LABEL_MAP[label_int]
    if hasattr(model, "predict_proba"):
        probas     = model.predict_proba(X_scaled)[0]
        proba_dict = {LABEL_MAP[i]: round(float(p), 4) for i, p in enumerate(probas)}
    else:
        proba_dict = {name: (1.0 if name == label_str else 0.0) for name in LABEL_MAP.values()}
    return label_int, label_str, proba_dict


# ── Demo / synthetic data generator (when CSV not present) ───
@st.cache_data
def make_demo_data() -> pd.DataFrame:
    """Generate synthetic monthly LNG data for demonstration when real data is absent."""
    rng   = np.random.default_rng(42)
    dates = pd.date_range("2014-01-01", "2026-03-01", freq="MS")
    n     = len(dates)

    jkm   = 8 + np.cumsum(rng.normal(0, 0.4, n))
    ttf   = 6 + np.cumsum(rng.normal(0, 0.35, n))
    # inject 2022 crisis spike in TTF
    crisis = (dates.year == 2022)
    ttf[crisis] += 25
    hh    = 2.5 + rng.normal(0, 0.3, n)

    spread = jkm - ttf
    labels = np.where(spread > FREIGHT, 2, np.where(spread < -1.0, 0, 1))

    df = pd.DataFrame({
        "jkm_price":       jkm,
        "ttf_price":       ttf,
        "henry_price":     hh,
        "jkm_ttf_spread":  spread,
        "jkm_hh_spread":   jkm - hh,
        "jkm_lag1":        np.roll(jkm, 1),
        "jkm_lag3":        np.roll(jkm, 3),
        "jkm_rolling_3m":  pd.Series(jkm).rolling(3).mean().values,
        "lng_share":       rng.uniform(28, 42, n),
        "arbitrage_label": labels,
        "regime_cluster":  labels,
    }, index=dates)
    return df.dropna()


# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🚢 LNG Intelligence Engine")
    st.caption("Singapore Arbitrage Regime Classifier")
    st.divider()

    page = st.radio(
        "Navigate",
        ["📊 Dashboard", "🔍 Live Predictor", "📈 Model Performance", "ℹ️ About"],
        label_visibility="collapsed",
    )
    st.divider()

    st.markdown("**Feature Inputs**")
    st.caption("Adjust values to simulate any market environment.")

    jkm_ttf_spread  = st.slider("JKM–TTF Spread ($/MMBtu)",  -10.0, 30.0, 2.5,  0.1,
                                 help="> 1.5 → Asia Premium | -1.0 to 1.5 → Neutral | < -1.0 → Europe Premium")
    ttf_price       = st.slider("TTF Price ($/MMBtu)",         0.0,  80.0, 12.0, 0.5)
    jkm_lag1        = st.slider("JKM 1-Month Lag ($/MMBtu)",   0.0,  80.0, 14.0, 0.5)
    jkm_lag3        = st.slider("JKM 3-Month Lag ($/MMBtu)",   0.0,  80.0, 13.5, 0.5)
    jkm_rolling_3m  = st.slider("JKM 3-Month Rolling Avg",     0.0,  80.0, 13.8, 0.5)
    henry_price     = st.slider("Henry Hub ($/MMBtu)",          0.0,  15.0, 2.8,  0.1)
    jkm_hh_spread   = st.slider("JKM–HH Spread ($/MMBtu)",    -5.0,  40.0, 11.2, 0.1)
    lng_share       = st.slider("LNG Share of Gas Trade (%)",   0.0,  60.0, 34.0, 0.5)

    # Quick rule-based live regime indicator in sidebar
    quick_regime = (
        "🔵 Asia Premium"   if jkm_ttf_spread >  FREIGHT else
        "🔴 Europe Premium" if jkm_ttf_spread < -1.0     else
        "🟠 Neutral"
    )
    st.divider()
    st.markdown(f"**Rule-based signal:** {quick_regime}")
    st.caption(f"Spread = **{jkm_ttf_spread:.2f}** $/MMBtu")

sidebar_inputs = {
    "ttf_price": ttf_price, "jkm_lag1": jkm_lag1, "jkm_lag3": jkm_lag3,
    "jkm_rolling_3m": jkm_rolling_3m, "henry_price": henry_price,
    "jkm_ttf_spread": jkm_ttf_spread, "jkm_hh_spread": jkm_hh_spread,
    "lng_share": lng_share,
}


# ── Load data & model ─────────────────────────────────────────
data_ok  = os.path.exists(DATA_PATH)
model_ok = os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH)

df             = load_data(DATA_PATH) if data_ok else make_demo_data()
using_demo     = not data_ok
model, scaler  = (load_model(MODEL_PATH, SCALER_PATH) if model_ok else (None, None))


# ─────────────────────────────────────────────────────────────
# PAGE 1 — DASHBOARD
# ─────────────────────────────────────────────────────────────
if page == "📊 Dashboard":

    st.title("📊 LNG Arbitrage Regime Dashboard")
    st.caption("Monthly regime classification · Jan 2014 – Mar 2026  |  JKM–TTF Spread Dynamics")

    if using_demo:
        st.info("ℹ️ **Demo mode** — `data/lng_arbitrage_clean.csv` not found. Displaying synthetic data.", icon="ℹ️")

    if "arbitrage_label" not in df.columns:
        st.warning("Dataset missing `arbitrage_label` column.")
        st.stop()

    # ── KPI Cards ──────────────────────────────────────────────
    label_counts   = df["arbitrage_label"].value_counts()
    latest_spread  = df["jkm_ttf_spread"].iloc[-1]
    prior_spread   = df["jkm_ttf_spread"].iloc[-2]
    latest_date    = df.index[-1].strftime("%b %Y")
    total_months   = len(df)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🔵 Asia Premium",   f"{int(label_counts.get(2, 0))} months",
              f"{100*label_counts.get(2,0)/total_months:.0f}% of history")
    c2.metric("🟠 Neutral",        f"{int(label_counts.get(1, 0))} months",
              f"{100*label_counts.get(1,0)/total_months:.0f}% of history")
    c3.metric("🔴 Europe Premium", f"{int(label_counts.get(0, 0))} months",
              f"{100*label_counts.get(0,0)/total_months:.0f}% of history")
    c4.metric(f"Latest Spread ({latest_date})", f"{latest_spread:.2f} $/MMBtu",
              f"{latest_spread - prior_spread:+.2f} vs prior month")

    st.divider()

    # ── Spread Over Time ───────────────────────────────────────
    fig_spread = go.Figure()
    for lid, lname in LABEL_MAP.items():
        mask = df["arbitrage_label"] == lid
        fig_spread.add_trace(go.Scatter(
            x=df.index[mask], y=df.loc[mask, "jkm_ttf_spread"],
            mode="markers+lines", name=lname,
            marker=dict(color=COLORS[lname], size=6),
            line=dict(color=COLORS[lname], width=1.5),
        ))
    fig_spread.add_hline(y=FREIGHT,  line_dash="dash", line_color="#444",
                         annotation_text=f"Asia threshold ({FREIGHT} $/MMBtu)",
                         annotation_position="top left")
    fig_spread.add_hline(y=-1.0, line_dash="dot",  line_color="#888",
                         annotation_text="Europe threshold (−1.0 $/MMBtu)",
                         annotation_position="bottom left")
    fig_spread.add_vrect(x0="2022-01-01", x1="2022-12-01",
                         fillcolor="rgba(192,57,43,0.07)", line_width=0,
                         annotation_text="2022 Crisis", annotation_position="top left")
    fig_spread.update_layout(
        title="JKM–TTF Spread Over Time — Coloured by Arbitrage Regime",
        xaxis_title="Date", yaxis_title="Spread ($/MMBtu)",
        legend_title="Regime", hovermode="x unified", height=420,
        plot_bgcolor="#fafafa", paper_bgcolor="white",
    )
    st.plotly_chart(fig_spread, use_container_width=True)

    # ── Distribution + Heatmap ────────────────────────────────
    col_l, col_r = st.columns([1, 2])

    with col_l:
        fig_dist = go.Figure(go.Bar(
            x=[int(label_counts.get(2, 0)), int(label_counts.get(1, 0)), int(label_counts.get(0, 0))],
            y=["Asia Premium", "Neutral", "Europe Premium"],
            orientation="h",
            marker_color=[COLORS["Asia Premium"], COLORS["Neutral"], COLORS["Europe Premium"]],
            text=[int(label_counts.get(2,0)), int(label_counts.get(1,0)), int(label_counts.get(0,0))],
            textposition="outside",
        ))
        fig_dist.update_layout(title="Regime Distribution", xaxis_title="Months",
                                height=310, margin=dict(t=40, b=10, l=10, r=30),
                                plot_bgcolor="#fafafa", paper_bgcolor="white")
        st.plotly_chart(fig_dist, use_container_width=True)

    with col_r:
        df_heat = df[["arbitrage_label"]].copy()
        df_heat["year"]  = df_heat.index.year
        df_heat["month"] = df_heat.index.month
        pivot = df_heat.pivot(index="year", columns="month", values="arbitrage_label")
        month_labels = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        fig_heat = px.imshow(
            pivot,
            color_continuous_scale=[[0.0, COLORS["Europe Premium"]],[0.5, COLORS["Neutral"]],[1.0, COLORS["Asia Premium"]]],
            labels=dict(color="Regime (0=EP, 1=N, 2=AP)"),
            x=month_labels[:pivot.shape[1]], y=pivot.index.tolist(),
            aspect="auto", title="Monthly Regime Heatmap",
        )
        fig_heat.update_layout(height=310, margin=dict(t=40, b=10),
                                paper_bgcolor="white")
        st.plotly_chart(fig_heat, use_container_width=True)

    # ── JKM vs TTF price lines ────────────────────────────────
    st.markdown('<div class="section-header">JKM & TTF Price History</div>', unsafe_allow_html=True)
    if "jkm_price" in df.columns:
        fig_prices = go.Figure()
        fig_prices.add_trace(go.Scatter(x=df.index, y=df["jkm_price"],
                                        name="JKM", line=dict(color="#2980b9", width=2)))
        fig_prices.add_trace(go.Scatter(x=df.index, y=df["ttf_price"],
                                        name="TTF", line=dict(color="#c0392b", width=2)))
        if "henry_price" in df.columns:
            fig_prices.add_trace(go.Scatter(x=df.index, y=df["henry_price"],
                                            name="Henry Hub", line=dict(color="#27ae60", width=1.5, dash="dot")))
        fig_prices.update_layout(xaxis_title="Date", yaxis_title="$/MMBtu",
                                  hovermode="x unified", height=360,
                                  plot_bgcolor="#fafafa", paper_bgcolor="white",
                                  legend=dict(orientation="h", y=-0.15))
        st.plotly_chart(fig_prices, use_container_width=True)


# ─────────────────────────────────────────────────────────────
# PAGE 2 — LIVE PREDICTOR
# ─────────────────────────────────────────────────────────────
elif page == "🔍 Live Predictor":

    st.title("🔍 Live Regime Predictor")
    st.caption("Adjust the sidebar sliders to simulate any LNG market environment.")

    # ── Rule-based prediction (always available) ──────────────
    rule_label = (
        "Asia Premium"   if jkm_ttf_spread >  FREIGHT else
        "Europe Premium" if jkm_ttf_spread < -1.0     else
        "Neutral"
    )
    pill_map = {"Asia Premium": "pill-asia", "Neutral": "pill-neutral", "Europe Premium": "pill-europe"}

    col_badge, col_info = st.columns([1, 2])

    with col_badge:
        st.markdown('<div class="section-header">Rule-Based Prediction</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="pill {pill_map[rule_label]}">{rule_label}</div>',
            unsafe_allow_html=True,
        )
        st.caption("Threshold logic: spread > 1.5 → AP | < −1.0 → EP | else → Neutral")

        # Confidence gauge using spread distance from nearest boundary
        if rule_label == "Asia Premium":
            margin = jkm_ttf_spread - FREIGHT
            conf   = min(margin / 5.0, 1.0)
        elif rule_label == "Europe Premium":
            margin = -1.0 - jkm_ttf_spread
            conf   = min(margin / 5.0, 1.0)
        else:
            d1   = jkm_ttf_spread - (-1.0)
            d2   = FREIGHT - jkm_ttf_spread
            conf = 1 - (min(d1, d2) / 1.25)

        conf = max(conf, 0.05)
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=round(conf * 100, 1),
            title={"text": "Signal Strength (%)"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar":  {"color": COLORS[rule_label]},
                "steps": [
                    {"range": [0,  40], "color": "#f8f9fa"},
                    {"range": [40, 70], "color": "#e9ecef"},
                    {"range": [70,100], "color": "#dee2e6"},
                ],
            },
            number={"suffix": "%"},
        ))
        fig_gauge.update_layout(height=260, margin=dict(t=40, b=10, l=20, r=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col_info:
        st.markdown('<div class="section-header">Input Summary</div>', unsafe_allow_html=True)
        input_df = pd.DataFrame([{
            "Feature":           f,
            "Value":             round(sidebar_inputs[f], 3),
        } for f in FEATURES])
        st.dataframe(input_df.set_index("Feature"), use_container_width=True, height=290)

    st.divider()

    # ── ML Model prediction (when model loaded) ───────────────
    if model is not None and scaler is not None:
        st.markdown('<div class="section-header">ML Model Prediction (XGBoost)</div>', unsafe_allow_html=True)
        label_int, label_str, proba_dict = predict(model, scaler, sidebar_inputs)

        col_ml, col_proba = st.columns([1, 2])
        with col_ml:
            st.markdown(
                f'<div class="pill {pill_map[label_str]}">{label_str}</div>',
                unsafe_allow_html=True,
            )
            st.caption(f"Model confidence: **{proba_dict[label_str]*100:.1f}%**")

        with col_proba:
            prob_df = pd.DataFrame({
                "Regime":      list(proba_dict.keys()),
                "Probability": list(proba_dict.values()),
            })
            fig_proba = go.Figure(go.Bar(
                x=prob_df["Probability"], y=prob_df["Regime"],
                orientation="h",
                marker_color=[COLORS[r] for r in prob_df["Regime"]],
                text=[f"{v:.1%}" for v in prob_df["Probability"]],
                textposition="outside",
            ))
            fig_proba.update_layout(
                title="Class Probabilities", xaxis=dict(range=[0, 1.05]),
                height=250, margin=dict(t=40, b=10, l=10, r=60),
                plot_bgcolor="#fafafa", paper_bgcolor="white",
            )
            st.plotly_chart(fig_proba, use_container_width=True)
    else:
        st.markdown('<div class="section-header">ML Model Prediction</div>', unsafe_allow_html=True)
        st.warning("Model files not found — showing rule-based prediction only.\n\n"
                   "Place `models/best_model.pkl` and `models/scaler.pkl` to enable ML predictions.",
                   icon="⚠️")

    st.divider()

    # ── Spread sensitivity chart ──────────────────────────────
    st.markdown('<div class="section-header">Spread Sensitivity — How Prediction Changes</div>',
                unsafe_allow_html=True)
    spread_vals = np.linspace(-8, 25, 200)
    regime_at_spread = np.where(spread_vals > FREIGHT, 2,
                        np.where(spread_vals < -1.0, 0, 1))
    colors_line = [COLORS[LABEL_MAP[r]] for r in regime_at_spread]

    fig_sens = go.Figure()
    for lid, lname in LABEL_MAP.items():
        mask = regime_at_spread == lid
        fig_sens.add_trace(go.Scatter(
            x=spread_vals[mask], y=[lid] * mask.sum(),
            mode="markers", name=lname,
            marker=dict(color=COLORS[lname], size=5, symbol="square"),
        ))
    fig_sens.add_vline(x=jkm_ttf_spread, line_dash="dash", line_color="#333",
                       annotation_text=f"Current: {jkm_ttf_spread:.2f}",
                       annotation_position="top right")
    fig_sens.update_layout(
        title="Predicted Regime vs JKM–TTF Spread",
        xaxis_title="JKM–TTF Spread ($/MMBtu)",
        yaxis=dict(tickvals=[0,1,2], ticktext=["Europe Prem.", "Neutral", "Asia Prem."]),
        height=280, legend_title="Regime",
        plot_bgcolor="#fafafa", paper_bgcolor="white",
    )
    st.plotly_chart(fig_sens, use_container_width=True)

    # ── Procurement guidance ──────────────────────────────────
    guidance = {
        "Asia Premium":   ("⚠️ Supply Tightening",
                           "Consider accelerating spot procurement or locking in forward contracts. "
                           "JKM cargoes are being diverted to Asia — supply availability is constrained."),
        "Neutral":        ("ℹ️ Balanced Market",
                           "Standard procurement scheduling is appropriate. "
                           "Monitor for directional shifts; spread is within normal operating range."),
        "Europe Premium": ("✅ Buying Opportunity",
                           "Consider deferring spot purchases where operationally possible. "
                           "Cargo diversion to Europe suggests loose Asia supply and softening prices."),
    }
    title_g, body_g = guidance[rule_label]
    st.markdown(f'<div class="info-box"><b>{title_g}</b><br>{body_g}</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# PAGE 3 — MODEL PERFORMANCE
# ─────────────────────────────────────────────────────────────
elif page == "📈 Model Performance":

    st.title("📈 Model Performance")
    st.caption("Comparative evaluation of all models trained across the 7-day pipeline.")

    # ── Model comparison table ────────────────────────────────
    st.markdown('<div class="section-header">Classifier Comparison (Test Set: Jan 2023 – Mar 2026)</div>',
                unsafe_allow_html=True)

    perf_data = {
        "Model":          ["KNN (K=3)", "SVM (RBF)", "Logistic Regression",
                           "Decision Tree", "Random Forest", "XGBoost ✅", "AdaBoost", "Stacking"],
        "Accuracy":       [0.800, 0.770, 0.670, 0.923, 0.950, 0.962, 0.920, 0.957],
        "Weighted F1":    [0.742, 0.670, 0.680, 0.938, 0.955, 0.962, 0.920, 0.958],
        "AP Recall":      [0.14,  0.00,  0.17,  1.00,  1.00,  1.00,  0.85,  1.00],
        "EP Precision":   [0.00,  0.00,  1.00,  0.40,  0.55,  0.60,  0.33,  0.58],
        "CV Mean F1":     ["—",   "—",   "—",   "0.91", "0.92","0.93","0.85","0.92"],
        "Day":            [3, 3, 3, 3, 4, 4, 4, 4],
    }
    perf_df = pd.DataFrame(perf_data)

    def highlight_best(row):
        styles = [""] * len(row)
        if row["Model"] == "XGBoost ✅":
            styles = ["background-color:#dbeafe; font-weight:600"] * len(row)
        return styles

    styled_df = (
        perf_df.style
        .apply(highlight_best, axis=1)
        .format({"Accuracy": "{:.3f}", "Weighted F1": "{:.3f}",
                 "AP Recall": "{:.2f}", "EP Precision": "{:.2f}"})
        .bar(subset=["Weighted F1"], color="#bfdbfe")
    )
    st.dataframe(styled_df, use_container_width=True, height=330)

    st.divider()

    # ── Weighted F1 bar chart ─────────────────────────────────
    col_f1, col_recall = st.columns(2)

    with col_f1:
        fig_f1 = go.Figure(go.Bar(
            y=perf_df["Model"], x=perf_df["Weighted F1"],
            orientation="h",
            marker_color=["#2980b9" if "XGBoost" in m else "#94a3b8" for m in perf_df["Model"]],
            text=[f"{v:.3f}" for v in perf_df["Weighted F1"]],
            textposition="outside",
        ))
        fig_f1.add_vline(x=0.938, line_dash="dash", line_color="#e67e22",
                         annotation_text="DT baseline", annotation_position="top right")
        fig_f1.update_layout(title="Weighted F1 — All Models",
                              xaxis=dict(range=[0.5, 1.05]),
                              height=380, margin=dict(t=40, b=10, l=10, r=60),
                              plot_bgcolor="#fafafa", paper_bgcolor="white")
        st.plotly_chart(fig_f1, use_container_width=True)

    with col_recall:
        fig_ap = go.Figure(go.Bar(
            y=perf_df["Model"], x=perf_df["AP Recall"],
            orientation="h",
            marker_color=["#2980b9" if v == 1.0 else "#f87171" for v in perf_df["AP Recall"]],
            text=[f"{v:.2f}" for v in perf_df["AP Recall"]],
            textposition="outside",
        ))
        fig_ap.add_vline(x=1.0, line_dash="dash", line_color="#ef4444",
                         annotation_text="Required = 1.00", annotation_position="top left")
        fig_ap.update_layout(title="Asia Premium Recall (must = 1.00)",
                              xaxis=dict(range=[0, 1.15]),
                              height=380, margin=dict(t=40, b=10, l=10, r=60),
                              plot_bgcolor="#fafafa", paper_bgcolor="white")
        st.plotly_chart(fig_ap, use_container_width=True)

    # ── Feature importance ────────────────────────────────────
    st.markdown('<div class="section-header">Feature Importances (XGBoost)</div>',
                unsafe_allow_html=True)

    fi_data = {
        "Feature":    ["jkm_ttf_spread", "ttf_price", "jkm_rolling_3m", "jkm_lag1",
                        "henry_price", "jkm_lag3", "jkm_hh_spread", "lng_share"],
        "Importance": [0.712, 0.082, 0.058, 0.051, 0.038, 0.029, 0.018, 0.012],
    }
    fi_df = pd.DataFrame(fi_data).sort_values("Importance")

    fig_fi = go.Figure(go.Bar(
        y=fi_df["Feature"], x=fi_df["Importance"],
        orientation="h",
        marker_color=["#1e40af" if f == "jkm_ttf_spread" else "#93c5fd" for f in fi_df["Feature"]],
        text=[f"{v:.3f}" for v in fi_df["Importance"]],
        textposition="outside",
    ))
    fig_fi.update_layout(title="XGBoost Feature Importances",
                          xaxis_title="Importance Score",
                          xaxis=dict(range=[0, 0.85]),
                          height=320, margin=dict(t=40, b=10, l=10, r=60),
                          plot_bgcolor="#fafafa", paper_bgcolor="white")
    st.plotly_chart(fig_fi, use_container_width=True)

    # ── Sequence model comparison ─────────────────────────────
    st.divider()
    st.markdown('<div class="section-header">Sequence Model Comparison (JKM Price Forecast)</div>',
                unsafe_allow_html=True)

    seq_data = {
        "Model":   ["GRU", "LSTM", "Sentiment-Enhanced LSTM ✅", "Transformer"],
        "MAE":     [1.58, 1.42, 1.31, 1.47],
        "RMSE":    [2.14, 1.89, 1.74, 1.96],
        "Notes":   ["Fastest training", "Best base model",
                    "Best overall; +12% at transitions", "Underperforms at small N"],
    }
    seq_df = pd.DataFrame(seq_data)

    def highlight_best_seq(row):
        if "✅" in row["Model"]:
            return ["background-color:#dcfce7; font-weight:600"] * len(row)
        return [""] * len(row)

    st.dataframe(
        seq_df.style.apply(highlight_best_seq, axis=1)
              .format({"MAE": "{:.2f}", "RMSE": "{:.2f}"}),
        use_container_width=True, height=200,
    )

    fig_seq = go.Figure()
    fig_seq.add_trace(go.Bar(name="MAE",  x=seq_df["Model"], y=seq_df["MAE"],
                             marker_color="#93c5fd"))
    fig_seq.add_trace(go.Bar(name="RMSE", x=seq_df["Model"], y=seq_df["RMSE"],
                             marker_color="#fca5a5"))
    fig_seq.update_layout(barmode="group", title="Sequence Model Errors ($/MMBtu)",
                           yaxis_title="$/MMBtu", height=320,
                           plot_bgcolor="#fafafa", paper_bgcolor="white")
    st.plotly_chart(fig_seq, use_container_width=True)


# ─────────────────────────────────────────────────────────────
# PAGE 4 — ABOUT
# ─────────────────────────────────────────────────────────────
elif page == "ℹ️ About":

    st.title("ℹ️ About This Project")

    st.markdown("""
    ## Singapore LNG Market Intelligence Engine

    A 7-day end-to-end data science capstone project built to classify LNG arbitrage regimes
    and forecast JKM spot prices for Singapore energy procurement decision support.

    ---

    ### Arbitrage Regime Labels

    | Label | Condition | Interpretation |
    |-------|-----------|----------------|
    | 🔵 **Asia Premium** | JKM − TTF > 1.50 $/MMBtu | Cargoes favour Asia; tight Singapore supply |
    | 🟠 **Neutral** | −1.00 ≤ JKM − TTF ≤ 1.50 | No directional pull |
    | 🔴 **Europe Premium** | JKM − TTF < −1.00 $/MMBtu | Cargoes divert to Europe; loose Asia supply |

    The 1.50 $/MMBtu threshold represents the typical Asia–Europe LNG freight differential.

    ---

    ### Pipeline Summary

    | Day | Techniques |
    |-----|-----------|
    | Day 1 | Web scraping, UTF-8/Regex, API/CSV ingestion |
    | Day 2 | Data cleaning, outlier detection, EDA, PCA, K-Means clustering |
    | Day 3 | Feature selection (SelectKBest + RFE), Logistic Regression, SVM, KNN |
    | Day 4 | Random Forest, XGBoost, AdaBoost, Stacking, SHAP, reproducible Pipeline |
    | Day 5 | Sentiment analysis (VADER + FinBERT), CNN text classification |
    | Day 6 | GRU, LSTM, Sentiment-enhanced LSTM, Transformer |
    | Day 7 | End-to-end pipeline, this Streamlit app, project report |

    ---

    ### Key Finding

    The **JKM–TTF spread is the sole sufficient feature** for regime classification.
    A single engineered variable explains 71.2% of XGBoost feature importance and fully
    defines the Decision Tree split structure. All linear models (KNN, SVM, LR) fail because
    the decision rule is piecewise constant on one variable — not a linear hyperplane.

    **Selected models:**
    - **Regime Classifier:** XGBoost — Weighted F1 = 0.962, Asia Premium Recall = 1.00
    - **Price Forecast:** Sentiment-Enhanced LSTM — MAE = 1.31 $/MMBtu

    ---

    ### File Layout

    ```
    ├── app.py                        ← this file
    ├── data/
    │   └── lng_arbitrage_clean.csv   ← processed dataset
    ├── models/
    │   ├── best_model.pkl            ← serialised XGBoost pipeline
    │   └── scaler.pkl                ← fitted StandardScaler
    └── notebooks/
        ├── 01_ingestion.ipynb
        ├── 02_clustering.ipynb
        ├── 03_eda.ipynb
        ├── 04_features.ipynb
        ├── 05_baseline_models.ipynb
        ├── 06_modelling.ipynb
        ├── 07_nlp_sentiment.ipynb
        ├── 08_sequence_models.ipynb
        └── 09_app.py.ipynb
    ```

    ---

    ### Run Instructions

    ```bash
    # Install dependencies
    pip install streamlit pandas numpy plotly joblib scikit-learn xgboost

    # Launch the app
    streamlit run app.py
    ```

    > **Demo mode:** If `data/lng_arbitrage_clean.csv` is not present, the app runs on
    > synthetic data. The Live Predictor page works without any data file using rule-based
    > threshold logic; ML predictions require `models/best_model.pkl` and `models/scaler.pkl`.
    """)

    st.divider()
    st.caption("Capstone Project — Singapore LNG Market Intelligence Engine · June 2026")
