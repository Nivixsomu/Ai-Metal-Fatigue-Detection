from streamlit_autorefresh import st_autorefresh
import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import numpy as np
from datetime import datetime

# ---------------------------
# CONFIG
# ---------------------------
MODEL_PATH = "fatigue_demo_model.pkl"
DATA_LOG = "fatigue_scores_log.csv"

# Load model
model = joblib.load(MODEL_PATH)

# ---------------------------
# PAGE SETTINGS
# ---------------------------
st.set_page_config(
    page_title="AI Mental Fatigue Detection",
    page_icon="🧠",
    layout="wide"
)

# Auto-refresh every 60s
st_autorefresh(interval=60 * 1000, limit=None, key="refresh")
st.sidebar.markdown("⏱️ Dashboard auto-updates every **60 seconds**.")


# ---------------------------
# LOAD DATA
# ---------------------------
try:
    df = pd.read_csv(DATA_LOG)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
except Exception:
    df = pd.DataFrame(columns=["timestamp", "score"])

# ---------------------------
# HEADER
# ---------------------------
st.markdown("<h1 style='text-align: center; color: cyan;'>🧠 Real-Time Mental Fatigue & Burnout Detection</h1>", unsafe_allow_html=True)
st.write("---")

# ---------------------------
# ---------------------------
# METRICS CARDS (Styled)
# ---------------------------
st.subheader("🧠 Current Status")

if not df.empty:
    latest_score = df["score"].iloc[-1]
else:
    latest_score = 0.0

if latest_score >= 0.7:
    risk_level = "High Fatigue"
    color = "#ff4d4d"   # 🔴 Red
elif latest_score >= 0.5:
    risk_level = "Moderate Fatigue"
    color = "#ffcc00"   # 🟡 Yellow
else:
    risk_level = "Low Fatigue"
    color = "#4CAF50"   # 🟢 Green

st.markdown(
    f"""
    <div style="padding:20px; border-radius:10px; background-color:{color}; text-align:center;">
        <h2 style="color:white;">Fatigue Score: {latest_score:.2f}</h2>
        <h3 style="color:white;">{risk_level}</h3>
    </div>
    """,
    unsafe_allow_html=True
)


# ---------------------------
# TREND CHART
# ---------------------------
st.subheader("📈 7-Day Fatigue Trend")

if not df.empty:
    cutoff = pd.Timestamp.now() - pd.Timedelta(days=7)
    df_trend = df[df["timestamp"] >= cutoff]
    fig = px.line(df_trend, x="timestamp", y="score", title="Fatigue Score Trend", markers=True)
    fig.update_yaxes(range=[0, 1])
    fig.update_traces(line_color="cyan")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No fatigue data available yet.")

# ---------------------------
# ---------------------------
# BEHAVIORAL METRICS (Styled)
# ---------------------------
st.subheader("📊 Behavioral Metrics")

def colored_progress(label, value, color):
    st.markdown(
        f"""
        <div style="margin-bottom:15px; padding:15px; border-radius:10px; background:#ffffff; box-shadow:0px 2px 5px rgba(0,0,0,0.1);">
            <p style="margin:0; font-weight:bold; color:#212529;">{label}</p>
            <div style="background:#e0e0e0; border-radius:10px; height:20px; overflow:hidden;">
                <div style="width:{int(value*100)}%; background:{color}; height:100%; border-radius:10px;"></div>
            </div>
            <p style="margin:5px 0 0; color:#212529;">{int(value*100)}%</p>
        </div>
        """,
        unsafe_allow_html=True
    )

colA, colB, colC, colD = st.columns(4)
with colA:
    colored_progress("⚡ Keystroke Velocity", 0.8, "#0077b6")
with colB:
    colored_progress("🖱️ Mouse Precision", 0.6, "#4CAF50")
with colC:
    colored_progress("⏱️ Response Time", 0.7, "#ffcc00")
with colD:
    colored_progress("📌 Consistency Score", 0.5, "#ff4d4d")


# ---------------------------
# RECOMMENDATIONS
# ---------------------------
st.subheader("💡 AI Recommendations")

if latest_score >= 0.7:
    st.error("⚠️ High fatigue detected! Please take a 5–10 minute break.")
    st.markdown("- Stand up, stretch, and walk around")
    st.markdown("- Try deep breathing (inhale 4s, exhale 6s)")
    st.markdown("- Drink water to rehydrate")
elif latest_score >= 0.5:
    st.warning("🙂 Moderate fatigue. Stay mindful!")
    st.markdown("- Take a 2–3 minute micro-break")
    st.markdown("- Stretch shoulders and wrists")
else:
    st.success("✅ Low fatigue. You're doing great!")
    st.markdown("- Keep hydrated")
    st.markdown("- Follow the 20–20–20 eye rule")

# ---------------------------
# QUICK ACTIONS
# ---------------------------
st.subheader("⚡ Quick Actions")
colA, colB, colC, colD = st.columns(4)

colA.button("💧 Hydrate")
colB.button("🧘 Stretch")
colC.button("🌿 Fresh Air")
colD.button("🕑 Take Break")
