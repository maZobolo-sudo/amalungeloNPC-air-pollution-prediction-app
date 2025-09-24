import streamlit as st, pandas as pd, joblib, matplotlib.pyplot as plt
from pathlib import Path
from src.model import FEATURES, forecast
st.title("🧪 Forecast + Compliance")
WORKSPACE = st.secrets.get("workspace_key","default")
mp = Path(f"tenants/{WORKSPACE}/models/pm25_model.joblib")
if not mp.exists(): st.warning("Train a model first."); st.stop()
model = joblib.load(mp)
up = st.file_uploader("Upload station-day CSV to forecast", type=["csv"])
naa_qs = st.number_input("NAAQS limit for PM2.5 (µg/m³)", value=25.0, step=1.0)
if up:
    df = pd.read_csv(up)
    pred = forecast(model, df)
    df["PM25_pred"] = pred
    df["will_exceed_pm25"] = (df["PM25_pred"] > naa_qs).astype(int)
    st.dataframe(df.head())
    rate = df["will_exceed_pm25"].mean()
    st.metric("Forecast exceedance share", f"{rate:.0%}")
    fig = plt.figure(figsize=(6,3)); df.groupby("station")["will_exceed_pm25"].mean().mul(100).plot(kind="bar"); plt.ylabel("% forecast exceed")
    st.pyplot(fig)
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download forecast CSV", csv, "pm25_forecast.csv", "text/csv")
