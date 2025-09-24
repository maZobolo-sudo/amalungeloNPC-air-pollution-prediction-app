import streamlit as st, pandas as pd, joblib
from pathlib import Path
from src.model import train, FEATURES, TARGET, forecast
st.title("🤖 Train Regressor")
WORKSPACE = st.secrets.get("workspace_key","default")
p = Path(f"tenants/{WORKSPACE}/data/pm25.csv")
if not p.exists(): st.warning("No dataset. Use Data Intake."); st.stop()
df = pd.read_csv(p)
if TARGET not in df.columns:
    st.error("Dataset must include 'PM25_next' (µg/m³)."); st.stop()
if st.button("Train"):
    m, metrics = train(df); st.json(metrics)
    mp = Path(f"tenants/{WORKSPACE}/models/pm25_model.joblib"); mp.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(m, mp); st.success(f"Saved model to {mp}")
