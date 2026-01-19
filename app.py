# =========================
# app.py — M5 Forecasting App
# =========================

import streamlit as st
import pandas as pd
import numpy as np
import pickle

# =========================
# Configuración Streamlit
# =========================
st.set_page_config(
    page_title="M5 Sales Forecasting",
    layout="wide"
)

st.title("📈 M5 Sales Forecasting")
st.caption("HistGradientBoosting — Forecast 28 días")

# =========================
# Cargar modelo y encoder
# =========================
@st.cache_resource
def load_model():
    with open("hgb_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("encoder.pkl", "rb") as f:
        encoder = pickle.load(f)
    return model, encoder

model, encoder = load_model()
st.success("✅ Modelo cargado correctamente")

# =========================
# Cargar datos
# =========================
@st.cache_data
def load_data():
    df = pd.read_parquet("df_model.parquet")
    df["date"] = pd.to_datetime(df["date"])
    return df

df_model = load_data()

# =========================
# Features
# =========================
features = [
    "store_id", "dept_id",
    "year", "month", "dayofweek", "weekofyear",
    "is_event",
    "snap_CA", "snap_TX", "snap_WI",
    "lag_1", "lag_7", "lag_14",
    "rolling_7", "rolling_14"
]

cat_features = ["store_id", "dept_id"]

# =========================
# Sidebar
# =========================
st.sidebar.header("Filtros")

store_id = st.sidebar.selectbox(
    "Store",
    sorted(df_model["store_id"].unique())
)

dept_id = st.sidebar.selectbox(
    "Department",
    sorted(
        df_model[df_model["store_id"] == store_id]["dept_id"].unique()
    )
)

# =========================
# Histórico
# =========================
df_hist = df_model[
    (df_model["store_id"] == store_id) &
    (df_model["dept_id"] == dept_id)
].sort_values("date")

st.subheader("📊 Histórico")
st.line_chart(df_hist.set_index("date")["sales"])

# =========================
# Calendario futuro
# =========================
def build_future(last_date):
    dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=28)
    df = pd.DataFrame({"date": dates})

    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["dayofweek"] = df["date"].dt.dayofweek
    df["weekofyear"] = df["date"].dt.isocalendar().week.astype(int)

    df["store_id"] = store_id
    df["dept_id"] = dept_id

    df["is_event"] = 0
    df["snap_CA"] = int(store_id.startswith("CA"))
    df["snap_TX"] = int(store_id.startswith("TX"))
    df["snap_WI"] = int(store_id.startswith("WI"))

    return df

# =========================
# Forecast autoregresivo
# =========================
def forecast_28(model, history, future):
    hist = history.copy()
    preds = []

    for i in range(28):
        row = future.iloc[i].copy()

        row["lag_1"] = hist["sales"].iloc[-1]
        row["lag_7"] = hist["sales"].iloc[-7]
        row["lag_14"] = hist["sales"].iloc[-14]

        row["rolling_7"] = hist["sales"].iloc[-7:].mean()
        row["rolling_14"] = hist["sales"].iloc[-14:].mean()

        X = row[features].to_frame().T
        X[cat_features] = encoder.transform(X[cat_features])

        y_pred = model.predict(X)[0]
        row["sales"] = y_pred

        preds.append(row)
        hist = pd.concat([hist, row[["date", "sales"]].to_frame().T])

    return pd.DataFrame(preds)

# =========================
# Ejecutar forecast
# =========================
st.markdown("---")
st.subheader("🔮 Forecast 28 días")

if st.button("Generar forecast"):
    history = df_hist[["date", "sales"]].tail(14)
    future = build_future(history["date"].max())

    forecast_df = forecast_28(model, history, future)

    st.dataframe(forecast_df[["date", "sales"]])

    combined = pd.concat([history, forecast_df])
    st.line_chart(combined.set_index("date")["sales"])

