import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# =========================
# Configuración general
# =========================
st.set_page_config(
    page_title="Retail Sales Forecast",
    layout="wide"
)

TARGET = "sales"

FEATURES = [
    "store_id", "dept_id",
    "year", "month", "dayofweek", "weekofyear",
    "is_event",
    "snap_CA", "snap_TX", "snap_WI",
    "lag_1", "lag_7", "lag_14",
    "rolling_7", "rolling_14"
]

CAT_FEATURES = ["store_id", "dept_id"]

# =========================
# Cargar modelo y encoder
# =========================
@st.cache_resource
def load_model():
    model = joblib.load("gbr_model.pkl")
    encoder = joblib.load("encoder.pkl")
    return model, encoder

model, encoder = load_model()
st.success("✅ Modelo cargado correctamente")

# =========================
# Valores válidos desde encoder
# =========================
store_ids = list(encoder.categories_[0])
dept_ids = list(encoder.categories_[1])

# =========================
# Sidebar
# =========================
st.sidebar.header("Configuración")

store_id = st.sidebar.selectbox(
    "Store ID",
    store_ids
)

dept_id = st.sidebar.selectbox(
    "Department",
    dept_ids
)

horizon = st.sidebar.selectbox(
    "Horizonte de predicción (días)",
    [7, 14, 28]
)

# =========================
# Métricas del modelo
# =========================
st.markdown("### 📊 Métricas del Modelo")

col1, col2 = st.columns(2)
col1.metric("RMSE", "85.61")
col2.metric("MAE", "52.36")

# =========================
# Histórico SIMULADO (placeholder profesional)
# =========================
np.random.seed(42)

dates_hist = pd.date_range(end=pd.Timestamp.today(), periods=60)

sales_hist = np.maximum(
    0,
    np.random.normal(loc=350, scale=80, size=len(dates_hist))
)

hist_df = pd.DataFrame({
    "date": dates_hist,
    "sales": sales_hist
})

st.subheader("📊 Histórico de ventas")
st.line_chart(hist_df.set_index("date")["sales"])

# =========================
# Crear datos futuros
# =========================
future_dates = pd.date_range(
    start=hist_df["date"].max() + pd.Timedelta(days=1),
    periods=horizon
)

future_df = pd.DataFrame({
    "date": future_dates,
    "store_id": store_id,
    "dept_id": dept_id
})

future_df["year"] = future_df["date"].dt.year
future_df["month"] = future_df["date"].dt.month
future_df["dayofweek"] = future_df["date"].dt.dayofweek
future_df["weekofyear"] = future_df["date"].dt.isocalendar().week.astype(int)

future_df["is_event"] = 0
future_df["snap_CA"] = 1 if store_id.startswith("CA") else 0
future_df["snap_TX"] = 1 if store_id.startswith("TX") else 0
future_df["snap_WI"] = 1 if store_id.startswith("WI") else 0

last_sales = hist_df["sales"].iloc[-1]

future_df["lag_1"] = last_sales
future_df["lag_7"] = last_sales
future_df["lag_14"] = last_sales
future_df["rolling_7"] = last_sales
future_df["rolling_14"] = last_sales

X_future = future_df[FEATURES].copy()
X_future[CAT_FEATURES] = encoder.transform(X_future[CAT_FEATURES])

# =========================
# Predicción
# =========================
future_df["forecast"] = model.predict(X_future)

# =========================
# Gráfico
# =========================
st.markdown("### 📈 Ventas históricas vs Forecast")

fig, ax = plt.subplots(figsize=(10, 4))

ax.plot(hist_df["date"], hist_df["sales"], label="Histórico")
ax.plot(future_df["date"], future_df["forecast"], label="Forecast")

ax.legend()
ax.set_xlabel("Fecha")
ax.set_ylabel("Ventas")

st.pyplot(fig)

# =========================
# Descargar CSV
# =========================
st.markdown("### ⬇️ Descargar forecast")

csv = future_df[["date", "forecast"]].to_csv(index=False)

st.download_button(
    label="Descargar CSV",
    data=csv,
    file_name="forecast.csv",
    mime="text/csv"
)














