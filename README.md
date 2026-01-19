🛒 Retail Sales Forecast App (M5)

Aplicación interactiva construida con **Streamlit** para la predicción de ventas minoristas
usando un modelo de **Machine Learning (Gradient Boosting Regressor)** entrenado sobre datos
inspirados en el dataset **M5 Forecasting**.


🚀 Demo
👉 Streamlit Cloud App  
https://sl-m5-forecasting-app.streamlit.app/


🎯 Objetivo del proyecto
Permitir a usuarios de negocio:
- Seleccionar **Store** y **Department**
- Definir un **horizonte de predicción**
- Visualizar ventas históricas vs forecast
- Descargar predicciones en CSV


🧠 Modelo
- Algoritmo: **Gradient Boosting Regressor**
- Features:
  - Variables temporales (año, mes, semana)
  - Lags y rolling features
  - Variables categóricas codificadas con `OrdinalEncoder`
- Métricas:
  - RMSE ≈ 85
  - MAE ≈ 52


🛠️ Tecnologías
- Python
- Pandas / NumPy
- Scikit-learn
- Streamlit
- Matplotlib




⚠️ Notas
Este repositorio contiene **solo el código de inferencia**.
El entrenamiento del modelo se realizó por separado.



👤 Autor
**Steve Loveday**  
Data Scientist | Business Analytics | Machine Learning
