import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px

# === Título de la App ===
st.title("🧠 App de Predicción de Riesgo")

# === Cargar modelo ===
@st.cache_resource
def cargar_modelo(path_modelo):
    with open(path_modelo, "rb") as f:
        modelo = pickle.load(f)
    return modelo

modelo = cargar_modelo("modelo_lightgbm_final_bkp.pkl")

# === Subir archivo ===
archivo = st.file_uploader("📤 Sube un archivo .parquet con tus datos", type=["parquet"])

if archivo:
    # === Leer archivo ===
    datos = pd.read_parquet(archivo)
    st.write("✅ Datos cargados:")
    st.dataframe(datos.head())

    # === Preprocesamiento mínimo ===
    datos.columns = datos.columns.str.lower()
    datos.fillna(0, inplace=True)
    datos["codzona_x"] = datos["codzona_x"].replace("", -1).astype(int)
    datos["codregion"] = datos["codregion"].replace("", -1).astype(int)
    datos = datos.astype({
        "codterritorio": int,
        "codzona_x": int,
        "codregion": int,
        "aniocampanaingreso": int,
        "campanaingreso": int,
        "aniocampana": int,
    })

    # === Predicción ===
    X_test = datos.drop(["codvendedora", "clase", "aniocampana"], axis=1)
    X_test = X_test.apply(pd.to_numeric, errors='coerce').fillna(0)
    probs = modelo.predict(X_test)

    datos["probs"] = probs

    # === Clasificación en deciles y riesgo ===
    datos["proba_decil"] = pd.qcut(
        datos["probs"].rank(method="first"),
        10,
        labels=False
    )

    datos["Perfil_Riesgo"] = np.where(
        datos.proba_decil < 5, "1 - Riesgo Bajo",
        np.where(datos.proba_decil < 8, "2 - Riesgo Medio", "3 - Riesgo Alto")
    )

    st.success("✅ Predicciones realizadas")

    # === Filtros ===
    with st.sidebar:
        st.header("📊 Filtros")
        riesgo_sel = st.multiselect(
            "Selecciona perfil de riesgo:",
            options=datos["Perfil_Riesgo"].unique(),
            default=datos["Perfil_Riesgo"].unique()
        )

        zona_sel = st.multiselect(
            "Selecciona zona:",
            options=sorted(datos["codzona_x"].unique()),
            default=sorted(datos["codzona_x"].unique())
        )

    datos_filtrados = datos[
        datos["Perfil_Riesgo"].isin(riesgo_sel) &
        datos["codzona_x"].isin(zona_sel)
    ]

    # === Resultados ===
    st.subheader("📈 Resultados filtrados")
    st.dataframe(datos_filtrados)

    st.subheader("📊 Distribución de perfiles de riesgo")
    fig = px.histogram(datos_filtrados, x="Perfil_Riesgo", color="Perfil_Riesgo")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("🗺️ Perfiles de riesgo por zona")

    fig2 = px.bar(
        datos_filtrados.groupby(["codzona_x", "Perfil_Riesgo"]).size().reset_index(name="conteo"),
        y="codzona_x",
        x="conteo",
        color="Perfil_Riesgo",
        barmode="group",
        orientation="h",
        title="Cantidad de perfiles de riesgo por zona",
        labels={"codzona_x": "Zona", "conteo": "Cantidad"}
    )
    st.plotly_chart(fig2, use_container_width=True)
    
    # === Descargar resultados ===
    csv = datos_filtrados.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Descargar resultados en CSV",
        data=csv,
        file_name="predicciones_filtradas.csv",
        mime="text/csv"
    )
