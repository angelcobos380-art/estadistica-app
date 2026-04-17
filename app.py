import streamlit as st
import pandas as pd
import numpy as np

st.markdown("""
    <style>
    /* Fondo principal: Azul pizarra muy oscuro y elegante */
    .stApp {
        background-color: #0B0F19;
    }
    
    /* Panel lateral: Ligeramente más claro para dar profundidad */
    [data-testid="stSidebar"] {
        background-color: #111827;
        border-right: 1px solid #1F2937;
    }
    
    /* Títulos: Efecto de texto con gradiente (Cian a Púrpura) */
    h1, h2, h3 {
        background: -webkit-linear-gradient(45deg, #38BDF8, #818CF8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 700;
        padding-bottom: 10px;
    }
    
    /* Texto normal más legible en fondos oscuros */
    p, li, .stRadio > label {
        color: #94A3B8 !important;
    }
    
    /* Tarjetas de información y métricas (donde saldrán tus resultados) */
    [data-testid="stMetric"], .stAlert {
        background-color: #1E293B !important;
        border-radius: 10px;
        border: 1px solid #334155;
    }
    
    /* Botones: Diseño limpio con efecto al pasar el mouse */
    .stButton>button {
        background-color: #1E293B;
        color: #38BDF8;
        border: 1px solid #38BDF8;
        border-radius: 6px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #38BDF8;
        color: #0B0F19;
        box-shadow: 0 0 10px rgba(56, 189, 248, 0.4);
    }
    </style>
    """, unsafe_allow_html=True)

# Configuración de la página
st.set_page_config(page_title="App Estadística - Datos", layout="wide")

st.title(" Fase 2: Módulo de Gestión de Datos")
st.markdown("---")

# Sidebar para configuración
st.sidebar.header("Configuración de Datos")
metodo_datos = st.sidebar.radio(
    "Selecciona el origen de los datos:",
    ["Subir archivo CSV", "Generar datos aleatorios"]
)

df = None

# Lógica de Carga de Datos
if metodo_datos == "Subir archivo CSV":
    archivo_subido = st.file_uploader("Carga tu archivo .csv aquí", type=["csv"])
    if archivo_subido is not None:
        df = pd.read_csv(archivo_subido)
        st.success("✅ Archivo cargado con éxito.")
else:
    # Generación de datos sintéticos (Útil para pruebas rápidas)
    st.sidebar.subheader("Parámetros de datos sintéticos")
    n_puntos = st.sidebar.slider("Número de datos", 30, 500, 100)
    media_sintetica = st.sidebar.number_input("Media deseada", value=100.0)
    desv_sintetica = st.sidebar.number_input("Desviación estándar", value=15.0)
    
    if st.sidebar.button("Generar Datos"):
        datos = np.random.normal(loc=media_sintetica, scale=desv_sintetica, size=n_puntos)
        df = pd.DataFrame({"Valores": datos})
        st.info(f"✨ Se generaron {n_puntos} datos aleatorios.")

# Visualización y Validación (Solo si hay datos cargados)
if df is not None:
    st.header(" Vista Previa y Resumen Estadístico")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Primeros registros")
        st.write(df.head(10))
    
    with col2:
        st.subheader("Estadísticas descriptivas")
        # Esto calcula automáticamente media, mediana, min, max, etc.
        st.write(df.describe())
        
    # Selección de columna para futuros pasos
    columnas_num = df.select_dtypes(include=[np.number]).columns.tolist()
    if columnas_num:
        st.session_state['columna_seleccionada'] = st.selectbox(
            "Selecciona la columna para el análisis:", 
            columnas_num
        )
    else:
        st.error("⚠️ El archivo no contiene columnas numéricas.")
else:
    st.warning("Aún no hay datos. Por favor, sube un archivo o genera datos en el menú lateral.")

