import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff

st.set_page_config(page_title="Análisis Estadístico - UP Chiapas", layout="wide")

# --- DISEÑO MIDNIGHT AURORA (CSS) ---
st.markdown("""
    <style>
    /* 1. FONDO TOTAL */
    .stApp {
        background-color: #0B0F19 !important;
    }

    /* 2. FORZAR BRILLO EN MÉTRICAS */
    [data-testid="stMetric"] {
        background-color: #1E293B !important;
        border: 1px solid #38BDF8 !important;
        border-radius: 12px !important;
        padding: 20px !important;
        opacity: 1 !important;
    }

    [data-testid="stMetricLabel"] p {
        color: #38BDF8 !important;
        font-weight: bold !important;
        font-size: 1.1rem !important;
        opacity: 1 !important;
    }

    [data-testid="stMetricValue"] div {
        color: #FFFFFF !important;
        font-weight: 800 !important;
        font-size: 2rem !important;
        opacity: 1 !important;
    }

    /* 3. TÍTULOS Y TEXTO GENERAL */
    h1, h2, h3 {
        background: -webkit-linear-gradient(45deg, #38BDF8, #818CF8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    p, li, label {
        color: #E2E8F0 !important;
    }

    /* 4. SIDEBAR */
    [data-testid="stSidebar"] {
        background-color: #020617 !important;
    }

    /* 5. ARREGLO PARA MENÚS DESPLEGABLES (NUEVO) */
    div[data-baseweb="popover"] {
        background-color: #1E293B !important;
    }
    
    div[data-baseweb="popover"] li {
        background-color: #1E293B !important;
        color: #F8FAFC !important;
    }

    div[data-baseweb="popover"] li:hover {
        background-color: #38BDF8 !important;
        color: #0B0F19 !important;
    }

    div[data-baseweb="popover"] ul {
        background-color: #1E293B !important;
        border: 1px solid #334155 !important;
    }
    </style>
    """, unsafe_allow_html=True)

# --- NAVEGACIÓN ---
modulo = st.sidebar.radio(
    "Módulos del Proyecto",
    ["📂 Carga de datos", "📊 Visualización", "🔬 Prueba de hipótesis", "🤖 Asistente IA"]
)

# --- MÓDULO 1: CARGA DE DATOS ---
if modulo == "📂 Carga de datos":
    st.header("Carga y Gestión de Datos")
    fuente = st.radio("Fuente de datos", ["Subir CSV", "Generar datos sintéticos"])

    if fuente == "Subir CSV":
        archivo = st.file_uploader("Sube tu archivo CSV", type=["csv"])
        if archivo:
            df = pd.read_csv(archivo)
            st.session_state["df"] = df
            st.success("✅ Archivo cargado correctamente")
    else:
        col1, col2, col3 = st.columns(3)
        with col1:
            distribucion = st.selectbox("Tipo de Distribución", ["Normal", "Sesgada (Exponencial)", "Uniforme"])
        with col2:
            n = st.slider("Tamaño de muestra (n)", 30, 500, 100)
        with col3:
            semilla = st.number_input("Semilla", value=42)

        if st.button("Generar Datos"):
            np.random.seed(int(semilla))
            if "Normal" in distribucion:
                datos = np.random.normal(loc=50, scale=10, size=n)
            elif "Sesgada" in distribucion:
                datos = np.random.exponential(scale=10, size=n) + 30
            else:
                datos = np.random.uniform(low=20, high=80, size=n)
            
            st.session_state["df"] = pd.DataFrame({"valor": datos})
            st.info(f"✨ Datos generados con éxito ({distribucion})")

    if "df" in st.session_state:
        st.subheader("Vista Previa")
        st.dataframe(st.session_state["df"].head(10), use_container_width=True)

# --- MÓDULO 2: VISUALIZACIÓN (FASE 3 - HOY) ---
elif modulo == "📊 Visualización":
    st.header("Análisis Visual de la Distribución")
    
    if "df" in st.session_state:
        df = st.session_state["df"]
        columnas_num = df.select_dtypes(include=[np.number]).columns.tolist()
        variable = st.selectbox("Selecciona la variable a graficar:", columnas_num)

        col_izq, col_der = st.columns(2)

        with col_izq:
            st.subheader("Histograma y Densidad")
            # Usamos Plotly para que sea interactivo
            fig_hist = px.histogram(df, x=variable, marginal="box", 
                                  title=f"Distribución de {variable}",
                                  color_discrete_sequence=['#38BDF8'])
            st.plotly_chart(fig_hist, use_container_width=True)

        with col_der:
            st.subheader("Gráfico de Probabilidad (Violín)")
            fig_violin = px.violin(df, y=variable, box=True, points="all",
                                 title=f"Dispersión y Outliers de {variable}",
                                 color_discrete_sequence=['#818CF8'])
            st.plotly_chart(fig_violin, use_container_width=True)

        # Análisis rápido para el reporte
        st.divider()
        st.subheader("Métricas Críticas")
        m1, m2, m3 = st.columns(3)
        m1.metric("Media", f"{df[variable].mean():.2f}")
        m2.metric("Desviación Estándar", f"{df[variable].std():.2f}")
        m3.metric("Sesgo (Skewness)", f"{df[variable].skew():.2f}")
    else:
        st.warning("⚠️ Primero carga datos en el módulo anterior.")

# --- MÓDULOS EN DESARROLLO ---
else:
    st.info(f"Módulo {modulo} en desarrollo.")