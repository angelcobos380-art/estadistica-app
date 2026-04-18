import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats

st.set_page_config(page_title="Análisis Estadístico - UP Chiapas", layout="wide")

# --- DISEÑO MIDNIGHT AURORA (CSS) ---
st.markdown("""
    <style>
    .stApp {
        background-color: #0B0F19 !important;
    }
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
    h1, h2, h3 {
        background: -webkit-linear-gradient(45deg, #38BDF8, #818CF8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    p, li, label {
        color: #E2E8F0 !important;
    }
    [data-testid="stSidebar"] {
        background-color: #020617 !important;
    }
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

# --- MÓDULO 2: VISUALIZACIÓN ---
elif modulo == "📊 Visualización":
    st.header("Análisis Visual de la Distribución")

    if "df" in st.session_state:
        df = st.session_state["df"]
        columnas_num = df.select_dtypes(include=[np.number]).columns.tolist()
        variable = st.selectbox("Selecciona la variable a graficar:", columnas_num)

        col_izq, col_der = st.columns(2)

        with col_izq:
            st.subheader("Histograma y Densidad")
            fig_hist = px.histogram(df, x=variable, marginal="box",
                                    title=f"Distribución de {variable}",
                                    color_discrete_sequence=['#38BDF8'])
            st.plotly_chart(fig_hist, use_container_width=True)

        with col_der:
            st.subheader("Boxplot y Outliers")
            fig_box = px.box(df, y=variable, points="all",
                             title=f"Boxplot de {variable}",
                             color_discrete_sequence=['#818CF8'])
            st.plotly_chart(fig_box, use_container_width=True)

        st.divider()
        st.subheader("Métricas Críticas")
        m1, m2, m3, m4 = st.columns(4)
        media    = df[variable].mean()
        std      = df[variable].std()
        sesgo    = df[variable].skew()
        q1       = df[variable].quantile(0.25)
        q3       = df[variable].quantile(0.75)
        iqr      = q3 - q1
        outliers = df[(df[variable] < q1 - 1.5*iqr) | (df[variable] > q3 + 1.5*iqr)].shape[0]

        m1.metric("Media", f"{media:.2f}")
        m2.metric("Desv. Estándar", f"{std:.2f}")
        m3.metric("Sesgo", f"{sesgo:.2f}")
        m4.metric("Outliers detectados", outliers)

        st.divider()
        st.subheader("Interpretación automática")

        if abs(sesgo) < 0.5:
            st.success("✅ La distribución parece aproximadamente normal (sesgo cercano a 0).")
        elif abs(sesgo) < 1:
            st.warning("⚠️ La distribución tiene sesgo moderado, podría no ser normal.")
        else:
            st.error("❌ La distribución tiene sesgo alto, probablemente no es normal.")

        if sesgo > 0.5:
            st.info("📈 Sesgo positivo: la cola se extiende hacia la derecha.")
        elif sesgo < -0.5:
            st.info("📉 Sesgo negativo: la cola se extiende hacia la izquierda.")

        if outliers == 0:
            st.success("✅ No se detectaron outliers.")
        else:
            st.warning(f"⚠️ Se detectaron {outliers} outlier(s) usando el criterio IQR.")

    else:
        st.warning("⚠️ Primero carga datos en el módulo anterior.")

# --- MÓDULO 3: PRUEBA Z ---
elif modulo == "🔬 Prueba de hipótesis":
    st.header("Prueba de Hipótesis — Prueba Z")
    st.markdown("Varianza poblacional conocida · Muestra grande (n ≥ 30)")

    if "df" not in st.session_state:
        st.warning("⚠️ Primero carga datos en el módulo de carga de datos.")
    else:
        df = st.session_state["df"]
        columnas_num = df.select_dtypes(include=[np.number]).columns.tolist()

        st.divider()
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Parámetros de la prueba")
            variable = st.selectbox("Variable a analizar:", columnas_num)
            mu0      = st.number_input("Hipótesis nula H₀ (μ₀):", value=50.0)
            sigma    = st.number_input("Desviación estándar poblacional (σ):", value=10.0, min_value=0.01)
            alpha    = st.selectbox("Nivel de significancia (α):", [0.01, 0.05, 0.10])
            tipo     = st.radio("Tipo de prueba:", ["Bilateral", "Cola derecha", "Cola izquierda"])

        with col2:
            st.subheader("Planteamiento de hipótesis")
            if tipo == "Bilateral":
                st.latex(r"H_0: \mu = \mu_0")
                st.latex(r"H_1: \mu \neq \mu_0")
            elif tipo == "Cola derecha":
                st.latex(r"H_0: \mu = \mu_0")
                st.latex(r"H_1: \mu > \mu_0")
            else:
                st.latex(r"H_0: \mu = \mu_0")
                st.latex(r"H_1: \mu < \mu_0")

        st.divider()

        if st.button("Ejecutar prueba Z"):
            datos  = df[variable].dropna()
            n      = len(datos)
            x_bar  = datos.mean()
            z_obs  = (x_bar - mu0) / (sigma / np.sqrt(n))

            # p-value según tipo de prueba
            if tipo == "Bilateral":
                p_value  = 2 * (1 - stats.norm.cdf(abs(z_obs)))
                z_crit_i = stats.norm.ppf(alpha / 2)
                z_crit_d = stats.norm.ppf(1 - alpha / 2)
            elif tipo == "Cola derecha":
                p_value  = 1 - stats.norm.cdf(z_obs)
                z_crit_i = None
                z_crit_d = stats.norm.ppf(1 - alpha)
            else:
                p_value  = stats.norm.cdf(z_obs)
                z_crit_i = stats.norm.ppf(alpha)
                z_crit_d = None

            rechazar = p_value < alpha

            # Resultados
            st.subheader("Resultados")
            r1, r2, r3, r4 = st.columns(4)
            r1.metric("n", n)
            r2.metric("Media muestral (x̄)", f"{x_bar:.4f}")
            r3.metric("Z calculado", f"{z_obs:.4f}")
            r4.metric("p-value", f"{p_value:.4f}")

            st.divider()

            if rechazar:
                st.error(f"❌ Se rechaza H₀ al nivel α = {alpha}. Hay evidencia estadística suficiente.")
            else:
                st.success(f"✅ No se rechaza H₀ al nivel α = {alpha}. No hay evidencia suficiente para rechazarla.")

            # Curva normal con rango dinámico (mejora #2)
            st.subheader("Curva normal con zona de rechazo")
            x_min = min(-4, z_obs - 1)
            x_max = max(4, z_obs + 1)
            x = np.linspace(x_min, x_max, 400)
            y = stats.norm.pdf(x)

            fig = go.Figure()

            fig.add_trace(go.Scatter(x=x, y=y, mode='lines',
                                     line=dict(color='#38BDF8', width=2),
                                     name='N(0,1)'))

            if tipo in ["Bilateral", "Cola izquierda"]:
                x_left = x[x <= z_crit_i]
                y_left = stats.norm.pdf(x_left)
                fig.add_trace(go.Scatter(
                    x=np.concatenate([x_left, x_left[::-1]]),
                    y=np.concatenate([y_left, np.zeros(len(y_left))]),
                    fill='toself', fillcolor='rgba(239,68,68,0.4)',
                    line=dict(color='rgba(0,0,0,0)'), name='Zona de rechazo'))

            if tipo in ["Bilateral", "Cola derecha"]:
                x_right = x[x >= z_crit_d]
                y_right = stats.norm.pdf(x_right)
                fig.add_trace(go.Scatter(
                    x=np.concatenate([x_right, x_right[::-1]]),
                    y=np.concatenate([y_right, np.zeros(len(y_right))]),
                    fill='toself', fillcolor='rgba(239,68,68,0.4)',
                    line=dict(color='rgba(0,0,0,0)'), name='Zona de rechazo'))

            fig.add_vline(x=z_obs, line_dash="dash", line_color="#FACC15",
                          annotation_text=f"Z obs = {z_obs:.2f}",
                          annotation_font_color="#FACC15")

            fig.update_layout(
                paper_bgcolor='#0B0F19',
                plot_bgcolor='#0B0F19',
                font_color='#E2E8F0',
                title="Distribución Normal Estándar — Región Crítica",
                xaxis_title="Estadístico Z",
                yaxis_title="Densidad"
            )

            st.plotly_chart(fig, use_container_width=True)

            st.session_state["resultado_z"] = {
                "variable": variable, "n": n, "x_bar": x_bar,
                "mu0": mu0, "sigma": sigma, "alpha": alpha,
                "tipo": tipo, "z_obs": z_obs,
                "p_value": p_value, "rechazar": rechazar
            }

# --- MÓDULO 4: ASISTENTE IA ---
elif modulo == "🤖 Asistente IA":
    st.header("Asistente con IA (Gemini)")
    st.info("Módulo en desarrollo — Viernes 18")