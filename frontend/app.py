import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates
import pandas as pd
import numpy as np
import altair as alt
import joblib
import sqlite3
import os
import html
import datetime
from PIL import Image, ImageDraw, ImageOps
from transformers import pipeline

# ==========================================
# 1. CONFIGURACIÓN Y ESTILOS
# ==========================================
st.set_page_config(
    page_title="AstroVital | NASA Space Apps",
    layout="wide",
    initial_sidebar_state="expanded"
)

def cargar_estilos():
    ruta_css = "frontend/style.css"
    if os.path.exists(ruta_css):
        with open(ruta_css, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

cargar_estilos()

FECHA_INICIO_MISION = datetime.date(2026, 6, 1)

def dia_a_fecha_str(dia):
    return (FECHA_INICIO_MISION + datetime.timedelta(days=int(dia))).strftime("%Y-%m-%d")

# ==========================================
# 2. UMBRALES Y PARÁMETRO DE CONTROL SANO (ESTÁNDAR NASA)
# ==========================================
CONTROL_SANO_NASA = {
    "id_astronauta": "CONTROL_NASA",
    "nombre": "Parámetro Control Sano (NASA)",
    "rol": "Referencia Fisiológica Ideal",
    "dia_mision": 150,
    "frecuencia_cardiaca": 68,
    "presion_sistolica": 115,
    "horas_sueno": 7.8,
    "densidad_osea": 0.25,
    "radiacion_diaria": 0.75,
    "radiacion_acumulada": 112.5,
    "nivel_riesgo": 0
}

UMBRALES = {
    "frecuencia_cardiaca": {
        "nombre": "Frecuencia cardíaca", "unidad": "bpm", "rango_normal": "60 – 85 bpm",
        "warn_min": 52, "warn_max": 85, "crit_min": 45, "crit_max": 100, "zona": "Torso",
        "control": 68.0
    },
    "presion_sistolica": {
        "nombre": "Presión sistólica", "unidad": "mmHg", "rango_normal": "95 – 125 mmHg",
        "warn_min": 90, "warn_max": 125, "crit_min": 85, "crit_max": 140, "zona": "Torso",
        "control": 115.0
    },
    "horas_sueno": {
        "nombre": "Horas de sueño", "unidad": "h", "rango_normal": "6.0 – 8.5 h",
        "warn_min": 5.0, "warn_max": 10.0, "crit_min": 4.5, "crit_max": 11.5, "zona": "Cabeza",
        "control": 7.8
    },
    "densidad_osea": {
        "nombre": "Densidad ósea (T-Score)", "unidad": "T-Score", "rango_normal": ">= -1.0",
        "warn_min": -1.0, "warn_max": 2.0, "crit_min": -2.5, "crit_max": 3.0, "zona": "Extremidades",
        "control": 0.25
    },
    "radiacion_diaria": {
        "nombre": "Radiación diaria", "unidad": "mSv", "rango_normal": "0.5 – 2.5 mSv/día",
        "warn_min": 0.0, "warn_max": 2.5, "crit_min": 0.0, "crit_max": 20.0, "zona": "Entorno",
        "control": 0.75
    },
    "radiacion_acumulada": {
        "nombre": "Radiación acumulada", "unidad": "mSv", "rango_normal": "< 150 mSv",
        "warn_min": 0.0, "warn_max": 150.0, "crit_min": 0.0, "crit_max": 300.0, "zona": "Entorno",
        "control": 112.5
    }
}

def evaluar_metrica(clave, valor):
    u = UMBRALES[clave]
    if valor < u["crit_min"] or valor > u["crit_max"]:
        return 2, "crit", f"Crítico · Rango normal: {u['rango_normal']}"
    if valor < u["warn_min"] or valor > u["warn_max"]:
        return 1, "warn", f"Preventivo · Rango normal: {u['rango_normal']}"
    return 0, "ok", f"Nominal ({u['rango_normal']})"

def explicar_causas_riesgo(datos):
    causas = []
    for clave, meta in UMBRALES.items():
        if clave in datos:
            niv, _, _ = evaluar_metrica(clave, float(datos[clave]))
            if niv == 2:
                causas.append(f"{meta['nombre']} ({datos[clave]} {meta['unidad']}) [crítico]")
            elif niv == 1:
                causas.append(f"{meta['nombre']} ({datos[clave]} {meta['unidad']}) [preventivo]")
    return " · ".join(causas) if causas else "Parámetros en rango nominal"

# ==========================================
# 3. CORRELACIÓN CAUSAL
# ==========================================
def analizar_correlacion_causal(clave_mod, nivel_sev, fila_bio, fecha_str):
    if fila_bio is None:
        return "No existen registros biométricos en esa fecha para establecer causalidad."

    fc = int(fila_bio["frecuencia_cardiaca"])
    ps = int(fila_bio["presion_sistolica"])
    sueno = float(fila_bio["horas_sueno"])
    osea = float(fila_bio["densidad_osea"])
    rad_d = float(fila_bio["radiacion_diaria"])
    rad_a = float(fila_bio["radiacion_acumulada"])

    hallazgos = []
    if clave_mod == "huesos":
        if osea < -1.0:
            hallazgos.append(f"el T-Score sistémico cayó a {osea} (osteopenia por microgravedad)")
        if sueno < 5.5:
            hallazgos.append(f"el déficit de sueño ({sueno} h) redujo la regeneración osteoblástica")
        if rad_a > 150:
            hallazgos.append(f"la radiación acumulada ({rad_a} mSv) aceleró la pérdida trabecular")
    elif clave_mod == "tomografia":
        if rad_d > 10.0 or rad_a > 200.0:
            hallazgos.append(f"exposición aguda a radiación espacial ({rad_d} mSv/día · {rad_a} mSv acum.)")
        if ps > 125 or fc > 85:
            hallazgos.append(f"hipertensión intracraneal/sistólica ({ps} mmHg) y taquicardia ({fc} bpm) por desplazamiento cefálico de fluidos")
    elif clave_mod == "animo":
        if sueno < 5.5:
            hallazgos.append(f"privación crónica de sueño ({sueno} h registradas el {fecha_str})")
        if fc > 85 or ps > 125:
            hallazgos.append(f"hiperactividad simpática por estrés fisiológico ({fc} bpm / {ps} mmHg)")
        if rad_d > 10.0:
            hallazgos.append(f"alerta de tormenta solar activa ({rad_d} mSv/día)")

    if nivel_sev == 0 and not hallazgos:
        return f"Estudio normal el {fecha_str}. Todos los signos vitales concurrentes se encuentran en rango saludable."
    if nivel_sev > 0 and hallazgos:
        return f" Correlación positiva ({fecha_str}): La anomalía visual fue detonada/agravada porque " + " y ".join(hallazgos) + "."
    if nivel_sev > 0 and not hallazgos:
        return f"Hallazgo primario aislado el {fecha_str}: los signos vitales sistémicos aún no reflejan descompensación."
    return f"Estudio morfológicamente estable el {fecha_str}, pero se vigila: " + ", ".join(hallazgos) + "."

# ==========================================
# 4. CACHÉ DE MODELOS (RANDOM FOREST + HUGGING FACE) Y SQLITE
# ==========================================
@st.cache_resource
def cargar_modelos():
    try:
        modelo_rf = joblib.load("ml_models/estimador_riesgo.pkl")
        columnas_rf = joblib.load("ml_models/columnas.pkl")
    except FileNotFoundError:
        modelo_rf, columnas_rf = None, None

    modelos_cnn = {}
    try:
        modelos_cnn["animo"] = pipeline("image-classification", model="dima806/facial_emotions_image_detection")
        # Modelo especializado en fracturas óseas
        modelos_cnn["huesos"] = pipeline("image-classification", model="Hemg/bone-fracture-detection-using-x-rays")
        modelos_cnn["tomografia"] = pipeline("image-classification", model="google/vit-base-patch16-224")
    except Exception as e:
        st.warning(f"No se pudieron cargar los modelos de visión de Hugging Face: {e}")
        modelos_cnn = None

    return modelo_rf, columnas_rf, modelos_cnn

def conectar_db():
    return sqlite3.connect("data/astrovital.db")

@st.cache_data(ttl=60)
def obtener_astronautas():
    with conectar_db() as conn:
        return pd.read_sql("SELECT id_astronauta, nombre, edad, rol FROM astronautas", conn)

@st.cache_data(ttl=60)
def obtener_historial_tripulante(id_astronauta):
    with conectar_db() as conn:
        df = pd.read_sql(
            """SELECT id_registro, id_astronauta, dia_mision, frecuencia_cardiaca, presion_sistolica,
                      horas_sueno, densidad_osea, radiacion_diaria, radiacion_acumulada,
                      nivel_riesgo, fecha_registro
               FROM historial_medico
               WHERE id_astronauta = ?
               ORDER BY dia_mision ASC, id_registro ASC""",
            conn, params=(id_astronauta,)
        )
    if not df.empty:
        df["fecha_calendario"] = df["dia_mision"].apply(dia_a_fecha_str)
    return df

@st.cache_data(ttl=60)
def obtener_historial_completo():
    with conectar_db() as conn:
        df = pd.read_sql(
            """SELECT h.*, a.nombre, a.rol
               FROM historial_medico h
               JOIN astronautas a ON h.id_astronauta = a.id_astronauta
               ORDER BY h.dia_mision ASC, h.id_registro ASC""",
            conn
        )
    if not df.empty:
        df["fecha_calendario"] = df["dia_mision"].apply(dia_a_fecha_str)
    return df

@st.cache_data(ttl=60)
def obtener_estudios_imagen(id_astronauta=None):
    with conectar_db() as conn:
        if id_astronauta:
            return pd.read_sql(
                """SELECT e.*, a.nombre
                   FROM estudios_imagen e
                   JOIN astronautas a ON e.id_astronauta = a.id_astronauta
                   WHERE e.id_astronauta = ?
                   ORDER BY e.dia_mision DESC, e.id_estudio DESC""",
                conn, params=(id_astronauta,)
            )
        return pd.read_sql(
            """SELECT e.*, a.nombre
               FROM estudios_imagen e
               JOIN astronautas a ON e.id_astronauta = a.id_astronauta
               ORDER BY e.dia_mision DESC, e.id_estudio DESC""",
            conn
        )

def guardar_diagnostico_db(id_astronauta, datos, riesgo):
    with conectar_db() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO historial_medico
            (id_astronauta, dia_mision, frecuencia_cardiaca, presion_sistolica, horas_sueno,
             densidad_osea, radiacion_diaria, radiacion_acumulada, nivel_riesgo)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            id_astronauta, int(datos['dia_mision']), int(datos['frecuencia_cardiaca']),
            int(datos['presion_sistolica']), float(datos['horas_sueno']),
            float(datos['densidad_osea']), float(datos['radiacion_diaria']),
            float(datos['radiacion_acumulada']), int(riesgo)
        ))
        conn.commit()
    st.cache_data.clear()

def guardar_estudio_imagen_db(id_astro, dia, fecha_str, modalidad, pred_txt, confianza, sev, corr, ruta_img):
    with conectar_db() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO estudios_imagen
            (id_astronauta, dia_mision, fecha_estudio, modalidad, prediccion_cnn,
             confianza, nivel_severidad, correlacion_clinica, ruta_imagen)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (id_astro, int(dia), fecha_str, modalidad, pred_txt, float(confianza), int(sev), corr, ruta_img))
        conn.commit()
    st.cache_data.clear()

# Mostrar spinner mientras descarga los modelos pesados la primera vez
with st.spinner("Inicializando sistemas de Inteligencia Artificial (Random Forest + Hugging Face)..."):
    modelo, columnas, modelos_cnn = cargar_modelos()

if not modelo:
    st.error("Error crítico: el modelo de signos vitales no está disponible.")
    st.stop()

# ==========================================
# 5. COMPONENTES VISUALES Y GRÁFICOS
# ==========================================
def render_header(titulo_seccion, subtitulo):
    st.markdown(
        f'<div class="hero"><div>'
        f'<h1>AstroVital · {html.escape(titulo_seccion)}</h1>'
        f'<p>{html.escape(subtitulo)}</p>'
        f'</div><div class="hero-tag">NASA Space Apps · Deep Space Medical OS</div></div>',
        unsafe_allow_html=True
    )

def veredicto_html(riesgo, causales="", origen="Diagnóstico activo"):
    mapa = {
        0: ("ok", "Estado Nominal", "Operaciones normales. Los biomarcadores se mantienen dentro de los márgenes seguros para vuelo espacial."),
        1: ("warn", "Alerta Preventiva", "Desviación fisiológica detectada. Ajuste higiene de sueño, hidratación y contramedidas de resistencia ARED."),
        2: ("crit", "Estado Crítico", "Peligro biológico inminente. Exposición radiológica o desgaste sistémico fuera de límite operativo. Active protocolo de refugio.")
    }
    clase, titulo, texto = mapa.get(int(riesgo), mapa[0])
    segmentos = "".join(f'<i class="{"on" if i <= riesgo else ""}"></i>' for i in range(3))
    extra = f'<p style="margin-top:.45rem;font-size:.84rem;color:#475569;"><strong>Factores detectados:</strong> {causales}</p>' if causales else ""
    return (
        f'<div class="verdict verdict-{clase}">'
        f'<div class="verdict-head"><div><span class="verdict-title">{titulo}</span>'
        f'<div style="font-size:.78rem;color:#5B6878;">{html.escape(origen)}</div></div>'
        f'<span class="scale">{segmentos}</span></div>'
        f'<p>{texto}</p>{extra}</div>'
    )

def indicador_campo_html(clave, valor):
    _, clase, msg = evaluar_metrica(clave, valor)
    st.markdown(
        f'<div class="field-ctx ctx-{clase}"><span>{msg}</span>'
        f'<span><strong>{valor} {UMBRALES[clave]["unidad"]}</strong></span></div>',
        unsafe_allow_html=True
    )

def grafico_progresion_limpio(df, col_y, titulo_y, color_linea, umbral_warn, umbral_crit, dominio_y=None):
    escala_y = alt.Scale(domain=dominio_y, zero=False) if dominio_y else alt.Scale(zero=False)
    base = alt.Chart(df).encode(
        x=alt.X("dia_mision:Q", title="Día de misión", axis=alt.Axis(grid=False))
    )
    area = base.mark_area(opacity=0.12, color=color_linea).encode(
        y=alt.Y(f"{col_y}:Q", title=titulo_y, scale=escala_y)
    )
    linea = base.mark_line(color=color_linea, strokeWidth=2.5).encode(
        y=alt.Y(f"{col_y}:Q", scale=escala_y),
        tooltip=[
            alt.Tooltip("dia_mision:Q", title="Día"),
            alt.Tooltip("fecha_calendario:N", title="Fecha"),
            alt.Tooltip(f"{col_y}:Q", title=titulo_y, format=".2f"),
            alt.Tooltip("nivel_riesgo:Q", title="Nivel Riesgo IA")
        ]
    )
    df_crit = df[df["nivel_riesgo"] == 2]
    puntos_crit = alt.Chart(df_crit).mark_circle(size=45, color="#DC2626").encode(
        x="dia_mision:Q",
        y=alt.Y(f"{col_y}:Q", scale=escala_y),
        tooltip=[alt.Tooltip("fecha_calendario:N", title="Fecha crítica"), alt.Tooltip(f"{col_y}:Q", title=titulo_y)]
    )
    regla_w = alt.Chart(pd.DataFrame({"y": [umbral_warn]})).mark_rule(
        color="#B45309", strokeDash=[5, 4], strokeWidth=1.5
    ).encode(y="y:Q")
    regla_c = alt.Chart(pd.DataFrame({"y": [umbral_crit]})).mark_rule(
        color="#DC2626", strokeDash=[4, 3], strokeWidth=1.8
    ).encode(y="y:Q")

    return (
        alt.layer(area, linea, puntos_crit, regla_w, regla_c)
        .properties(height=260, background="transparent")
        .configure_view(strokeWidth=0)
        .configure_axis(labelColor="#5B6878", titleColor="#0E1B2C", gridColor="#EEF2F6")
    )

def grafico_vital_con_media_movil(df, col_y, titulo_y, color_principal, umbral_warn, umbral_crit):
    df_plot = df.copy()
    df_plot["media_7d"] = df_plot[col_y].rolling(window=7, min_periods=1).mean().round(2)

    base = alt.Chart(df_plot).encode(
        x=alt.X("dia_mision:Q", title="Día de misión", axis=alt.Axis(grid=False))
    )
    linea_diaria = base.mark_line(color="#94A3B8", strokeWidth=1.2, opacity=0.45).encode(
        y=alt.Y(f"{col_y}:Q", title=titulo_y, scale=alt.Scale(zero=False))
    )
    linea_tendencia = base.mark_line(color=color_principal, strokeWidth=2.8).encode(
        y=alt.Y("media_7d:Q", scale=alt.Scale(zero=False)),
        tooltip=[
            alt.Tooltip("dia_mision:Q", title="Día"),
            alt.Tooltip("fecha_calendario:N", title="Fecha"),
            alt.Tooltip(f"{col_y}:Q", title="Valor del día", format=".1f"),
            alt.Tooltip("media_7d:Q", title="Promedio 7 días", format=".1f")
        ]
    )
    regla_w = alt.Chart(pd.DataFrame({"y": [umbral_warn]})).mark_rule(
        color="#B45309", strokeDash=[5, 4], strokeWidth=1.5
    ).encode(y="y:Q")
    regla_c = alt.Chart(pd.DataFrame({"y": [umbral_crit]})).mark_rule(
        color="#DC2626", strokeDash=[4, 3], strokeWidth=1.8
    ).encode(y="y:Q")

    return (
        alt.layer(linea_diaria, linea_tendencia, regla_w, regla_c)
        .properties(height=280, background="transparent")
        .configure_view(strokeWidth=0)
        .configure_axis(labelColor="#5B6878", titleColor="#0E1B2C", gridColor="#EEF2F6")
    )

# ==========================================
# 6. ESTADO DE SESIÓN Y BARRA LATERAL
# ==========================================
if "datos_medicos" not in st.session_state:
    st.session_state.datos_medicos = {
        "dia_mision": 150, "frecuencia_cardiaca": 72, "presion_sistolica": 118,
        "horas_sueno": 6.5, "densidad_osea": -0.5, "radiacion_diaria": 1.5, "radiacion_acumulada": 120.0
    }
if "tripulante_actual" not in st.session_state:
    st.session_state.tripulante_actual = None
if "ultimo_diagnostico" not in st.session_state:
    st.session_state.ultimo_diagnostico = {}
if "zona_seleccionada" not in st.session_state:
    st.session_state.zona_seleccionada = "Cabeza"

def fila_telemetria(etiqueta, valor, unidad=""):
    u = f'<span class="tele-u">{unidad}</span>' if unidad else ""
    return f'<div class="tele-row"><span class="tele-k">{etiqueta}</span><span class="tele-v">{valor}{u}</span></div>'

def ejecutar_inferencia_y_guardar(id_astro, guardar=True):
    with st.spinner("Procesando telemetría con modelo Random Forest..."):
        d = st.session_state.datos_medicos
        datos_entrada = pd.DataFrame([[
            d["dia_mision"], d["frecuencia_cardiaca"], d["presion_sistolica"],
            d["horas_sueno"], d["densidad_osea"], d["radiacion_diaria"], d["radiacion_acumulada"]
        ]], columns=columnas)
        riesgo = int(modelo.predict(datos_entrada)[0])
        causas = explicar_causas_riesgo(d)
        ahora = datetime.datetime.now().strftime("%H:%M:%S")
        st.session_state.ultimo_diagnostico[id_astro] = {
            "riesgo": riesgo,
            "causas": causas,
            "hora": f"Día {d['dia_mision']} ({dia_a_fecha_str(d['dia_mision'])}) · Evaluado a las {ahora}"
        }
        if guardar:
            guardar_diagnostico_db(id_astro, d, riesgo)
            st.toast(f"Diagnóstico guardado para {id_astro} (Nivel {riesgo})")

def render_sidebar():
    df_astronautas = obtener_astronautas()
    if df_astronautas.empty:
        st.sidebar.error("La tabla 'astronautas' está vacía.")
        st.stop()

    st.sidebar.markdown('<div class="side-title">Control de Misión</div>', unsafe_allow_html=True)

    nombres_lista = df_astronautas["nombre"].tolist()
    idx_default = 0
    if st.session_state.tripulante_actual:
        astro_actual_df = df_astronautas[df_astronautas["id_astronauta"] == st.session_state.tripulante_actual]
        if not astro_actual_df.empty:
            nombre_actual = astro_actual_df["nombre"].values[0]
            if nombre_actual in nombres_lista:
                idx_default = nombres_lista.index(nombre_actual)

    nombre_sel = st.sidebar.selectbox("Tripulante activo", nombres_lista, index=idx_default)
    fila_astro = df_astronautas[df_astronautas["nombre"] == nombre_sel].iloc[0]
    id_sel = fila_astro["id_astronauta"]

    if st.session_state.tripulante_actual != id_sel:
        st.session_state.tripulante_actual = id_sel
        df_hist = obtener_historial_tripulante(id_sel)
        if not df_hist.empty:
            reg = df_hist.iloc[-1]
            st.session_state.datos_medicos = {
                "dia_mision": int(reg["dia_mision"]),
                "frecuencia_cardiaca": int(reg["frecuencia_cardiaca"]),
                "presion_sistolica": int(reg["presion_sistolica"]),
                "horas_sueno": float(reg["horas_sueno"]),
                "densidad_osea": float(reg["densidad_osea"]),
                "radiacion_diaria": float(reg["radiacion_diaria"]),
                "radiacion_acumulada": float(reg["radiacion_acumulada"])
            }
            st.session_state.ultimo_diagnostico[id_sel] = {
                "riesgo": int(reg["nivel_riesgo"]),
                "causas": explicar_causas_riesgo(st.session_state.datos_medicos),
                "hora": f"Último corte en BD · Día {int(reg['dia_mision'])} ({dia_a_fecha_str(reg['dia_mision'])})"
            }

    st.sidebar.markdown(
        f'<div class="crew"><div class="crew-name">{html.escape(str(fila_astro["nombre"]))}</div>'
        f'<div class="crew-role">{html.escape(str(fila_astro["rol"]))} · {int(fila_astro["edad"])} años</div></div>',
        unsafe_allow_html=True
    )

    d = st.session_state.datos_medicos
    st.sidebar.markdown('<div class="side-title">Telemetría en Memoria</div>', unsafe_allow_html=True)
    st.sidebar.markdown(
        '<div class="tele">'
        + fila_telemetria("Fecha misión", dia_a_fecha_str(d["dia_mision"]))
        + fila_telemetria("Día operativo", d["dia_mision"])
        + fila_telemetria("Frecuencia cardíaca", d["frecuencia_cardiaca"], "bpm")
        + fila_telemetria("Presión sistólica", d["presion_sistolica"], "mmHg")
        + fila_telemetria("Horas de sueño", d["horas_sueno"], "h")
        + fila_telemetria("Densidad ósea", d["densidad_osea"], "T")
        + fila_telemetria("Radiación acum.", d["radiacion_acumulada"], "mSv")
        + '</div>',
        unsafe_allow_html=True
    )

    if st.sidebar.button("Diagnosticar y Guardar", type="primary", width="stretch"):
        ejecutar_inferencia_y_guardar(id_sel, guardar=True)

# ==========================================
# 7. PÁGINAS DE LA APLICACIÓN
# ==========================================

# --- PÁGINA 1: RESUMEN DE MISIÓN ---
def page_resumen():
    render_header("Resumen de Misión", "Estado bioastronáutico en tiempo real de toda la tripulación.")
    df_astro = obtener_astronautas()
    df_all = obtener_historial_completo()

    if df_all.empty:
        st.info("No existen registros médicos aún.")
        return

    ultimos = df_all.sort_values("dia_mision").groupby("id_astronauta").last().reset_index()
    n_crit = int((ultimos["nivel_riesgo"] == 2).sum())
    n_warn = int((ultimos["nivel_riesgo"] == 1).sum())
    rad_max = float(ultimos["radiacion_acumulada"].max())

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        with st.container(border=True):
            st.metric("Tripulantes en misión", len(df_astro))
    with k2:
        with st.container(border=True):
            st.metric("En estado crítico", n_crit)
    with k3:
        with st.container(border=True):
            st.metric("En alerta preventiva", n_warn)
    with k4:
        with st.container(border=True):
            st.metric("Máx. radiación acumulada", f"{rad_max:.1f} mSv")

    st.markdown("#### Tarjetas clínicas por tripulante")
    cols = st.columns(len(df_astro))

    for idx, (_, astro) in enumerate(df_astro.iterrows()):
        id_a = astro["id_astronauta"]
        sub = ultimos[ultimos["id_astronauta"] == id_a]
        with cols[idx]:
            with st.container(border=True):
                if sub.empty:
                    st.markdown(f'<div class="panel-title">{html.escape(astro["nombre"])}</div>', unsafe_allow_html=True)
                    st.info("Sin registros iniciales.")
                    continue

                u = sub.iloc[0]
                riesgo = int(u["nivel_riesgo"])
                badge_cls = "badge-ok" if riesgo == 0 else ("badge-warn" if riesgo == 1 else "badge-crit")
                badge_txt = "NOMINAL" if riesgo == 0 else ("PREVENTIVO" if riesgo == 1 else "CRÍTICO")
                causas = explicar_causas_riesgo(u)

                st.markdown(
                    f'<div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:.75rem;">'
                    f'<div><div class="panel-title">{html.escape(astro["nombre"])}</div>'
                    f'<div style="font-size:.8rem;color:#5B6878;">{html.escape(astro["rol"])} · {dia_a_fecha_str(u["dia_mision"])}</div></div>'
                    f'<span class="badge {badge_cls}">{badge_txt}</span></div>',
                    unsafe_allow_html=True
                )
                c_a, c_b = st.columns(2)
                c_a.metric("Pulso", f"{int(u['frecuencia_cardiaca'])} bpm")
                c_b.metric("Presión", f"{int(u['presion_sistolica'])} mmHg")
                c_c, c_d = st.columns(2)
                c_c.metric("Densidad Ósea", f"{float(u['densidad_osea']):.2f} T")
                c_d.metric("Radiación", f"{float(u['radiacion_acumulada']):.1f} mSv")
                st.caption(f"**Diagnóstico:** {causas}")

# --- PÁGINA 2: INGESTA BIOMÉTRICA ---
def page_ingesta():
    id_sel = st.session_state.tripulante_actual
    df_astro = obtener_astronautas()
    nombre_astro = df_astro[df_astro["id_astronauta"] == id_sel]["nombre"].values[0]
    render_header("Ingesta Biométrica", f"Captura de signos vitales y evaluación clínica para {nombre_astro}.")

    if id_sel in st.session_state.ultimo_diagnostico:
        diag = st.session_state.ultimo_diagnostico[id_sel]
        st.markdown(veredicto_html(diag["riesgo"], diag["causas"], diag["hora"]), unsafe_allow_html=True)

    modo = st.radio(
        "Modo de captura:",
        ["Selector Anatómico Interactivo", "Formulario Clínico Unificado"],
        horizontal=True,
        label_visibility="collapsed"
    )

    d = st.session_state.datos_medicos
    niv_cabeza, _, _ = evaluar_metrica("horas_sueno", d["horas_sueno"])
    niv_torso = max(evaluar_metrica("frecuencia_cardiaca", d["frecuencia_cardiaca"])[0],
                    evaluar_metrica("presion_sistolica", d["presion_sistolica"])[0])
    niv_ext, _, _ = evaluar_metrica("densidad_osea", d["densidad_osea"])
    niv_ent = max(evaluar_metrica("radiacion_diaria", d["radiacion_diaria"])[0],
                  evaluar_metrica("radiacion_acumulada", d["radiacion_acumulada"])[0])
    estados_zona = {
        "Cabeza": ("Salud Conductual y Sueño", niv_cabeza),
        "Torso": ("Sistema Cardiovascular", niv_torso),
        "Extremidades": ("Sistema Osteomuscular", niv_ext),
        "Entorno": ("Dosimetría Radiológica", niv_ent)
    }

    if "Selector" in modo:
        col_img, col_form = st.columns([1, 1.85])
        with col_img:
            with st.container(border=True):
                st.markdown('<div class="panel-title">Modelo del Tripulante</div>'
                            '<div class="panel-note">Haz clic sobre el astronauta o elige el sistema abajo.</div>',
                            unsafe_allow_html=True)

                img_path = "frontend/astronauta.png"
                if not os.path.exists(img_path):
                    img_path = "https://cdn-icons-png.flaticon.com/512/1055/1055661.png"

                coordenadas = streamlit_image_coordinates(img_path, width=260, key="astronauta_limpio")

                if coordenadas is not None:
                    y = coordenadas["y"]
                    if y < 95:
                        st.session_state.zona_seleccionada = "Cabeza"
                    elif y < 185:
                        st.session_state.zona_seleccionada = "Torso"
                    else:
                        st.session_state.zona_seleccionada = "Extremidades"

                etiquetas_sem = {0: ("badge-ok", "Nominal"), 1: ("badge-warn", "Preventivo"), 2: ("badge-crit", "Crítico")}
                html_lista = '<div class="zone-list">'
                for z_key, (z_desc, z_niv) in estados_zona.items():
                    cls_b, txt_b = etiquetas_sem[z_niv]
                    act_cls = "active" if st.session_state.zona_seleccionada == z_key else ""
                    html_lista += (
                        f'<div class="zone-item {act_cls}">'
                        f'<span>{z_key} · <small style="color:#5B6878;">{z_desc}</small></span>'
                        f'<span class="badge {cls_b}">{txt_b}</span></div>'
                    )
                html_lista += '</div>'
                st.markdown(html_lista, unsafe_allow_html=True)

        with col_form:
            with st.container(border=True):
                st.markdown('<div class="panel-title">Parámetros por Sistema Anatómico</div>'
                            '<div class="panel-note">Cambia de región haciendo clic en el astronauta o con este selector:</div>',
                            unsafe_allow_html=True)

                zona = st.segmented_control(
                    "Región anatómica",
                    options=["Cabeza", "Torso", "Extremidades", "Entorno"],
                    default=st.session_state.zona_seleccionada,
                    label_visibility="collapsed"
                )
                if zona and zona != st.session_state.zona_seleccionada:
                    st.session_state.zona_seleccionada = zona

                z_actual = st.session_state.zona_seleccionada

                if z_actual == "Cabeza":
                    st.markdown("#### Salud Conductual y Fatiga Circadiana")
                    d["horas_sueno"] = st.slider("Horas de sueño (últimas 24h)", 0.0, 12.0, float(d["horas_sueno"]), 0.5)
                    indicador_campo_html("horas_sueno", d["horas_sueno"])

                elif z_actual == "Torso":
                    st.markdown("#### Sistema Cardiovascular (Desplazamiento de Fluidos)")
                    d["frecuencia_cardiaca"] = st.number_input("Frecuencia cardíaca (BPM)", 30, 200, int(d["frecuencia_cardiaca"]))
                    indicador_campo_html("frecuencia_cardiaca", d["frecuencia_cardiaca"])
                    d["presion_sistolica"] = st.number_input("Presión sistólica (mmHg)", 80, 250, int(d["presion_sistolica"]))
                    indicador_campo_html("presion_sistolica", d["presion_sistolica"])

                elif z_actual == "Extremidades":
                    st.markdown("#### Sistema Osteomuscular (Microgravedad)")
                    d["densidad_osea"] = st.slider("Densidad ósea (T-Score)", -4.0, 1.0, float(d["densidad_osea"]), 0.1)
                    indicador_campo_html("densidad_osea", d["densidad_osea"])

                else:
                    st.markdown("#### Dosimetría Espacial y Día de Misión")
                    d["radiacion_diaria"] = st.number_input("Radiación diaria (mSv)", 0.0, 150.0, float(d["radiacion_diaria"]), step=0.5)
                    indicador_campo_html("radiacion_diaria", d["radiacion_diaria"])
                    d["radiacion_acumulada"] = st.number_input("Radiación acumulada (mSv)", 0.0, 1500.0, float(d["radiacion_acumulada"]), step=5.0)
                    indicador_campo_html("radiacion_acumulada", d["radiacion_acumulada"])
                    d["dia_mision"] = st.number_input("Día actual de la misión", 1, 1000, int(d["dia_mision"]))

                st.markdown("---")
                if st.button("Ejecutar Diagnóstico IA y Guardar en BD", type="primary"):
                    ejecutar_inferencia_y_guardar(id_sel, guardar=True)
                    st.rerun()

    else:
        with st.container(border=True):
            st.markdown('<div class="panel-title">Formulario Clínico Unificado</div>', unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                d["dia_mision"] = st.number_input("Día actual de la misión", 1, 1000, int(d["dia_mision"]))
                d["frecuencia_cardiaca"] = st.number_input("Frecuencia cardíaca (BPM)", 30, 200, int(d["frecuencia_cardiaca"]))
                indicador_campo_html("frecuencia_cardiaca", d["frecuencia_cardiaca"])
                d["presion_sistolica"] = st.number_input("Presión sistólica (mmHg)", 80, 250, int(d["presion_sistolica"]))
                indicador_campo_html("presion_sistolica", d["presion_sistolica"])
                d["horas_sueno"] = st.slider("Horas de sueño (24h)", 0.0, 12.0, float(d["horas_sueno"]), 0.5)
                indicador_campo_html("horas_sueno", d["horas_sueno"])
            with c2:
                d["densidad_osea"] = st.slider("Densidad ósea (T-Score)", -4.0, 1.0, float(d["densidad_osea"]), 0.1)
                indicador_campo_html("densidad_osea", d["densidad_osea"])
                d["radiacion_diaria"] = st.number_input("Radiación diaria (mSv)", 0.0, 150.0, float(d["radiacion_diaria"]), step=0.5)
                indicador_campo_html("radiacion_diaria", d["radiacion_diaria"])
                d["radiacion_acumulada"] = st.number_input("Radiación acumulada (mSv)", 0.0, 1500.0, float(d["radiacion_acumulada"]), step=5.0)
                indicador_campo_html("radiacion_acumulada", d["radiacion_acumulada"])

            if st.button("Evaluar y Registrar Check-in", type="primary"):
                ejecutar_inferencia_y_guardar(id_sel, guardar=True)
                st.rerun()

# --- PÁGINA 3: EVOLUCIÓN TEMPORAL ---
def page_ficha():
    id_sel = st.session_state.tripulante_actual
    df_astro = obtener_astronautas()
    nombre_astro = df_astro[df_astro["id_astronauta"] == id_sel]["nombre"].values[0]
    render_header("Evolución Temporal", f"Monitoreo histórico con fechas, umbrales clínicos y tendencias de {nombre_astro}.")

    df_hist = obtener_historial_tripulante(id_sel)
    if df_hist.empty:
        st.info("No existen registros históricos para este tripulante.")
        return

    st.markdown(
        '<div class="chart-legend">'
        '<span><strong>Referencias:</strong></span>'
        '<span><i class="leg-dot" style="background:#16794B;"></i>Nominal (Nivel 0)</span>'
        '<span><i class="leg-dot" style="background:#B45309;"></i>Preventivo (Nivel 1)</span>'
        '<span><i class="leg-dot" style="background:#DC2626;"></i>Crítico (Nivel 2)</span>'
        '<span><i class="leg-line" style="border-color:#B45309;"></i>Umbral Preventivo</span>'
        '<span><i class="leg-line" style="border-color:#DC2626;"></i>Umbral Crítico</span>'
        '</div>',
        unsafe_allow_html=True
    )

    with st.container(border=True):
        st.markdown('<div class="panel-title">Cronograma de Nivel de Riesgo (Clasificación IA)</div>', unsafe_allow_html=True)
        escala_riesgo = alt.Scale(domain=[0, 1, 2], range=["#16794B", "#D97706", "#DC2626"])
        chart_riesgo = (
            alt.Chart(df_hist)
            .mark_rect()
            .encode(
                x=alt.X("dia_mision:O", title="Día de misión", axis=alt.Axis(values=list(range(10, 160, 10)), labelAngle=0)),
                color=alt.Color("nivel_riesgo:N", scale=escala_riesgo, legend=None),
                tooltip=[
                    alt.Tooltip("dia_mision:Q", title="Día"),
                    alt.Tooltip("fecha_calendario:N", title="Fecha"),
                    alt.Tooltip("nivel_riesgo:Q", title="Nivel de Riesgo"),
                    alt.Tooltip("radiacion_acumulada:Q", title="Radiación (mSv)")
                ]
            )
            .properties(height=68, background="transparent")
            .configure_view(strokeWidth=0)
        )
        st.altair_chart(chart_riesgo, width="stretch")

    g1, g2 = st.columns(2)
    with g1:
        with st.container(border=True):
            st.markdown('<div class="panel-title">Exposición a Radiación Acumulada (mSv)</div>'
                        '<div class="panel-note">Umbral preventivo: 150 mSv · Límite crítico operativo: 300 mSv</div>',
                        unsafe_allow_html=True)
            st.altair_chart(
                grafico_progresion_limpio(df_hist, "radiacion_acumulada", "mSv acumulados", "#DC2626", 150, 300),
                width="stretch"
            )

    with g2:
        with st.container(border=True):
            st.markdown('<div class="panel-title">Degradación de Densidad Ósea (T-Score)</div>'
                        '<div class="panel-note">Umbral de osteopenia: -1.0 · Umbral de osteoporosis: -2.5</div>',
                        unsafe_allow_html=True)
            st.altair_chart(
                grafico_progresion_limpio(df_hist, "densidad_osea", "T-Score", "#0B3D91", -1.0, -2.5, dominio_y=[-3.0, 0.8]),
                width="stretch"
            )

    with st.container(border=True):
        c_head, c_sel = st.columns([1.3, 1])
        with c_head:
            st.markdown('<div class="panel-title">Monitor de Signos Vitales y Tendencia Semanal</div>'
                        '<div class="panel-note">Línea sólida: tendencia media de 7 días · Línea gris tenue: fluctuación diaria.</div>',
                        unsafe_allow_html=True)
        with c_sel:
            vital_sel = st.segmented_control(
                "Biomarcador",
                options=["Frecuencia Cardíaca", "Presión Sistólica", "Horas de Sueño"],
                default="Frecuencia Cardíaca",
                label_visibility="collapsed"
            )

        if vital_sel and "Presión" in vital_sel:
            st.altair_chart(
                grafico_vital_con_media_movil(df_hist, "presion_sistolica", "Presión Sistólica (mmHg)", "#0B3D91", 125, 140),
                width="stretch"
            )
        elif vital_sel and "Sueño" in vital_sel:
            st.altair_chart(
                grafico_vital_con_media_movil(df_hist, "horas_sueno", "Horas de Sueño (24h)", "#2563EB", 5.0, 4.5),
                width="stretch"
            )
        else:
            st.altair_chart(
                grafico_vital_con_media_movil(df_hist, "frecuencia_cardiaca", "Frecuencia Cardíaca (bpm)", "#16794B", 85, 100),
                width="stretch"
            )

    with st.container(border=True):
        st.markdown('<div class="panel-title">Registro Histórico Fechado y Causa Raíz</div>', unsafe_allow_html=True)
        df_tabla = df_hist.sort_values("dia_mision", ascending=False).copy()
        df_tabla["Estado"] = df_tabla["nivel_riesgo"].map({0: "Nominal", 1: "Preventivo", 2: "Crítico"})
        df_tabla["Factores causales"] = df_tabla.apply(explicar_causas_riesgo, axis=1)
        columnas_vista = [
            "fecha_calendario", "dia_mision", "Estado", "Factores causales",
            "frecuencia_cardiaca", "presion_sistolica", "horas_sueno", "densidad_osea", "radiacion_acumulada"
        ]
        st.dataframe(df_tabla[columnas_vista], width="stretch", hide_index=True)

# --- PÁGINA 4: COMPARADOR MULTITRIPULANTE CON VALOR DE CONTROL SANO ---
def page_comparador():
    render_header(
        "Comparador Clínico y Control Sano",
        "Visualiza todos los datos de la tripulación y compara cualquier parámetro frente al Estándar de Control Sano."
    )
    df_astro = obtener_astronautas()
    df_all = obtener_historial_completo()

    if df_all.empty:
        st.info("No hay datos suficientes para comparar.")
        return

    ultimos = df_all.sort_values("dia_mision").groupby("id_astronauta").last().reset_index()

    with st.container(border=True):
        st.markdown(
            '<div class="panel-title">1. Matriz General de la Tripulación vs. Parámetro de Control Sano</div>'
            '<div class="panel-note">Estado actual de todos los astronautas comparado con los valores fisiológicos ideales (Estándar NASA).</div>',
            unsafe_allow_html=True
        )

        filas_matriz = [{
            " Sujeto / Referencia": CONTROL_SANO_NASA["nombre"],
            "Rol": CONTROL_SANO_NASA["rol"],
            "Fecha": dia_a_fecha_str(CONTROL_SANO_NASA["dia_mision"]),
            "Día": CONTROL_SANO_NASA["dia_mision"],
            "Estado": "CONTROL SANO",
            "Frec. Cardíaca (bpm)": CONTROL_SANO_NASA["frecuencia_cardiaca"],
            "Presión Sist. (mmHg)": CONTROL_SANO_NASA["presion_sistolica"],
            "Sueño (h)": CONTROL_SANO_NASA["horas_sueno"],
            "Densidad Ósea (T-Score)": CONTROL_SANO_NASA["densidad_osea"],
            "Rad. Diaria (mSv)": CONTROL_SANO_NASA["radiacion_diaria"],
            "Rad. Acumulada (mSv)": CONTROL_SANO_NASA["radiacion_acumulada"],
            "Diagnóstico / Causa": "Parámetros ideales de referencia"
        }]

        for _, u in ultimos.iterrows():
            riesgo = int(u["nivel_riesgo"])
            est_str = "Nominal" if riesgo == 0 else ("Preventivo" if riesgo == 1 else "Crítico")
            filas_matriz.append({
                " Sujeto / Referencia": u["nombre"],
                "Rol": u["rol"],
                "Fecha": dia_a_fecha_str(u["dia_mision"]),
                "Día": int(u["dia_mision"]),
                "Estado": est_str,
                "Frec. Cardíaca (bpm)": int(u["frecuencia_cardiaca"]),
                "Presión Sist. (mmHg)": int(u["presion_sistolica"]),
                "Sueño (h)": float(u["horas_sueno"]),
                "Densidad Ósea (T-Score)": float(u["densidad_osea"]),
                "Rad. Diaria (mSv)": float(u["radiacion_diaria"]),
                "Rad. Acumulada (mSv)": float(u["radiacion_acumulada"]),
                "Diagnóstico / Causa": explicar_causas_riesgo(u)
            })

        df_matriz = pd.DataFrame(filas_matriz)
        st.dataframe(df_matriz, width="stretch", hide_index=True)

    with st.container(border=True):
        st.markdown(
            '<div class="panel-title">2. Comparador Selectivo por Biomarcador y Tripulantes</div>'
            '<div class="panel-note">Elige qué valor clínico deseas analizar y selecciona libremente qué astronautas comparar contra el Control Sano.</div>',
            unsafe_allow_html=True
        )

        mapa_metricas = {
            "Radiación acumulada (mSv)": "radiacion_acumulada",
            "Densidad ósea (T-Score)": "densidad_osea",
            "Frecuencia cardíaca (bpm)": "frecuencia_cardiaca",
            "Presión sistólica (mmHg)": "presion_sistolica",
            "Horas de sueño (h)": "horas_sueno",
            "Radiación diaria (mSv)": "radiacion_diaria"
        }

        c_sel1, c_sel2, c_sel3 = st.columns([1.1, 1.6, 0.9])
        with c_sel1:
            etiqueta_metrica = st.selectbox("Valor clínico a comparar:", list(mapa_metricas.keys()))
            col_metrica = mapa_metricas[etiqueta_metrica]
        with c_sel2:
            nombres_disponibles = df_astro["nombre"].tolist()
            tripulantes_elegidos = st.multiselect(
                "Tripulantes a comparar (elige 1, 2, 3 o los 4):",
                options=nombres_disponibles,
                default=nombres_disponibles
            )
        with c_sel3:
            st.write("")
            incluir_control = st.checkbox("Incluir Control Sano (NASA)", value=True)

        if not tripulantes_elegidos:
            st.warning("Selecciona al menos un tripulante en el selector superior para graficar la comparativa.")
            return

        valor_control = UMBRALES[col_metrica]["control"]
        unidad_m = UMBRALES[col_metrica]["unidad"]

        dias_unicos = sorted(df_all["dia_mision"].unique())
        df_plot_hist = df_all[df_all["nombre"].isin(tripulantes_elegidos)][
            ["dia_mision", "fecha_calendario", "nombre", col_metrica]
        ].copy()

        if incluir_control:
            if col_metrica == "radiacion_acumulada":
                valores_ctrl_serie = [round(d * 0.75, 2) for d in dias_unicos]
            elif col_metrica == "densidad_osea":
                valores_ctrl_serie = [round(0.40 - 0.15 * (d / 150.0), 2) for d in dias_unicos]
            else:
                valores_ctrl_serie = [valor_control for _ in dias_unicos]

            df_ctrl_hist = pd.DataFrame({
                "dia_mision": dias_unicos,
                "fecha_calendario": [dia_a_fecha_str(d) for d in dias_unicos],
                "nombre": CONTROL_SANO_NASA["nombre"],
                col_metrica: valores_ctrl_serie
            })
            df_plot_hist = pd.concat([df_plot_hist, df_ctrl_hist], ignore_index=True)

        g_curva, g_barras = st.columns([1.45, 1])

        with g_curva:
            st.markdown(f"**Evolución histórica: {etiqueta_metrica}**")
            ch_comp_linea = (
                alt.Chart(df_plot_hist)
                .mark_line(strokeWidth=2.5)
                .encode(
                    x=alt.X("dia_mision:Q", title="Día de misión"),
                    y=alt.Y(f"{col_metrica}:Q", title=etiqueta_metrica, scale=alt.Scale(zero=False)),
                    color=alt.Color("nombre:N", title="Sujeto"),
                    strokeDash=alt.condition(
                        alt.datum.nombre == CONTROL_SANO_NASA["nombre"],
                        alt.value([6, 4]),
                        alt.value([1, 0])
                    ),
                    tooltip=["nombre", "fecha_calendario", "dia_mision", alt.Tooltip(f"{col_metrica}:Q", format=".2f")]
                )
                .properties(height=300, background="transparent")
                .configure_view(strokeWidth=0)
            )
            st.altair_chart(ch_comp_linea, width="stretch")

        with g_barras:
            st.markdown(f"**Corte actual vs. Control Sano ({valor_control} {unidad_m})**")
            filas_bar = []
            if incluir_control:
                filas_bar.append({
                    "Sujeto": CONTROL_SANO_NASA["nombre"],
                    "Valor Actual": float(valor_control),
                    "Diferencia vs Control": 0.0,
                    "Tipo": "Control Sano"
                })
            for nom in tripulantes_elegidos:
                u_row = ultimos[ultimos["nombre"] == nom].iloc[0]
                val_act = float(u_row[col_metrica])
                filas_bar.append({
                    "Sujeto": nom,
                    "Valor Actual": val_act,
                    "Diferencia vs Control": round(val_act - valor_control, 2),
                    "Tipo": "Tripulante"
                })
            df_bar = pd.DataFrame(filas_bar)

            ch_bar = (
                alt.Chart(df_bar)
                .mark_bar(cornerRadiusTopRight=5, cornerRadiusBottomRight=5, height=28)
                .encode(
                    x=alt.X("Valor Actual:Q", title=etiqueta_metrica),
                    y=alt.Y("Sujeto:N", sort=None, title=None),
                    color=alt.Color(
                        "Tipo:N",
                        scale=alt.Scale(domain=["Control Sano", "Tripulante"], range=["#16794B", "#0B3D91"]),
                        legend=None
                    ),
                    tooltip=["Sujeto", "Valor Actual", "Diferencia vs Control"]
                )
                .properties(height=300, background="transparent")
                .configure_view(strokeWidth=0)
            )
            st.altair_chart(ch_bar, width="stretch")

        st.dataframe(df_bar, width="stretch", hide_index=True)

# --- PÁGINA 5: VISIÓN COMPUTACIONAL (HUGGING FACE + FECHAS + CAUSALIDAD) ---
def page_vision():
    id_sel = st.session_state.tripulante_actual
    df_astro = obtener_astronautas()
    nombre_astro = df_astro[df_astro["id_astronauta"] == id_sel]["nombre"].values[0]
    df_hist = obtener_historial_tripulante(id_sel)

    render_header(
        "Visión Computacional (IA Preentrenada)",
        f"Modelos de Transfer Learning (Hugging Face) para evaluación médica offline de {nombre_astro}."
    )

    if not modelos_cnn:
        st.error("No se pudieron cargar los modelos de visión de Hugging Face. Revisa tu conexión a internet para la descarga inicial.")
        return

    m1, m2, m3 = st.columns(3)
    with m1:
        with st.container(border=True):
            st.metric("Huesos (ViT X-Ray)", "Transfer Learning", "Modelo: Hemg/bone-fracture")
    with m2:
        with st.container(border=True):
            st.metric("Tomografía (ViT Brain)", "Transfer Learning", "Modelo: google/vit-base")
    with m3:
        with st.container(border=True):
            st.metric("Ánimo (ViT Facial)", "Transfer Learning", "Modelo: dima806/facial_emotions")

    tab_analizador, tab_historial_img = st.tabs([
        "Analizador Diagnóstico y Correlación Causal",
        "Historial Fechado de Estudios de Imagen"
    ])

    with tab_analizador:
        with st.container(border=True):
            c_cfg1, c_cfg2 = st.columns([1.5, 1])
            with c_cfg1:
                mod_etiqueta = st.selectbox(
                    "1. Tipo de estudio médico a procesar:",
                    [
                        "Densitometría Ósea (Rayos X)",
                        "Tomografía Axial (TAC)",
                        "Biometría de Estado de Ánimo"
                    ]
                )
                clave_mod = "huesos" if "Ósea" in mod_etiqueta else ("tomografia" if "Tomografía" in mod_etiqueta else "animo")

            with c_cfg2:
                dia_estudio = st.slider("2. Día de misión del estudio:", 1, 150, 150)
                fecha_estudio_str = dia_a_fecha_str(dia_estudio)
                st.caption(f"**Fecha del estudio:** `{fecha_estudio_str}`")

            img_analizar = None
            ruta_guardado = "muestra_subida.png"

            # CORRECCIÓN: Etiqueta integrada directamente en el componente para evitar advertencias de accesibilidad
            up = st.file_uploader("3. Sube la imagen médica real (PNG/JPG):", type=["png", "jpg", "jpeg"])
            
            if up is not None:
                img_analizar = Image.open(up).convert("RGB")
                os.makedirs("data/muestras_vision", exist_ok=True)
                ruta_guardado = f"data/muestras_vision/hf_{id_sel}_dia{dia_estudio}_{clave_mod}.png"
                img_analizar.save(ruta_guardado)

            if img_analizar is not None:
                # 1. Hacer predicción con Hugging Face
                clasificador_hf = modelos_cnn[clave_mod]
                resultado_hf = clasificador_hf(img_analizar)
                mejor_prediccion = resultado_hf[0]
                label_hf = mejor_prediccion['label'].lower()
                confianza = round(mejor_prediccion['score'] * 100, 1)

                # 2. Mapear la etiqueta de HF a nuestro sistema de gravedad (0, 1, 2)
                if clave_mod == "huesos":
                    if "not fractur" in label_hf or "normal" in label_hf:
                        pred_clase, pred_nombre = 0, "Densidad Ósea Nominal"
                    elif "fractur" in label_hf or "abnormal" in label_hf:
                        pred_clase, pred_nombre = 2, f"Lesión Ósea / Fractura ({label_hf})"
                    else:
                        pred_clase, pred_nombre = 0, f"Densidad Ósea Nominal ({label_hf})"
                        
                elif clave_mod == "tomografia":
                    if "normal" in label_hf:
                        pred_clase, pred_nombre = 0, "Tomografía Axial Normal"
                    else:
                        pred_clase, pred_nombre = 2, f"Anomalía Encefálica ({label_hf})"
                        
                elif clave_mod == "animo":
                    if label_hf in ["happy", "neutral"]:
                        pred_clase, pred_nombre = 0, f"Estado Anímico Nominal ({label_hf})"
                    elif label_hf in ["surprise", "sad"]:
                        pred_clase, pred_nombre = 1, f"Fatiga Circadiana ({label_hf})"
                    else:
                        pred_clase, pred_nombre = 2, f"Agotamiento Severo / Estrés ({label_hf})"

                # 3. CORRECCIÓN CRÍTICA: Declarar e instanciar fila_bio_dia antes de usarla
                fila_bio_dia = None
                if not df_hist.empty:
                    sub_b = df_hist[df_hist["dia_mision"] == dia_estudio]
                    if not sub_b.empty:
                        fila_bio_dia = sub_b.iloc[-1]

                texto_correlacion = analizar_correlacion_causal(clave_mod, pred_clase, fila_bio_dia, fecha_estudio_str)

                st.markdown("---")
                col_v1, col_v3 = st.columns([1, 1.4])
                with col_v1:
                    st.markdown('<div class="panel-title">Muestra Analizada</div>', unsafe_allow_html=True)
                    st.image(img_analizar.resize((280, 280)))
                    st.caption(f"Tripulante: **{nombre_astro}** · `{fecha_estudio_str}`")

                with col_v3:
                    cls_ver = "ok" if pred_clase == 0 else ("warn" if pred_clase == 1 else "crit")
                    st.markdown(
                        f'<div class="verdict verdict-{cls_ver}">'
                        f'<div class="verdict-head"><span class="verdict-title">{html.escape(pred_nombre)}</span>'
                        f'<span class="badge badge-{cls_ver}">Precisión HF: {confianza}%</span></div>'
                        f'<p style="margin-top:.5rem;"><strong>Análisis de Causalidad ({fecha_estudio_str} · Día {dia_estudio}):</strong><br>{html.escape(texto_correlacion)}</p>'
                        f'</div>',
                        unsafe_allow_html=True
                    )

                    if fila_bio_dia is not None:
                        st.markdown(
                            f"**Biometría concurrente del {fecha_estudio_str}:** "
                            f"Pulso `{int(fila_bio_dia['frecuencia_cardiaca'])} bpm` · "
                            f"Presión `{int(fila_bio_dia['presion_sistolica'])} mmHg` · "
                            f"Sueño `{float(fila_bio_dia['horas_sueno'])} h` · "
                            f"T-Score `{float(fila_bio_dia['densidad_osea'])}` · "
                            f"Radiación `{float(fila_bio_dia['radiacion_acumulada'])} mSv`"
                        )

                    df_p_cnn = pd.DataFrame(resultado_hf)
                    df_p_cnn['score'] = (df_p_cnn['score'] * 100).round(1)
                    st.dataframe(df_p_cnn.rename(columns={'label': 'Clase (Hugging Face)', 'score': 'Probabilidad (%)'}), width="stretch", hide_index=True)

                    if st.button("Registrar Estudio Fechado en Expediente Clínico", type="primary"):
                        guardar_estudio_imagen_db(
                            id_sel, dia_estudio, fecha_estudio_str, mod_etiqueta,
                            pred_nombre, confianza, pred_clase, texto_correlacion, ruta_guardado
                        )
                        st.toast(f"Estudio de {mod_etiqueta} registrado con fecha {fecha_estudio_str}.")
                        st.rerun()

    with tab_historial_img:
        with st.container(border=True):
            filtro_alcance = st.radio(
                "Mostrar estudios de:",
                [f"Solo {nombre_astro}", "Toda la tripulación"],
                horizontal=True
            )
            df_est = obtener_estudios_imagen(id_sel if "Solo" in filtro_alcance else None)

            if df_est.empty:
                st.info("No hay estudios de imagen registrados.")
            else:
                df_est_vista = df_est.copy()
                df_est_vista["Severidad"] = df_est_vista["nivel_severidad"].map({
                    0: "Nominal", 1: "Preventivo", 2: "Crítico"
                })
                cols_mostrar = [
                    "fecha_estudio", "dia_mision", "nombre", "modalidad",
                    "prediccion_cnn", "confianza", "Severidad", "correlacion_clinica"
                ]
                st.dataframe(df_est_vista[cols_mostrar], width="stretch", hide_index=True)
                
# --- PÁGINA 6: CENTRO DE ALERTAS ---
def page_alertas():
    render_header("Centro de Alertas", "Anomalías biométricas y de visión computacional ordenadas por prioridad clínica.")
    df_all = obtener_historial_completo()
    if df_all.empty:
        st.info("No hay registros activos.")
        return

    ultimos = df_all.sort_values("dia_mision").groupby("id_astronauta").last().reset_index()
    alertas = []

    for _, fila in ultimos.iterrows():
        for clave, meta in UMBRALES.items():
            val = float(fila[clave])
            niv, cls, _ = evaluar_metrica(clave, val)
            if niv > 0:
                alertas.append({
                    "gravedad": niv, "clase": cls,
                    "etiqueta": "CRÍTICO" if niv == 2 else "PREVENTIVO",
                    "tripulante": fila["nombre"],
                    "fecha": dia_a_fecha_str(fila["dia_mision"]),
                    "dia": int(fila["dia_mision"]),
                    "variable": meta["nombre"],
                    "valor": f"{val} {meta['unidad']}",
                    "rango_normal": meta["rango_normal"],
                    "zona": meta["zona"]
                })

    alertas = sorted(alertas, key=lambda x: x["gravedad"], reverse=True)

    if not alertas:
        st.success("Todos los tripulantes se encuentran dentro de los umbrales nominales.")
        return

    for a in alertas:
        cls_badge = "badge-crit" if a["gravedad"] == 2 else "badge-warn"
        st.markdown(
            f'<div class="alert-card alert-{a["clase"]}">'
            f'<div><span class="badge {cls_badge}">{a["etiqueta"]}</span> '
            f'<strong style="margin-left:.5rem;font-size:.98rem;">{html.escape(a["tripulante"])} · {html.escape(a["variable"])}</strong>'
            f'<div style="font-size:.83rem;color:#5B6878;margin-top:.2rem;">'
            f'Sistema: <strong>{a["zona"]}</strong> · Fecha: {a["fecha"]} (Día {a["dia"]}) · Rango Control Sano: {a["rango_normal"]}</div></div>'
            f'<div style="text-align:right;"><div style="font-size:1.15rem;font-weight:700;">{a["valor"]}</div>'
            f'<div style="font-size:.76rem;color:#5B6878;">Fuera de umbral</div></div></div>',
            unsafe_allow_html=True
        )

# --- PÁGINA 7: REPORTE IMPRIMIBLE ---
def page_reporte():
    id_sel = st.session_state.tripulante_actual
    df_astro = obtener_astronautas()
    astro = df_astro[df_astro["id_astronauta"] == id_sel].iloc[0]
    df_hist = obtener_historial_tripulante(id_sel)
    df_est = obtener_estudios_imagen(id_sel)

    render_header("Reporte Clínico (PDF)", f"Expediente médico oficial de vuelo para {astro['nombre']}.")

    if df_hist.empty:
        st.info("Este tripulante aún no cuenta con registros.")
        return

    u = df_hist.iloc[-1]
    riesgo = int(u["nivel_riesgo"])
    estado_txt = "NOMINAL (Nivel 0)" if riesgo == 0 else ("ALERTA PREVENTIVA (Nivel 1)" if riesgo == 1 else "ESTADO CRÍTICO (Nivel 2)")
    causas = explicar_causas_riesgo(u)
    fecha_rep = datetime.datetime.now().strftime("%Y-%m-%d %H:%M UTC")

    filas_img_html = ""
    for _, est in df_est.head(4).iterrows():
        filas_img_html += (
            f"<tr><td style='padding:8px;border:1px solid #E2E8F0;'>{est['fecha_estudio']} (Día {est['dia_mision']})</td>"
            f"<td style='padding:8px;border:1px solid #E2E8F0;'>{html.escape(est['modalidad'])}</td>"
            f"<td style='padding:8px;border:1px solid #E2E8F0;'><strong>{html.escape(est['prediccion_cnn'])}</strong> ({est['confianza']}%)</td>"
            f"<td style='padding:8px;border:1px solid #E2E8F0;'>{html.escape(est['correlacion_clinica'])}</td></tr>"
        )

    reporte_html = f"""
    <div class="print-box">
        <div style="display:flex;justify-content:space-between;border-bottom:2px solid #0B3D91;padding-bottom:1rem;margin-bottom:1.2rem;">
            <div>
                <h2 style="margin:0;color:#0B3D91;">ASTROVITAL · EXPEDIENTE MÉDICO DE VUELO</h2>
                <div style="color:#5B6878;font-size:.88rem;">Sistema Autónomo de Evaluación Biomédica y Visión Computacional</div>
            </div>
            <div style="text-align:right;font-size:.84rem;color:#5B6878;">
                <div><strong>Folio:</strong> {id_sel}-D{int(u['dia_mision'])}</div>
                <div><strong>Fecha clínica:</strong> {dia_a_fecha_str(u['dia_mision'])}</div>
                <div><strong>Emisión:</strong> {fecha_rep}</div>
            </div>
        </div>
        <p><strong>Tripulante:</strong> {html.escape(astro['nombre'])} &nbsp;|&nbsp;
           <strong>Rol:</strong> {html.escape(astro['rol'])} &nbsp;|&nbsp;
           <strong>Edad:</strong> {int(astro['edad'])} años &nbsp;|&nbsp;
           <strong>Día de misión:</strong> {int(u['dia_mision'])}</p>
        <p><strong>Dictamen IA:</strong> <span style="font-weight:700;">{estado_txt}</span></p>
        <p><strong>Factores causales:</strong> {causas}</p>
        <hr style="border-color:#E2E8F0;margin:1rem 0;">
        <h4>1. Signos Vitales vs. Parámetro de Control Sano (NASA)</h4>
        <table style="width:100%;border-collapse:collapse;font-size:.9rem;margin-bottom:1.2rem;">
            <thead>
                <tr style="background:#F4F6F9;text-align:left;">
                    <th style="padding:9px;border:1px solid #E2E8F0;">Parámetro</th>
                    <th style="padding:9px;border:1px solid #E2E8F0;">Valor Tripulante</th>
                    <th style="padding:9px;border:1px solid #E2E8F0;">Control Sano (NASA)</th>
                    <th style="padding:9px;border:1px solid #E2E8F0;">Rango Nominal</th>
                </tr>
            </thead>
            <tbody>
                <tr><td style="padding:9px;border:1px solid #E2E8F0;">Frecuencia cardíaca</td><td style="padding:9px;border:1px solid #E2E8F0;">{int(u['frecuencia_cardiaca'])} bpm</td><td style="padding:9px;border:1px solid #E2E8F0;">68 bpm</td><td style="padding:9px;border:1px solid #E2E8F0;">60 – 85 bpm</td></tr>
                <tr><td style="padding:9px;border:1px solid #E2E8F0;">Presión sistólica</td><td style="padding:9px;border:1px solid #E2E8F0;">{int(u['presion_sistolica'])} mmHg</td><td style="padding:9px;border:1px solid #E2E8F0;">115 mmHg</td><td style="padding:9px;border:1px solid #E2E8F0;">95 – 125 mmHg</td></tr>
                <tr><td style="padding:9px;border:1px solid #E2E8F0;">Horas de sueño (24h)</td><td style="padding:9px;border:1px solid #E2E8F0;">{float(u['horas_sueno']):.1f} h</td><td style="padding:9px;border:1px solid #E2E8F0;">7.8 h</td><td style="padding:9px;border:1px solid #E2E8F0;">6.0 – 8.5 h</td></tr>
                <tr><td style="padding:9px;border:1px solid #E2E8F0;">Densidad ósea (T-Score)</td><td style="padding:9px;border:1px solid #E2E8F0;">{float(u['densidad_osea']):.2f}</td><td style="padding:9px;border:1px solid #E2E8F0;">+0.25</td><td style="padding:9px;border:1px solid #E2E8F0;">&gt;= -1.0</td></tr>
                <tr><td style="padding:9px;border:1px solid #E2E8F0;">Radiación acumulada</td><td style="padding:9px;border:1px solid #E2E8F0;">{float(u['radiacion_acumulada']):.2f} mSv</td><td style="padding:9px;border:1px solid #E2E8F0;">112.5 mSv</td><td style="padding:9px;border:1px solid #E2E8F0;">&lt; 150 mSv</td></tr>
            </tbody>
        </table>
        <h4>2. Estudios de Visión Computacional (IA Preentrenada) y Correlación Causal</h4>
        <table style="width:100%;border-collapse:collapse;font-size:.88rem;">
            <thead>
                <tr style="background:#F4F6F9;text-align:left;">
                    <th style="padding:8px;border:1px solid #E2E8F0;">Fecha</th>
                    <th style="padding:8px;border:1px solid #E2E8F0;">Modalidad</th>
                    <th style="padding:8px;border:1px solid #E2E8F0;">Dictamen IA</th>
                    <th style="padding:8px;border:1px solid #E2E8F0;">Correlación Clínica</th>
                </tr>
            </thead>
            <tbody>{filas_img_html}</tbody>
        </table>
    </div>
    """
    st.markdown(reporte_html, unsafe_allow_html=True)
    st.write("")
    st.download_button(
        "Descargar Expediente Imprimible (.html / PDF)",
        data=f"<html><head><meta charset='utf-8'><title>Reporte {id_sel}</title></head><body style='font-family:sans-serif;padding:2rem;'>{reporte_html}</body></html>",
        file_name=f"Reporte_{id_sel}_Dia_{int(u['dia_mision'])}.html",
        mime="text/html"
    )

# --- PÁGINA 8: ACERCA DEL SISTEMA ---
def page_acerca():
    render_header("Acerca del Sistema", "Fundamentación científica, arquitectura de IA (Vision Transformers + Random Forest) y estándares NASA.")
    c1, c2 = st.columns(2)
    with c1:
        with st.container(border=True):
            st.markdown("#### 1. Autonomía Médica Multimodal en Espacio Profundo")
            st.write(
                "En misiones hacia Marte, el retardo de comunicación de hasta 22 minutos impide depender "
                "de diagnósticos desde la Tierra. AstroVital combina telemetría fisiológica diaria con "
                "modelos preentrenados (Transfer Learning) para diagnosticar causas clínicas offline a bordo."
            )
            st.markdown("#### 2. Modelos de Visión (Hugging Face - Transfer Learning)")
            st.markdown(
                "- **ViT Densitometría Ósea (`luke14/vit-xray-bone-fracture`):** Detecta pérdida trabecular y microfracturas.\n"
                "- **ViT Tomografía Axial (`Falah/brain_ct_scan`):** Detección de desplazamiento cefálico de fluidos (SANS) y lesiones.\n"
                "- **ViT Biometría de Ánimo (`dima806/facial_emotions_image_detection`):** Evaluación de fatiga circadiana y estrés agudo."
            )
    with c2:
        with st.container(border=True):
            st.markdown("#### 3. Estándares Clínicos y Parámetro de Control Sano")
            st.markdown(
                "- **Parámetro de Control Sano (NASA):** Pulso 68 bpm, Presión 115 mmHg, Sueño 7.8 h, T-Score +0.25 y Radiación basal controlada.\n"
                "- **Límites Radiológicos (NASA / CSA):** Nominal (< 150 mSv), Preventivo (150 – 300 mSv), Crítico (> 300 mSv).\n"
                "- **Correlación Temporal Fechada:** Cruce automático por fecha (`YYYY-MM-DD`) entre hallazgos de imagen y signos vitales."
            )

# ==========================================
# 8. NAVEGACIÓN MULTIPÁGINA
# ==========================================
render_sidebar()

paginas = [
    st.Page(page_resumen, title="1. Resumen de Misión", url_path="resumen"),
    st.Page(page_ingesta, title="2. Ingesta Biométrica", url_path="ingesta"),
    st.Page(page_ficha, title="3. Evolución Temporal", url_path="ficha"),
    st.Page(page_comparador, title="4. Comparador y Control Sano", url_path="comparador"),
    st.Page(page_vision, title="5. Visión Computacional (IA)", url_path="vision"),
    st.Page(page_alertas, title="6. Centro de Alertas", url_path="alertas"),
    st.Page(page_reporte, title="7. Reporte Clínico (PDF)", url_path="reporte"),
    st.Page(page_acerca, title="8. Acerca del Sistema", url_path="acerca"),
]

nav = st.navigation(paginas)
nav.run()