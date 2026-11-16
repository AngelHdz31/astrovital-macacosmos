import os
import sqlite3
import datetime
import joblib
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

np.random.seed(42)

# ==========================================
# 1. ARQUITECTURA CONVOLUCIONAL 2D (Conv2D -> ReLU -> MaxPool2D)
# ==========================================
KERNELS_CONV2D = [
    np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32),      # Gradiente horizontal (trabéculas/ojos)
    np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32),      # Gradiente vertical (cortical/simetría)
    np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.float32),     # Laplaciano (microfracturas/lesiones)
    np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=np.float32) / 9.0,   # Densidad media local
    np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]], dtype=np.float32)       # Textura diagonal
]

def extraer_mapa_y_vector_cnn(img_gray_64):
    """Aplica banco de filtros Convolucionales 2D + activación ReLU + MaxPooling 2D."""
    arr = np.asarray(img_gray_64, dtype=np.float32) / 255.0
    H, W = arr.shape
    patches = np.stack([
        arr[i:H - 2 + i, j:W - 2 + j]
        for i in range(3) for j in range(3)
    ], axis=-1)  # (62, 62, 9)

    mapas_pool = []
    mapa_energia = np.zeros((62, 62), dtype=np.float32)

    for k in KERNELS_CONV2D:
        conv = np.tensordot(patches, k.flatten(), axes=([2], [0]))
        relu = np.maximum(0.0, conv)
        mapa_energia += relu
        # MaxPooling 2D (bloques 4x4 sobre recorte 60x60 -> 15x15 por filtro)
        pooled = relu[:60, :60].reshape(15, 4, 15, 4).max(axis=(1, 3))
        mapas_pool.append(pooled.flatten())

    Estadisticas_globales = np.array([
        arr.mean(), arr.std(),
        arr[16:48, 16:48].mean(), arr[16:48, 16:48].std()
    ], dtype=np.float32)

    vector_caracteristicas = np.concatenate(mapas_pool + [Estadisticas_globales])
    return vector_caracteristicas, mapa_energia

# ==========================================
# 2. GENERADORES SINTÉTICOS DE LAS 3 MODALIDADES (64x64)
# ==========================================
def sintetizar_hueso(clase):
    """0: Nominal, 1: Osteopenia Trabecular, 2: Microfractura / Osteoporosis"""
    img = Image.new("L", (64, 64), int(np.random.uniform(18, 30)))
    draw = ImageDraw.Draw(img)
    # Diáfisis ósea
    densidad_base = 210 if clase == 0 else (155 if clase == 1 else 115)
    draw.rectangle([18, 4, 46, 60], fill=int(densidad_base + np.random.uniform(-10, 10)))
    # Paredes corticales
    grosor = 5 if clase == 0 else (3 if clase == 1 else 2)
    draw.rectangle([18, 4, 18 + grosor, 60], fill=240)
    draw.rectangle([46 - grosor, 4, 46, 60], fill=240)
    # Malla trabecular interna
    n_lineas = 18 if clase == 0 else (10 if clase == 1 else 5)
    for _ in range(n_lineas):
        y1 = np.random.randint(8, 56)
        y2 = y1 + np.random.randint(-6, 7)
        draw.line([(20, y1), (44, y2)], fill=int(np.random.uniform(170, 225)), width=1)
    # Si es clase 2, insertar fisura/reabsorción radiolúcida
    if clase == 2:
        cy = np.random.randint(22, 42)
        draw.ellipse([22, cy - 6, 42, cy + 6], fill=int(np.random.uniform(35, 55)))
        draw.line([(18, cy), (45, cy + 3)], fill=25, width=2)
    arr = np.clip(np.asarray(img, dtype=np.float32) + np.random.normal(0, 6, (64, 64)), 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="L")

def sintetizar_tomografia(clase):
    """0: Normal, 1: Edema / Desplazamiento de Fluidos (SANS), 2: Lesión Hiperdensa por Radiación"""
    img = Image.new("L", (64, 64), int(np.random.uniform(10, 20)))
    draw = ImageDraw.Draw(img)
    # Cráneo y parénquima cerebral
    draw.ellipse([8, 6, 56, 58], fill=235)
    draw.ellipse([11, 9, 53, 55], fill=int(np.random.uniform(105, 120)))
    # Ventrículos laterales
    if clase == 0:
        draw.ellipse([25, 24, 31, 38], fill=45)
        draw.ellipse([33, 24, 39, 38], fill=45)
    elif clase == 1:
        # Dilatación ventricular y halo de edema por microgravedad
        draw.ellipse([21, 20, 31, 42], fill=165)
        draw.ellipse([33, 20, 43, 42], fill=165)
        draw.ellipse([16, 15, 48, 49], outline=175, width=2)
    elif clase == 2:
        # Focos hiperdensos asimétricos por radiación
        draw.ellipse([25, 24, 31, 38], fill=50)
        draw.ellipse([33, 24, 39, 38], fill=50)
        rx, ry = np.random.randint(18, 40), np.random.randint(18, 40)
        draw.ellipse([rx - 6, ry - 6, rx + 6, ry + 6], fill=248)
        draw.ellipse([rx - 10, ry - 10, rx + 10, ry + 10], outline=200, width=1)
    arr = np.clip(np.asarray(img, dtype=np.float32) + np.random.normal(0, 5, (64, 64)), 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="L")

def sintetizar_animo(clase):
    """0: Ánimo Nominal / Alerta, 1: Fatiga Circadiana, 2: Agotamiento Severo / Estrés Agudo"""
    img = Image.new("L", (64, 64), int(np.random.uniform(25, 40)))
    draw = ImageDraw.Draw(img)
    # Contorno facial biométrico
    tono_base = 175 if clase == 0 else (145 if clase == 1 else 115)
    draw.ellipse([12, 8, 52, 56], fill=int(tono_base + np.random.uniform(-8, 8)), outline=210, width=1)
    if clase == 0:
        # Ojos abiertos y comisura labial neutra/positiva
        draw.ellipse([20, 22, 28, 28], fill=245, outline=40)
        draw.ellipse([36, 22, 44, 28], fill=245, outline=40)
        draw.arc([22, 36, 42, 48], start=15, end=165, fill=235, width=2)
    elif clase == 1:
        # Ptosis palpebral (ojos semi-cerrados por falta de sueño) y ojeras oscuras
        draw.rectangle([19, 26, 29, 31], fill=70)
        draw.rectangle([35, 26, 45, 31], fill=70)
        draw.line([(20, 25), (28, 25)], fill=220, width=2)
        draw.line([(36, 25), (44, 25)], fill=220, width=2)
        draw.line([(24, 43), (40, 43)], fill=190, width=2)
    elif clase == 2:
        # Tensión frontal severa, ojos cerrados/hundidos y comisura caída
        draw.line([(18, 19), (29, 23)], fill=30, width=2)
        draw.line([(46, 19), (35, 23)], fill=30, width=2)
        draw.ellipse([19, 23, 29, 31], fill=40)
        draw.ellipse([35, 23, 45, 31], fill=40)
        draw.arc([22, 41, 42, 52], start=195, end=345, fill=30, width=2)
    arr = np.clip(np.asarray(img, dtype=np.float32) + np.random.normal(0, 5, (64, 64)), 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="L")

# ==========================================
# 3. ENTRENAMIENTO DE LOS 3 MODELOS CNN
# ==========================================
def entrenar_modelos_cnn():
    os.makedirs("ml_models", exist_ok=True)
    os.makedirs("data/muestras_vision", exist_ok=True)

    config_modalidades = {
        "huesos": {
            "fn": sintetizar_hueso,
            "clases": {0: "Densidad Ósea Nominal", 1: "Osteopenia Trabecular", 2: "Microfractura / Osteoporosis"}
        },
        "tomografia": {
            "fn": sintetizar_tomografia,
            "clases": {0: "Tomografía Axial Normal", 1: "Desplazamiento de Fluidos (SANS)", 2: "Lesión Hiperdensa por Radiación"}
        },
        "animo": {
            "fn": sintetizar_animo,
            "clases": {0: "Estado Anímico Nominal", 1: "Fatiga Circadiana Moderada", 2: "Agotamiento Severo / Estrés"}
        }
    }

    modelos_entrenados = {}

    for clave, cfg in config_modalidades.items():
        X, y = [], []
        for clase_id in [0, 1, 2]:
            for _ in range(180):
                img = cfg["fn"](clase_id)
                vec, _ = extraer_mapa_y_vector_cnn(img)
                X.append(vec)
                y.append(clase_id)

        X = np.array(X)
        y = np.array(y)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        clf = MLPClassifier(hidden_layer_sizes=(128, 64), activation="relu", max_iter=300, random_state=42)
        clf.fit(X_tr, y_tr)
        acc = accuracy_score(y_te, clf.predict(X_te))
        print(f"✅ CNN [{clave.upper()}] entrenada · Precisión en validación: {acc * 100:.1f}%")

        modelos_entrenados[clave] = {
            "modelo": clf,
            "clases": cfg["clases"],
            "precision": round(acc * 100, 1)
        }

    joblib.dump(modelos_entrenados, "ml_models/cnn_vision_models.pkl")
    return modelos_entrenados

# ==========================================
# 4. POBLAR ESTUDIOS FECHADOS Y CORRELACIONADOS EN SQLITE
# ==========================================
def inicializar_tabla_estudios(modelos):
    conn = sqlite3.connect("data/astrovital.db")
    cursor = conn.cursor()

    cursor.execute("DROP TABLE IF EXISTS estudios_imagen")
    cursor.execute('''
    CREATE TABLE estudios_imagen (
        id_estudio INTEGER PRIMARY KEY AUTOINCREMENT,
        id_astronauta TEXT,
        dia_mision INTEGER,
        fecha_estudio TEXT,
        modalidad TEXT,
        prediccion_cnn TEXT,
        confianza REAL,
        nivel_severidad INTEGER,
        correlacion_clinica TEXT,
        ruta_imagen TEXT,
        FOREIGN KEY (id_astronauta) REFERENCES astronautas(id_astronauta)
    )
    ''')

    fecha_base = datetime.date(2026, 6, 1)

    # Casos clínicos fechados para los 4 tripulantes
    casos_iniciales = [
        # ASTRO_01 (Elena Rosas - Sufre tormenta solar y crisis hipertensiva en días 142-150)
        ("ASTRO_01", 60, "Densitometría Ósea (Rayos X)", "huesos", 0),
        ("ASTRO_01", 142, "Tomografía Axial (TAC)", "tomografia", 1),
        ("ASTRO_01", 150, "Tomografía Axial (TAC)", "tomografia", 2),
        ("ASTRO_01", 150, "Biometría de Estado de Ánimo", "animo", 2),
        # ASTRO_02 (Marcus Vance - Sano control)
        ("ASTRO_02", 90, "Densitometría Ósea (Rayos X)", "huesos", 0),
        ("ASTRO_02", 150, "Tomografía Axial (TAC)", "tomografia", 0),
        ("ASTRO_02", 150, "Biometría de Estado de Ánimo", "animo", 0),
        # ASTRO_03 (Sarah Chen - Sufre degradación ósea progresiva)
        ("ASTRO_03", 105, "Densitometría Ósea (Rayos X)", "huesos", 1),
        ("ASTRO_03", 150, "Densitometría Ósea (Rayos X)", "huesos", 2),
        ("ASTRO_03", 150, "Biometría de Estado de Ánimo", "animo", 0),
        # ASTRO_04 (David Morales - Sufre privación de sueño / fatiga circadiana)
        ("ASTRO_04", 110, "Biometría de Estado de Ánimo", "animo", 1),
        ("ASTRO_04", 150, "Biometría de Estado de Ánimo", "animo", 2),
    ]

    generadores = {
        "huesos": sintetizar_hueso,
        "tomografia": sintetizar_tomografia,
        "animo": sintetizar_animo
    }

    for idx, (id_astro, dia, nombre_mod, clave_mod, clase_real) in enumerate(casos_iniciales, start=1):
        fecha_str = (fecha_base + datetime.timedelta(days=dia)).strftime("%Y-%m-%d")
        img = generadores[clave_mod](clase_real)
        ruta_archivo = f"data/muestras_vision/{id_astro}_dia{dia}_{clave_mod}.png"
        img.resize((240, 240), Image.Resampling.NEAREST).save(ruta_archivo)

        vec, _ = extraer_mapa_y_vector_cnn(img)
        clf = modelos[clave_mod]["modelo"]
        probs = clf.predict_proba([vec])[0]
        pred_id = int(np.argmax(probs))
        confianza = round(float(probs[pred_id]) * 100, 1)
        etiqueta_pred = modelos[clave_mod]["clases"][pred_id]

        # Consultar signos vitales de ese mismo día para construir la correlación causal
        df_bio = pd.read_sql(
            "SELECT * FROM historial_medico WHERE id_astronauta = ? AND dia_mision = ? LIMIT 1",
            conn, params=(id_astro, dia)
        )
        if not df_bio.empty:
            b = df_bio.iloc[0]
            if clave_mod == "huesos" and pred_id > 0:
                corr = f"Causado por desmineralización en microgravedad: T-Score cayó a {b['densidad_osea']} en el Día {dia}."
            elif clave_mod == "tomografia" and pred_id > 0:
                corr = f"Secundario a pico radiológico ({b['radiacion_acumulada']} mSv acum.) e hipertensión sistólica ({b['presion_sistolica']} mmHg)."
            elif clave_mod == "animo" and pred_id > 0:
                corr = f"Inducido por déficit de sueño ({b['horas_sueno']} h/día) o estrés autonómico ({b['frecuencia_cardiaca']} bpm)."
            else:
                corr = f"Sin anomalías asociadas. Signos vitales del {fecha_str} en rango fisiológico nominal."
        else:
            corr = "Estudio de referencia basal."

        cursor.execute('''
            INSERT INTO estudios_imagen
            (id_astronauta, dia_mision, fecha_estudio, modalidad, prediccion_cnn,
             confianza, nivel_severidad, correlacion_clinica, ruta_imagen)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (id_astro, dia, fecha_str, nombre_mod, etiqueta_pred, confianza, pred_id, corr, ruta_archivo))

    conn.commit()
    conn.close()
    print("✅ Tabla 'estudios_imagen' creada con muestras fechadas y correlacionadas con signos vitales.")

if __name__ == "__main__":
    modelos = entrenar_modelos_cnn()
    inicializar_tabla_estudios(modelos)
