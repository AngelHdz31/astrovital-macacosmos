import sqlite3
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

np.random.seed(42)

def etiquetar_riesgo(fc, ps, sueno, osea, rad_dia, rad_acum):
    """Regla clínica alineada con los umbrales de la interfaz."""
    if (rad_acum > 300 or rad_dia > 20 or osea < -2.5 or 
        sueno < 4.5 or fc > 100 or fc < 45 or ps > 140 or ps < 85):
        return 2
    if (rad_acum > 150 or rad_dia > 2.5 or osea < -1.0 or 
        sueno < 5.0 or fc > 85 or fc < 52 or ps > 125 or ps < 90):
        return 1
    return 0

def generar_datos_cohorte():
    perfiles = [
        ("ASTRO_01", "Cmdr. Elena Rosas", 38, "Comandante · Crisis Radiológica"),
        ("ASTRO_02", "Piloto Marcus Vance", 42, "Piloto · Salud Nominal"),
        ("ASTRO_03", "Esp. Sarah Chen", 31, "Especialista · Osteopenia"),
        ("ASTRO_04", "Ing. David Morales", 35, "Ingeniero · Fatiga Circadiana")
    ]

    registros = []
    dias = 150

    for id_astro, _, _, _ in perfiles:
        rad_acum = 0.0
        for d in range(1, dias + 1):
            progreso = d / dias  # 0.0 a 1.0

            if id_astro == "ASTRO_02":
                # 100% NOMINAL (Todo en rango verde siempre)
                rad_dia = round(np.random.uniform(0.72, 0.85), 2)
                rad_acum = round(rad_acum + rad_dia, 2)
                osea = round(0.40 - (0.65 * progreso) + np.random.normal(0, 0.015), 2) # Termina en -0.25
                sueno = round(np.clip(np.random.normal(7.5, 0.25), 6.8, 8.2), 1)
                fc = int(np.clip(np.random.normal(68, 2.0), 63, 74))
                ps = int(np.clip(np.random.normal(114, 2.5), 108, 120))

            elif id_astro == "ASTRO_01":
                # CRÍTICO AL FINAL: Normal hasta día 140, Tormenta Solar (SPE) en días 141-150
                if d <= 140:
                    rad_dia = round(np.random.uniform(0.75, 0.90), 2)
                    fc = int(np.clip(np.random.normal(71, 2.0), 65, 77))
                    ps = int(np.clip(np.random.normal(116, 2.5), 110, 122))
                elif d <= 145:
                    rad_dia = round(np.random.uniform(14.0, 18.0), 2)
                    fc = int(np.clip(np.random.normal(89, 2.0), 86, 94))
                    ps = int(np.clip(np.random.normal(130, 2.0), 126, 135))
                else:
                    rad_dia = round(np.random.uniform(28.0, 33.0), 2)
                    fc = int(np.clip(np.random.normal(106, 2.0), 102, 112))
                    ps = int(np.clip(np.random.normal(143, 2.0), 141, 148))

                rad_acum = round(rad_acum + rad_dia, 2)
                osea = round(0.30 - (0.70 * progreso) + np.random.normal(0, 0.015), 2) # Termina en -0.40 (Normal)
                sueno = round(np.clip(np.random.normal(7.1, 0.25), 6.4, 7.8), 1)

            elif id_astro == "ASTRO_03":
                # PREVENTIVO ÓSEO: Degradación ósea progresiva hasta -1.68 (Osteopenia), resto normal
                rad_dia = round(np.random.uniform(0.75, 0.88), 2)
                rad_acum = round(rad_acum + rad_dia, 2)
                osea = round(0.35 - (2.03 * progreso) + np.random.normal(0, 0.02), 2) # Cruza -1.0 en día ~100, termina en -1.68
                sueno = round(np.clip(np.random.normal(7.3, 0.25), 6.6, 8.0), 1)
                fc = int(np.clip(np.random.normal(72, 2.0), 66, 78))
                ps = int(np.clip(np.random.normal(112, 2.5), 106, 118))

            elif id_astro == "ASTRO_04":
                # PREVENTIVO CONDUCTUAL: Privación progresiva de sueño hasta 4.7h, resto normal
                rad_dia = round(np.random.uniform(0.70, 0.82), 2)
                rad_acum = round(rad_acum + rad_dia, 2)
                osea = round(0.25 - (0.65 * progreso) + np.random.normal(0, 0.015), 2) # Termina en -0.40 (Normal)
                base_sueno = 7.4 - (2.7 * (progreso ** 1.3)) # Baja gradualmente hasta 4.7h
                sueno = round(np.clip(np.random.normal(base_sueno, 0.15), 4.6, 8.0), 1)
                fc = int(np.clip(np.random.normal(74 + (6 * progreso), 2.0), 68, 83))
                ps = int(np.clip(np.random.normal(117 + (4 * progreso), 2.0), 110, 123))

            riesgo = etiquetar_riesgo(fc, ps, sueno, osea, rad_dia, rad_acum)

            registros.append({
                "id_astronauta": id_astro,
                "dia_mision": d,
                "frecuencia_cardiaca": fc,
                "presion_sistolica": ps,
                "horas_sueno": sueno,
                "densidad_osea": osea,
                "radiacion_diaria": rad_dia,
                "radiacion_acumulada": rad_acum,
                "nivel_riesgo": riesgo
            })

    return perfiles, pd.DataFrame(registros)

def reconstruir_todo():
    os.makedirs("data", exist_ok=True)
    os.makedirs("ml_models", exist_ok=True)

    perfiles, df_cohorte = generar_datos_cohorte()

    # 1. Reconstruir Base de Datos SQLite limpia
    conn = sqlite3.connect("data/astrovital.db")
    cursor = conn.cursor()
    cursor.execute("DROP TABLE IF EXISTS historial_medico")
    cursor.execute("DROP TABLE IF EXISTS astronautas")

    cursor.execute('''
    CREATE TABLE astronautas (
        id_astronauta TEXT PRIMARY KEY,
        nombre TEXT,
        edad INTEGER,
        rol TEXT
    )
    ''')

    cursor.execute('''
    CREATE TABLE historial_medico (
        id_registro INTEGER PRIMARY KEY AUTOINCREMENT,
        id_astronauta TEXT,
        dia_mision INTEGER,
        frecuencia_cardiaca INTEGER,
        presion_sistolica INTEGER,
        horas_sueno REAL,
        densidad_osea REAL,
        radiacion_diaria REAL,
        radiacion_acumulada REAL,
        nivel_riesgo INTEGER,
        fecha_registro TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (id_astronauta) REFERENCES astronautas(id_astronauta)
    )
    ''')

    cursor.executemany("INSERT INTO astronautas VALUES (?, ?, ?, ?)", perfiles)
    df_cohorte.to_sql("historial_medico", conn, if_exists="append", index=False)
    conn.commit()
    conn.close()
    print("✅ Base de datos 'astrovital.db' reconstruida con 4 perfiles clínicos distintos.")

    # 2. Guardar CSV y entrenar el modelo con cobertura completa de casos
    df_csv = df_cohorte.rename(columns={
        "frecuencia_cardiaca": "frecuencia_cardiaca_bpm",
        "presion_sistolica": "presion_sistolica_mmhg",
        "densidad_osea": "densidad_osea_tscore",
        "radiacion_diaria": "dosis_radiacion_diaria_msv",
        "radiacion_acumulada": "radiacion_acumulada_msv"
    })
    df_csv.to_csv("data/dataset_historico.csv", index=False)

    # Añadir muestras sintéticas de refuerzo para que el Random Forest cubra todo el simulador
    extras = []
    for _ in range(600):
        dia_s = np.random.randint(1, 300)
        fc_s = np.random.randint(50, 130)
        ps_s = np.random.randint(90, 165)
        sueno_s = round(np.random.uniform(3.5, 9.0), 1)
        osea_s = round(np.random.uniform(-3.2, 0.6), 2)
        r_dia_s = round(np.random.uniform(0.5, 35.0), 2)
        r_acum_s = round(np.random.uniform(10.0, 420.0), 2)
        r_s = etiquetar_riesgo(fc_s, ps_s, sueno_s, osea_s, r_dia_s, r_acum_s)
        extras.append([dia_s, fc_s, ps_s, sueno_s, osea_s, r_dia_s, r_acum_s, r_s])

    columnas = [
        "dia_mision", "frecuencia_cardiaca_bpm", "presion_sistolica_mmhg",
        "horas_sueno", "densidad_osea_tscore", "dosis_radiacion_diaria_msv",
        "radiacion_acumulada_msv"
    ]
    df_extras = pd.DataFrame(extras, columns=columnas + ["nivel_riesgo"])
    df_train = pd.concat([df_csv[columnas + ["nivel_riesgo"]], df_extras], ignore_index=True)

    modelo = RandomForestClassifier(n_estimators=150, random_state=42, class_weight="balanced")
    modelo.fit(df_train[columnas], df_train["nivel_riesgo"])

    joblib.dump(modelo, "ml_models/estimador_riesgo.pkl")
    joblib.dump(columnas, "ml_models/columnas.pkl")
    print("✅ Modelo Random Forest reentrenado y sincronizado con los nuevos perfiles.")

if __name__ == "__main__":
    reconstruir_todo()
