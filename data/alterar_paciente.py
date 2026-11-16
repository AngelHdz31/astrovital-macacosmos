import sqlite3

def inyectar_anomalia():
    conexion = sqlite3.connect("data/astrovital.db")
    cursor = conexion.cursor()
    
    print("🛰️ Simulando Tormenta Solar (SPE) sobre la Cmdr. Elena Rosas (ASTRO_01)...")
    
    # Valores: (radiacion_acumulada, radiacion_diaria, frecuencia_cardiaca, alerta_riesgo, ID, dia_mision)
    valores_alterados = [
        (160.0, 25.0, 85, 1, "ASTRO_01", 146),
        (220.0, 60.0, 95, 2, "ASTRO_01", 147),
        (290.0, 70.0, 110, 2, "ASTRO_01", 148),
        (370.0, 80.0, 125, 2, "ASTRO_01", 149),
        (450.0, 80.0, 140, 2, "ASTRO_01", 150) # Nivel crítico (Casi infarto / Enfermedad por radiación)
    ]
    
    query = """
    UPDATE historial_medico 
    SET radiacion_acumulada = ?, radiacion_diaria = ?, frecuencia_cardiaca = ?, nivel_riesgo = ?
    WHERE id_astronauta = ? AND dia_mision = ?
    """
    
    cursor.executemany(query, valores_alterados)
    conexion.commit()
    conexion.close()
    print("✅ Anomalía inyectada con éxito. ASTRO_01 ahora presenta un cuadro clínico crítico.")

if __name__ == "__main__":
    inyectar_anomalia()
