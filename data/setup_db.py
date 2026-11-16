import sqlite3
import os

def inicializar_base_datos():
    os.makedirs("data", exist_ok=True)
    conexion = sqlite3.connect("data/astrovital.db")
    cursor = conexion.cursor()

    # Crear tabla de Astronautas
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS astronautas (
        id_astronauta TEXT PRIMARY KEY,
        nombre TEXT,
        edad INTEGER,
        rol TEXT
    )
    ''')

    # Crear tabla de Historial Médico
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS historial_medico (
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

    # Insertar perfiles base si la tabla está vacía
    cursor.execute("SELECT COUNT(*) FROM astronautas")
    if cursor.fetchone()[0] == 0:
        astronautas = [
            ("ASTRO_01", "Cmdr. Elena Rosas", 38, "Comandante"),
            ("ASTRO_02", "Piloto Marcus Vance", 42, "Piloto"),
            ("ASTRO_03", "Esp. Sarah Chen", 31, "Especialista de Misión")
        ]
        cursor.executemany("INSERT INTO astronautas VALUES (?, ?, ?, ?)", astronautas)
        conexion.commit()

    conexion.close()
    print("✅ Base de datos relacional 'astrovital.db' inicializada correctamente.")

if __name__ == "__main__":
    inicializar_base_datos()
