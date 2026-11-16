import pandas as pd
import sqlite3

def poblar_base_datos():
    print("Leyendo el dataset histórico...")
    try:
        df = pd.read_csv("data/dataset_historico.csv")
    except FileNotFoundError:
        print("Error: No se encontró el dataset en data/dataset_historico.csv")
        return

    # Filtrar solo hasta el día 150 para la demostración
    df = df[df["dia_mision"] <= 150]

    # Renombrar las columnas del CSV para que coincidan exactamente con la tabla SQL
    df = df.rename(columns={
        "frecuencia_cardiaca_bpm": "frecuencia_cardiaca",
        "presion_sistolica_mmhg": "presion_sistolica",
        "densidad_osea_tscore": "densidad_osea",
        "dosis_radiacion_diaria_msv": "radiacion_diaria",
        "radiacion_acumulada_msv": "radiacion_acumulada"
    })

    print(f"Insertando {len(df)} registros en la base de datos SQLite...")
    
    conexion = sqlite3.connect("data/astrovital.db")
    
    # Inyectar los datos en la tabla 'historial_medico'
    df.to_sql("historial_medico", conexion, if_exists="append", index=False)
    
    conexion.commit()
    conexion.close()
    print("✅ Historial de 150 días anexado correctamente para todos los astronautas.")

if __name__ == "__main__":
    poblar_base_datos()
