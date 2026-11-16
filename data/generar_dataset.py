import pandas as pd
import numpy as np
import datetime
import random

def generar_dataset_realista():
    astronautas = [
        {"id": "ASTRO_01", "nombre": "Cmdr. Elena Rosas", "edad": 38, "sexo": "F"},
        {"id": "ASTRO_02", "nombre": "Piloto Marcus Vance", "edad": 42, "sexo": "M"},
        {"id": "ASTRO_03", "nombre": "Esp. Sarah Chen", "edad": 31, "sexo": "F"}
    ]
    
    dias_mision = 180 
    fecha_inicio = datetime.date(2026, 6, 1)
    registros = []

    for astro in astronautas:
        densidad_osea_t_score = 0.5  
        radiacion_acumulada = 0.0    
        
        for dia in range(1, dias_mision + 1):
            fecha_actual = fecha_inicio + datetime.timedelta(days=dia)
            
            if random.random() < 0.02:
                dosis_diaria = np.random.uniform(15.0, 35.0) 
            else:
                dosis_diaria = np.random.uniform(1.2, 2.2) 
            
            radiacion_acumulada += dosis_diaria
            densidad_osea_t_score -= np.random.uniform(0.001, 0.003)
            horas_sueno = np.clip(np.random.normal(7.5 - (dia * 0.008), 1.2), 3.5, 9.0)
            
            estres = 1.0 if horas_sueno < 5.0 else 0.0
            frecuencia_cardiaca = int(np.random.normal(68 + (dia * 0.05) + (estres * 15), 5))
            presion_sistolica = int(np.random.normal(118 + (estres * 10), 6))
            
            if radiacion_acumulada > 300 or densidad_osea_t_score < -2.5 or dosis_diaria > 20:
                riesgo = 2 
            elif radiacion_acumulada > 150 or densidad_osea_t_score < -1.0 or horas_sueno < 4.5:
                riesgo = 1 
            else:
                riesgo = 0 

            registros.append({
                "id_astronauta": astro["id"],
                "dia_mision": dia,
                "frecuencia_cardiaca_bpm": frecuencia_cardiaca,
                "presion_sistolica_mmhg": presion_sistolica,
                "horas_sueno": round(horas_sueno, 1),
                "densidad_osea_tscore": round(densidad_osea_t_score, 2),
                "dosis_radiacion_diaria_msv": round(dosis_diaria, 2),
                "radiacion_acumulada_msv": round(radiacion_acumulada, 2),
                "nivel_riesgo": riesgo
            })

    df = pd.DataFrame(registros)
    df.to_csv("data/dataset_historico.csv", index=False)
    print(f"Dataset generado: data/dataset_historico.csv ({len(df)} registros)")

if __name__ == "__main__":
    generar_dataset_realista()
