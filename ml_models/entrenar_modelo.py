import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import os

def entrenar_estimador():
    print("Cargando dataset histórico...")
    try:
        df = pd.read_csv("data/dataset_historico.csv")
    except FileNotFoundError:
        print("Error: No se encontró el dataset. Ejecuta generar_dataset.py primero.")
        return

    # Definir características de entrada (X) y la etiqueta a predecir (y)
    caracteristicas = [
        "dia_mision", 
        "frecuencia_cardiaca_bpm", 
        "presion_sistolica_mmhg", 
        "horas_sueno", 
        "densidad_osea_tscore", 
        "dosis_radiacion_diaria_msv", 
        "radiacion_acumulada_msv"
    ]
    X = df[caracteristicas]
    y = df["nivel_riesgo"]

    # Dividir el dataset: 80% para entrenar, 20% para probar
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Entrenando modelo de Inteligencia Artificial (Estimador de Riesgo)...")
    # class_weight='balanced' ayuda a que el modelo preste atención a los casos críticos (2) que son menos comunes
    modelo = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    modelo.fit(X_train, y_train)

    # Evaluar la precisión del modelo
    y_pred = modelo.predict(X_test)
    print("\n--- Precisión del Modelo ---")
    print(classification_report(y_test, y_pred))

    # Exportar el modelo preentrenado
    os.makedirs("ml_models", exist_ok=True)
    joblib.dump(modelo, "ml_models/estimador_riesgo.pkl")
    # Guardamos también las columnas para asegurar que el frontend envíe los datos en el mismo orden
    joblib.dump(caracteristicas, "ml_models/columnas.pkl")
    
    print("✅ Modelo de IA guardado exitosamente en ml_models/estimador_riesgo.pkl")

if __name__ == "__main__":
    entrenar_estimador()
