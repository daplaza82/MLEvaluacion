# predict_prices.py
# Script simplificado para cargar y usar el modelo

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import glob
from datetime import datetime, timedelta

# Asegurar que existen las carpetas necesarias
os.makedirs('reports/figures', exist_ok=True)

def load_most_recent_model(model_type):
    """Carga el modelo más reciente del tipo especificado"""
    # Buscar archivos que coincidan con el patrón
    pattern = f"models/{model_type}_simple_*.pkl"
    files = glob.glob(pattern)
    
    if not files:
        raise FileNotFoundError(f"No se encontró ningún modelo {model_type}")
    
    # Ordenar por fecha (asumiendo que la fecha está en el nombre del archivo)
    latest_model_path = sorted(files)[-1]
    
    # Cargar el modelo
    with open(latest_model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    print(f"Modelo cargado: {latest_model_path}")
    print(f"Características del modelo: {model_data['features']}")
    
    return model_data

def predict_future_prices(model_data, data, days_to_predict=30):
    """Predice precios futuros para los próximos N días"""
    # Extraer componentes del modelo guardado
    model = model_data['model']
    features = model_data['features']
    scaler = model_data['scaler']
    
    # Preparar datos - ordenar por fecha
    df = data.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    
    # Encontrar la última fecha con datos
    last_date = df['Date'].max()
    
    # Crear las características básicas para las fechas futuras
    future_dates = [last_date + timedelta(days=i) for i in range(1, days_to_predict + 1)]
    future_df = pd.DataFrame({'Date': future_dates})
    future_df['Year'] = future_df['Date'].dt.year
    future_df['Month'] = future_df['Date'].dt.month
    
    # Lista para almacenar las predicciones
    future_prices = []
    
    # Obtener los últimos precios conocidos para los rezagos iniciales
    last_prices = list(df['Price'].tail(5))
    
    # Predecir cada día futuro
    for i in range(days_to_predict):
        # Configurar valores para este día
        current_row = {}
        
        # Características de fecha
        current_row['Year'] = future_df.loc[i, 'Year']
        current_row['Month'] = future_df.loc[i, 'Month']
        
        # Rezagos de precios (usando históricos o predicciones anteriores)
        if i == 0:  # Primer día a predecir
            current_row['Price_Lag_2'] = last_prices[-2]
            current_row['Price_Lag_3'] = last_prices[-3]
            
            # Promedio móvil de 7 días
            window_prices = last_prices[-7:] if len(last_prices) >= 7 else last_prices
            current_row['Price_Rolling_7'] = sum(window_prices) / len(window_prices)
            
        elif i == 1:  # Segundo día a predecir
            current_row['Price_Lag_2'] = last_prices[-1]
            current_row['Price_Lag_3'] = last_prices[-2]
            
            # Promedio móvil de 7 días
            window_prices = last_prices[-6:] + [future_prices[0]]
            current_row['Price_Rolling_7'] = sum(window_prices) / len(window_prices)
            
        else:  # Tercer día en adelante
            current_row['Price_Lag_2'] = future_prices[i-2]
            current_row['Price_Lag_3'] = future_prices[i-3] if i >= 3 else last_prices[-3-(i-3)]
            
            # Promedio móvil de 7 días
            lookback = min(i, 6)  # Cuántos días atrás podemos ver en las predicciones
            remaining = 7 - lookback - 1  # Cuántos días históricos necesitamos
            
            window_prices = []
            if remaining > 0:
                window_prices.extend(last_prices[-remaining:])
            window_prices.extend(future_prices[max(0, i-lookback):i])
            
            current_row['Price_Rolling_7'] = sum(window_prices) / len(window_prices)
        
        # Preparar la entrada para el modelo
        X_pred = pd.DataFrame([current_row], columns=features)
        
        # Normalizar
        X_pred_scaled = scaler.transform(X_pred)
        
        # Predecir
        pred_price = model.predict(X_pred_scaled)[0]
        future_prices.append(pred_price)
    
    # Crear DataFrame de resultados
    result_df = pd.DataFrame({
        'Date': future_dates,
        'Predicted_Price': future_prices
    })
    
    return result_df

# Código principal
try:
    # Cargar datos
    print("Cargando datos...")
    data = pd.read_parquet("data/raw/oil_prices.parquet")
    print(f"Dataset cargado. Registros: {len(data)}")
    
    # Asegurarse de que las fechas estén en formato datetime
    if not pd.api.types.is_datetime64_any_dtype(data['Date']):
        print("Convirtiendo 'Date' a formato datetime...")
        data['Date'] = pd.to_datetime(data['Date'])
    
    # Ordenar por fecha
    data = data.sort_values('Date')
    
    # Cargar modelo (puedes cambiar a "random_forest" si prefieres)
    model_data = load_most_recent_model("linear_regression")
    
    # Predecir precios para los próximos 30 días
    print("\nGenerando predicciones...")
    predictions = predict_future_prices(model_data, data, days_to_predict=30)
    
    # Mostrar predicciones
    print("\nPredicciones para los próximos 30 días:")
    print(predictions.head(10))
    
    # Visualizar resultados
    plt.figure(figsize=(12, 6))
    
    # Datos históricos recientes (últimos 90 días)
    historical = data.sort_values('Date').tail(90).copy()
    
    # Asegurarse de que las fechas estén en formato datetime
    if not pd.api.types.is_datetime64_any_dtype(historical['Date']):
        historical['Date'] = pd.to_datetime(historical['Date'])
    
    # Convertir predicciones a formato datetime si es necesario
    if not pd.api.types.is_datetime64_any_dtype(predictions['Date']):
        predictions['Date'] = pd.to_datetime(predictions['Date'])
    
    # Graficar directamente (matplotlib maneja pandas datetime)
    plt.plot(historical['Date'], historical['Price'], 'b-', label='Precios Históricos')
    plt.plot(predictions['Date'], predictions['Predicted_Price'], 'r--', label='Predicciones')
    
    plt.title('Predicción de Precios del Petróleo')
    plt.xlabel('Fecha')
    plt.ylabel('Precio ($)')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Guardar y mostrar
    plt.savefig('reports/figures/future_predictions.png', dpi=300)
    plt.close()
    
    print("\nGráfico de predicciones guardado en 'reports/figures/future_predictions.png'")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()