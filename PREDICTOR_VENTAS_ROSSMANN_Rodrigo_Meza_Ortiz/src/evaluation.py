import pandas as pd
import numpy as np
import joblib

# Cargar los datos de validación
X_val = pd.read_csv('C:/Users/rodri/OneDrive/Escritorio/PREDICTOR_VENTAS_ROSSMANN/data/test/X_val.csv')
y_val = pd.read_csv('C:/Users/rodri/OneDrive/Escritorio/PREDICTOR_VENTAS_ROSSMANN/data/test/y_val.csv')

# Cargar los modelos entrenados
model = joblib.load('C:/Users/rodri/OneDrive/Escritorio/PREDICTOR_VENTAS_ROSSMANN/models/final_model.pkl')

# Evaluar el modelo
predictions = model.predict(X_val)
rmspe = np.sqrt(np.mean(((y_val - predictions) / y_val) ** 2))
print(f'Random Forest RMSPE: {rmspe:.4f}')

# Guardar las métricas de evaluación en un archivo
with open('C:/Users/rodri/OneDrive/Escritorio/PREDICTOR_VENTAS_ROSSMANN/models/model_evaluation.txt', 'w') as file:
    file.write(f'Random Forest RMSPE: {rmspe:.4f}\n') 
