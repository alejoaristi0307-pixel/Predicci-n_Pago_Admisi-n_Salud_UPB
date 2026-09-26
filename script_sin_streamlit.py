import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# 1. Cargamos el modelo entrenado
filename = 'modelo-cla.pkl'
modelo_pipeline = pickle.load(open(filename, 'rb'))
print("Modelo cargado exitosamente:", modelo_pipeline)

# 2. Cargamos los datos futuros (o el dataset completo)
df = pd.read_excel('Admisiones_Posgrado_salud.xlsx', sheet_name='Datos')
print(f"Datos cargados. Total de registros: {len(df)}")

# 3. Seleccionamos una muestra o los datos a predecir
# Definimos las características esperadas por el modelo
features = [
    'Periodo', 'Programa Admision', 'Presentan examen', 'Genero', 
    'Edad', 'Pais Nacimiento', 'Tipo Documento', 
    'Departamento Nacimiento', 'Ciudad Residencia', 
    'Naturaleza Origen Universidad'
]

X_nuevos = df[features].copy()

# 4. Generamos las predicciones
# Nota: Si tu modelo es un pipeline completo, puedes hacer directo: predicciones = modelo_pipeline.predict(X_nuevos)
print("Características listas para la predicción.")
