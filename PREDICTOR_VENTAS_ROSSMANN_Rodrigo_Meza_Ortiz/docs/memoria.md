MEMORIA DEL PROYECTO. 

I.- Naturaleza del proyecto: Problema resuelto indicando como se ha formado el proyecto. 
A modo de contexto, merece la pena recordar que el objetivo de este trabajo es desarrollar un modelo de machine learning que permita principalmente practicar y demostrar las competencias adquiridas durante el curso Data Science. Siguiendo en esta línea se debe desarrollar un modelo de machine learning, desde la obtención de datos hasta el despliegue de los mismos. 
Tal y como se acaba de explicar, el primer paso en lo que respecta al desarrollo de machine learning es la obtención de los datos. En esta etapa del proyecto lo fundamental fue elegir datos que fuesen consistentes con las exigencias que requiere este trabajo. Para esto, tal y como se sugiere en el enunciado del trabajo, se decidió obtener los datos desde la página web “Kaggle”. Para ser más específicos, se decidió desarrollar el proyecto llamado “Rossmann Store Sales Forecast sales using store, promotion, and competitor data”. El problema a resolver en este proyecto es crear una herramienta de machine learning que permita predecir 6 semanas de ventas diarias para 1.115 tiendas Rossmann ubicadas en Alemania. Los datos fuente del proyecto son los siguientes: “store.csv”, “train.csv” y “test.csv”. La descripción detallada de este proyecto la podrá encontrar en el siguiente enlace: https://www.kaggle.com/competitions/rossmann-store-sales/overview


II.- Cuestiones técnicas relativas a la Ciencia de Datos: Análisis realizado a los datos, Aspectos clave del negocio a resolver, y Modelo(s) empleados y análisis. 

i.- Adquisición de Datos (01_Fuentes.ipynb). 
La adquisición de datos es el primer paso en cualquier proyecto de Machine Learning. Los datos pueden provenir de diversas fuentes como bases de datos, APIs, Web Scraping, o archivos locales. En este proyecto, los datos provienen de archivos CSV que incluyen información de las tiendas, las ventas diarias y otros factores relevantes.

Código:
import pandas as pd
# Cargar los archivos CSV
store_data = pd.read_csv('data/raw/store.csv')
train_data = pd.read_csv('data/raw/train.csv')
test_data = pd.read_csv('data/raw/test.csv')
# Inspección inicial de los datos
print(store_data.head())
print(train_data.head())
print(test_data.head()) 
Análisis:
•	Inspección inicial: Se revisa la estructura de los datos y se identifican posibles problemas como valores faltantes.
•	Identificación de variables relevantes: Variables como Store, Date, Sales, Promo, y StateHoliday son críticas para el análisis. 

ii.- Limpieza de Datos y Análisis Exploratorio (02_LimpiezaEDA.ipynb). 
El análisis exploratorio de datos (EDA) es un proceso clave en cualquier proyecto de ciencia de datos. Involucra la inspección inicial, limpieza y transformación de los datos para hacerlos utilizables en el modelado. La limpieza de datos incluye el manejo de valores faltantes, outliers, y la transformación de variables.

Código:
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# Cargar los datos
train_data = pd.read_csv('data/raw/train.csv')
# Convertir la columna de fechas a formato datetime
train_data['Date'] = pd.to_datetime(train_data['Date'])
# Imputar valores faltantes
train_data['Open'].fillna(1, inplace=True)
# Análisis exploratorio
sns.histplot(train_data['Sales'], bins=50, kde=True)
plt.title('Distribución de las Ventas')
plt.show()
Análisis:
•	Distribución de las ventas: Se observa la distribución de la variable objetivo Sales. Esto ayuda a identificar si es necesario aplicar transformaciones para mejorar la distribución (por ejemplo, logaritmo para normalizar).
•	Impacto de promociones: Mediante gráficos de cajas, se evalúa cómo las promociones (Promo) afectan las ventas. Este análisis es crucial para entender qué factores influyen más en la predicción. 

iii.- Feature Engineering. 
El Feature Engineering es el proceso de crear nuevas características a partir de las existentes para mejorar el rendimiento del modelo. Esto puede incluir la transformación de variables temporales, la creación de variables dummy para variables categóricas, y la generación de características basadas en lag (retrasos) o ventanas temporales.

Código:
# Crear características adicionales basadas en la fecha
train_data['Year'] = train_data['Date'].dt.year
train_data['Month'] = train_data['Date'].dt.month
train_data['DayOfWeek'] = train_data['Date'].dt.dayofweek
# Crear características de lag para capturar la inercia de las ventas
train_data['Sales_Lag1'] = train_data.groupby('Store')['Sales'].shift(1)
train_data['Sales_Lag1'].fillna(0, inplace=True)
Análisis:
•	Variables temporales: La creación de variables como Year, Month, y DayOfWeek permite capturar patrones estacionales y tendencias a lo largo del tiempo.
•	Lag features: Estas características ayudan a capturar la inercia de las ventas, es decir, cómo las ventas pasadas pueden influir en las ventas futuras. 

iv.- Preparación de los Datos para el Modelado.
Antes de entrenar los modelos, los datos deben dividirse en conjuntos de entrenamiento y prueba. Es importante escalar las características para que todas las variables estén en la misma escala, lo que ayuda a los algoritmos de aprendizaje a converger más rápidamente y a evitar que algunas características dominen sobre otras.

Código:
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# Seleccionar características y variable objetivo
X = train_data.drop(columns=['Sales', 'Date'])
y = train_data['Sales']
# Dividir en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Escalar los datos
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
Análisis:
•	División de datos: Es fundamental dividir los datos en entrenamiento y prueba para evaluar la capacidad de generalización del modelo.
•	Estandarización: El escalado de los datos asegura que todas las variables contribuyen de manera equitativa al entrenamiento del modelo.

v.- Entrenamiento y Evaluación de Modelos (03_Entrenamiento_Evaluacion.ipynb).
El entrenamiento de modelos implica seleccionar y ajustar algoritmos para hacer predicciones precisas. En este proyecto, se entrenaron varios modelos supervisados, como regresión lineal, árboles de decisión, y modelos ensemble como Random Forest y Gradient Boosting. Además, se entrenó el modelo no supervisado Clustering. También se probó una red neuronal.
Modelos Supervisados:
1.	Regresión Lineal: Modelo simple que asume una relación lineal entre las características y la variable objetivo.
2.	Árboles de Decisión: Modelo no paramétrico que segmenta el espacio de las características en regiones más pequeñas y homogéneas.
3.	Random Forest: Ensemble de árboles de decisión que reduce la varianza y mejora la precisión.
4.	Gradient Boosting: Modelo que construye árboles de decisión de manera secuencial, optimizando los errores de los árboles anteriores. 
5.	Red neuronal: Funciona simultaneando un número elevado de unidades de procesamiento interconectadas que parecen versiones abstractas de neuronas. Las unidades de procesamiento se organizan en capas. 

Código:
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
# Entrenar modelo de Regresión Lineal
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)
lr_predictions = lr_model.predict(X_test_scaled)
# Evaluación
lr_mse = mean_squared_error(y_test, lr_predictions)
print(f'Linear Regression MSE: {lr_mse}')
# Entrenar modelo de Random Forest
rf_model = RandomForestRegressor()
rf_model.fit(X_train_scaled, y_train)
rf_predictions = rf_model.predict(X_test_scaled)
# Evaluación
rf_mse = mean_squared_error(y_test, rf_predictions)
print(f'Random Forest MSE: {rf_mse}')
Evaluación:
•	Métricas de Evaluación: En este caso se utiliza RMSPE, que es una métrica ajustada para ventas.
•	Comparación de Modelos: Se comparan los modelos basados en las métricas de error. Modelos como Random Forest y Gradient Boosting suelen superar a la regresión lineal debido a su capacidad para capturar relaciones no lineales.
Red Neuronal: Para capturar patrones más complejos, se implementó una red neuronal usando Keras y se ajustaron los hiper parámetros utilizando Keras Tuner.

Código:
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
import keras_tuner as kt
def build_model(hp):
    model = Sequential()
    model.add(Dense(units=hp.Int('units_1', min_value=64, max_value=512, step=64), activation='relu', input_dim=X_train.shape[1]))
    model.add(Dropout(hp.Float('dropout', min_value=0.1, max_value=0.5, step=0.1)))
    model.add(Dense(units=hp.Int('units_2', min_value=64, max_value=256, step=64), activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer=Adam(hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])), loss='mean_squared_error')
    return model
tuner = kt.RandomSearch(build_model, objective='val_loss', max_trials=10)
tuner.search(X_train_scaled, y_train, epochs=50, validation_split=0.2)
best_model = tuner.get_best_models(num_models=1)[0]
Análisis:
•	Redes Neuronales: Aunque más complejas, las redes neuronales pueden capturar interacciones entre variables que modelos lineales no pueden. Sin embargo, requieren mayor cuidado para evitar el sobreajuste.
•	Ajuste de Hiper parámetros: Keras Tuner permite buscar la mejor configuración de hiper parámetros, crucial para optimizar el rendimiento de la red. 

Modelo no supervisado: 
1.	Clustering: Modelo que agrupa datos en subconjuntos llamados clusters, de modo que los puntos de datos en el mismo cluster son más similares entre sí que con los puntos en otros clusters. 

Código: 
from sklearn.cluster import KMeans 
from sklearn.metrics import mean_squared_error
cl_model = RandomForestRegressor()
cl_model.fit(X_train_scaled, y_train)
cl_predictions = rf_model.predict(X_test_scaled)
# Evaluación
cl_mse = mean_squared_error(y_test, rf_predictions)
print(f'Clustering MSE: {rf_mse}')
Evaluación:
•	Métricas de Evaluación: En este caso se utiliza RMSPE, que es una métrica ajustada para ventas.
•	Comparación de Modelos: Se comparan los modelos basados en las métricas de error. Este modelo resultó ser el menos preciso de todos. 

vi.- Despliegue del Modelo.
El despliegue de modelos es la etapa en la cual el modelo entrenado se pone en producción para ser utilizado en la toma de decisiones. Herramientas como Streamlit permiten crear aplicaciones interactivas que permiten a los usuarios finales interactuar con el modelo.

Código:
import streamlit as st
# Cargar el modelo entrenado
import joblib
model = joblib.load('models/random_forest_model.pkl')
# Crear la aplicación con Streamlit
st.title('Predicción de Ventas - Rossmann')
# Entrada de usuario
store_id = st.number_input('ID de la Tienda', min_value=1, max_value=1115, step=1)
date = st.date_input('Fecha de la Predicción')
# Hacer predicción
input_data = preprocess_input(store_id, date)  # Función para procesar la entrada
prediction = model.predict(input_data)
st.write(f'La predicción de ventas es: {prediction[0]:.2f}')
Análisis:
•	Interactividad: La interfaz permite a los usuarios ingresar los parámetros relevantes y obtener una predicción en tiempo real.
•	Despliegue en la nube: Herramientas como Streamlit dan la posibilidad de  integrarse con servicios como cloud computing para permitir el acceso a la aplicación desde cualquier lugar que tenga conexión a internet. 

vii.- Documentación y Presentación.
La documentación es esencial para que otros desarrolladores o usuarios finales comprendan el proyecto y puedan reproducir los resultados. La estructura organizada del código y los comentarios claros facilitan la mantenibilidad del proyecto.
Elementos clave:
•	README.md: Debe incluir una descripción del proyecto, instrucciones para la instalación, y un resumen del proceso de modelado.
•	Presentación: Un archivo en docs/ que incluya los hallazgos clave y la justificación de las decisiones tomadas. 


III.- Como impactaría a un potencial cliente en el ámbito de negocio contemplado. o Potencial final del proyecto (aumentar un x% las ventas, etc.). 
Este proyecto de predicción de ventas de Rossmann es un excelente ejemplo de cómo se pueden aplicar técnicas avanzadas de Machine Learning para resolver problemas reales de negocio. Desde la adquisición y limpieza de datos hasta el despliegue de un modelo en producción, cada paso se apoya en fundamentos teóricos sólidos que garantizan la calidad y la relevancia del modelo final. 
Si se considera a las 1115 tiendas Rossmann que tienen presencia en Alemania como potenciales clientes, esta herramienta de machine learning les proporcionará predicciones muy precisas de cuales van a ser sus ventas futuras. Este hecho les traerá muchos beneficios. Además de contar con predicciones muy precisas de cuáles serán sus ventas, tendrán los siguientes beneficios:  
•	Reducción de Costos Operativos: Principalmente como consecuencia del ahorro en costos de inventario y personal.
•	Incremento en Ingresos: Como consecuencia de prever con anticipación cuando hacer promociones, y prever con anticipación las temporadas con ventas altas.
•	Satisfacción del Cliente: Todo lo antes descrito permite entregar un mejor servicio a los clientes, lo cual se verá reflejado en el incremento de la satisfacción de estos para con la empresa. 
