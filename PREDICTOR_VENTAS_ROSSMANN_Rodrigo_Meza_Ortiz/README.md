# Predicción de Ventas de Rossmann

## Descripción

Este proyecto tiene como objetivo predecir las ventas diarias de las tiendas Rossmann utilizando técnicas de Machine Learning. Se han empleado varios algoritmos de regresión y métodos avanzados como `RandomizedSearchCV` y `SHAP` para optimizar el rendimiento del modelo y proporcionar interpretaciones detalladas sobre la importancia de las características.

La aplicación final ha sido implementada utilizando `Streamlit`, lo que permite a los usuarios ingresar datos específicos de una tienda y obtener predicciones de ventas en tiempo real.

## Estructura del Proyecto

- **app**: Contiene la aplicación `Streamlit` para la predicción de ventas.
    - `app.py`: Código fuente de la aplicación.
    - `requirements.txt`: Dependencias del proyecto.

- **data**: Contiene los datos utilizados para entrenar y probar el modelo.
    - `raw`: Datos originales sin procesar.
    - `processed`: Datos preprocesados y listos para ser usados en el modelo.
    - `train`: Conjunto de entrenamiento.
    - `test`: Conjunto de prueba.

- **models**: Contiene los modelos entrenados y los archivos de configuración.
    - `final_model.pkl`: Modelo final entrenado.
    - `model_config.yaml`: Configuración avanzada del modelo.

- **docs**: Documentación relacionada con el proyecto.
    - `negocio.ppt`: Presentación de negocio.
    - `ds.ppt`: Presentación de Data Science.
    - `memoria.md`: Memoria del proyecto.

- **README.md**: Este archivo sirve como guía del proyecto.

## Instalación

Para ejecutar este proyecto en tu entorno local, sigue estos pasos:

1. Clona este repositorio:
    ```bash
    # git clone https://github.com/rodrigomezaortiz/PREDICTOR_VENTAS_ROSSMANN.git 
    # cd PREDICTOR_VENTAS_ROSSMANN 
    ```

2. Instala las dependencias del proyecto utilizando `pip`:
    ```bash
    pip install -r requirements.txt
    ```

3. Asegúrate de tener los datos en la carpeta `data/raw`.

## Uso

Para ejecutar la aplicación `Streamlit`, sigue estos pasos:

1. Inicia la aplicación:
    ```bash
    streamlit run app/app.py
    ```

2. Abre tu navegador y ve a `http://localhost:8501` para interactuar con la aplicación.

3. Ingresa los datos solicitados en la barra lateral y obtén la predicción de ventas para la tienda seleccionada.

## Modelos Implementados

En este proyecto se han utilizado varios modelos de regresión para predecir las ventas, incluyendo:

- **XGBoost**
- **Regresión Lineal**
- **Árboles de Decisión**
- **Random Forest**
- **Gradient Boosting**
- **Regresión Ridge**
- **KMeans Clustering**
- **Red Neuronal**

Cada modelo ha sido entrenado y evaluado. El mejor rendimiento se obtuvo con `Random Forest`, el cuál se optimizó mediante `RandomizedSearchCV`. 
