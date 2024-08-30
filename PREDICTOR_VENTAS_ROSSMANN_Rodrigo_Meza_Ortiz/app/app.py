import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor

def main():
    st.title("Rossmann Store Sales Predictor")
    
    st.write("""
    ### Descripción
    Esta aplicación predice las ventas diarias de las tiendas Rossmann en base a varios factores.
    Cargue un archivo CSV con datos de prueba para predecir las ventas.
    """)

    st.markdown("""
    #### Información sobre la columna **StateHoliday**:
    - **0**: No es un día festivo.
    - **a**: Feriado público (ej., feriado nacional).
    - **b**: Feriado relacionado con la Pascua.
    - **c**: Día de Navidad u otro feriado especial.
    """)
    
    st.markdown("""
    #### Columnas requeridas en el archivo CSV:
    - **Store**: ID de la tienda.
    - **DayOfWeek**: Día de la semana (1 = lunes, 7 = domingo).
    - **Date**: Fecha en formato YYYY-MM-DD.
    - **Sales**: Ventas del día (opcional si solo se desea predecir).
    - **Customers**: Número de clientes en la tienda.
    - **Open**: Si la tienda está abierta (1 = sí, 0 = no).
    - **Promo**: Si hay promoción activa (1 = sí, 0 = no).
    - **SchoolHoliday**: Si hay vacaciones escolares (1 = sí, 0 = no).
    - **Year**: Año de la fecha.
    - **Month**: Mes de la fecha.
    - **WeekOfYear**: Número de la semana en el año.
    - **IsHoliday**: Indica si es un día festivo (1 = sí, 0 = no).
    - **Sales_Lag1**: Ventas del día anterior.
    - **StateHoliday_0**: Indica si es un feriado estatal tipo 0.
    - **StateHoliday_a**: Indica si es un feriado estatal tipo a.
    - **StateHoliday_b**: Indica si es un feriado estatal tipo b.
    - **StateHoliday_c**: Indica si es un feriado estatal tipo c.
    """)

    model = joblib.load('C:/Users/rodri/OneDrive/Escritorio/PREDICTOR_VENTAS_ROSSMANN_Rodrigo_Meza_Ortiz/models/final_model.pkl', mmap_mode='r')
    
    uploaded_file = st.file_uploader("Suba el archivo CSV con los datos de prueba", type="csv")
    
    if uploaded_file is not None:
        test_data = pd.read_csv(uploaded_file)
        st.write("Archivo cargado:")
        st.write(test_data.head())
        
        if 'Unnamed: 0' in test_data.columns:
            test_data = test_data.drop(columns=['Unnamed: 0'])
        
        expected_columns = ['Store', 'DayOfWeek', 'Date', 'Sales', 'Customers', 'Open', 'Promo', 'SchoolHoliday',
                            'Year', 'Month', 'WeekOfYear', 'IsHoliday', 'Sales_Lag1', 
                            'StateHoliday_0', 'StateHoliday_a', 'StateHoliday_b', 'StateHoliday_c']
        missing_columns = [col for col in expected_columns if col not in test_data.columns]
        
        if missing_columns:
            st.error(f"Faltan las siguientes columnas en los datos cargados: {missing_columns}")
        else:

            test_data['Date'] = pd.to_datetime(test_data['Date'])
            test_data['DayOfWeek'] = test_data['Date'].dt.dayofweek
            test_data['WeekOfYear'] = test_data['Date'].dt.isocalendar().week
            test_data['IsHoliday'] = np.where(
                (test_data[['StateHoliday_a', 'StateHoliday_b', 'StateHoliday_c']].sum(axis=1) > 0) | 
                (test_data['SchoolHoliday'] == 1), 
                1, 
                0
            )
            
            if 'Sales' in test_data.columns:
                X_test = test_data.drop(columns=['Date', 'Sales'])
            else:
                X_test = test_data.drop(columns=['Date'])
            
            predictions = model.predict(X_test)
            
            test_data['Predicted Sales'] = predictions
            st.write("Predicciones de ventas:")
            st.write(test_data[['Store', 'Date', 'Predicted Sales']].head())
            
            st.download_button(
                label="Descargar predicciones",
                data=test_data[['Store', 'Date', 'Predicted Sales']].to_csv(index=False),
                file_name="sales_predictions.csv",
                mime="text/csv"
            )

if __name__ == "__main__":
    main() 
