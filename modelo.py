import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error, r2_score
import streamlit as st

def download_template():
    with open("plantilla.xlsx", "rb") as file:
        st.sidebar.download_button(
            label="Descargar Plantilla",
            data=file,
            file_name="plantilla.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

def load_data():
    uploaded_file = st.sidebar.file_uploader("Cargar archivo Excel", type=['xlsx'])
    if uploaded_file is not None:
      
        data = pd.read_excel(uploaded_file)
        data.fillna(0, inplace=True)  
        return data
    else:
        # Puedes devolver un DataFrame vacío o algún valor por defecto si prefieres
        return pd.DataFrame()

def create_card(title, value, value_format="{:,.2f}", color="#03bb85"):
    formatted_value = value_format.format(value)
    return f"""
        <style>
        .card {{
            border: 1px solid #f0f0f0;
            border-radius: 5px;
            padding: 20px;
            margin: 10px 0;
            box-shadow: 2px 2px 12px rgba(0,0,0,0.1);
            background-color: #333; /* El color de fondo de la tarjeta */
        }}
        .big-font {{
            font-size: 20px !important;
            color: #FFF; /* El color de texto general de la tarjeta */
        }}
        .highlighted-value {{
            color: {color}; /* Código de color hexadecimal para el color del valor */
        }}
        </style>
        <div class="card big-font">
            <p>{title}: <span class="highlighted-value">{formatted_value}</span></p>
        </div>
    """


# Función para entrenar el modelo de 'Universo de empleados cotizables'
def train_model_universo(data):
    from sklearn.linear_model import LinearRegression
    # Asegúrate de que las columnas coincidan con las de tus datos
    features = ['PEA', 'PET', 'Secundario2', 'Universitario3']
    target = 'Universo de empleados contizables'  # Asegúrate de que este sea el nombre correcto de la columna objetivo
    
    # Dividimos los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(data[features], data[target], test_size=0.2, random_state=42)
    
    # Entrenamos el modelo
    modelo_universo = LinearRegression().fit(X_train, y_train)
    
    # Predecimos en los conjuntos de entrenamiento y prueba
    y_train_pred = modelo_universo.predict(X_train)
    y_test_pred = modelo_universo.predict(X_test)
    
    # Calculamos y devolvemos las métricas
    train_mse = mean_squared_error(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    return modelo_universo, train_mse, train_r2, test_mse, test_r2


# Función para entrenar el modelo de 'Capitas pagadas en el año'
def train_model_capitas_pagadas(data):
    # Definimos las características y la variable objetivo
    features = ['PEA', 'PET','Total Afiliados Contribuyentes', 'Media ingresos mensual']
    target = 'Capitas pagadas en el año'

    # Dividimos los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(data[features], data[target], test_size=0.2, random_state=42)

    # Entrenamos el modelo de regresión lineal
    modelo_capitas = LinearRegression().fit(X_train, y_train)

    # Predecimos en los conjuntos de entrenamiento y prueba
    y_train_pred = modelo_capitas.predict(X_train)
    y_test_pred = modelo_capitas.predict(X_test)

    # Calculamos las métricas de rendimiento
    train_mse = mean_squared_error(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    return modelo_capitas, train_mse, train_r2, test_mse, test_r2

# Función para entrenar el modelo de 'Gasto de Salud'
def train_model_gasto_salud(data, dispersiones_pred):
    # Definimos las características y agregamos las dispersiones como una nueva característica
    features = ['Total Afiliados Contribuyentes', 'Media ingresos mensual']
    X = data[features].copy()  # Creamos una copia para evitar modificar el DataFrame original
    X['Disperciones Actuales'] = dispersiones_pred  # Asumimos que es un valor o una serie de valores
    
    y = data['Gasto en Salud Actual']
    
    # Dividimos los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Entrenamos el modelo de regresión lineal
    modelo_gasto_salud = LinearRegression().fit(X_train, y_train)

    # Predecimos en los conjuntos de entrenamiento y prueba
    y_train_pred = modelo_gasto_salud.predict(X_train)
    y_test_pred = modelo_gasto_salud.predict(X_test)

    # Calculamos las métricas de rendimiento
    train_mse = mean_squared_error(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    return modelo_gasto_salud, train_mse, train_r2, test_mse, test_r2


def train_model_dispersiones_actuales(data, gasto_salud_pred, capitas_pred):
    # Definimos las características y agregamos las predicciones como nuevas características
    features = ['Total Afiliados Contribuyentes', 'Media ingresos mensual']
    X = data[features].copy()  # Creamos una copia para evitar modificar el DataFrame original
    # Asegúrate de que gasto_salud_pred y capitas_pred son adecuados para la longitud de X
    X['Gasto en Salud Actual'] = gasto_salud_pred
    X['Capitas pagadas en el año'] = capitas_pred
    
    y = data['Disperciones Actuales']
    
    # Dividimos los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Entrenamos el modelo de regresión lineal
    modelo_dispersiones_actuales = LinearRegression().fit(X_train, y_train)

    # Predecimos en los conjuntos de entrenamiento y prueba
    y_train_pred = modelo_dispersiones_actuales.predict(X_train)
    y_test_pred = modelo_dispersiones_actuales.predict(X_test)

    # Calculamos las métricas de rendimiento
    train_mse = mean_squared_error(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    return modelo_dispersiones_actuales, train_mse, train_r2, test_mse, test_r2
    


# Función principal que ejecuta la aplicación Streamlit
def modulo_modelo():

    from sklearn.model_selection import train_test_split

 

    st.header("Modelo predictivo")
    download_template() 
    data = load_data()


    # Preparación de entradas de usuario
    user_inputs = {
        'Ocupados': st.sidebar.number_input('Ocupados', value=0),
        'Cesantes': st.sidebar.number_input('Cesantes', value=0),
        'PEA': st.sidebar.number_input('PEA', value=0),
        'PET': st.sidebar.number_input('PET', value=0),
        'Secundario2': st.sidebar.number_input('Secundario2', value=0),
        'Universitario3': st.sidebar.number_input('Universitario3', value=0),
        'Media ingresos mensual': st.sidebar.number_input('Media ingresos mensual', value=0),
        'Total Afiliados Contribuyentes': st.sidebar.number_input('Total Afiliados Contribuyentes', value=0),
        'Dependencias': st.sidebar.number_input('Dependencias', value=0.0, format="%.2f")
        #'Monto por Capita': st.sidebar.number_input('Monto por Capita', value=0), 
        
    }

    if st.sidebar.button('Predecir'):

        # MODELO universo de empleados elegibles
        if 'modelo_universo' not in st.session_state:
            modelo, train_mse, train_r2, test_mse, test_r2 = train_model_universo(data)
            st.session_state['modelo_universo'] = modelo
            st.session_state['test_mse'] = test_mse
            st.session_state['test_r2'] = test_r2
        
        universo_input = np.array([[
            user_inputs['PEA'], 
            user_inputs['PET'], 
            user_inputs['Secundario2'], 
            user_inputs['Universitario3'],
        
        ]])
        
        universo_pred = st.session_state['modelo_universo'].predict(universo_input)[0]
        




        # MODELO CAPITAS
        if 'modelo_capitas' not in st.session_state:
            modelo, train_mse, train_r2, test_mse, test_r2 = train_model_capitas_pagadas(data)
            st.session_state['modelo_capitas'] = modelo
            # Almacenamos las métricas en el session state para poder mostrarlas después
            st.session_state['capitas_train_mse'] = train_mse
            st.session_state['capitas_train_r2'] = train_r2
            st.session_state['capitas_test_mse'] = test_mse
            st.session_state['capitas_test_r2'] = test_r2


        capita_input = np.array([[
            user_inputs['PEA'], 
            user_inputs['PET'],
            user_inputs['Total Afiliados Contribuyentes'],  # Este campo debe coincidir con la entrada del modelo
            user_inputs['Media ingresos mensual'],  # Este campo debe coincidir con la entrada del modelo
            # Este campo debe coincidir con la entrada del modelo
        ]])

        
        capitas_pred = st.session_state['modelo_capitas'].predict(capita_input)[0]




        #Disperciones
        if 'modelo_dispersiones_actuales' not in st.session_state:
            # Ajustando la llamada a la función para incluir las métricas
            modelo, train_mse, train_r2, test_mse, test_r2 = train_model_dispersiones_actuales(data, universo_pred, capitas_pred)
            st.session_state['modelo_dispersiones_actuales'] = modelo
            # Almacenando las métricas de rendimiento en el estado de sesión para su uso posterior
            st.session_state['dispersiones_train_mse'] = train_mse
            st.session_state['dispersiones_train_r2'] = train_r2
            st.session_state['dispersiones_test_mse'] = test_mse
            st.session_state['dispersiones_test_r2'] = test_r2

        disp_input = np.array([[
            user_inputs['Total Afiliados Contribuyentes'], 
            user_inputs['Media ingresos mensual'], 
            universo_pred,  # Asegúrate de que este valor se ha calculado previamente
            capitas_pred  # Asegúrate de que este valor se ha calculado previamente
        ]])

        # Haciendo la predicción con el modelo de dispersiones actuales
        disp_pred = st.session_state['modelo_dispersiones_actuales'].predict(disp_input)[0]



        #Gasto Salud
        if 'modelo_gasto_salud' not in st.session_state:
       
            disp_pred_array = np.full((data.shape[0],), disp_pred)  # Crea un array lleno de disp_pred con el mismo tamaño que tus datos
            
            modelo, train_mse, train_r2, test_mse, test_r2 = train_model_gasto_salud(data, disp_pred_array)
            st.session_state['modelo_gasto_salud'] = modelo
            st.session_state['gasto_salud_train_mse'] = train_mse
            st.session_state['gasto_salud_train_r2'] = train_r2
            st.session_state['gasto_salud_test_mse'] = test_mse
            st.session_state['gasto_salud_test_r2'] = test_r2

        gasto_salud_input = np.array([[
            user_inputs['Total Afiliados Contribuyentes'],
            user_inputs['Media ingresos mensual'],
            disp_pred  # Asegúrate de que disp_pred es el valor correcto a utilizar aquí
        ]])

        gasto_salud_pred = st.session_state['modelo_gasto_salud'].predict(gasto_salud_input)[0]

    

        tab1, tab2 = st.tabs(["Predicciones", "Métricas de Rendimiento"])
        # TAB1: Predicciones
        with tab1:
            card_html = create_card("Predicción de Universo de empleados cotizables", universo_pred)
            st.markdown(card_html, unsafe_allow_html=True)

        # Columna para la predicción de Capitas pagadas en el año
    
            card_html = create_card("Predicción de Capitas pagadas en el año", capitas_pred)
            st.markdown(card_html, unsafe_allow_html=True)

        # Columna para la predicción de Disperciones Actuales
    
            card_html = create_card("Predicción de Disperciones Actuales", disp_pred)
            st.markdown(card_html, unsafe_allow_html=True)

            card_html = create_card("Predicción de Gasto en Salud Actual", gasto_salud_pred)
            st.markdown(card_html, unsafe_allow_html=True)

        # Sinistros 
            sini = (gasto_salud_pred / disp_pred  ) * 100
            card_html = create_card("Porcentaje de Siniestrabilidad", sini)
            st.markdown(card_html, unsafe_allow_html=True)
        #Disperciones Neta
            dn = (disp_pred - gasto_salud_pred)*0.9
            card_html = create_card("Disperciones netas (Excluyendo el 10% de gastos operacionales)", dn)
            st.markdown(card_html, unsafe_allow_html=True)

        #No Afiliados
            
            total_afiliados_contribuyentes = user_inputs['Total Afiliados Contribuyentes']
            NF = universo_pred - total_afiliados_contribuyentes
            card_html = create_card("cantidad de personas no afiliadas", NF)
            st.markdown(card_html, unsafe_allow_html=True)

        #Aporte NO Afiliados
            
            APN = ((user_inputs['Media ingresos mensual']*0.01013)*NF)*12
            card_html = create_card("Aporte no recibido por No Afiliados", APN)
            st.markdown(card_html, unsafe_allow_html=True)

        #Posibes New AF
            PAF = (NF * user_inputs['Dependencias']+NF)
            card_html = create_card("Aporte no recibido por No Afiliados", PAF)
            st.markdown(card_html, unsafe_allow_html=True)
            

        # TAB2: Métricas de Rendimiento
        with tab2:
            mse_test = st.session_state['test_mse']
            r2_test = st.session_state['test_r2']


            card_html = create_card("Error cuadrático medio (MSE) para el conjunto de prueba", mse_test)
            st.markdown(card_html, unsafe_allow_html=True)
            card_html = create_card("Coeficiente de determinación (R²) para el conjunto de prueba", r2_test)
            st.markdown(card_html, unsafe_allow_html=True)

            capitas_mse_test = st.session_state['capitas_test_mse']
            capitas_r2_test = st.session_state['capitas_test_r2']
            
        
            card_html = create_card("Error cuadrático medio (MSE) para el conjunto de prueba de Capitas", capitas_mse_test)
            st.markdown(card_html, unsafe_allow_html=True)
            card_html = create_card("Coeficiente de determinación (R²) para el conjunto de prueba de Capitas", capitas_r2_test)
            st.markdown(card_html, unsafe_allow_html=True)
            
            dispersiones_mse_test = st.session_state['dispersiones_test_mse']
            dispersiones_r2_test = st.session_state['dispersiones_test_r2']
        
            card_html = create_card("Error cuadrático medio (MSE) para el conjunto de prueba de Disperciones", dispersiones_mse_test)
            st.markdown(card_html, unsafe_allow_html=True)
            card_html = create_card("Coeficiente de determinación (R²) para el conjunto de prueba de Disperciones", dispersiones_r2_test)
            st.markdown(card_html, unsafe_allow_html=True)

            gasto_salud_mse_test = st.session_state['gasto_salud_test_mse']
            gasto_salud_r2_test = st.session_state['gasto_salud_test_r2']
            
        
            card_html = create_card("Error cuadrático medio (MSE) para el conjunto de prueba de Gasto Salud", gasto_salud_mse_test)
            st.markdown(card_html, unsafe_allow_html=True)
            card_html = create_card("Coeficiente de determinación (R²) para el conjunto de prueba de Gasto Salud", gasto_salud_r2_test)
            st.markdown(card_html, unsafe_allow_html=True)            




   








        

        

    
    
        
        

        
    
    



    

    



  



    



 