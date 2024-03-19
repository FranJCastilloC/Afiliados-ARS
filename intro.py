
def modulo_overview():

    import streamlit as st
    import pandas as pd
    import plotly.express as px
    import os


    st.header("Análisis Exploratorio de Datos")
    st.write("Este módulo se puede dedicar al análisis exploratorio de los datos relacionados con las ARS.")

    # Función para cargar los datos
    def load_data(carpeta):

        df = {}
        for archivo in os.listdir(carpeta):
            if archivo.endswith('.csv') or archivo.endswith('.xlsx'):
                variable = os.path.splitext(archivo)[0]
                ruta = os.path.join(carpeta, archivo)

                if archivo.endswith('.csv'):
                    df[variable] = pd.read_csv(ruta)
                elif archivo.endswith('.xlsx'):
                    # Leer todas las hojas del archivo Excel
                    xls = pd.ExcelFile(ruta)
                    for sheet_name in xls.sheet_names:
                        # Crear un nombre único para cada hoja/variable
                        variable_hoja = f"{variable}_{sheet_name}"
                        df[variable_hoja] = pd.read_excel(ruta, sheet_name=sheet_name)
        return df

    carpeta_ruta = "C:/Users/franc/OneDrive - INTEC/Escritorio/Proyecto ARS/Bases de datos/BD FINAL"
    df = load_data(carpeta_ruta)


    def asignar_dfs_a_variables_globales(df):
        for clave, df in df.items():
            # Establece cada DataFrame en el diccionario como una variable global
            globals()[clave] = df
            
    asignar_dfs_a_variables_globales(df)

    # Visualización de los DataFrames
    if 'Poblacion_Ocupacion' in globals():
        st.write("### Población y Ocupación")
        st.dataframe(Poblacion_Ocupacion)
    
    if 'lista_Afiliados_1' in globals():
        st.write("### Lista de Afiliados")
        st.dataframe(lista_Afiliados_1)

    if 'Poblacion_Empleado_Educacion'in globals():
        st.write("### Afiliados por Edad y Sexo")
        st.dataframe(Poblacion_Empleado_Educacion)

    