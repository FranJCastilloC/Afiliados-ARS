
def modulo_corr():

    import streamlit as st
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    st.header("Correlacion de los datos")
    st.write("Este módulo se puede dedicar al análisis exploratorio de los datos relacionados con las ARS.")
    

 
    # Carga o define tu DataFrame aquí, por ejemplo:
    df = pd.read_excel('C:/Users/franc/OneDrive - INTEC/Escritorio/Proyecto ARS/Dashboards/BD FINAL/corr2.xlsx', engine='openpyxl')


    # Titulo de tu aplicación
    st.title('Visualizador de Correlaciones')

    # Selector para elegir variables/columnas del DataFrame
    selected_columns = st.multiselect('Selecciona las variables para correlacionar', df.columns)

    # Verificar si se seleccionaron columnas
    if selected_columns:
        # Filtrar el DataFrame con las columnas seleccionadas
        df_filtered = df[selected_columns]
        
        # Calcular la matriz de correlación
        corr_matrix = df_filtered.corr()

        # Configurar tamaño de figura en Seaborn
        fig, ax = plt.subplots(figsize=(10, 8)) # Ajusta el tamaño según necesites
        
        # Crear el heatmap
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, ax=ax)
        
        # Titulo del heatmap
        plt.title('Correlación entre variables seleccionadas')
        
        # Mostrar el gráfico en Streamlit
        st.pyplot(fig)
    else:
        st.write('Por favor, selecciona al menos una variable.')