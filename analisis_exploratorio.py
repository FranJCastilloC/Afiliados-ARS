

def main():
    #Definicion del modulo de Exploracion de datos
    def modulo_analisis_exploratorio():

        import streamlit as st
        import pandas as pd
        import plotly.graph_objs as go
        import plotly.express as px
        import os
        import seaborn as sns 
        import matplotlib.pyplot as plt
        from plotly.subplots import make_subplots
        import json
        import pydeck as pdk
        import numpy as np
        import matplotlib.colors as mcolors

        st.header("Análisis Exploratorio de Datos")
        st.write("Este apartado se enfoca en la exploración de datos desde cuatro perspectivas distintas, abordando el planteamiento e interpretación del problema, y el análisis e interpretación de datos asociados con los afiliados al Sistema de Seguridad Social (SFS), la población en general, y las finanzas del sistema dentro de regimen contributivo de salud.")


    #Cargar Datos
        def load_data(folder_path):
        
            data_frames = {}
            for file_name in os.listdir(folder_path):
                if file_name.endswith('.csv') or file_name.endswith('.xlsx'):
                    variable_name = os.path.splitext(file_name)[0]
                    file_path = os.path.join(folder_path, file_name)
                    if file_name.endswith('.csv'):
                        data_frames[variable_name] = pd.read_csv(file_path)
                    else:  # Para archivos XLSX
                        xls = pd.ExcelFile(file_path)
                        for sheet_name in xls.sheet_names:
                            variable_sheet_name = f"{variable_name}_{sheet_name}"
                            data_frames[variable_sheet_name] = pd.read_excel(file_path, sheet_name=sheet_name)
            return data_frames

    #Preparando Data Frames
        def prepare_dataframes(data_frames):
            """
            Realiza transformaciones necesarias en los DataFrames para el análisis.
            """

            for key, df in data_frames.items():
                if 'lista_Afiliados' in key:
                    df['Año'] = pd.to_datetime(df['Fecha'], format='%Y%m').dt.year
                    df['Mes'] = pd.to_datetime(df['Fecha'], format='%Y%m').dt.month
                    # Asumiendo que el 'key' contiene algún identificador para las entradas de diciembre
                    if 'diciembre' in key:
                        df_diciembre = df[df['Mes'] == 12]
                        df_afiliados_agrupados = df_diciembre.groupby('Año').agg({'Afiliados C': 'sum'}).reset_index()
                        data_frames[key + '_agrupados'] = df_afiliados_agrupados



            
            if 'lista_Afiliados_1' in data_frames:
                df = data_frames['lista_Afiliados_1']
                # Suponiendo que ya has agregado las columnas 'Año' y 'Mes' como en tu código anterior
                df_diciembre = df[df['Mes'] == 12]  
                df_afiliados_agrupados = df_diciembre.groupby('Año').agg({'Afiliados C': 'sum'}).reset_index()
                df_afiliados_agrupados = df_afiliados_agrupados.melt(id_vars='Año', var_name='Categoría', value_name='Cantidad')
                data_frames['df_afiliados_agrupados'] = df_afiliados_agrupados

            
            if 'Poblacion_Ocupacion' in data_frames:
                df = data_frames['Poblacion_Ocupacion']
                filtro_ocupacion = df['Ocupacion'].isin(['Empleado privado', 'Empleado del estado', 'Monto por capita'])
                df_filtrado = df[filtro_ocupacion]
                df_filtrado = df_filtrado.groupby('Año')['Poblacion'].sum().reset_index()
                df_filtrado['Categoría'] = 'Universo Elegible RC'
                df_filtrado = df_filtrado.rename(columns={'Poblacion': 'Cantidad'})
                data_frames['df_poblacion_filtrado'] = df_filtrado

                
            if 'df_afiliados_agrupados' in data_frames and 'df_poblacion_filtrado' in data_frames:
                df_unificado = pd.concat([data_frames['df_afiliados_agrupados'], data_frames['df_poblacion_filtrado']], ignore_index=True)
                data_frames['df_unificado'] = df_unificado

            if 'Afiliados_Por_Edad_1' in data_frames:
                df = data_frames['Afiliados_Por_Edad_1']
                is_numeric_fecha = pd.to_numeric(df['Fecha'], errors='coerce').notnull()
                df = df[is_numeric_fecha]
                df['Fecha'] = pd.to_datetime(df['Fecha'], format='%Y%m', errors='coerce')
                df.dropna(subset=['Fecha'], inplace=True)
                df['Año'] = df['Fecha'].dt.year
                edad_c = df.groupby(['Año', 'Edad'])['Afiliados C'].sum().unstack()
                edad_no_c = df.groupby(['Año', 'Edad'])['Afiliados NO C'].sum().unstack()
                edad = pd.concat([edad_c.add_suffix(' C'), edad_no_c.add_suffix(' NO C')], axis=1)
                data_frames['edad'] = edad


            if 'Poblacion_Empleado_Educacion' in data_frames:
                educacion = data_frames['Poblacion_Empleado_Educacion']
                data_frames['educacion'] = educacion

            if 'Poblacion_Empleado_' in data_frames:
                educacion = data_frames['Poblacion_Empleado_Educacion']
                data_frames['educacion'] = educacion

            if 'Poblacion_Indicadores' in data_frames:
                Indicador = data_frames['Poblacion_Indicadores']
                data_frames['Indicador'] = Indicador

            if 'Poblacion_Igresos_Establecimiento' in data_frames:
                Sector = data_frames['Poblacion_Igresos_Establecimiento']
                data_frames['Sector'] = Sector

            if 'Poblacion_Edad' in data_frames:
                Eda = data_frames['Poblacion_Edad']
                data_frames['Eda'] = Eda

            if 'Poblacion_Igresos_Sector' in data_frames:
                sec = data_frames['Poblacion_Igresos_Sector']
                data_frames['sec'] = sec

            if 'Finanzas_1' in data_frames:
                Fina = data_frames['Finanzas_1']
                data_frames['Fina'] = Fina

            if "IPC_IPC" in data_frames:
                ipc = data_frames['IPC_IPC']
                data_frames['ipc'] = ipc

            if "Prestaciones_1" in data_frames:
                pres = data_frames['Prestaciones_1']
                data_frames['pres'] = pres

            if 'lista_Afiliados_1' in data_frames:
                lista_Afiliados_1 = data_frames['lista_Afiliados_1']
                lista_Afiliados_1['Fecha'] = pd.to_datetime(lista_Afiliados_1['Fecha'], format='%Y%m')
                lista_Afiliados_1.sort_values('Fecha', inplace=True)
                data_frames['lista_Afiliados_1'] = lista_Afiliados_1

    #Visualizacion de datos
        def display_data_visualizations(df_unificado, educacion, Sector, edad, lista_Afiliados_1, Indicador, Eda, sec, Fina, ipc, pres):
            """
            Muestra visualizaciones de datos utilizando Plotly.
            """
        #Definiendo las secciones
            tab1, tab2, tab3,tab4 = st.tabs(["Datos Afiliados", "Datos Poblacion","Finanzas","Planteamiento del problema"])


        #Seccion 4, Panteamiento del problema

            with tab4:

                #Grafico de empleados totales vs afiliados
                fig_tendencia = px.line(df_unificado, x='Año', y='Cantidad', color='Categoría', title='Afiliados contribuyentes vs Total de empleados aplicables')
                fig_tendencia.update_layout(plot_bgcolor='black', paper_bgcolor='black', font_color='white')
                

                col1, col2 = st.columns([1, 1])
                with col1:
                    
                    data_2022 = {
                        'Menos de RD$5,000': [13172],
                        'De RD$5,000 a RD$10,000': [199255],
                        'De RD$10,000 a RD$15,000': [577944],
                        'De RD$15,000 a RD$30,000': [849295],
                        'De RD$30,000 a RD$50,000': [265327],
                        'Más de RD$50,000': [314346],
                        'Total': [2219339]
                    }

                    # Create the DataFrame
                    df_22 = pd.DataFrame(data_2022)

                    # Sumar las cantidades de trabajadores que ganan menos de RD$30,000
                    trabajadores_menos_30000 = df_22['Menos de RD$5,000'] + df_22['De RD$5,000 a RD$10,000'] + df_22['De RD$10,000 a RD$15,000'] + df_22['De RD$15,000 a RD$30,000']
                    trabajadores_menos_30000_con_no_aportantes = trabajadores_menos_30000 + 635092  
                    total_mercado_laboral = df_22['Total']
                    probabilidad_menos_30000 = trabajadores_menos_30000 / total_mercado_laboral
                    probabilidad_menos_30000_con_no_aportantes = trabajadores_menos_30000_con_no_aportantes / total_mercado_laboral

                    probabilidad_menos_30000.values[0], probabilidad_menos_30000_con_no_aportantes.values[0]
                
                    años_disponibles = df_unificado['Año'].unique()
                    año_seleccionado = st.select_slider('Selecciona un año', options=sorted(años_disponibles))
                    df_filtrado = df_unificado[df_unificado['Año'] == año_seleccionado]
                    fig_brecha = go.Figure()
                    cantidad_afiliados_c = None
                    cantidad_universo_elegible_rc = None
                    porcentaje_diferencia = None

                    
                    # Para la gráfica
                    categorias_salario = df_22.columns[:-1]
                    cantidad_trabajadores = df_22.iloc[0][:-1]
                    # Calculando los porcentajes
                    porcentajes = (cantidad_trabajadores / cantidad_trabajadores.sum()) * 100

                    # Crear un gráfico de barras con Plotly
                    fig2 = px.bar(
                        x=categorias_salario, 
                        y=cantidad_trabajadores,
                        text=porcentajes.apply(lambda x: '{:.2f}%'.format(x)),
                        labels={'x': 'Rango Salarial', 'y': 'Cantidad de trabajadores'},
                        title='Distribución de trabajadores por rango salarial'
                    )

                    # Actualizar el diseño para tener un fondo negro y ajustes para mostrar los porcentajes
                    fig2.update_layout(
                        plot_bgcolor='rgba(0,0,0,1)', 
                        paper_bgcolor='rgba(0,0,0,1)', 
                        font=dict(color='white'),
                        xaxis_tickangle=-45
                    )

                    # Configuración para mostrar los textos (porcentajes) sobre las barras
                    fig2.update_traces(texttemplate='%{text}', textposition='outside')

                    # Añadimos las barras al gráfico
                    df_afiliados_c = df_filtrado[df_filtrado['Categoría'].str.contains('Afiliados C')]
                    if not df_afiliados_c.empty:
                        cantidad_afiliados_c = df_afiliados_c['Cantidad'].values[0]
                        fig_brecha.add_trace(go.Bar(
                            x=['Empleados afiliados al RC'], y=[cantidad_afiliados_c],
                            name='Afiliados RC', marker_color='blue'
                        ))

                    df_universo_elegible_rc = df_filtrado[df_filtrado['Categoría'].str.contains('Universo Elegible RC')]
                    if not df_universo_elegible_rc.empty:
                        cantidad_universo_elegible_rc = df_universo_elegible_rc['Cantidad'].values[0]
                        fig_brecha.add_trace(go.Bar(
                            x=['Total empleadados elegibles al RC'], y=[cantidad_universo_elegible_rc],
                            name='Total Empleadados', marker_color='red'
                        ))

                    # Calculamos la diferencia y la añadimos como una nueva barra encima de la barra 'Afiliados C'
                    if cantidad_afiliados_c is not None and cantidad_universo_elegible_rc is not None:
                        diferencia = cantidad_universo_elegible_rc - cantidad_afiliados_c
                        porcentaje_diferencia = (diferencia / cantidad_universo_elegible_rc) * 100  # Calculamos el porcentaje
                        fig_brecha.add_trace(go.Bar(
                            x=['Empleados afiliados al RC'], y=[diferencia],
                            name='Afiliados RC', marker_color='white'
                        ))

                        # Añadimos anotación con el porcentaje de la diferencia
                        fig_brecha.add_annotation(
                            x='Empleados afiliados al RC',
                            y=cantidad_afiliados_c + diferencia/2,  # Posiciona la anotación en el medio de la barra de diferencia
                            text=f"{porcentaje_diferencia:.1f}%",  # Formatea el texto con 1 decimal
                            showarrow=False,
                            font=dict(
                                color='white',
                                size=14
                            ),
                            bgcolor='black'
                        )

                        # Asegurarse de que el gráfico sea apilado
                        fig_brecha.update_layout(barmode='stack')

                        # El resto de tu código para actualizar y mostrar el gráfico
                        fig_brecha.update_layout(
                            title_text='Brecha de empleados que evaden el regimen contributivo', 
                            plot_bgcolor='black', 
                            paper_bgcolor='black', 
                            font_color='white'
                        )
                    st.plotly_chart(fig_brecha, use_container_width=True)
                    st.header("Que pasaria si no existe dicha brecha?")
                    st.plotly_chart(fig2, use_container_width=True)

                with col2:
                        
                    # Crear una fila de columnas para las métricas y mostrarlas horizontalmente
                    metrics_col1, metrics_col2, metrics_col3 = st.columns(3)

                    def format_number(num):

                        if num >= 1_000_000:
                            return f"{num / 1_000_000:.2f}M"
                        elif num >= 1_000:
                            return f"{num / 1_000:.2f}K"
                        else:
                            return f"{num:.2f}"

                    metrics_col1, metrics_col2, metrics_col3 = st.columns(3)

                    with metrics_col1:
                        if cantidad_afiliados_c is not None:
                            formatted_afiliados_c = format_number(cantidad_afiliados_c)
                            st.metric(label="Afiliados RC", value=formatted_afiliados_c)

                    with metrics_col2:
                        if cantidad_universo_elegible_rc is not None:
                            formatted_universo_elegible_rc = format_number(cantidad_universo_elegible_rc)
                            st.metric(label="Total empleados elegibles al R", value=formatted_universo_elegible_rc)

                    with metrics_col3:
                        if porcentaje_diferencia is not None and cantidad_universo_elegible_rc is not None:
                            formatted_diferencia = format_number(diferencia)
                            st.metric(label="Empleados NO Afiliados al RC", value=f"{formatted_diferencia} ")

                    st.plotly_chart(fig_tendencia, use_container_width=True)

    
                with col2:

                    st.write("Al interactuar con el gráfico que ilustra la brecha de empleados que no realizan sus aportes a la Tesorería de la Seguridad Social (TSS), se observa que, para el año 2022, aproximadamente 635,000 personas no cumplían con sus obligaciones de pago.")

                    st.dataframe(df_22)

                    st.header("Resumen de Salarios y Aportes")

                    # Creación de columnas
                    col1, col2, col3 = st.columns(3)

                    # Métricas de salarios
                    with col1:
                        st.metric(label="Salario < RD$15,000", value="35.59%")
                    with col2:
                        st.metric(label="Salario entre RD$15,000 y RD$30,000", value="38.30%")
                    with col3:
                        st.metric(label="Salario > RD$30,000", value="26.12%")

                    # Segunda fila de columnas para más detalles
                    col4, col5, col6 = st.columns(3)

                    with col4:
                        st.metric(label="Salario < RD$30,000", value="73.88%")
                    with col5:
                        st.metric(label="Salario medio esperado", value="RD$25,126.04")
                    with col6:
                        st.metric(label="Aporteanual esperado por persona", value="RD$2,545.27")

                    # Para el aporte total esperado de los no afiliados, dado que es una cifra importante, podría ocupar su propia sección o estar destacada de otra manera
                    st.metric(label="Aporte total esperado de no afiliados en el año 2022", value="RD$1,616,479,141.53")
                    st.write("Estos cálculos se obtuvieron mediante la aplicación de una distribución de densidad kernel")
                
                st.header("Resultados con los nuevos aportes")


                    

            with tab1:

                st.markdown("<h1 style='text-align: center; color: white;'>Informaciones relevantes sobre los empleados afiliados al régimen contributivo</h1>", unsafe_allow_html=True)

                col1, col2 = st.columns([1, 1,])

            
                afiliados_diciembre = lista_Afiliados_1[lista_Afiliados_1['Fecha'].dt.month == 12]
                afiliados_c_por_año = afiliados_diciembre.groupby(afiliados_diciembre['Fecha'].dt.year)['Afiliados C'].sum()
                afiliados_no_c_por_año = afiliados_diciembre.groupby(afiliados_diciembre['Fecha'].dt.year)['Afiliados NO C'].sum()

                with col1:

                    # Carga el GeoJSON
                    geojson_path = 'C:/Users/franc/OneDrive - INTEC/Escritorio/Proyecto ARS\Dashboards\PROV.geojson'
                    with open(geojson_path) as geojson_file:
                        geojson_data = json.load(geojson_file)


                    # Preparación de los datos
                    lista_Afiliados_1['Fecha'] = pd.to_datetime(lista_Afiliados_1['Fecha'])
                    lista_Afiliados_1['Total_Afiliados'] = lista_Afiliados_1['Afiliados C'] + lista_Afiliados_1['Afiliados NO C']
                    df_diciembre = lista_Afiliados_1[lista_Afiliados_1['Fecha'].dt.month == 12]
                    df_suma = df_diciembre.groupby([df_diciembre['Fecha'].dt.year, 'Provincia']).agg({
                        'Total_Afiliados': 'sum',
                        'Afiliados C': 'sum',
                        'Afiliados NO C': 'sum'
                    }).reset_index()
                    

                    # Slider para seleccionar el año
                    año_seleccionado = st.slider(
                        'Seleccione el año',
                        min_value=int(df_suma['Fecha'].min()),
                        max_value=int(df_suma['Fecha'].max()),
                        step=1
                    )

                    # Añadimos una columna con la suma de 'Afiliados C' y 'Afiliados NO C'
                    df_filtrado_por_año = df_suma[df_suma['Fecha'] == año_seleccionado]
                    max_afiliados = df_filtrado_por_año['Total_Afiliados'].max()

                    # Función para calcular el color
                    def calculate_color(total_afiliados, max_afiliados):
                        """Genera un color basado en la cantidad de afiliados."""
                        intensity = total_afiliados / max_afiliados
                        return [255, 255 * (1 - intensity), 255 * (1 - intensity), 150]

                    # Actualiza el GeoJSON basado en df_filtrado_por_año
                    for feature in geojson_data['features']:
                        provincia = feature['properties']['province_name']  
                        match = df_filtrado_por_año[df_filtrado_por_año['Provincia'] == provincia]
                        if not match.empty:
                            total_afiliados = match['Total_Afiliados'].iloc[0]
                            feature['properties']['fillColor'] = calculate_color(total_afiliados, max_afiliados)
                        else:
                            feature['properties']['fillColor'] = [0, 0, 0, 0]


                    # Crear el mapa con PyDeck
                    view_state = pdk.ViewState(latitude=18.486058, longitude=-69.931212, zoom=7, min_zoom=5, max_zoom=15, pitch=0, bearing=0)
                    layer_provincias = pdk.Layer(
                        'GeoJsonLayer',
                        geojson_data,
                        stroked=False,
                        filled=True,
                        extruded=True,
                        get_fill_color='properties.fillColor',
                        get_line_color=[255, 255, 255],
                    )

                    # Renderizar el mapa en Streamlit
                    st.pydeck_chart(pdk.Deck(
                        map_style='mapbox://styles/mapbox/light-v9',
                        initial_view_state=view_state,
                        layers=[layer_provincias],
                    ))

                    # Función para dibujar la barra de colores
                    def draw_colorbar(max_value, title="Cantidad de Afiliados"):
                        fig, ax = plt.subplots(figsize=(6, 1))
                        fig.subplots_adjust(bottom=0.5)

                        # Define la paleta de colores que se correlacione con el mapa
                        cmap = mcolors.LinearSegmentedColormap.from_list("", [(1, 1, 1), calculate_color(max_value, max_value)[:3]])
                        norm = mcolors.Normalize(vmin=0, vmax=max_value)

                        cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                                        cax=ax, orientation='horizontal', ticklocation='bottom')
                        cb.set_label(title)
                        st.pyplot(fig)


                    # Llama a esta función con el valor máximo de afiliados
                    draw_colorbar(max_afiliados)

                with col2:
                        
                        # Crear el gráfico de barras
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=afiliados_c_por_año.index,
                            y=afiliados_c_por_año,
                            name='Afiliados C',
                            marker_color='blue'
                        ))
                        fig.add_trace(go.Scatter(
                            x=afiliados_no_c_por_año.index,
                            y=afiliados_no_c_por_año,
                            name='Afiliados NO C',
                            marker_color='orange'
                        ))

                        fig.update_layout(
                            title='Comparación de Afiliados que contribuyen y Afiliados no contribuyentes',
                            xaxis_tickangle=-45,
                            xaxis_title='Año',
                            yaxis_title='Cantidad de Afiliados',
                            barmode='group',  
                        )

                        # Mostrar el gráfico en Streamlit
                        st.plotly_chart(fig, use_container_width=True)  

                        altura_para_6_filas = 10 * 30   
                        st.dataframe(df_filtrado_por_año, height=altura_para_6_filas)

                st.markdown("<h1 style='text-align: center;'>Afiliados Contribuyentes y No Contribuyentes divididos por edad</h1>", unsafe_allow_html=True)

                tipo_afiliado = st.radio("Selecciona el tipo de afiliado:", ('Afiliados Contribuyentes', 'Afiliados NO Contribuyentes'))

                if tipo_afiliado == 'Afiliados C':
                    columnas_filtradas = [col for col in edad.columns if 'C' in col and 'NO C' not in col] 
                else:
                    columnas_filtradas = [col for col in edad.columns if 'NO C' in col] 

                fig = go.Figure()
                colores_base = px.colors.qualitative.Plotly + px.colors.qualitative.Alphabet + px.colors.qualitative.Light24
                colores_filtrados = [colores_base[i % len(colores_base)] for i in range(len(columnas_filtradas))]

                # Agregar una línea por cada columna de edad dentro del grupo seleccionado
                for i, columna in enumerate(columnas_filtradas):
                    nombre_legible = columna.replace(' NO C', '')
                    nombre_legible = columna.replace(' C', '')  
                    fig.add_trace(go.Scatter(
                        x=edad.index,
                        y=edad[columna],
                        mode='lines+markers',
                        name=nombre_legible,  
                        line=dict(color=colores_filtrados[i], width=4),
                        marker=dict(symbol='circle')
        ))

                # Actualizar los detalles del gráfico
                fig.update_layout(
                    title={
                        'text': f'{tipo_afiliado} por Edad a lo Largo del Tiempo',
                        'y':0.9, 
                        'x':0.5,  
                        'xanchor': 'center',
                        'yanchor': 'top'
                    },
                    xaxis_title='Año',
                    yaxis_title=f'Cantidad de {tipo_afiliado}',
                    legend_title='Edad',
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=-0.17,
                        xanchor="center",
                        x=0.5,
                        font=dict(size=14)
                    ),
                    height=800  
                )

                fig.update_xaxes(showline=True, linewidth=2, linecolor='gray', gridcolor='lightgray')
                fig.update_yaxes(showline=True, linewidth=2, linecolor='gray', gridcolor='lightgray')

                # Mostrar el gráfico en Streamlit
                st.plotly_chart(fig, use_container_width=True)


            with tab2:

                texto = "Esta seccion estaremos analizando diversos indicadores y factores respecto a la empleomania de la Republica Dominicana. Estos factores de vital importancia para nuestro analisi ya influyen directamente en el aumentos de afiliados y los ingresos al SFS "
                st.info(texto)


                explicaciones = {
                    "PET": "Constituye la oferta disponible de fuerza de trabajo de la economía naciona",
                    "PEA": " también se llama fuerza laboral y son las personas en edad de trabajar, que trabajan o están buscando empleo.",
                    "NO PET": "Constituye la oferta disponible de fuerza de trabajo de la economía naciona",
                    "Ocupados": "Constituye a las peronas con empleo",
                    "Cesantes": "Personas sin empleo pero que estan buscando empleo",
                    "Desocupados abiertos": "Constituye a las peronas sin con empleo",
                    "Nuevos": "Personas que ingresan al mercado laboral por primera vez",
                    "Inactivos": "Personas que no están trabajando ni buscando activamente empleo",
                    "Tasa Global de Participación3": "Proporción de la población en edad de trabajar que está en la fuerza laboral",
                    "   Tasa de Ocupación  ": "Porcentaje de la fuerza laboral que está empleada",
                    "Tasa desocupación abierta4": "Proporción de la fuerza laboral activa que busca empleo sin éxito",
                    "Tasa de cesantía": "Proporción de la población económicamente activa que está desempleada",
                    "Tasa de nuevos": "Proporción de personas que ingresaron al mercado laboral en un período específico",
                    "Tasa inactividad": "Proporción de la población en edad de trabajar que no está en la fuerza laboral."

                }

            

                df_agrupado = Indicador.groupby(['Indicadores', 'Tipo'])['Poblacion'].sum().reset_index()

                selected_indicadores = st.multiselect(
                    'Favor seleccionar un indicador ', 
                    options=df_agrupado['Indicadores'].unique(), 
                    default=df_agrupado['Indicadores'].unique()
                )

                # Filtrar el DataFrame basado en la selección
                if 'Todos' in selected_indicadores or len(selected_indicadores) == len(df_agrupado['Indicadores'].unique()):
                    filtered_df = df_agrupado
                else:
                    filtered_df = df_agrupado[df_agrupado['Indicadores'].isin(selected_indicadores)]
                
                col1, col2 = st.columns([1, 1,])

                with col1:

                    fig = go.Figure()

                    for indicador in filtered_df['Indicadores'].unique():
                        df_temporal = filtered_df[filtered_df['Indicadores'] == indicador]
                        fig.add_trace(go.Scatter(
                            x=df_temporal['Tipo'], 
                            y=df_temporal['Poblacion'],
                            mode='lines+markers',
                            name=indicador
                        ))

                    # Mostrar el gráfico en Streamlit
                    st.plotly_chart(fig, use_container_width=True)

                with col2:
                    st.write("Definicioin de cada indicador")
                    
                    if selected_indicadores:
                        for indicador in selected_indicadores:
                            
                            if indicador in explicaciones:
                                
                                st.write(f"**{indicador}:** {explicaciones[indicador]}")
                            else:
                                
                                st.write(f"**{indicador}:** No hay explicación disponible.")

                col1, col2 = st.columns([1, 1])

                df_agrupado = educacion.groupby(['Año', 'Nivel educativo'])['Poblacion'].sum().reset_index()

                with col1:
                    fig = px.bar(
                        df_agrupado,
                        x='Año',
                        y='Poblacion',
                        color='Nivel educativo',  
                        title='Distribución de empleados por nivel educativo',
                        labels={'Poblacion': 'Población', 'Año': 'Año', 'Nivel educativo': 'Nivel Educativo'},
                        barmode='group'
                    )

                    fig.update_layout(
                        xaxis_title='Año',
                        yaxis_title='Población',
                        legend_title='Nivel Educativo',
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                    )

                    fig.update_xaxes(tickangle=45)
                    st.plotly_chart(fig, use_container_width=True)

            #Media de ingresos
                with col2:

                    df_agrupado = Eda.groupby(['Año', 'Edad'])['Poblacion'].sum().reset_index()
                    fig = px.bar(
                    df_agrupado,
                    x='Año',
                    y='Poblacion',
                    color='Edad',  
                    title='Distribución de empleados por edad',
                    labels={'Poblacion': 'Población', 'Año': 'Año'},
                    barmode='group'
                )

                    fig.update_layout(
                        xaxis_title='Año',
                        yaxis_title='Población',
                        legend_title='Nivel Educativo',
                        plot_bgcolor='rgba(0,0,0,0)', 
                        paper_bgcolor='rgba(0,0,0,0)', 
                    )

                    fig.update_xaxes(tickangle=45)

                    st.plotly_chart(fig, use_container_width=True)



        #Ingresos por poblacion 
                Sector = Sector.groupby(['Años', 'Tipo'])['Poblacion'].mean().reset_index()
                sec = sec.groupby(['Años', 'Tipo'])['Poblacion'].mean().reset_index()

                data_source = st.radio("Selecciona Sector empleador y tipo de empleo", ('Sector Empleador', 'Tipo de Empleo'))
                df_selected = Sector if data_source == 'Sector Empleador' else sec
                col1, col2 = st.columns([1, 1])

                with col1:
                    fig = px.bar(
                    df_selected,
                    x='Años',
                    y='Poblacion',
                    color='Tipo', 
                    title='Distribución de ingresos por hora segun el sector de empleo',
                    labels={'Tipo': 'Población', 'Año': 'Año'},
                    barmode='group'
                )

                    fig.update_layout(
                        xaxis_title='Año',
                        yaxis_title='Población',
                        legend_title='Sector Empleador',
                        plot_bgcolor='rgba(0,0,0,0)',  
                        paper_bgcolor='rgba(0,0,0,0)',  
                    )

                    fig.update_xaxes(tickangle=45)
                    st.plotly_chart(fig, use_container_width=True)

                with col2:
                    
                    df_selected['Años'] = df_selected['Años'].astype(int)
                    years = df_selected['Años'].unique()
                    min_year = min(years)
                    max_year = max(years)

                    selected_year = st.slider(
                        'Selecciona un año', 
                        min_value=min_year, 
                        max_value=max_year, 
                        value=min_year 
                    )

                    Sector_filtered = df_selected[df_selected['Años'] == selected_year]

                    st.write(f"Datos para el año {selected_year}")

                    col1, col2 = st.columns([1, 1])
                    with col1:
                        total_poblacion = 0  
                        for tipo in Sector_filtered['Tipo'].unique():
                            poblacion_tipo = Sector_filtered[Sector_filtered['Tipo'] == tipo]['Poblacion'].sum()
                            total_poblacion += poblacion_tipo  
                            st.metric(
                                label=tipo,
                                value=f"{poblacion_tipo:.1f} $"
                            )
                    with col2:
                    # Calcular la media de ingresos para el año seleccionado
                        if len(df_selected['Tipo'].unique()) > 0:  
                            media_ingresos = total_poblacion / len(df_selected['Tipo'].unique())
                            # HTML y CSS personalizado para mostrar la métrica más grande
                            html = f"""
                            <div style='text-align: center; padding: 10px;'>
                                <h1 style='color: white; margin-bottom: 0px;'>Media de ingresos</h1>
                                <h2 style='color: red; margin-top: 0px;'>{media_ingresos:.1f} $</h2>
                            </div>
                            """
                            st.markdown(html, unsafe_allow_html=True)


            with tab3:
                Fina['Fecha'] = pd.to_datetime(Fina['Periodo'], format='%Y%m', errors='coerce')
                Finanzas_2 = Fina.groupby('Fecha')[['Ingresos en Salud', 'Gasto en Salud', 'Siniestralidad','Monto per Capita']].sum().reset_index()

                texto = "En esta sección, estaremos analizando diversos factores financieros asociados al Sistema Financiero (SFS) con el fin de entender y analizar las tendencias financieras que impactan en dicho sistema"
                st.info(texto)
            
                color_map = {'Gasto en Salud': 'blue', 'Ingresos en Salud': 'yellow'}

                opcion = st.selectbox('Selecciona el tipo de agrupación temporal:', ['Mensual', 'Anual'])

                if opcion == 'Mensual':
                    # Código para el gráfico mensual
                    
                    Finanzas_2_melted = Finanzas_2.melt(id_vars=['Fecha'], value_vars=['Gasto en Salud', 'Ingresos en Salud'],
                                                        var_name='Categoría', value_name='Valor')
                    color_map = {'Gasto en Salud': 'blue', 'Ingresos en Salud': 'yellow'}
                    fig = px.line(Finanzas_2_melted, x='Fecha', y='Valor', color='Categoría',
                                color_discrete_map=color_map, markers=True, line_shape='linear',
                                title='Tendencia Temporal de Ingresos al SFS vs Gastos en Salud')

                else:
                    # Código para el gráfico anual
                    Fina['Año'] = Fina['Fecha'].dt.year  # Asegúrate de usar 'Fecha' después de que ya haya sido definida
                    Finanzas_anual = Fina.groupby('Año')[['Ingresos en Salud', 'Gasto en Salud', 'Siniestralidad', 'Monto per Capita']].sum().reset_index()
                    Finanzas_anual_melted = Finanzas_anual.melt(id_vars=['Año'], value_vars=['Gasto en Salud', 'Ingresos en Salud'],
                                                                var_name='Categoría', value_name='Valor')
                    fig = px.line(Finanzas_anual_melted, x='Año', y='Valor', color='Categoría',
                                color_discrete_map=color_map, markers=True, line_shape='linear',
                                title='Tendencia Temporal de Ingresos al SFS vs Gastos en Salud - Totales Anuales')

                # Actualizaciones comunes al gráfico
                fig.update_traces(line=dict(width=4))
                fig.update_layout(xaxis_title='Año', yaxis_title='Valor',
                                legend_title='Indicador', xaxis_tickangle=-45,
                                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))

                # Muestra el gráfico seleccionado
                st.plotly_chart(fig, use_container_width=True)


                col1, col2 = st.columns([1, 1])
                with col1:

                
                    fig = px.line(Finanzas_2, x='Fecha', y='Siniestralidad', markers=True, line_shape='linear',
                                title='Tendencia Temporal de Siniestralidad')

                
                    fig.update_traces(line=dict(width=4))
                    fig.update_layout(xaxis_title='Año', yaxis_title='Siniestralidad', 
                                    legend_title='Indicador', xaxis_tickangle=-45,
                                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                    st.plotly_chart(fig, use_container_width=True)

                with col2: 


                    Finanzas_2['Fecha'] = pd.to_datetime(Finanzas_2['Fecha']).dt.date

                    fecha_seleccionada = st.selectbox(
                        'Selecciona una fecha:',
                        options=Finanzas_2['Fecha'].unique(),
                        index=0  
                    )

                    datos_filtrados = Finanzas_2[Finanzas_2['Fecha'] == fecha_seleccionada]
                    resumen_valores = datos_filtrados[['Ingresos en Salud', 'Gasto en Salud', 'Siniestralidad','Monto per Capita']].mean()

                    def format_value(value, is_percentage=False):
                        if is_percentage:  # Formatear como porcentaje
                            return f"{value:.2f}%"
                        elif value >= 1000000:  # Más de un millón
                            return f"${value/1000000:.2f}M"
                        elif value >= 1000:  # Más de mil
                            return f"${value/1000:.2f}K"
                        else:
                            return f"${value:.2f}"

                    # Usar markdown para hacer los cards más grandes
                    st.markdown(f"""
                        <style>
                            .big-font {{
                                font-size:30px !important;
                                margin-bottom: 1rem;
                            }}
                            .container {{
                                display: flex;
                                justify-content: space-around;
                                flex-wrap: wrap;
                                gap: 20px;
                                padding: 20px;
                            }}
                            .item {{
                                flex: 1;
                                min-width: 150px; /* Ajusta este valor como lo necesites para evitar que los elementos sean demasiado estrechos */
                            }}
                        </style>
                        <div class="container">
                            <div class="big-font item"><strong>Ingresos en Salud:</strong> {format_value(resumen_valores['Ingresos en Salud'])}</div>
                            <div class="big-font item"><strong>Gasto en Salud:</strong> {format_value(resumen_valores['Gasto en Salud'])}</div>
                            <div class="big-font item"><strong>Siniestralidad:</strong> {format_value(resumen_valores['Siniestralidad'], is_percentage=True)}</div>
                        </div>
                        <div class="container" style="padding-top: 0;">
                            <div class="big-font item"><strong>Monto per Capita:</strong> {format_value(resumen_valores['Monto per Capita'])}</div>
                        </div>
                    """, unsafe_allow_html=True)

                Fina['Fecha'] = pd.to_datetime(Fina['Periodo'], format='%Y%m')
                Fina['Año'] = Fina['Fecha'].dt.year

            
                datos_agrupados = Fina.groupby('Año')[['TitularesM', 'Dependientes DirectosM', 'Dependientes AdicionalesM']].sum().reset_index()

                datos_melted = datos_agrupados.melt(id_vars=['Año'], value_vars=['TitularesM', 'Dependientes DirectosM', 'Dependientes AdicionalesM'],
                                                    var_name='Categoría', value_name='Valor')
                
                datos_agrupados1 = Fina.groupby('Año')[['Titulares', 'Dependientes Directos', 'Dependientes Adicionales']].sum().reset_index()

                datos_melted1 = datos_agrupados1.melt(id_vars=['Año'], value_vars=['Titulares', 'Dependientes Directos', 'Dependientes Adicionales'],
                                                    var_name='Categoría', value_name='Valor')
                
                # Crea el selector para las gráficas
                opcion_grafica = st.selectbox(
                    'Seleccionae entre Total de Capitas dispersadas y Total de Montos dispersados',
                    ('Montos distribuidos a las ARS por Año', 'Total de capitas distribuidos a las ARS por Año')
                )

                # Dependiendo de la selección, muestra la gráfica correspondiente
                if opcion_grafica == 'Montos distribuidos a las ARS por Año':
                    # Tus cálculos previos para datos_melted aquí...
                    
                    # Gráfica
                    fig = px.bar(datos_melted, x='Año', y='Valor', color='Categoría', barmode='group', title='Montos distribuidos a las ARS por Año')
                    st.plotly_chart(fig, use_container_width=True)
                elif opcion_grafica == 'Total de capitas distribuidos a las ARS por Año':
                    # Tus cálculos previos para datos_melted1 aquí...
                    
                    # Gráfica
                    fig1 = px.bar(datos_melted1, x='Año', y='Valor', color='Categoría', barmode='group', title='Total de capitas distribuidos a las ARS por Año')
                    st.plotly_chart(fig1, use_container_width=True)


                col1, col2 = st.columns([1, 1])

                with col1:
                #Inflacion en Salud
                    
                    ipc['Fecha'] = pd.to_datetime(ipc['Fecha'], format='%Y%m', errors='coerce')
                    ipc2 = ipc.groupby('Fecha')[['Salud']].sum().reset_index()

                    # Crear el gráfico de líneas con Plotly Express para mostrar 'Ingresos en Salud', 'Gasto en Salud'
                    fig = px.line(ipc2, x='Fecha', y='Salud',
                                markers=True, line_shape='linear',
                                title='Tendencia Temporal de inflacion en salud')

                    fig.update_traces(line=dict(width=4))
                    fig.update_layout(xaxis_title='Año', yaxis_title='Valor', 
                                    legend_title='Indicador', xaxis_tickangle=-45,
                                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))

                    st.plotly_chart(fig, use_container_width=True)
                    

                with col2:
                    

                    Finanzas_3 = Fina[Fina["Fecha"].dt.month == 12]

                    # Crear el gráfico de líneas con Plotly Express para mostrar 'Ingresos en Salud', 'Gasto en Salud'
                    fig = px.line(Finanzas_3, x='Fecha', y='Monto per Capita', markers=True, line_shape='linear',
                                title='Tendencia del valor del monto per capita')

                    # Actualizar el layout para líneas más gruesas
                    fig.update_traces(line=dict(width=4))
                    fig.update_layout(xaxis_title='Año', yaxis_title='Monto per Capita', 
                                    legend_title='Indicador', xaxis_tickangle=-45,
                                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                    st.plotly_chart(fig, use_container_width=True)


                # Limpiar espacios adicionales en los nombres de las columnas
                pres.columns = [col.strip() for col in pres.columns]

                # Limpiar espacios adicionales en los nombres de las columnas
                pres['Fecha'] = pres['Fecha'].astype(str)

                # Obtener la lista de valores únicos de la columna 'Fecha', que representan años
                valores_unicos_fecha = pres['Fecha'].unique()
                valores_unicos_fecha.sort()

                # Selector de fecha/año
                fecha_seleccionada = st.selectbox(
                    'Selecciona una fecha:',
                    options=valores_unicos_fecha
                )

                # Filtrar por la fecha seleccionada
                datos_filtrados = pres[pres['Fecha'] == fecha_seleccionada]

                # Divide la pantalla en dos columnas para los gráficos de pastel
                col1, col2 = st.columns(2)

                # Gráfico de pastel para 'Porcentaje Capita'
                with col1:
                    fig_capita = px.pie(datos_filtrados, values='Porcentaje Capita', names='Descripcion',
                                        title=f'Distribución de Capita - {fecha_seleccionada}')
                    st.plotly_chart(fig_capita, use_container_width=True)

                # Gráfico de pastel para 'Porcentaje Monto'
                with col2:
                    fig_monto = px.pie(datos_filtrados, values='Porcentaje Monto', names='Descripcion',
                                    title=f'Distribución de Monto - {fecha_seleccionada}')
                    st.plotly_chart(fig_monto, use_container_width=True)

        folder_path = "C:/Users/franc/OneDrive - INTEC/Escritorio/Proyecto ARS/Dashboards/BD FINAL" 
        data_frames = load_data(folder_path)
        prepare_dataframes(data_frames)
            
        if 'df_unificado' in data_frames and 'educacion' in data_frames:
            display_data_visualizations(data_frames['df_unificado'], data_frames['educacion'],data_frames['Sector'],data_frames['edad'],data_frames['lista_Afiliados_1'], data_frames['Indicador'],data_frames['Eda'],
                                        data_frames['sec'], data_frames['Fina'],data_frames['ipc'], data_frames['pres']) 
if __name__ == "__main__":
    main()