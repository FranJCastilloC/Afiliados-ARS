import streamlit as st
from analisis_exploratorio import modulo_analisis_exploratorio
from intro import modulo_overview
from corr import modulo_corr
from modelo import modulo_modelo

def main():
    # Configura la página para utilizar todo el ancho disponible
    st.set_page_config(layout="wide")

    # Creación del menú lateral para la navegación
    st.sidebar.title("Menú")
    opcion = st.sidebar.radio(
        "Selecciona un módulo",
        ("Análisis Exploratorio de Datos", "Correlación de Datos", "Modelo", "Base de datos")
    )

    # Lógica para mostrar el módulo seleccionado
    if opcion == "Análisis Exploratorio de Datos":
        modulo_analisis_exploratorio()
    elif opcion == "Correlación de Datos":
        modulo_corr()
    elif opcion == "Modelo":
        modulo_modelo()
    elif opcion == "Base de datos":
        modulo_overview()

if __name__ == "__main__":
    main()