import streamlit as st
from analisis_exploratorio import modulo_analisis_exploratorio
from intro import modulo_overview
from corr import modulo_corr
from modelo import modulo_modelo

# Configura la página para utilizar todo el ancho disponible
st.set_page_config(layout="wide")

def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

load_css('c:/Users/franc/OneDrive - INTEC/Escritorio/Proyecto ARS/Dashboards/Assets/style.css')


# Creación del menú lateral para la navegación
st.sidebar.title("Menú")
opcion = st.sidebar.radio(
    "Selecciona un módulo",
    ("Análisis Exploratorio de Datos", "Correlacion de Datos", "Modelo", "Base de datos")
)

# Lógica para mostrar el módulo seleccionado
if opcion == "Análisis Exploratorio de Datos":
    modulo_analisis_exploratorio()
elif opcion == "Correlacion de Datos":
    modulo_corr()
elif opcion == "Modelo":
    modulo_modelo()
elif opcion == "Base de datos":
    modulo_overview()
