# -*- coding: utf-8 -*-
"""
Despliegue del modelo Random Forest con Streamlit
Ejecutar con:  streamlit run app.py
"""

import pickle
import pandas as pd
import streamlit as st


# ------------------------------------------------------------------
# 1. Cargar el modelo (una sola vez, sin mostrarlo en la app)
# ------------------------------------------------------------------
@st.cache_resource
def cargar_modelo(ruta="modelo-cla.pkl"):
    with open(ruta, "rb") as f:
        return pickle.load(f)

modelo = cargar_modelo()
clf = modelo[0]                    # RandomForestClassifier
label_encoder = modelo[1]          # LabelEncoder de la variable objetivo
columnas_originales = list(modelo[2])
scaler = modelo[3]                 # MinMaxScaler
columnas_modelo = list(clf.feature_names_in_)


# ------------------------------------------------------------------
# 2. Interfaz gráfica
# ------------------------------------------------------------------
st.title("Estimar anticipadamente cuántas personas que diligencian "
         "el formulario terminarán pagando")

programas = [
    'Esp en Psiquiatria-Med', 'Esp Medicina Interna-Med', 'Esp Med Dol y Cui Pal-Med',
    'Esp Dermatologia-Med', 'Esp Radiologia e Imag Diag-Med', 'Esp en Oftalmologia-Med',
    'Esp Ortopedia y Traumatol-Med', 'Especializac. Anestesiolog-Med',
    'Esp Ginecolo y Obstetricia-Med', 'Esp Cirugia General-Med', 'Esp en Pediatria-Med',
    'Esp Med Critic Cuid Intens-Med', 'Esp en Cardiologia Adultos-Med',
    'Esp. Med Activ Fis y Depor-Med', 'Esp Medicina Materno Fetal-Med',
    'Esp Cirugia Cardiovascular-Med', 'Esp. En Reumatologia-Med', 'Esp en Endocrinologia-Med',
    'Esp. en Neonatologia-Med', 'Esp en Infectologia-Med', 'Esp Anestesia Cardiovascul-Med',
    'Esp en Nefrologia-Med', 'Esp en Ecocardiografia-Med', 'Esp Cardiol Interv y Hemod-Med',
    'Esp Cardiologia Pediatrica-Med', 'Esp.Psiquiatria de Enlace-Med',
    'Esp.Ortop y Traumat Pediat-Med', 'Esp. en Cirugia de Mano-Med',
]

Edad = st.slider("Edad", min_value=18, max_value=70, value=20, step=1)
Genero = st.selectbox("Genero", ["M", "F"])
Naturaleza = st.selectbox("Naturaleza Origen Universidad", ["Publica", "Privada", "Otro"])
Tipo_Documento = st.selectbox("Tipo Documento", ["CC", "CE", "PA", "DE", "TI"])
Programa = st.selectbox("Programa", programas)


# ------------------------------------------------------------------
# 3. Preparación de datos
# ------------------------------------------------------------------
def preparar_datos():
    data = pd.DataFrame([{
        "Edad": Edad,
        "Genero": Genero,
        "Naturaleza Origen Universidad": Naturaleza,
        "Tipo Documento": Tipo_Documento,
        "Programa Admision": Programa,
        # Valores por defecto: deben escribirse EXACTAMENTE como en el entrenamiento
        "Periodo": 202401,
        "Presentan examen": 0,
        "Pais Nacimiento": "Colombia",
        "Departamento Nacimiento": "Bogota D.C.",
        "Ciudad Residencia": "Bogota D.C.",
    }])
    data = data[columnas_originales]

    categoricas = [
        "Periodo", "Programa Admision", "Genero", "Pais Nacimiento", "Tipo Documento",
        "Departamento Nacimiento", "Ciudad Residencia", "Naturaleza Origen Universidad",
    ]
    dummies = pd.get_dummies(data, columns=categoricas, drop_first=False, dtype=int)

    # Columnas dummy generadas que el modelo NO conoce (valor mal escrito o nuevo)
    desconocidas = [c for c in dummies.columns if c not in columnas_modelo]

    # Alinear con las columnas del modelo
    dummies = dummies.reindex(columns=columnas_modelo, fill_value=0).astype(float)

    # Aplicar el MinMaxScaler SOLO si fue entrenado (fit) antes de guardarlo
    if hasattr(scaler, "data_min_"):
        n_cols = len(scaler.data_min_)
        if hasattr(scaler, "feature_names_in_"):
            cols = list(scaler.feature_names_in_)
        elif n_cols == len(columnas_modelo):
            cols = columnas_modelo
        elif n_cols == 1:
            cols = ["Edad"]
        else:
            st.error(f"No se pudo determinar qué columnas escaló el MinMaxScaler "
                     f"(espera {n_cols} columnas). Revisa el notebook de entrenamiento.")
            st.stop()
        dummies[cols] = scaler.transform(dummies[cols])

    return data, dummies, desconocidas


# ------------------------------------------------------------------
# 4. Predicción
# ------------------------------------------------------------------
if st.button("Predecir"):
    data, X, desconocidas = preparar_datos()

    y_pred = clf.predict(X)
    etiqueta = label_encoder.inverse_transform(y_pred)[0]
    probabilidades = clf.predict_proba(X)[0]

    st.success(f"Predicción: **{etiqueta}**")

    prob_df = pd.DataFrame({
        "Clase": label_encoder.inverse_transform(clf.classes_),
        "Probabilidad": probabilidades,
    })
    st.dataframe(prob_df, hide_index=True)

    if desconocidas:
        st.warning("Estos valores no existían en el entrenamiento y se ignoraron: "
                   + ", ".join(desconocidas))

    with st.expander("Ver datos ingresados"):
        st.dataframe(data, hide_index=True)

st.info("Recuerda que el modelo es un clasificador: su desempeño se mide con "
        "métricas como accuracy, precisión, recall o F1.")
