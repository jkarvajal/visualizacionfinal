# app.py

import os
import pandas as pd
import numpy as np

from dash import Dash, html, dcc
import plotly.express as px
import plotly.graph_objects as go


# ============================================================
# 1. Carga y limpieza de datos
# ============================================================

def cargar_y_preparar_datos():
    # Ruta segura al CSV (carpeta data/)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, "data", "Student_Mental_health.csv")

    # Cargar el CSV
    df = pd.read_csv(csv_path)

    # Estandarizar nombres de columnas
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("-", "_")
    )

    # Limpieza básica de nulos
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].fillna("Nulo")
        else:
            if df[col].notna().any():
                df[col] = df[col].fillna(df[col].median())

    # Eliminar duplicados
    df = df.drop_duplicates()

    # Diccionario de normalización de cursos (usar todo en minúsculas)
    course_map = {
        "bcs": "Computer Science",
        "bit": "Information Technology",
        "engine": "Engineering",
        "engin": "Engineering",
        "engineering": "Engineering",
        "mhsc": "Health Sciences",
        "biomedical science": "Biomedical Science",
        "koe": "Education",
        "benl": "English",
        "ala": "Arts and Letters",
        "psychology": "Psychology",
        "irkhs": "Islamic Studies",
        "kirkhs": "Islamic Studies",
        "kirkhs ": "Islamic Studies",
        "islamic education": "Islamic Education",
        "pendidikan islam": "Islamic Education",
        "fiqh": "Islamic Jurisprudence",
        "fiqh fatwa": "Islamic Jurisprudence",
        "nursing": "Nursing",
        "diploma nursing": "Nursing",
        "marine science": "Marine Science",
        "banking studies": "Banking Studies",
        "mathemathics": "Mathematics",
        "communication": "Communication",
        "cts": "Computer Technology",
    }

    # Normalizar columna "what_is_your_course?"
    if "what_is_your_course?" in df.columns:
        df["what_is_your_course?"] = (
            df["what_is_your_course?"]
            .astype(str)
            .str.strip()
            .str.lower()
            .replace(course_map)
        )

    # Normalizar columna "your_current_year_of_study"
    if "your_current_year_of_study" in df.columns:
        df["your_current_year_of_study"] = (
            df["your_current_year_of_study"]
            .astype(str)
            .str.lower()
            .str.replace("year", "")
            .str.strip()
            .replace({"1": "Year 1", "2": "Year 2", "3": "Year 3", "4": "Year 4"})
        )

    # Normalizar columna "what_is_your_cgpa?"
    if "what_is_your_cgpa?" in df.columns:
        df["what_is_your_cgpa?"] = df["what_is_your_cgpa?"].astype(str).str.strip()
        df["what_is_your_cgpa?"] = df["what_is_your_cgpa?"].replace({
            "3.50 - 4.00": "3.50 - 4.00",
            "3.50-4.00": "3.50 - 4.00"
        })

    # Convertir columnas Yes/No a 1/0
    yn_cols = [
        "marital_status",
        "do_you_have_depression?",
        "do_you_have_anxiety?",
        "do_you_have_panic_attack?",
        "did_you_seek_any_specialist_for_a_treatment?"
    ]

    for col in yn_cols:
        if col in df.columns:
            df[col] = df[col].map({"Yes": 1, "No": 0})

    # Convertir timestamp a datetime
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    # Normalizar edad
    if "age" in df.columns:
        df["age"] = df["age"].astype(float).astype(int)

    # Traducir nombres de columnas
    column_translate = {
        "timestamp": "fecha_registro",
        "choose_your_gender": "genero",
        "age": "edad",
        "what_is_your_course?": "programa_academico",
        "your_current_year_of_study": "año_estudio",
        "what_is_your_cgpa?": "Promedio_de_calificaciones",
        "marital_status": "estado_civil",
        "do_you_have_depression?": "tiene_depresion",
        "do_you_have_anxiety?": "tiene_ansiedad",
        "do_you_have_panic_attack?": "tiene_ataques_panico",
        "did_you_seek_any_specialist_for_a_treatment?": "busco_tratamiento_especialista"
    }

    df = df.rename(columns=column_translate)

    return df


df = cargar_y_preparar_datos()


# ============================================================
# 2. Figuras con Plotly
# ============================================================

def fig_distribucion_depresion(df):
    counts = df["tiene_depresion"].value_counts().sort_index()
    labels = ["No (0)", "Sí (1)"]
    values = [counts.get(0, 0), counts.get(1, 0)]

    fig = px.bar(
        x=labels,
        y=values,
        labels={"x": "Respuesta", "y": "Cantidad de estudiantes"},
        title="Distribución de estudiantes con síntomas de depresión"
    )

    fig.update_traces(text=values, textposition="outside")
    fig.update_layout(yaxis=dict(title="Cantidad"), xaxis=dict(title="Respuesta"))
    return fig


def fig_tendencia_sintomas(df):
    counts = pd.Series({
        "Depresión (1)": df["tiene_depresion"].sum(),
        "Ansiedad (1)": df["tiene_ansiedad"].sum(),
        "Pánico (1)": df["tiene_ataques_panico"].sum()
    })

    fig = px.line(
        x=counts.index,
        y=counts.values,
        markers=True,
        labels={"x": "Síntoma", "y": "Número de estudiantes"},
        title="Tendencia de síntomas emocionales en estudiantes"
    )
    fig.update_traces(line=dict(width=3))
    return fig


def fig_mapa_calor_corr(df):
    cols_binarias = [
        "tiene_depresion",
        "tiene_ansiedad",
        "tiene_ataques_panico",
        "busco_tratamiento_especialista"
    ]
    numeric_cols = ["edad"] + cols_binarias
    numeric_cols = [c for c in numeric_cols if c in df.columns]

    corr = df[numeric_cols].corr()

    fig = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale="Oranges",
        title="Mapa de calor de correlaciones entre síntomas emocionales y edad"
    )
    fig.update_xaxes(side="top")
    return fig


def fig_programa_depresion(df):
    if "programa_academico" not in df.columns:
        return go.Figure()

    program_counts = df.groupby("programa_academico")["tiene_depresion"].sum().sort_values(ascending=False)

    fig = px.bar(
        x=program_counts.index,
        y=program_counts.values,
        labels={"x": "Programa académico", "y": "Cantidad de estudiantes con depresión"},
        title="Relación entre Programa Académico y Síntoma de Depresión"
    )
    fig.update_layout(xaxis_tickangle= -45)
    return fig


def fig_cgpa_depresion(df):
    if "Promedio_de_calificaciones" not in df.columns:
        return go.Figure()

    serie = df.groupby("Promedio_de_calificaciones")["tiene_depresion"].sum()

    fig = px.bar(
        x=serie.index,
        y=serie.values,
        labels={"x": "Promedio de calificaciones", "y": "Estudiantes con depresión"},
        title="Relación entre Promedio de Calificaciones y Depresión"
    )
    return fig


def fig_estado_civil_ansiedad(df):
    if "estado_civil" not in df.columns:
        return go.Figure()

    serie = df.groupby("estado_civil")["tiene_ansiedad"].sum()

    fig = px.bar(
        x=serie.index,
        y=serie.values,
        labels={"x": "Estado civil", "y": "Estudiantes con ansiedad"},
        title="Relación entre Estado Civil y Ansiedad"
    )
    return fig


def fig_genero_panico(df):
    if "genero" not in df.columns:
        return go.Figure()

    serie = df.groupby("genero")["tiene_ataques_panico"].sum()

    fig = px.bar(
        x=serie.index,
        y=serie.values,
        labels={"x": "Género", "y": "Estudiantes con ataques de pánico"},
        title="Relación entre Género y ataques de pánico"
    )
    return fig


# ============================================================
# 3. App Dash
# ============================================================

app = Dash(__name__)

app.layout = html.Div([
    html.H1("Análisis de Salud Mental en Estudiantes", style={"textAlign": "center"}),

    dcc.Tabs([
        dcc.Tab(label="Distribución de Depresión", children=[
            html.Br(),
            dcc.Graph(figure=fig_distribucion_depresion(df)),
            html.P(
                "Este gráfico muestra cuántos estudiantes reportan síntomas de depresión "
                "y permite dimensionar la magnitud del problema en la población analizada."
            )
        ]),

        dcc.Tab(label="Tendencia de Síntomas Emocionales", children=[
            html.Br(),
            dcc.Graph(figure=fig_tendencia_sintomas(df)),
            html.P(
                "Aquí se observa la cantidad de estudiantes que presentan depresión, ansiedad "
                "y ataques de pánico, lo que ayuda a identificar qué síntomas son más frecuentes."
            )
        ]),

        dcc.Tab(label="Correlaciones y Edad", children=[
            html.Br(),
            dcc.Graph(figure=fig_mapa_calor_corr(df)),
            html.P(
                "El mapa de calor muestra cómo se relacionan la edad y los distintos síntomas emocionales, "
                "incluyendo la búsqueda de tratamiento especializado."
            )
        ]),

        dcc.Tab(label="Programa vs Depresión", children=[
            html.Br(),
            dcc.Graph(figure=fig_programa_depresion(df)),
            html.P(
                "Este gráfico permite explorar en qué programas académicos se concentran más casos "
                "de estudiantes que reportan síntomas de depresión."
            )
        ]),

        dcc.Tab(label="Calificaciones y Estado Civil", children=[
            html.Br(),
            dcc.Graph(figure=fig_cgpa_depresion(df)),
            html.Br(),
            dcc.Graph(figure=fig_estado_civil_ansiedad(df)),
            html.P(
                "Se exploran posibles patrones entre el rendimiento académico, el estado civil "
                "y la presencia de síntomas emocionales como depresión y ansiedad."
            )
        ]),

        dcc.Tab(label="Género y Ataques de Pánico", children=[
            html.Br(),
            dcc.Graph(figure=fig_genero_panico(df)),
            html.P(
                "Este gráfico contrasta el género de los estudiantes con la frecuencia de ataques de pánico reportados."
            )
        ]),
    ])
])


#if __name__ == "__main__":
#    # Ejecución local
#    app.run(host="0.0.0.0", port=8050, debug=True)
if __name__ == "__main__":
    import os

    # Render define la variable de entorno PORT.
    # En local usamos 8050 por defecto.
    port = int(os.environ.get("PORT", 8050))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=True  # Si quieres, luego lo pones en False para producción
    )