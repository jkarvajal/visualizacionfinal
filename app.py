import pandas as pd
import plotly.express as px
from dash import Dash, dcc, html
import os

# --------------------------------------------------------
# Cargar datos
# --------------------------------------------------------
df = pd.read_csv("data/Student Mental health.csv")

df = df.rename(columns={
    'Choose your gender': 'gender',
    'Age': 'age',
    'What is your course?': 'course',
    'Your current year of Study': 'year_study',
    'What is your CGPA?': 'cgpa',
    'Marital status': 'marital_status',
    'Do you have Depression?': 'depression',
    'Do you have Anxiety?': 'anxiety',
    'Do you have Panic attack?': 'panic',
    'Did you seek any specialist for a treatment?': 'seek_treatment'
})

map_yn = {'Yes': 1, 'No': 0, 'YES': 1, 'NO': 0}

for col in ['depression', 'anxiety', 'panic', 'seek_treatment']:
    df[col + '_bin'] = df[col].map(map_yn)

df['any_issue'] = df[['depression_bin', 'anxiety_bin', 'panic_bin']].max(axis=1)
mask_issue = df['any_issue'] == 1

# --------------------------------------------------------
# Gráficas
# --------------------------------------------------------
prev_genero = df.groupby('gender')['any_issue'].mean().reset_index()
fig1 = px.bar(prev_genero, x='gender', y='any_issue',
              title="Prevalencia de problemas por género")
fig1.update_yaxes(tickformat=".0%")

prev_cgpa = df.groupby('cgpa')['any_issue'].mean().reset_index()
fig2 = px.bar(prev_cgpa, x='cgpa', y='any_issue',
              title="Salud mental según CGPA")
fig2.update_yaxes(tickformat=".0%")

help_df = df[mask_issue].groupby('seek_treatment')['any_issue'].count().reset_index()
fig3 = px.pie(help_df, names='seek_treatment', values='any_issue',
              title="¿Buscaron ayuda profesional?")

help_gender = df[mask_issue].groupby(['gender', 'seek_treatment'])['any_issue'].count().reset_index()
fig4 = px.bar(help_gender, x='gender', y='any_issue', color='seek_treatment',
              barmode='group',
              title="Búsqueda de ayuda por género")

# --------------------------------------------------------
# Dash App
# --------------------------------------------------------
app = Dash(__name__)

app.layout = html.Div([
    html.H1("Dashboard Salud Mental Estudiantes"),
    dcc.Tabs([
        dcc.Tab(label="Por género", children=[dcc.Graph(figure=fig1)]),
        dcc.Tab(label="Por CGPA", children=[dcc.Graph(figure=fig2)]),
        dcc.Tab(label="Búsqueda de ayuda", children=[dcc.Graph(figure=fig3)]),
        dcc.Tab(label="Ayuda por género", children=[dcc.Graph(figure=fig4)]),
    ])
])

# --------------------------------------------------------
# Iniciar servidor en Render
# --------------------------------------------------------
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8050))
    app.run(host="0.0.0.0", port=port, debug=False)
