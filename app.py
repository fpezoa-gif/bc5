# ============================================================
# CABECERA
# ============================================================
# Alumno: Felipe Pezoa
# URL Streamlit Cloud: https://...streamlit.app
# URL GitHub: https://github.com/...

# ============================================================
# IMPORTS
# ============================================================
# Streamlit: framework para crear la interfaz web
# pandas: manipulación de datos tabulares
# plotly: generación de gráficos interactivos
# openai: cliente para comunicarse con la API de OpenAI
# json: para parsear la respuesta del LLM (que llega como texto JSON)
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from openai import OpenAI
import json

# ============================================================
# CONSTANTES
# ============================================================
# Modelo de OpenAI. No lo cambies.
MODEL = "gpt-4.1-mini"

# -------------------------------------------------------
# >>> SYSTEM PROMPT — TU TRABAJO PRINCIPAL ESTÁ AQUÍ <<<
# -------------------------------------------------------
# El system prompt es el conjunto de instrucciones que recibe el LLM
# ANTES de la pregunta del usuario. Define cómo se comporta el modelo:
# qué sabe, qué formato debe usar, y qué hacer con preguntas inesperadas.
#
# Puedes usar estos placeholders entre llaves — se rellenan automáticamente
# con información real del dataset cuando la app arranca:
#   {fecha_min}             → primera fecha del dataset
#   {fecha_max}             → última fecha del dataset
#   {plataformas}           → lista de plataformas (Android, iOS, etc.)
#   {reason_start_values}   → valores posibles de reason_start
#   {reason_end_values}     → valores posibles de reason_end
#
# IMPORTANTE: como el prompt usa llaves para los placeholders,
# si necesitas escribir llaves literales en el texto (por ejemplo para
# mostrar un JSON de ejemplo), usa doble llave: {{ y }}
#
SYSTEM_PROMPT = """
Eres un Analista de Datos Senior en Spotify.
Tu objetivo es generar código Python para analizar un DataFrame llamado `df` y devolver una visualización clara en formato JSON.

CONTEXTO DEL DATASET:
- Rango de datos: desde {fecha_min} hasta {fecha_max}.
- Plataformas: {plataformas}.
- Semestres: {semester_values}.
- Estaciones: {season_values}.
- Tramos horarios: {day_part_values}.

ESTRUCTURA DE `df`:
- `track`, `artist`, `album`: Identificadores de la música.
- `minutes`: Tiempo escuchado en minutos (métrica principal).
- `ts`: Datetime original.
- `month_name`, `year_month`, `hour`, `day_name`, `is_weekend`, `semester`, `season`, `day_part`: Dimensiones temporales.
- `skipped`, `shuffle`: Booleanos de comportamiento.

REGLAS DE NEGOCIO:
1. "Más escuchado" siempre suma la columna `minutes`.
2. "Canciones nuevas": Cuenta canciones cuya primera aparición (`ts.min()`) ocurre en el periodo consultado.
3. Estaciones: Verano = `season == 'Summer'`, Invierno = `season == 'Winter'`.
4. Fin de semana: `is_weekend == True`.
5. Evolución temporal: Usa `year_month` para el eje X para asegurar orden cronológico.
6. Rankings: Por defecto usa Top 10 y ordena de mayor a menor.

REGLAS DE SALIDA:
Responde ÚNICAMENTE un JSON con:
{{
    "tipo": "grafico" | "fuera_de_alcance",
    "codigo": "Código Python que cree la variable `fig` con px. No uses st.plotly_chart.",
    "interpretacion": "Resumen ejecutivo en español (evita excederte más alla de 2 lineas)."
}}
"""


# ============================================================
# CARGA Y PREPARACIÓN DE DATOS
# ============================================================
# Esta función se ejecuta UNA SOLA VEZ gracias a @st.cache_data.
# Lee el fichero JSON y prepara el DataFrame para que el código
# que genere el LLM sea lo más simple posible.
#
@st.cache_data
def load_data():
    df = pd.read_json("streaming_history.json")

    # ----------------------------------------------------------
    # >>> TU PREPARACIÓN DE DATOS ESTÁ AQUÍ <<<
    # ----------------------------------------------------------
    # Transforma el dataset para facilitar el trabajo del LLM.
    # Lo que hagas aquí determina qué columnas tendrá `df`,
    # y tu system prompt debe describir exactamente esas columnas.
    #
    # Cosas que podrías considerar:
    # - Convertir 'ts' de string a datetime
    # - Crear columnas derivadas (hora, día de la semana, mes...)
    # - Convertir milisegundos a unidades más legibles
    # - Renombrar columnas largas para simplificar el código generado
    # - Filtrar registros que no aportan al análisis (podcasts, etc.)
    # ----------------------------------------------------------

    # 1. Nos quedamos solamente con valores distintos de vacíos.
    df = df[df["master_metadata_track_name"].notna()].copy()

    # 2. Renombrado de columnas y conversión de ts
    df = df.rename(columns={
        'master_metadata_track_name': 'track',
        'master_metadata_album_artist_name': 'artist',
        'master_metadata_album_album_name': 'album',
        'ms_played': 'ms'})
    df['ts'] = pd.to_datetime(df['ts'], utc=True)
    df['minutes'] = df['ms'] / 60000

    # 3. Feature engineering
    df['month_name'] = df['ts'].dt.month_name()
    df['year_month'] = df['ts'].dt.to_period("M").astype(str) # Para evolución mensual
    df['hour'] = df['ts'].dt.hour
    df['day_name'] = df['ts'].dt.day_name()
    df['is_weekend'] = df['ts'].dt.weekday >= 5
    df['semester'] = df['ts'].dt.month.apply(lambda x: '1st Semester' if x <= 6 else '2nd Semester')
    
    # Mapeo de estaciones
    season_map = {12:"Winter", 1:"Winter", 2:"Winter", 3:"Spring", 4:"Spring", 5:"Spring",
                  6:"Summer", 7:"Summer", 8:"Summer", 9:"Autumn", 10:"Autumn", 11:"Autumn"}
    df["season"] = df["ts"].dt.month.map(season_map)

    # Tramos Horarios
    df["day_part"] = pd.cut(df["hour"], bins=[0, 6, 12, 18, 24], 
                            labels=["Night", "Morning", "Afternoon", "Evening"], 
                            right=False, include_lowest=True)

    # 4. Nos quedamos solamente con regitros que fueron de más de 10 segundos (economía de "attention-span")
    df = df[df['ms'] > 10000]

    return df.sort_values("ts").reset_index(drop=True)


def build_prompt(df):
    """
    Sincroniza la realidad del dataset con el conocimiento del LLM.
    """
    # 1. Límites temporales limpios
    fecha_min = df["ts"].min().date()
    fecha_max = df["ts"].max().date()

    # 2. Valores únicos para que el LLM no "alucine" nombres de categorías
    plataformas = df["platform"].unique().tolist()
    reason_start_values = df["reason_start"].unique().tolist()
    reason_end_values = df["reason_end"].unique().tolist()
    
    # 3. Nuevas columnas
    semester_values = df["semester"].unique().tolist()
    season_values = df["season"].unique().tolist()
    day_part_values = df["day_part"].unique().tolist()

    # 4. Inyección en los placeholders del SYSTEM_PROMPT
    return SYSTEM_PROMPT.format(
        fecha_min=fecha_min,
        fecha_max=fecha_max,
        plataformas=plataformas,
        reason_start_values=reason_start_values,
        reason_end_values=reason_end_values,
        semester_values=semester_values,
        season_values=season_values,
        day_part_values=day_part_values
    )


# ============================================================
# FUNCIÓN DE LLAMADA A LA API
# ============================================================
# Esta función envía DOS mensajes a la API de OpenAI:
# 1. El system prompt (instrucciones generales para el LLM)
# 2. La pregunta del usuario
#
# El LLM devuelve texto (que debería ser un JSON válido).
# temperature=0.2 hace que las respuestas sean más predecibles.
#
# No modifiques esta función.
#
def get_response(user_msg, system_prompt):
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content


# ============================================================
# PARSING DE LA RESPUESTA
# ============================================================
# El LLM devuelve un string que debería ser un JSON con esta forma:
#
#   {"tipo": "grafico",          "codigo": "...", "interpretacion": "..."}
#   {"tipo": "fuera_de_alcance", "codigo": "",    "interpretacion": "..."}
#
# Esta función convierte ese string en un diccionario de Python.
# Si el LLM envuelve el JSON en backticks de markdown (```json...```),
# los limpia antes de parsear.
#
# No modifiques esta función.
#
def parse_response(raw):
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

    return json.loads(cleaned)


# ============================================================
# EJECUCIÓN DEL CÓDIGO GENERADO
# ============================================================
# El LLM genera código Python como texto. Esta función lo ejecuta
# usando exec() y busca la variable `fig` que el código debe crear.
# `fig` debe ser una figura de Plotly (px o go).
#
# El código generado tiene acceso a: df, pd, px, go.
#
# No modifiques esta función.
#
def execute_chart(code, df):
    local_vars = {"df": df, "pd": pd, "px": px, "go": go}
    exec(code, {}, local_vars)
    return local_vars.get("fig")


# ============================================================
# INTERFAZ STREAMLIT
# ============================================================
# Toda la interfaz de usuario. No modifiques esta sección.
#

# Configuración de la página
st.set_page_config(page_title="Spotify Analytics", layout="wide")

# --- Control de acceso ---
# Lee la contraseña de secrets.toml. Si no coincide, no muestra la app.
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.title("🔒 Acceso restringido")
    pwd = st.text_input("Contraseña:", type="password")
    if pwd:
        if pwd == st.secrets["PASSWORD"]:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Contraseña incorrecta.")
    st.stop()

# --- App principal ---
st.title("🎵 Spotify Analytics Assistant")
st.caption("Pregunta lo que quieras sobre tus hábitos de escucha")

# Cargar datos y construir el prompt con información del dataset
df = load_data()
system_prompt = build_prompt(df)

# Caja de texto para la pregunta del usuario
if prompt := st.chat_input("Ej: ¿Cuál es mi artista más escuchado?"):

    # Mostrar la pregunta en la interfaz
    with st.chat_message("user"):
        st.write(prompt)

    # Generar y mostrar la respuesta
    with st.chat_message("assistant"):
        with st.spinner("Analizando..."):
            try:
                # 1. Enviar pregunta al LLM
                raw = get_response(prompt, system_prompt)

                # 2. Parsear la respuesta JSON
                parsed = parse_response(raw)

                if parsed["tipo"] == "fuera_de_alcance":
                    # Pregunta fuera de alcance: mostrar solo texto
                    st.write(parsed["interpretacion"])
                else:
                    # Pregunta válida: ejecutar código y mostrar gráfico
                    fig = execute_chart(parsed["codigo"], df)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                        st.write(parsed["interpretacion"])
                        st.code(parsed["codigo"], language="python")
                    else:
                        st.warning("El código no produjo ninguna visualización. Intenta reformular la pregunta.")
                        st.code(parsed["codigo"], language="python")

            except json.JSONDecodeError:
                st.error("No he podido interpretar la respuesta. Intenta reformular la pregunta.")
            except Exception as e:
                st.error("Ha ocurrido un error al generar la visualización. Intenta reformular la pregunta.")


# ============================================================
# REFLEXIÓN TÉCNICA (máximo 30 líneas)
# ============================================================
#
# Responde a estas tres preguntas con tus palabras. Sé concreto
# y haz referencia a tu solución, no a generalidades.
# No superes las 30 líneas en total entre las tres respuestas.
#
# 1. ARQUITECTURA TEXT-TO-CODE
#   1.1 ¿Cómo funciona la arquitectura de tu aplicación?
#   1.2 ¿Qué recibe el LLM?
#   1.3 ¿Qué devuelve?
#   1.4 ¿Dónde se ejecuta el código generado?
#   1.5 ¿Por qué el LLM no recibe los datos directamente?
#
#   [Tu respuesta aquí]
#   1.1 El LLM actúa como un traductor de lenguaje natural a código ejecutable, pero no toca los datos reales.
#   1.2 Recibe metadatos (nombres de columnas, tipos de datos, valores únicos) y el esquema del df.
#   1.3 Un objeto JSON estructurado que contiene código Python (vease seccion del codigo PARSING DE LA RESPUESTA).
#   1.4 El código generado se ejecuta localmente en el servidor de Streamlit.
#   1.5 
#
# 2. EL SYSTEM PROMPT COMO PIEZA CLAVE
#   ¿Qué información le das al LLM y por qué?
#   Pon un ejemplo concreto de una pregunta que funciona gracias a algo específico de tu prompt,
#   y otro de una que falla o fallaría si quitases una instrucción.
#
#   [Tu respuesta aquí]
#   Información proporcionada: Comportamiento esperado, descripción de columnas, formatos de salida (JSON) y límites operativos.
#       De esta manera evitamos posibles alucinaciones (es muy dificil que una IA responde "no sé")
#   Ejemplo de funcionamiento: "¿Cuál es mi top 5 de artistas?" funciona porque el prompt define explícitamente la columna artist y
#       la instrucción de usar .head(). Sin esto, el modelo podría intentar usar una columna inexistente como artist_name.
#   Ejemplo de fallo: Si eliminamos la instrucción "Responde exclusivamente en formato JSON",
#       el LLM podría incluir texto como "Aquí tienes tu gráfico".
#       Esto causaría un error crítico mas adelante en la función json.loads(), rompiendo la aplicación por una falla de formato.
#
# 3. EL FLUJO COMPLETO
#   Describe paso a paso qué ocurre desde que el usuario escribe una pregunta hasta que ve el gráfico en pantalla.
#
#   [Tu respuesta aquí]
#   1. El usuario escribe su pregunta en la interfaz de Streamlit.
#   2. Vease seccion INTERFAZ STREAMLIT
#       Primero se ejecuta load_data, luego función build_prompt inyecta los metadatos actuales en el SYSTEM_PROMPT.
#   3. Se envía la pregunta y el prompt a la API de OpenAI (gpt-4o-mini).
#   4. La aplicación recibe el JSON del LLM y extrae el bloque de codigo.
#   5. La función execute_chart ejecuta ese código usando exec(), aplicandolo directamente sobre df que cargamos en load_data.
#   6. El objeto fig resultante se muestra en pantalla mediante st.plotly_chart, junto con la interpretación textual del modelo.