"""
Pipeline de análisis de DOS preguntas abiertas de encuesta usando:

- LangGraph para orquestar 2 "agentes" (nodos):
    1) Agente analista: sintetiza insights a partir de las respuestas.
    2) Agente redactor: convierte esos insights en un párrafo final estructurado.

- Azure OpenAI (AzureChatOpenAI) o Groq (ChatGroq) como LLM subyacente.

Escenario:
---------
Tenemos DOS preguntas abiertas en la encuesta:

1) Propósito ACTUAL de la empresa.
2) Propósito de la empresa dentro de 18 años.

Queremos:
- Para cada pregunta, generar un PÁRRAFO que exprese un "propósito común
  del empresariado colombiano" para ese horizonte temporal.
- Ejecutar ambos procesos en PARALELO (asincronía) para ir más rápido.
"""

# ============================
# IMPORTS
# ============================

from typing import List, TypedDict
import os                    # Para leer variables de entorno
import asyncio               # Para ejecutar procesos en paralelo (asincronía)

from dotenv import load_dotenv           # Carga variables de entorno desde .env
from langgraph.graph import StateGraph, END   # Núcleo de LangGraph (grafo de estado)

# Modelos de chat de LangChain para Azure OpenAI y Groq
from langchain_openai import AzureChatOpenAI
from langchain_groq import ChatGroq

# Tipos de mensajes estilo ChatGPT
from langchain_core.messages import SystemMessage, HumanMessage


# ============================
# DEFINICIÓN DEL ESTADO GLOBAL DEL GRAFO
# ============================

class SurveyState(TypedDict, total=False):
    """
    Estado que viajará entre los nodos (agentes) del grafo.

    Claves:
    - raw_answers: lista de respuestas en texto libre de la BD referentes
      a UNA pregunta concreta (p.ej., "propósito actual").
    - question_context: texto corto que describe el contexto de la pregunta,
      por ejemplo:
        - "propósito ACTUAL de las empresas colombianas"
        - "propósito de las empresas colombianas dentro de 18 años"
      Se usará para adaptar las instrucciones del LLM.
    - synthesized_insights: resumen intermedio (salida del agente analista).
    - final_report: párrafo final, bien redactado, que resume el "propósito común".
    """
    raw_answers: List[str]
    question_context: str
    synthesized_insights: str
    final_report: str


# ============================
# CONFIGURACIÓN DEL ENTORNO Y LLMs
# ============================

def load_environment() -> None:
    """
    Carga las variables de entorno desde .env (si existe) y valida lo mínimo
    necesario para Azure OpenAI. Si usas solo Groq, podrías relajar estos checks.
    """
    # Cargamos las variables declaradas en un archivo .env (si existe).
    load_dotenv()

    # Validar que la API key de Azure OpenAI está definida (opcional si no lo usas).
    # Puedes comentar estas validaciones si SOLO usas Groq.
    if not os.getenv("AZURE_OPENAI_API_KEY"):
        print(
            "[AVISO] No se encontró AZURE_OPENAI_API_KEY. "
            "Si vas a usar Azure OpenAI, configúrala en tu entorno o .env."
        )

    if not os.getenv("AZURE_OPENAI_ENDPOINT"):
        print(
            "[AVISO] No se encontró AZURE_OPENAI_ENDPOINT. "
            "Si vas a usar Azure OpenAI, configúralo en tu entorno o .env."
        )

    if not os.getenv("AZURE_OPENAI_API_VERSION"):
        print(
            "[AVISO] No se encontró AZURE_OPENAI_API_VERSION. "
            "Si vas a usar Azure OpenAI, configúralo en tu entorno o .env."
        )


def build_azure_chat_model(azure_deployment_env_var: str) -> AzureChatOpenAI:
    """
    Crea una instancia de AzureChatOpenAI a partir del nombre de la variable
    de entorno que contiene el deployment.

    Parámetros:
    - azure_deployment_env_var: nombre de la variable de entorno que guarda
      el nombre del deployment en Azure (por ejemplo 'AZURE_OPENAI_DEPLOYMENT_ANALYSIS').

    Devuelve:
    - Instancia configurada de AzureChatOpenAI (modelo de chat).
    """
    # Tomamos el nombre del deployment desde la variable de entorno.
    deployment_name = os.getenv(azure_deployment_env_var)

    # Si no está definida, lanzamos un error claro.
    if not deployment_name:
        raise RuntimeError(
            f"Falta la variable de entorno {azure_deployment_env_var} "
            f"con el nombre del deployment de Azure OpenAI."
        )

    # Creamos el modelo de chat de Azure OpenAI.
    llm = AzureChatOpenAI(
        azure_deployment=deployment_name,                     # Nombre del deployment
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),    # Versión de API
        temperature=0.2,                                      # Poco creativo, más estable
        max_tokens=None,                                      # Que decida el modelo
        timeout=30,                                           # Timeout en segundos
        max_retries=2                                         # Reintentos en caso de fallo transitorio
    )

    return llm


def build_groq_chat_model() -> ChatGroq:
    """
    Devuelve un cliente Groq Chat compatible con LangChain.
    Requiere la variable de entorno GROQ_API_KEY.
    """
    # Leemos la API key de Groq.
    api_key = os.getenv("GROQ_API_KEY")

    # Si no está definida, lanzamos un error claro.
    if not api_key:
        raise RuntimeError("Falta GROQ_API_KEY en tu .env o variables de entorno.")

    # Creamos el modelo de Groq (Llama 4 Scout en este caso).
    llm = ChatGroq(
        api_key=api_key,
        model="meta-llama/llama-4-scout-17b-16e-instruct",  # Modelo recomendado por Groq
        temperature=0.2,    # Poco creativo, más enfocado a síntesis estable
        timeout=30,         # Timeout en segundos
        max_retries=2,      # Reintentos
    )

    return llm


# ============================
# SIMULACIÓN DE LECTURA DE RESPUESTAS DESDE BD
# ============================

def load_purpose_now_answers_from_db() -> List[str]:
    """
    Simula la lectura de respuestas a la pregunta:
    '¿Cuál es el propósito ACTUAL de tu empresa?'

    En tu proyecto real, aquí pondrías la consulta a SQL, Cosmos, etc.
    """
    example_answers = [
        "Nuestro propósito es generar empleo de calidad y aportar al desarrollo local.",
        "Buscamos ofrecer soluciones tecnológicas que ayuden a las pymes colombianas a ser más competitivas.",
        "Queremos mejorar la calidad de vida de nuestros clientes a través de servicios financieros responsables.",
        "Nuestro foco es la sostenibilidad, promoviendo prácticas responsables en la cadena de suministro.",
        "Acompañamos a los agricultores con herramientas y formación para aumentar su productividad."
    ]
    return example_answers


def load_purpose_future_answers_from_db() -> List[str]:
    """
    Simula la lectura de respuestas a la pregunta:
    '¿Cuál crees que será el propósito de tu empresa dentro de 18 años?'

    De nuevo, en producción aquí iría la consulta real a tu BD.
    """
    example_answers = [
        "En 18 años queremos ser referentes regionales en innovación sostenible.",
        "Esperamos contribuir a que Colombia sea un hub de tecnología y talento digital.",
        "Nos vemos liderando proyectos de impacto social a gran escala en educación.",
        "Nuestro propósito será impulsar la transición energética en toda Latinoamérica.",
        "Queremos ser una empresa que combine rentabilidad con un fuerte impacto social."
    ]
    return example_answers


# ============================
# NODO 1: AGENTE ANALISTA (ASÍNCRONO)
# ============================

def make_analysis_node(llm):
    """
    Factoría de nodo para el 'agente analista'.

    IMPORTANTE: el nodo será ASÍNCRONO (async def) y usará await llm.ainvoke(...).
    Esto permite que, si lanzamos DOS grafos en paralelo, las llamadas al LLM
    se ejecuten concurrentemente, reduciendo el tiempo total.

    El nodo:
    - Toma las respuestas crudas del estado (raw_answers).
    - Usa question_context para adaptar el prompt (actual / futuro).
    - Llama al LLM para extraer patrones y temas clave.
    - Devuelve 'synthesized_insights' con un resumen estructurado.
    """

    async def analysis_node(state: SurveyState) -> dict:
        """
        Implementación concreta del nodo de análisis (ASÍNCRONA).
        """
        # Extraemos la lista de respuestas y el contexto de la pregunta del estado.
        raw_answers = state.get("raw_answers", [])
        question_context = state.get(
            "question_context",
            "propósito de las empresas colombianas"  # Valor por defecto si no viene nada.
        )

        # Si no hay respuestas, devolvemos un insight mínimo.
        if not raw_answers:
            return {
                "synthesized_insights": (
                    f"No se han recibido respuestas para el {question_context}."
                )
            }

        # Construimos una lista tipo bullet con todas las respuestas para dar
        # estructura y claridad al modelo.
        answers_bullet_list = "\n".join(f"- {answer}" for answer in raw_answers)

        # Mensaje de sistema: rol del modelo como analista de propósito empresarial.
        system_message = SystemMessage(
            content=(
                "Eres un analista de estrategia especializado en interpretar encuestas "
                "sobre el propósito empresarial en Colombia. A partir de las respuestas, "
                "debes identificar patrones comunes, matices y diferencias."
            )
        )

        # Mensaje humano: incluimos el contexto de la pregunta (actual / futuro).
        human_message = HumanMessage(
            content=(
                f"A continuación tienes una lista de respuestas textuales de una encuesta "
                f"en la que las empresas describen el {question_context}.\n\n"
                "RESPUESTAS:\n"
                f"{answers_bullet_list}\n\n"
                "Tu tarea es sintetizar los principales insights agregados, no caso a caso. "
                "Devuélveme:\n"
                "1) Los 3-5 ejes de propósito más repetidos.\n"
                "2) Cómo se complementan o se diferencian esos propósitos.\n"
                "3) Qué elementos parecen ser nucleares y compartidos por la mayoría.\n"
                "4) Una frase breve que resuma en una idea central el propósito común."
            )
        )

        # Llamamos al modelo de forma ASÍNCRONA.
        # Esto es clave para que dos ejecuciones del grafo puedan solaparse.
        ai_response = await llm.ainvoke([system_message, human_message])

        # Extraemos el texto de la respuesta.
        insights_text = ai_response.content

        # Devolvemos un fragmento de estado con los insights sintetizados.
        return {"synthesized_insights": insights_text}

    # Devolvemos la función que actuará como nodo dentro del grafo.
    return analysis_node


# ============================
# NODO 2: AGENTE REDACTOR (ASÍNCRONO)
# ============================

def make_writing_node(llm):
    """
    Factoría de nodo para el 'agente redactor'.

    También será ASÍNCRONO y llamará a await llm.ainvoke(...).

    El nodo:
    - Recibe 'synthesized_insights' y 'question_context'.
    - Redacta UN SOLO PÁRRAFO que exprese un propósito común del empresariado
      colombiano para ese contexto temporal (actual / dentro de 18 años).
    - Escribe en tono profesional y claro.
    """

    async def writing_node(state: SurveyState) -> dict:
        """
        Implementación concreta del nodo de redacción (ASÍNCRONA).
        """
        # Recuperamos el resumen intermedio y el contexto.
        synthesized_insights = state.get("synthesized_insights", "")
        question_context = state.get(
            "question_context",
            "propósito de las empresas colombianas"
        )

        # Si no hay insights, devolvemos un mensaje estándar.
        if not synthesized_insights:
            return {
                "final_report": (
                    f"No se dispone de insights agregados para el {question_context} "
                    "porque no se han recibido respuestas válidas."
                )
            }

        # Mensaje de sistema: estilo de redacción del párrafo final.
        system_message = SystemMessage(
            content=(
                "Eres un consultor senior que redacta resúmenes ejecutivos claros y concisos "
                "para directivos empresariales. Vas a recibir un resumen de insights agregados "
                "procedentes de una encuesta sobre el propósito de las empresas en Colombia. "
                "Debes transformarlo en UN SOLO PÁRRAFO bien redactado que exprese un propósito "
                "común del empresariado colombiano."
            )
        )

        # Mensaje humano: indicamos que queremos UN párrafo, sin bullets ni secciones.
        human_message = HumanMessage(
            content=(
                f"Contexto de la pregunta: {question_context}.\n\n"
                "A continuación tienes un resumen de insights agregados de la encuesta:\n\n"
                f"{synthesized_insights}\n\n"
                "Redacta UN SOLO PÁRRAFO (5-8 frases) que exprese de forma sintética "
                "un 'propósito común del empresariado colombiano' para este contexto. "
                "Debe sonar como una declaración de propósito compartida, en tono profesional "
                "y positivo, pero realista. No uses listas, ni viñetas, ni apartados, "
                "solo un párrafo continuo en español neutro."
            )
        )

        # Invocamos el LLM de forma ASÍNCRONA.
        ai_response = await llm.ainvoke([system_message, human_message])

        # Extraemos el texto del párrafo final.
        final_report_text = ai_response.content

        # Devolvemos el estado con el párrafo ya listo para el dashboard / informe.
        return {"final_report": final_report_text}

    # Devolvemos el nodo redactor.
    return writing_node


# ============================
# CONSTRUCCIÓN DEL GRAFO DE LANGGRAPH
# ============================

def build_survey_insight_graph(analysis_llm, writing_llm):
    """
    Construye y compila un StateGraph de LangGraph con dos nodos:

    1) 'analysis_agent' -> agente analista (genera synthesized_insights).
    2) 'writing_agent'  -> agente redactor (genera final_report).

    Flujo:
        ENTRY_POINT -> analysis_agent -> writing_agent -> END

    NOTA:
    - Los nodos son asíncronos (async def), por lo que el grafo se podrá ejecutar
      con graph.ainvoke(...) dentro de un contexto asyncio.
    """
    # Creamos el builder de grafo, indicando el tipo de estado que manejará.
    builder = StateGraph(SurveyState)

    # Creamos los nodos a partir de los modelos de LLM.
    analysis_node = make_analysis_node(analysis_llm)
    writing_node = make_writing_node(writing_llm)

    # Registramos los nodos en el grafo con un nombre identificador.
    builder.add_node("analysis_agent", analysis_node)
    builder.add_node("writing_agent", writing_node)

    # Establecemos el punto de entrada: el primer nodo en ejecutarse.
    builder.set_entry_point("analysis_agent")

    # Definimos que, tras el análisis, se ejecuta el nodo de redacción.
    builder.add_edge("analysis_agent", "writing_agent")

    # Y tras la redacción, el flujo termina.
    builder.add_edge("writing_agent", END)

    # Compilamos el grafo en un objeto ejecutable.
    graph = builder.compile()

    return graph


# ============================
# EJECUCIÓN ASÍNCRONA PARA CADA PREGUNTA
# ============================

async def run_pipeline_for_question(
    graph,
    raw_answers: List[str],
    question_context: str
) -> str:
    """
    Ejecuta el grafo (de forma ASÍNCRONA) para una pregunta concreta.

    Parámetros:
    - graph: grafo ya compilado de LangGraph.
    - raw_answers: lista de respuestas en texto para esa pregunta.
    - question_context: descripción corta del contexto de la pregunta
      (ej.: 'propósito ACTUAL de las empresas colombianas').

    Devuelve:
    - final_report (str): párrafo final listo para usar en dashboard / informe.
    """

    # Construimos el estado inicial del grafo.
    initial_state: SurveyState = {
        "raw_answers": raw_answers,          # Respuestas de esa pregunta
        "question_context": question_context # Contexto de esa pregunta
        # 'synthesized_insights' y 'final_report' se generarán en los nodos.
    }

    # Ejecutamos el grafo de forma ASÍNCRONA usando .ainvoke().
    final_state = await graph.ainvoke(initial_state)

    # Recuperamos el párrafo final del estado resultante.
    final_report = final_state.get("final_report", "")

    return final_report


# ============================
# FUNCIÓN PRINCIPAL ASÍNCRONA
# ============================

async def main_async():
    """
    Ejemplo de ejecución completa del pipeline, para DOS preguntas, en paralelo:

    1) Carga variables de entorno.
    2) Crea los LLMs (analista y redactor).
    3) Construye el grafo de LangGraph.
    4) Carga las respuestas de las dos preguntas (simuladas).
    5) Lanza DOS ejecuciones asíncronas del grafo (una por cada pregunta)
       usando asyncio.gather, para que corran en paralelo.
    6) Imprime los dos párrafos finales.
    """
    # 1) Cargar configuración del entorno (.env y validaciones).
    load_environment()

    # 2) Construir los modelos de LLM.
    #    Puedes usar Azure, Groq o una combinación (por ejemplo, Groq para análisis
    #    y Azure para redacción). Aquí, para simplificar, usamos Groq en ambos.
    #
    # Si quisieras usar Azure:
    # analysis_llm = build_azure_chat_model("AZURE_OPENAI_DEPLOYMENT_ANALYSIS")
    # writing_llm = build_azure_chat_model("AZURE_OPENAI_DEPLOYMENT_WRITING")
    #
    # En este ejemplo usamos Groq:
    analysis_llm = build_groq_chat_model()
    writing_llm = build_groq_chat_model()

    # 3) Construir el grafo de LangGraph que usaremos para AMBAS preguntas.
    graph = build_survey_insight_graph(
        analysis_llm=analysis_llm,
        writing_llm=writing_llm
    )

    # 4) Cargar las respuestas simuladas para cada pregunta.
    #    En tu entorno real, aquí llamarías a tu capa de acceso a datos.
    purpose_now_answers = load_purpose_now_answers_from_db()
    purpose_future_answers = load_purpose_future_answers_from_db()

    # 5) Definir los contextos de cada pregunta, que usarán los nodos en el prompt.
    context_now = "propósito ACTUAL de las empresas colombianas"
    context_future = "propósito de las empresas colombianas dentro de 18 años"

    # 6) Lanzar DOS ejecuciones del grafo en paralelo usando asyncio.gather.
    #    Esto es el corazón de la ASINCRONÍA:
    #    - Cada pipeline llamará al LLM de forma asíncrona.
    #    - Mientras el LLM procesa una pregunta, la otra puede avanzar.
    now_report, future_report = await asyncio.gather(
        run_pipeline_for_question(graph, purpose_now_answers, context_now),
        run_pipeline_for_question(graph, purpose_future_answers, context_future),
    )

    # 7) Imprimir resultados (o guardarlos en BD, enviarlos a Power BI, etc.).
    print("\n===== PROPÓSITO COMÚN ACTUAL DEL EMPRESARIADO COLOMBIANO =====\n")
    print(now_report)

    print("\n===== PROPÓSITO COMÚN A 18 AÑOS DEL EMPRESARIADO COLOMBIANO =====\n")
    print(future_report)
    print("\n==============================================================\n")


# Punto de entrada estándar de un script Python.
# Llamamos a main_async() usando asyncio.run para ejecutar el código asíncrono.
if __name__ == "__main__":
    asyncio.run(main_async())
