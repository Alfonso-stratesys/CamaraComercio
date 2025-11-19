"""
Pequeño pipeline de análisis de respuestas abiertas de encuesta usando:

- LangGraph para orquestar 2 "agentes":
    1) Agente analista: sintetiza insights a partir de las respuestas.
    2) Agente redactor: convierte esos insights en un texto final estructurado.

- Azure OpenAI (AzureChatOpenAI) como LLM subyacente.
"""

# ============================
# IMPORTS
# ============================

# Tipado básico de Python: listas y TypedDict para definir el estado del grafo
from typing import List, TypedDict

# Módulo estándar para trabajar con variables de entorno
import os

# Carga automática de variables definidas en un fichero .env
from dotenv import load_dotenv

# Núcleo de LangGraph para definir un grafo de estado (StateGraph)
from langgraph.graph import StateGraph, END

# Integración de LangChain con Azure OpenAI (chat models)
from langchain_openai import AzureChatOpenAI
#Version groq
#from groq import ChatGroq
from langchain_groq import ChatGroq

# Tipos de mensajes de LangChain (estilo ChatGPT: system / human / ai)
from langchain_core.messages import SystemMessage, HumanMessage


# ============================
# DEFINICIÓN DEL ESTADO GLOBAL DEL GRAFO
# ============================

class SurveyState(TypedDict, total=False):
    """
    Estado que viajará entre los nodos (agentes) del grafo.

    Claves:
    - raw_answers: lista de respuestas en texto libre procedentes de la BD.
    - synthesized_insights: resumen intermedio (salida del agente analista).
    - final_report: texto final, bien redactado, para usar en informe / dashboard.
    """
    raw_answers: List[str]
    synthesized_insights: str
    final_report: str


# ============================
# CONFIGURACIÓN DEL ENTORNO Y LLMs
# ============================

def load_environment() -> None:
    """
    Carga las variables de entorno desde .env (si existe) y valida lo mínimo.
    """
    # Cargar variables de entorno desde el fichero .env, si está presente.
    load_dotenv()

    # Validar que al menos tenemos la API key de Azure OpenAI.
    # En caso contrario, lanzamos un error claro.
    if not os.getenv("AZURE_OPENAI_API_KEY"):
        raise RuntimeError(
            "Falta AZURE_OPENAI_API_KEY. "
            "Configúrala en el entorno o en un archivo .env."
        )

    # Validar que el endpoint también está definido.
    if not os.getenv("AZURE_OPENAI_ENDPOINT"):
        raise RuntimeError(
            "Falta AZURE_OPENAI_ENDPOINT. "
            "Configúralo en el entorno o en un archivo .env."
        )

    # Validar que la versión de API de Azure OpenAI está configurada.
    if not os.getenv("AZURE_OPENAI_API_VERSION"):
        raise RuntimeError(
            "Falta AZURE_OPENAI_API_VERSION. "
            "Configúrala en el entorno o en un archivo .env."
        )


def build_azure_chat_model(azure_deployment_env_var: str) -> AzureChatOpenAI:
    """
    Crea una instancia de AzureChatOpenAI a partir del nombre de la variable
    de entorno que contiene el deployment.

    Parámetros:
    - azure_deployment_env_var: nombre de la variable de entorno que guarda
      el nombre del deployment en Azure (por ejemplo 'AZURE_OPENAI_DEPLOYMENT_ANALYSIS').

    Devuelve:
    - Instancia configurada de AzureChatOpenAI.
    """
    # Leer el nombre de deployment de la variable de entorno indicada.
    deployment_name = os.getenv(azure_deployment_env_var)

    # Si no está definida, lanzar un error explicativo.
    if not deployment_name:
        raise RuntimeError(
            f"Falta la variable de entorno {azure_deployment_env_var} "
            f"con el nombre del deployment de Azure OpenAI."
        )

    # Crear el modelo de chat de Azure OpenAI usando langchain-openai.
    # Usamos temperatura baja para tener respuestas estables y menos "creativas".
    llm = AzureChatOpenAI(
        azure_deployment=deployment_name,                     # nombre del deployment
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),    # versión de la API
        temperature=0.2,                                      # algo de variación, pero poco
        max_tokens=None,                                      # que decida el modelo
        timeout=30,                                           # timeout en segundos
        max_retries=2                                         # reintentos ante errores transitorios
    )

    # Devolvemos el modelo ya listo para usar.
    return llm

def build_groq_chat_model():
    """
    Devuelve un cliente Groq Chat compatible con LangChain.
    Requiere la variable de entorno GROQ_API_KEY.
    """

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise RuntimeError("Falta GROQ_API_KEY en tu .env")

    llm = ChatGroq(
        api_key=api_key,
        model="meta-llama/llama-4-scout-17b-16e-instruct",  # modelo habitual de Groq
        temperature=0.2,
        timeout=30,
        max_retries=2,
    )

    return llm
# ============================
# SIMULACIÓN DE LECTURA DE RESPUESTAS DESDE BD
# ============================

def load_survey_answers_from_db() -> List[str]:
    """
    Esta función simula la lectura de respuestas de una BD.
    En tu proyecto real, aquí pondrás la query a SQL, Cosmos, API, etc.

    Devuelve:
    - Lista de respuestas en texto libre, cada string es una respuesta de un participante.
    """
    # EJEMPLO simplificado de respuestas abiertas de la encuesta:
    # En producción, aquí vendría el resultado de tu consulta.
    example_answers = [
        "Creo que la inteligencia artificial va a transformar la productividad "
        "de las empresas en Colombia, pero me preocupa la falta de regulación clara.",

        "La IA generativa es una oportunidad enorme para mejorar la atención al cliente, "
        "pero muchas PYMEs no tienen todavía la capacidad para adoptarla de forma responsable.",

        "En mi opinión, el principal reto será la formación del talento humano para trabajar "
        "con IA. Sin eso, la brecha entre empresas grandes y pequeñas se va a ampliar.",

        "La IA puede ayudar a formalizar muchos negocios informales en Colombia, "
        "si se combina con políticas públicas adecuadas.",

        "Veo riesgos en el uso de datos personales sin suficiente transparencia, "
        "pero también oportunidades en salud, educación y sector público."
    ]

    # Devolvemos la lista de respuestas de ejemplo.
    return example_answers


# ============================
# NODO 1: AGENTE ANALISTA
# ============================

def make_analysis_node(llm: AzureChatOpenAI):
    """
    Factoría de nodo para el 'agente analista'.

    Este nodo:
    - Toma las respuestas crudas del estado (raw_answers).
    - Llama al LLM para que extraiga patrones, temas, oportunidades y riesgos.
    - Devuelve un diccionario con la clave 'synthesized_insights' rellenada.
    """

    def analysis_node(state: SurveyState) -> dict:
        """
        Implementación concreta del nodo de análisis.
        """
        # Recuperar del estado la lista de respuestas abiertas.
        raw_answers = state.get("raw_answers", [])

        # Si no hay respuestas, devolvemos un insight vacío para evitar errores.
        if not raw_answers:
            return {
                "synthesized_insights": "No se han recibido respuestas en la encuesta."
            }

        # Construir un texto concatenando todas las respuestas en formato de lista.
        # Esto facilita que el modelo vea cada respuesta claramente separada.
        answers_bullet_list = "\n".join(
            f"- {answer}" for answer in raw_answers
        )

        # Mensaje de sistema: explica el rol del modelo como analista.
        system_message = SystemMessage(
            content=(
                "Eres un analista de estrategia especializado en interpretar encuestas "
                "sobre el impacto de la inteligencia artificial en Colombia. "
                "Tienes que identificar patrones, consensos, desacuerdos, oportunidades "
                "y riesgos a partir de las respuestas de los participantes."
            )
        )

        # Mensaje humano: le pasamos las respuestas en bruto y lo que queremos a cambio.
        human_message = HumanMessage(
            content=(
                "A continuación tienes una lista de respuestas textuales de una encuesta "
                "sobre el rumbo de la IA en Colombia.\n\n"
                "RESPUESTAS:\n"
                f"{answers_bullet_list}\n\n"
                "Tu tarea es sintetizar los principales insights agregados, no caso a caso. "
                "Devuélveme:\n"
                "1) Los 3-5 temas más repetidos.\n"
                "2) Las principales oportunidades percibidas.\n"
                "3) Los principales riesgos y preocupaciones.\n"
                "4) Una frase breve que resuma el tono general (optimista, cauteloso, etc.)."
            )
        )

        # Invocar el modelo con los mensajes construidos.
        # El resultado es un AIMessage, del que extraemos el texto (content).
        ai_response = llm.invoke([system_message, human_message])

        # Extraemos el contenido textual de la respuesta.
        insights_text = ai_response.content

        # Devolvemos un diccionario parcial de estado con la clave 'synthesized_insights'.
        # LangGraph se encargará de mezclarlo con el estado existente.
        return {
            "synthesized_insights": insights_text
        }

    # Devolvemos la función que actuará como nodo dentro del grafo.
    return analysis_node


# ============================
# NODO 2: AGENTE REDACTOR
# ============================

def make_writing_node(llm: AzureChatOpenAI):
    """
    Factoría de nodo para el 'agente redactor'.

    Este nodo:
    - Recibe el texto con insights sintetizados (synthesized_insights).
    - Llama al LLM para convertirlo en un informe breve y estructurado.
    - Devuelve un diccionario con la clave 'final_report' rellenada.
    """

    def writing_node(state: SurveyState) -> dict:
        """
        Implementación concreta del nodo de redacción.
        """
        # Recuperar el resumen intermedio generado por el agente analista.
        synthesized_insights = state.get("synthesized_insights", "")

        # Si no hay insights (por ejemplo, si no había datos), devolvemos un informe mínimo.
        if not synthesized_insights:
            return {
                "final_report": (
                    "No se dispone de insights agregados porque no se han recibido "
                    "respuestas válidas en la encuesta."
                )
            }

        # Mensaje de sistema: define el estilo y estructura del informe final.
        system_message = SystemMessage(
            content=(
                "Eres un consultor senior que redacta resúmenes ejecutivos claros y concisos "
                "para directivos empresariales. Vas a recibir un resumen de insights agregados "
                "procedentes de una encuesta sobre IA en Colombia. "
                "Debes transformarlo en un texto listo para presentar en un dashboard "
                "o informe, con máximo 2 páginas de texto."
            )
        )

        # Mensaje humano: pasamos los insights y definimos la estructura deseada.
        human_message = HumanMessage(
            content=(
                "A continuación tienes un resumen de insights agregados de una encuesta:\n\n"
                f"{synthesized_insights}\n\n"
                "Redáctalo en un texto estructurado con el siguiente esquema:\n"
                "- Contexto general (2-3 frases).\n"
                "- Visión global de los participantes (3-5 frases).\n"
                "- Principales oportunidades identificadas (lista corta de bullets).\n"
                "- Principales riesgos y preocupaciones (lista corta de bullets).\n"
                "- Conclusión final en 2-3 frases, señalando el tono general.\n\n"
                "Lenguaje: español neutro, profesional, fácil de entender para directivos. "
                "Evita tecnicismos innecesarios."
            )
        )

        # Invocamos el LLM con los mensajes construidos.
        ai_response = llm.invoke([system_message, human_message])

        # Extraemos el contenido textual (el informe final).
        final_report_text = ai_response.content

        # Devolvemos un diccionario con la clave 'final_report' actualizada.
        return {
            "final_report": final_report_text
        }

    # Devolvemos la función que actuará como nodo dentro del grafo.
    return writing_node


# ============================
# CONSTRUCCIÓN DEL GRAFO DE LANGGRAPH
# ============================

def build_survey_insight_graph(
    analysis_llm: AzureChatOpenAI,
    writing_llm: AzureChatOpenAI
):
    """
    Construye y compila un StateGraph de LangGraph con dos nodos:

    1) 'analysis_agent' -> agente analista (genera synthesized_insights).
    2) 'writing_agent'  -> agente redactor (genera final_report).

    Flujo:
        ENTRY_POINT -> analysis_agent -> writing_agent -> END
    """
    # Crear un builder de grafo de estado indicando el tipo de estado que maneja.
    builder = StateGraph(SurveyState)

    # Crear el nodo de análisis a partir del LLM analista.
    analysis_node = make_analysis_node(analysis_llm)

    # Crear el nodo de redacción a partir del LLM redactor.
    writing_node = make_writing_node(writing_llm)

    # Registrar los nodos en el grafo, asignándoles un nombre.
    builder.add_node("analysis_agent", analysis_node)
    builder.add_node("writing_agent", writing_node)

    # Definir el punto de entrada: primer nodo que se ejecuta.
    builder.set_entry_point("analysis_agent")

    # Conectar el nodo de análisis con el nodo de redacción.
    builder.add_edge("analysis_agent", "writing_agent")

    # Conectar el nodo de redacción con el final del grafo (END).
    builder.add_edge("writing_agent", END)

    # Compilar el grafo en un objeto ejecutable.
    graph = builder.compile()

    # Devolver el grafo compilado.
    return graph


# ============================
# FUNCIÓN PRINCIPAL (EJEMPLO DE USO)
# ============================

def main():
    """
    Ejemplo de ejecución completa del pipeline:

    1) Carga variables de entorno.
    2) Crea dos LLMs (analista y redactor).
    3) Construye el grafo de LangGraph.
    4) Lee las respuestas de la BD (aquí simulado).
    5) Ejecuta el grafo para obtener el informe final.
    6) Imprime el resultado por consola (o lo podrías guardar en la BD, etc.).
    """
    # 1) Cargar configuración del entorno y validar que todo lo necesario está.
    load_environment()

    # 2) Construir el modelo de Azure OpenAI para el agente analista.
    #analysis_llm = build_azure_chat_model("AZURE_OPENAI_DEPLOYMENT_ANALYSIS")

    # 3) Construir el modelo de Azure OpenAI para el agente redactor.
    #    Podrías reutilizar el mismo deployment si quieres.
    #writing_llm = build_azure_chat_model("AZURE_OPENAI_DEPLOYMENT_WRITING")

    analysis_llm = build_groq_chat_model()
    writing_llm = build_groq_chat_model()
    # 4) Construir el grafo de LangGraph con ambos agentes.
    graph = build_survey_insight_graph(
        analysis_llm=analysis_llm,
        writing_llm=writing_llm
    )

    # 5) Leer las respuestas desde la base de datos (aquí usando datos simulados).
    raw_answers = load_survey_answers_from_db()

    # 6) Crear el estado inicial del grafo: solo necesitamos raw_answers.
    initial_state: SurveyState = {
        "raw_answers": raw_answers
        # El resto de claves (synthesized_insights, final_report) se rellenan en los nodos.
    }

    # 7) Ejecutar el grafo con el estado inicial.
    #    graph.invoke devuelve el estado final tras pasar por todos los nodos.
    final_state = graph.invoke(initial_state)

    # 8) Recuperar el informe final del estado resultante.
    final_report = final_state.get("final_report", "")

    # 9) Imprimir el informe por consola.
    #    En tu caso, aquí podrías guardarlo en la BD o exponerlo a otro servicio.
    print("\n================ INFORME FINAL PARA DASHBOARD ================\n")
    print(final_report)
    print("\n==============================================================\n")


# Punto de entrada estándar de un script de Python.
if __name__ == "__main__":
    main()
