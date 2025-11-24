"""
MISMO pipeline de análisis de dos preguntas abiertas,
pero ahora las respuestas se cargan desde Supabase.

- Lee proposito_hoy y proposito_18 desde tabla respuestas
- Corre el grafo de análisis/redacción en paralelo
- Genera conclusión conjunta final
"""

# ============================
# IMPORTS
# ============================
from typing import List, TypedDict
import os
import asyncio
import httpx  # <-- nuevo
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_openai import AzureChatOpenAI
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

# ============================
# DEFINICIÓN DEL ESTADO GLOBAL DEL GRAFO
# ============================
class SurveyState(TypedDict, total=False):
    raw_answers: List[str]
    question_context: str
    synthesized_insights: str
    final_report: str

class ComparisonState(TypedDict, total=False):
    purpose_now_report: str
    purpose_future_report: str
    final_joint_conclusion: str

# ============================
# CONFIGURACIÓN DEL ENTORNO Y LLMs
# ============================
def load_environment() -> None:
    load_dotenv()

    if not os.getenv("AZURE_OPENAI_API_KEY"):
        print("[AVISO] No se encontró AZURE_OPENAI_API_KEY (ok si usas Groq).")
    if not os.getenv("AZURE_OPENAI_ENDPOINT"):
        print("[AVISO] No se encontró AZURE_OPENAI_ENDPOINT (ok si usas Groq).")
    if not os.getenv("AZURE_OPENAI_API_VERSION"):
        print("[AVISO] No se encontró AZURE_OPENAI_API_VERSION (ok si usas Groq).")

    if not os.getenv("SUPABASE_URL") or not os.getenv("SUPABASE_KEY"):
        print("[AVISO] Falta SUPABASE_URL o SUPABASE_KEY para leer BD.")

def build_azure_chat_model(azure_deployment_env_var: str) -> AzureChatOpenAI:
    deployment_name = os.getenv(azure_deployment_env_var)
    if not deployment_name:
        raise RuntimeError(f"Falta {azure_deployment_env_var} con el deployment de Azure.")

    return AzureChatOpenAI(
        azure_deployment=deployment_name,
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        temperature=0.2,
        max_tokens=None,
        timeout=30,
        max_retries=2
    )

def build_groq_chat_model() -> ChatGroq:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("Falta GROQ_API_KEY.")
    return ChatGroq(
        api_key=api_key,
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        temperature=0.2,
        timeout=30,
        max_retries=2,
    )

# ============================
# NUEVO: CARGA REAL DESDE SUPABASE
# ============================
async def fetch_purpose_answers_from_supabase() -> tuple[List[str], List[str]]:
    """
    Lee de Supabase las columnas:
      - proposito_hoy
      - proposito_18
    desde tabla respuestas.

    Devuelve:
      (purpose_now_answers, purpose_future_answers)
    """
    URL = os.environ["SUPABASE_URL"]
    KEY = os.environ["SUPABASE_KEY"]
    TABLE = "respuestas"
    COL_NOW = "proposito_hoy"
    COL_FUTURE = "proposito_18"

    endpoint = f"{URL}/rest/v1/{TABLE}"
    headers = {
        "apikey": KEY,
        "Authorization": f"Bearer {KEY}",
        "Accept": "application/json",
    }

    # Traemos solo esas columnas
    params = {
        "select": f"{COL_NOW},{COL_FUTURE}",
        # si quieres limitar para pruebas:
        # "limit": "500"
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.get(endpoint, headers=headers, params=params)
        r.raise_for_status()
        rows = r.json()

    now_answers = []
    future_answers = []

    for row in rows:
        a_now = (row.get(COL_NOW) or "").strip()
        a_future = (row.get(COL_FUTURE) or "").strip()
        if a_now:
            now_answers.append(a_now)
        if a_future:
            future_answers.append(a_future)

    return now_answers, future_answers

def load_final_answers(proposito_hoy, proposito_futuro) -> List[str]:
    return [proposito_hoy, proposito_futuro]

# ============================
# NODO 1: AGENTE ANALISTA
# ============================
def make_analysis_node(llm):
    async def analysis_node(state: SurveyState) -> dict:
        raw_answers = state.get("raw_answers", [])
        question_context = state.get("question_context", "propósito de empresas colombianas")

        if not raw_answers:
            return {"synthesized_insights": f"No hay respuestas para {question_context}."}

        answers_bullets = "\n".join(f"- {a}" for a in raw_answers)

        system_message = SystemMessage(content=(
            "Eres un analista de estrategia especializado en interpretar encuestas "
            "sobre el propósito empresarial en Colombia."
        ))

        human_message = HumanMessage(content=(
            f"Estas son respuestas sobre el {question_context}:\n\n"
            f"{answers_bullets}\n\n"
            "Sintetiza insights agregados:\n"
            "Interpreta el porcentaje de aceptación de cada idea que te llegue al texto y muestralo claramente"
            "1) 3-5 ejes de propósito más repetidos.\n"
            "2) Diferencias/complementos.\n"
            "3) Núcleo compartido.\n"
            "4) Una frase central resumen."
        ))

        ai_response = await llm.ainvoke([system_message, human_message])
        return {"synthesized_insights": ai_response.content}

    return analysis_node

# ============================
# NODO 2: AGENTE REDACTOR
# ============================
def make_writing_node(llm):
    async def writing_node(state: SurveyState) -> dict:
        synthesized_insights = state.get("synthesized_insights", "")
        question_context = state.get("question_context", "propósito de empresas colombianas")

        if not synthesized_insights:
            return {"final_report": f"No hay insights para {question_context}."}

        system_message = SystemMessage(content=(
            "Eres un consultor senior que redacta resúmenes ejecutivos."
            "Tu tarea es identificar qué idea tiene un peso mayoritario sobre la totalidad de los encuestados (si la hay)"
            "Si has encntrado una idea claramente destacada exponla, si hay clara división indica cuáles son las ideas prncipales y su relevancia"
        ))

        human_message = HumanMessage(content=(
           f"Contexto: {question_context}\n\n"
            f"Insights:\n{synthesized_insights}\n\n"
            #"Redacta UN solo párrafo (1 línea) que exprese el propósito común."
            "Redacta 2-3 líneas que resuman los rasgos comunes entre todo el texto que recibes"
            "Si te llegan ideas divididas muestra claramente el porcentaje de aceptación de ambas"
        ))

        ai_response = await llm.ainvoke([system_message, human_message])
        return {"final_report": ai_response.content}

    return writing_node

# ============================
# NODO 3: COMPARADOR
# ============================
def make_comparison_node(llm):
    async def comparison_node(state: ComparisonState) -> dict:
        purpose_now = state.get("purpose_now_report", "")
        purpose_future = state.get("purpose_future_report", "")

        system_message = SystemMessage(content=(
            "Eres un estratega experto en evolución del propósito empresarial."
        ))
        human_message = HumanMessage(content=(
            "PROPÓSITO ACTUAL:\n"
            f"{purpose_now}\n\n"
            "PROPÓSITO 18 AÑOS:\n"
            f"{purpose_future}\n\n"
            "Compara y concluye en DOS líneas (máx 20 palabras)."
        ))

        ai_response = await llm.ainvoke([system_message, human_message])
        return {"final_joint_conclusion": ai_response.content}

    return comparison_node

# ============================
# GRAFOS
# ============================
def build_survey_insight_graph(analysis_llm, writing_llm):
    builder = StateGraph(SurveyState)
    builder.add_node("analysis_agent", make_analysis_node(analysis_llm))
    builder.add_node("writing_agent", make_writing_node(writing_llm))
    builder.set_entry_point("analysis_agent")
    builder.add_edge("analysis_agent", "writing_agent")
    builder.add_edge("writing_agent", END)
    return builder.compile()

def build_comparison_graph(comparison_llm):
    builder = StateGraph(ComparisonState)
    builder.add_node("comparison_agent", make_comparison_node(comparison_llm))
    builder.set_entry_point("comparison_agent")
    builder.add_edge("comparison_agent", END)
    return builder.compile()

# ============================
# EJECUCIÓN POR PREGUNTA
# ============================
async def run_pipeline_for_question(graph, raw_answers, question_context):
    initial_state: SurveyState = {
        "raw_answers": raw_answers,
        "question_context": question_context
    }
    final_state = await graph.ainvoke(initial_state)
    return final_state.get("final_report", "")

# ============================
# MAIN
# ============================
async def main_async():
    load_environment()

    #analysis_llm = build_groq_chat_model()
    #writing_llm = build_groq_chat_model()
    analysis_llm = build_azure_chat_model("AZURE_OPENAI_DEPLOYMENT_ANALYSIS")
    writing_llm = build_azure_chat_model("AZURE_OPENAI_DEPLOYMENT_WRITING")
    comparison_llm = writing_llm

    graph = build_survey_insight_graph(analysis_llm, writing_llm)
    comparison_graph = build_comparison_graph(comparison_llm)

    # ✅ ahora viene todo de la BD
    purpose_now_answers, purpose_future_answers = await fetch_purpose_answers_from_supabase()

    context_now = "propósito ACTUAL de las empresas colombianas"
    context_future = "propósito de las empresas colombianas dentro de 18 años"
    c=""
    p=""
    now_report, future_report = await asyncio.gather(
        run_pipeline_for_question(graph, purpose_now_answers,context_now),
        run_pipeline_for_question(graph, purpose_future_answers, context_future),
    )

    comparison_initial_state: ComparisonState = {
        "purpose_now_report": now_report,
        "purpose_future_report": future_report,
    }
    comparison_final_state = await comparison_graph.ainvoke(comparison_initial_state)
    final_joint_conclusion = comparison_final_state.get("final_joint_conclusion", "")

    print("\n===== PROPÓSITO COMÚN ACTUAL =====\n")
    print(now_report)

    print("\n===== PROPÓSITO COMÚN A 18 AÑOS =====\n")
    print(future_report)

    print("\n===== CONCLUSIÓN CONJUNTA (2 líneas) =====\n")
    print(final_joint_conclusion)

if __name__ == "__main__":
    asyncio.run(main_async())
