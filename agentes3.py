"""
Pipeline batched en 5 partes para evitar sesgo por saturación
+ medición de tiempo y coste estimado.

- Cuenta filas en Supabase e imprime el total.
- Divide respuestas en 5 grupos iguales (descarta resto).
- Corre 5 análisis en paralelo por pregunta.
- Redactor integra 5 resúmenes.
- No mezcla propósito actual con futuro.
- Mide tiempo total.
- Suma tokens LLM y estima coste si hay precios en .env.
"""

# ============================
# IMPORTS
# ============================
from typing import List, TypedDict, Tuple
import os
import asyncio
import httpx
import random
import time  # << NUEVO
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage


# ============================
# DEFINICIÓN DE ESTADOS
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
# CONTADOR GLOBAL DE TOKENS / COSTE
# ============================
USAGE_LOCK = asyncio.Lock()
USAGE = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

def _extract_usage(ai_msg) -> Tuple[int, int, int]:
    """
    Intenta leer tokens desde:
    - ai_msg.usage_metadata  (LangChain moderno)
    - ai_msg.response_metadata["token_usage"] / ["usage"]
    """
    u = getattr(ai_msg, "usage_metadata", None)
    if u:
        pt = u.get("input_tokens") or u.get("prompt_tokens") or 0
        ct = u.get("output_tokens") or u.get("completion_tokens") or 0
        tt = u.get("total_tokens") or (pt + ct)
        return int(pt), int(ct), int(tt)

    rm = getattr(ai_msg, "response_metadata", {}) or {}
    tu = rm.get("token_usage") or rm.get("usage") or {}
    if tu:
        pt = tu.get("prompt_tokens", 0)
        ct = tu.get("completion_tokens", 0)
        tt = tu.get("total_tokens", pt + ct)
        return int(pt), int(ct), int(tt)

    return 0, 0, 0

async def add_usage(ai_msg) -> None:
    pt, ct, tt = _extract_usage(ai_msg)
    if tt == 0:
        return
    async with USAGE_LOCK:
        USAGE["prompt_tokens"] += pt
        USAGE["completion_tokens"] += ct
        USAGE["total_tokens"] += tt

def estimate_cost_usd() -> float | None:
    """
    Coste simple (si defines precios en .env):
      PRICE_INPUT_PER_1M  y  PRICE_OUTPUT_PER_1M  (USD por 1M tokens)

    Devuelve None si no hay precios configurados.
    """
    p_in = os.getenv("PRICE_INPUT_PER_1M")
    p_out = os.getenv("PRICE_OUTPUT_PER_1M")
    if not p_in or not p_out:
        return None
    try:
        p_in = float(p_in)
        p_out = float(p_out)
    except ValueError:
        return None

    prompt = USAGE["prompt_tokens"]
    completion = USAGE["completion_tokens"]
    return (prompt / 1_000_000) * p_in + (completion / 1_000_000) * p_out


# ============================
# CONFIG ENTORNO Y LLMs
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
        temperature=0.1,
        max_tokens=None,
        timeout=60,
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
        timeout=60,
        max_retries=2,
    )


# =================================
# SUPABASE: CONTAR + LEER + GUARDAR
# =================================
async def count_rows_supabase(table: str) -> int:
    URL = os.environ["SUPABASE_URL"]
    KEY = os.environ["SUPABASE_KEY"]

    endpoint = f"{URL}/rest/v1/{table}"
    headers = {
        "apikey": KEY,
        "Authorization": f"Bearer {KEY}",
        "Accept": "application/json",
        "Prefer": "count=exact",
    }
    params = {"select": "id", "limit": "1"}

    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.get(endpoint, headers=headers, params=params)
        r.raise_for_status()

    content_range = r.headers.get("Content-Range", "")
    try:
        total = int(content_range.split("/")[-1])
    except Exception:
        total = 0

    return total

async def fetch_purpose_answers_from_supabase() -> Tuple[List[str], List[str]]:
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
    params = {"select": f"{COL_NOW},{COL_FUTURE}"}

    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.get(endpoint, headers=headers, params=params)
        r.raise_for_status()
        rows = r.json()

    now_answers, future_answers = [], []
    for row in rows:
        a_now = (row.get(COL_NOW) or "").strip()
        a_future = (row.get(COL_FUTURE) or "").strip()
        if a_now:
            now_answers.append(a_now)
        if a_future:
            future_answers.append(a_future)

    return now_answers, future_answers

async def insert_llm_results_to_supabase(
    pregunta_1: str,
    pregunta_2: str,
    resumen_final: str,
    table: str = "llm"
) -> None:
    """
    Inserta una fila en la tabla `llm` con columnas:
      - pregunta_1 (propósito actual)
      - pregunta_2 (propósito 18 años)
      - resumen_final (evolución / comparación)

    Crea 1 fila por ejecución.
    """
    URL = os.environ["SUPABASE_URL"]
    KEY = os.environ["SUPABASE_KEY"]
    endpoint = f"{URL}/rest/v1/{table}"

    headers = {
        "apikey": KEY,
        "Authorization": f"Bearer {KEY}",
        "Content-Type": "application/json",
        # opcional: evita volver con payload grande
        "Prefer": "return=minimal",
    }

    payload = {
        "pregunta_1": pregunta_1,
        "pregunta_2": pregunta_2,
        "resumen_final": resumen_final,
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.post(endpoint, headers=headers, json=payload)
        r.raise_for_status()

async def clear_llm_table(table: str = "llm") -> None:
    """
    Elimina todas las filas de la tabla `llm`.

    Asume que la tabla tiene una columna numérica `id` (la típica de Supabase).
    Si tu PK se llama distinto, cambia 'id' por el nombre real.
    """
    URL = os.environ["SUPABASE_URL"]
    KEY = os.environ["SUPABASE_KEY"]
    endpoint = f"{URL}/rest/v1/{table}"

    # 1) Leer los ids que existen
    headers_get = {
        "apikey": KEY,
        "Authorization": f"Bearer {KEY}",
        "Accept": "application/json",
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.get(
            endpoint,
            headers=headers_get,
            params={"select": "id"}
        )
        r.raise_for_status()
        rows = r.json()

    if not rows:
        print("[INFO] clear_llm_table: no hay filas que borrar.")
        return

    ids = [row["id"] for row in rows]
    print(f"[INFO] clear_llm_table: ids a borrar -> {ids}")

    # 2) Borrar esos ids explícitamente
    headers_del = {
        "apikey": KEY,
        "Authorization": f"Bearer {KEY}",
        "Accept": "application/json",
        # queremos que devuelva las filas borradas para contarlas
        "Prefer": "return=representation",
    }

    # Construimos 'in.(1,2,3,...)'
    ids_str = ",".join(str(i) for i in ids)
    params_del = {"id": f"in.({ids_str})"}

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.delete(
            endpoint,
            headers=headers_del,
            params=params_del,
        )
        resp.raise_for_status()
        try:
            deleted = resp.json()
            num_deleted = len(deleted)
        except ValueError:
            num_deleted = 0

    print(f"[INFO] clear_llm_table: filas borradas realmente = {num_deleted}")



# ============================
# AGENTE ANALISTA (1 chunk)
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
            "sobre el propósito de las diferentes empresas Colombianas."
            "Cíñete a analizar respuestas sobre esa temática, si alguna es sobre otra temática, ignórala."
            "Si llegases a no tener respuestas relacionadas con el tema de interés, indícalo amablemente."
        ))

        human_message = HumanMessage(content=(
            f"Estas son respuestas sobre el {question_context}:\n\n"
            f"{answers_bullets}\n\n"
            "Sintetiza insights agregados:\n"
            "Interpreta el porcentaje de aceptación de cada idea y muéstralo claramente.\n"
            "Incluye también:\n"
            "1) 3-5 ejes de propósito más repetidos.\n"
            "2) Diferencias/complementos.\n"
            "3) Núcleo compartido.\n"
            "4) Una frase central resumen."
        ))

        ai_response = await llm.ainvoke([system_message, human_message])
        await add_usage(ai_response)  # << NUEVO
        return {"synthesized_insights": ai_response.content}

    return analysis_node


# ============================
# AGENTE REDACTOR (5 resúmenes)
# ============================
def make_writing_node_from_summaries(llm):
    async def writing_node(state: SurveyState) -> dict:
        summaries = state.get("synthesized_insights", "")
        question_context = state.get("question_context", "propósito de empresas colombianas")

        if not summaries:
            return {"final_report": f"No hay insights para {question_context}."}

        system_message = SystemMessage(content=(
            "Eres un consultor senior que redacta resúmenes ejecutivos.\n"
            "Tu tarea es identificar qué idea tiene un peso mayoritario sobre la totalidad "
            "de los encuestados (si la hay). Trata de representar variedad de respuestas cuando las haya si eres capaz de hacerlo sin que el texto total supere 2-3 líneas" # Si hay división, indícalo y muestra porcentajes.
            "Si te llegan consultas vacías de contenido porque el contenido a resumir no tenía nada, indica claramente que no dispones de la información que precisas."
        ))

        human_message = HumanMessage(content=(
            f"Contexto: {question_context}\n\n"
            f"A continuación tienes {N_CHUNK} resúmenes parciales (cada uno de 1/{N_CHUNK} de la muestra).\n"
            "Integra TODOS sin sesgo de orden y redacta un resumen ejecutivo final:\n\n"
            f"{summaries}\n\n"
            "Redacta 2-3 líneas que resuman rasgos comunes.\n"
            "Si hay ideas dividias trata de que se pueda inferir del texto solamente si eso no te supondrá superar 2-3 líneas."
            #"Si hay ideas divididas muestra claramente el porcentaje de aceptación."
        ))

        ai_response = await llm.ainvoke([system_message, human_message])
        await add_usage(ai_response)  # << NUEVO
        return {"final_report": ai_response.content}

    return writing_node


# ============================
# COMPARADOR FINAL
# ============================
def make_comparison_node(llm):
    async def comparison_node(state: ComparisonState) -> dict:
        purpose_now = state.get("purpose_now_report", "")
        purpose_future = state.get("purpose_future_report", "")

        system_message = SystemMessage(content=(
            "Eres un estratega experto en la evolución del propósito de las empresas colombianas de ahora a los próximos 18 años."
        ))
        human_message = HumanMessage(content=(
            "PROPÓSITO ACTUAL:\n"
            f"{purpose_now}\n\n"
            "PROPÓSITO 18 AÑOS:\n"
            f"{purpose_future}\n\n"
            "Compara y concluye en DOS líneas (máx 25 palabras)."
        ))

        ai_response = await llm.ainvoke([system_message, human_message])
        await add_usage(ai_response)  # << NUEVO
        return {"final_joint_conclusion": ai_response.content}

    return comparison_node


# ============================
# UTIL: cortar y chunkear en 5
# ============================
N_CHUNK=10
N_CHUNK = 10

def chunk_in_five(answers: List[str]) -> List[List[str]]:
    """
    Divide 'answers' en N_CHUNK trozos lo más equilibrados posible
    SIN descartar respuestas.

    - Si hay menos respuestas que N_CHUNK, algunos chunks tendrán 1 respuesta
      y otros 0, pero no se pierde ninguna.
    - Si hay resto, los primeros chunks tienen 1 elemento más.
    """
    n = len(answers)
    if n == 0:
        return [[] for _ in range(N_CHUNK)]

    base = n // N_CHUNK           # tamaño mínimo de cada chunk
    resto = n % N_CHUNK           # cuántos chunks tendrán un elemento extra

    chunks = []
    start = 0
    for i in range(N_CHUNK):
        size = base + (1 if i < resto else 0)
        end = start + size
        chunks.append(answers[start:end])
        start = end

    return chunks



# ============================
# PIPELINE BATCHED POR PREGUNTA
# ============================
async def run_batched_pipeline_for_question(
    analysis_llm,
    writing_llm,
    raw_answers: List[str],
    question_context: str
) -> str:
    chunks = chunk_in_five(raw_answers)
    analysis_node = make_analysis_node(analysis_llm)

    analysis_tasks = [
        analysis_node({"raw_answers": ch, "question_context": question_context})
        for ch in chunks
    ]
    analysis_results = await asyncio.gather(*analysis_tasks)

    summaries_text = ""
    for idx, res in enumerate(analysis_results, start=1):
        summaries_text += f"RESUMEN {idx}:\n{res.get('synthesized_insights','')}\n\n"

    writing_node = make_writing_node_from_summaries(writing_llm)
    final_state = await writing_node({
        "synthesized_insights": summaries_text,
        "question_context": question_context
    })

    return final_state.get("final_report", "")


# ============================
# MAIN
# ============================
async def main_async():
    t_start = time.perf_counter()  # << NUEVO: inicio timer

    load_environment()

    analysis_llm = build_azure_chat_model("AZURE_OPENAI_DEPLOYMENT_ANALYSIS")
    writing_llm  = build_azure_chat_model("AZURE_OPENAI_DEPLOYMENT_WRITING")
    comparison_llm = writing_llm

    total_rows = await count_rows_supabase("respuestas")
    print(f"\n[INFO] Registros actuales en BD (respuestas): {total_rows}\n")

    purpose_now_answers, purpose_future_answers = await fetch_purpose_answers_from_supabase()

    context_now = "propósito ACTUAL de las empresas colombianas"
    context_future = "propósito de las empresas colombianas dentro de 18 años"

    now_report, future_report = await asyncio.gather(
        run_batched_pipeline_for_question(analysis_llm, writing_llm, purpose_now_answers, context_now),
        run_batched_pipeline_for_question(analysis_llm, writing_llm, purpose_future_answers, context_future),
    )

    comparison_node = make_comparison_node(comparison_llm)
    comparison_final_state = await comparison_node({
        "purpose_now_report": now_report,
        "purpose_future_report": future_report
    })
    final_joint_conclusion = comparison_final_state.get("final_joint_conclusion", "")

    print("\n===== PROPÓSITO COMÚN ACTUAL =====\n")
    print(now_report)

    print("\n===== PROPÓSITO COMÚN A 18 AÑOS =====\n")
    print(future_report)

    print("\n===== CONCLUSIÓN CONJUNTA (2 líneas) =====\n")
    print(final_joint_conclusion)

    # >>> AQUÍ: antes de insertar, limpiamos la tabla
    await clear_llm_table()

    # >>> NUEVO: guardar resultados en tabla llm
    await insert_llm_results_to_supabase(
        pregunta_1=now_report,
        pregunta_2=future_report,
        resumen_final=final_joint_conclusion
    )
    print("\n[INFO] Resultados guardados en la tabla 'llm'.\n")


    # << NUEVO: fin timer + tokens/coste
    t_end = time.perf_counter()
    elapsed = t_end - t_start

    print("\n===== MÉTRICAS DE EJECUCIÓN =====")
    print(f"[INFO] Tiempo total: {elapsed:.2f} s")
    print(
        "[INFO] Tokens LLM usados -> "
        f"input: {USAGE['prompt_tokens']} | "
        f"output: {USAGE['completion_tokens']} | "
        f"total: {USAGE['total_tokens']}"
    )

    cost = estimate_cost_usd()
    if cost is not None:
        print(f"[INFO] Coste estimado aprox: ${cost:.4f} USD")
    else:
        print(
            "[INFO] Para ver coste, define PRICE_INPUT_PER_1M y PRICE_OUTPUT_PER_1M "
            "en el .env según tu modelo/deployment."
        )

if __name__ == "__main__":
    asyncio.run(main_async())
