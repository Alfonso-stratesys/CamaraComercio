# nube_palabras_llm_filter.py
#
# Igual que nube_palabras.py, pero antes de subir a BD
# pasa las palabras por un filtro LLM que decide cuáles son
# relevantes en el contexto del propósito de empresas MEGA.

import os
import re
import string
import json
from collections import Counter
from typing import List, Dict

import httpx
from dotenv import load_dotenv

import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import TweetTokenizer

# LLM (mismo estilo que el resto de tu repo)
from langchain_groq import ChatGroq
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

# ===========================
# CONFIG / ENTORNO
# ===========================
TOP_N = 20
# cuántas palabras candidatas máximo mandamos al LLM
LLM_MAX_CANDIDATES = 5 * TOP_N  # por ejemplo, top 100

def load_environment() -> None:
    load_dotenv()
    if not os.getenv("SUPABASE_URL") or not os.getenv("SUPABASE_KEY"):
        raise RuntimeError("Faltan SUPABASE_URL o SUPABASE_KEY en el .env")

    # Descargas necesarias la primera vez
    nltk.download("stopwords", quiet=True)
    nltk.download("punkt", quiet=True)

    if not (os.getenv("GROQ_API_KEY") or os.getenv("AZURE_OPENAI_API_KEY")):
        print("[AVISO] No hay configuración de LLM (GROQ_API_KEY o AZURE_OPENAI_API_KEY).")


def build_llm():
    """
    Usa Azure si tiene deployment configurado, si no Groq.
    """
    if os.getenv("AZURE_OPENAI_API_KEY") and os.getenv("AZURE_OPENAI_DEPLOYMENT_ANALYSIS"):
        return AzureChatOpenAI(
            azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_ANALYSIS"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            temperature=0.0,
            timeout=30,
            max_retries=2,
        )
    elif os.getenv("GROQ_API_KEY"):
        return ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"),
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            temperature=0.0,
            timeout=30,
            max_retries=2,
        )
    else:
        raise RuntimeError("No se encontró configuración de LLM (ni Azure ni Groq).")

# ===========================
# LECTURA DE RESPUESTAS
# ===========================
def fetch_purpose_now_answers() -> List[str]:
    """
    Lee todas las respuestas de la columna `proposito_hoy`
    de la tabla `respuestas` en Supabase.
    """
    url = os.environ["SUPABASE_URL"]
    key = os.environ["SUPABASE_KEY"]

    endpoint = f"{url}/rest/v1/respuestas"
    headers = {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Accept": "application/json",
    }
    params = {"select": "proposito_hoy"}

    with httpx.Client(timeout=30.0) as client:
        r = client.get(endpoint, headers=headers, params=params)
        r.raise_for_status()
        rows = r.json()

    answers: List[str] = []
    for row in rows:
        txt = (row.get("proposito_hoy") or "").strip()
        if txt:
            answers.append(txt)

    print(f"[INFO] Respuestas de propósito actual leídas: {len(answers)}")
    return answers


# ===========================
# PREPROCESADO NLTK (tipo process_tweet)
# ===========================
_STEMMER = SnowballStemmer("spanish")
_STOPWORDS_ES = set(stopwords.words("spanish"))
_PUNCT = set(string.punctuation) | {"¿", "¡", "…", "“", "”", "«", "»"}
_TOKENIZER = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)


def process_text(text: str, use_stem: bool = False) -> List[str]:
    """
    Limpia y tokeniza texto en castellano usando NLTK.
    """
    text = text.lower()
    text = re.sub(r"https?://\S+", " ", text)   # URLs
    text = re.sub(r"@\w+", " ", text)           # @usuarios
    text = re.sub(r"\d+", " ", text)            # números
    text = re.sub(r"[\r\n\t]+", " ", text)      # saltos

    tokens = _TOKENIZER.tokenize(text)

    clean_tokens: List[str] = []
    for w in tokens:
        if w in _STOPWORDS_ES:
            continue
        if w in _PUNCT:
            continue
        if len(w) <= 1:
            continue
        if not any(ch.isalpha() for ch in w):
            continue

        if use_stem:
            w = _STEMMER.stem(w)
        clean_tokens.append(w)

    return clean_tokens


# ===========================
# FRECUENCIAS
# ===========================
def build_freqs(texts: List[str]) -> Dict[str, int]:
    freqs = Counter()
    for text in texts:
        for word in process_text(text):
            freqs[word] += 1
    print(f"[INFO] Palabras distintas encontradas: {len(freqs)}")
    return dict(freqs)


# ===========================
# FILTRADO LLM DE PALABRAS RELEVANTES
# ===========================
def filter_relevant_words_with_llm(
    candidate_words: List[str],
    llm,
) -> List[str]:
    """
    Envía una lista de palabras al LLM para que filtre SOLO las
    relevantes al contexto:
      - propósito de empresas colombianas
      - proyecto MEGA

    Devuelve una lista de palabras aceptadas.
    """

    if not candidate_words:
        return []

    # Por seguridad, eliminamos duplicados preservando orden
    seen = set()
    unique_words = []
    for w in candidate_words:
        if w not in seen:
            unique_words.append(w)
            seen.add(w)

    # Construimos prompt
    system = SystemMessage(content=(
        "Eres un analista de texto para una encuesta sobre el propósito "
        "de diferentes empresas colombianas que participan en el proyecto "
        "MEGA de la Cámara de Comercio.\n"
        "Te doy una lista de palabras que aparecen en las respuestas. "
        "Debes seleccionar SOLO aquellas que sean relevantes como conceptos "
        "de propósito empresarial, estrategia, impacto, valores, clientes, "
        "stakeholders, innovación, crecimiento, sostenibilidad, etc.\n"
        "Descarta palabras genéricas o poco informativas como 'respuesta', "
        "'pregunta', 'empresa', 'hoy', 'mañana', 'realizar', etc."
    ))

    lista_palabras = "\n".join(f"- {w}" for w in unique_words)

    human = HumanMessage(content=(
        "Lista de palabras detectadas (cada línea es una palabra):\n\n"
        f"{lista_palabras}\n\n"
        "Devuelve SOLO un JSON válido, sin markdown, con el siguiente formato:\n\n"
        "{\n"
        '  "palabras_relevantes": ["palabra1", "palabra2", ...]\n'
        "}\n\n"
        "Incluye únicamente las palabras que consideres relevantes según el contexto.\n"
        "No inventes palabras nuevas."
    ))

    resp = llm.invoke([system, human])
    text = resp.content

    try:
        data = json.loads(text)
    except Exception:
        # Por si el modelo añade texto extra, buscamos un objeto JSON en bruto
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            print("[WARN] No se pudo parsear JSON de filtro LLM. Se usan todas las candidatas.")
            return unique_words
        try:
            data = json.loads(match.group(0))
        except Exception:
            print("[WARN] JSON de filtro LLM inválido. Se usan todas las candidatas.")
            return unique_words

    palabras_relevantes = data.get("palabras_relevantes", [])
    if not isinstance(palabras_relevantes, list):
        print("[WARN] Formato inesperado en 'palabras_relevantes'. Se usan todas las candidatas.")
        return unique_words

    # Limpiamos un poco la salida
    cleaned = []
    for w in palabras_relevantes:
        if not isinstance(w, str):
            continue
        w2 = w.strip().lower()
        if w2:
            cleaned.append(w2)

    print(f"[INFO] Palabras aceptadas por LLM: {len(cleaned)} de {len(unique_words)} candidatas")
    return cleaned


# ===========================
# SUPABASE: LIMPIAR + INSERTAR
# ===========================
def clear_nube_palabras_table(table: str = "nube_palabras") -> None:
    url = os.environ["SUPABASE_URL"]
    key = os.environ["SUPABASE_KEY"]
    endpoint = f"{url}/rest/v1/{table}"

    headers = {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Accept": "application/json",
        "Prefer": "return=minimal",
    }

    params = {"id": "gt.0"}

    with httpx.Client(timeout=30.0) as client:
        resp = client.delete(endpoint, headers=headers, params=params)
        resp.raise_for_status()

    print("[INFO] Tabla 'nube_palabras' limpiada correctamente.")


def insert_word_frequencies(
    freqs: Dict[str, int],
    table: str = "nube_palabras",
    top_n: int | None = None
) -> None:
    if not freqs:
        print("[INFO] No hay frecuencias que insertar.")
        return

    # Ordenar de mayor a menor frecuencia
    items = sorted(freqs.items(), key=lambda x: x[1], reverse=True)

    # Quedarse solo con las N primeras (si se indica)
    if top_n is not None:
        items = items[:top_n]

    url = os.environ["SUPABASE_URL"]
    key = os.environ["SUPABASE_KEY"]
    endpoint = f"{url}/rest/v1/{table}"

    headers = {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Prefer": "return=minimal",
    }

    rows = [{"palabra": w, "frecuencia": int(f)} for w, f in items]

    with httpx.Client(timeout=60.0) as client:
        resp = client.post(endpoint, headers=headers, json=rows)
        resp.raise_for_status()

    print(f"[INFO] Filas insertadas en 'nube_palabras': {len(rows)}")


# ===========================
# MAIN
# ===========================
def main():
    load_environment()

    texts = fetch_purpose_now_answers()
    if not texts:
        print("[WARN] No hay respuestas de propósito actual.")
        return

    freqs = build_freqs(texts)

    # 1) Creamos lista de palabras candidatas ordenadas por frecuencia
    sorted_items = sorted(freqs.items(), key=lambda x: x[1], reverse=True)
    if LLM_MAX_CANDIDATES is not None:
        sorted_items = sorted_items[:LLM_MAX_CANDIDATES]

    candidate_words = [w for w, _ in sorted_items]

    # 2) LLM filtra palabras relevantes
    llm = build_llm()
    relevant_words = filter_relevant_words_with_llm(candidate_words, llm)

    # 3) Nos quedamos solo con las frecuencias de palabras aceptadas
    filtered_freqs = {w: f for w, f in freqs.items() if w in relevant_words}

    # opcional: mostrar top 20 filtrado por consola
    print("\n[TOP PALABRAS RELEVANTES DESPUÉS DE LLM]")
    for palabra, freq in sorted(filtered_freqs.items(), key=lambda x: x[1], reverse=True)[:TOP_N]:
        print(f"{palabra}: {freq}")

    clear_nube_palabras_table()
    insert_word_frequencies(filtered_freqs, top_n=TOP_N)


if __name__ == "__main__":
    main()
