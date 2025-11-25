# nube_palabras.py
#
# Lee `proposito_hoy` de la tabla `respuestas`,
# procesa el texto con NLTK (stopwords ES + stemming opcional),
# calcula frecuencias (Bag of Words) y las guarda en `nube_palabras`.

import os
import re
import string
from collections import Counter
from typing import List, Dict

import httpx
from dotenv import load_dotenv

import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import TweetTokenizer


# ===========================
# CONFIG / ENTORNO
# ===========================
def load_environment() -> None:
    load_dotenv()
    if not os.getenv("SUPABASE_URL") or not os.getenv("SUPABASE_KEY"):
        raise RuntimeError("Faltan SUPABASE_URL o SUPABASE_KEY en el .env")

    # Descargas necesarias la primera vez
    nltk.download("stopwords", quiet=True)
    nltk.download("punkt", quiet=True)


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
# preparamos objetos globales para no recrearlos en cada llamada
_STEMMER = SnowballStemmer("spanish")
_STOPWORDS_ES = set(stopwords.words("spanish"))
_PUNCT = set(string.punctuation) | {"¿", "¡", "…", "“", "”", "«", "»"}
_TOKENIZER = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)


def process_text(text: str, use_stem: bool = False) -> List[str]:
    """
    Limpia y tokeniza texto en castellano usando NLTK.

    - pasa a minúsculas
    - elimina URLs, menciones, números raros
    - tokeniza estilo tweet
    - elimina stopwords ES y puntuación
    - aplica stemming (opcional)
    """
    # minúsculas
    text = text.lower()

    # quitar URLs, menciones, números sueltos, etc.
    text = re.sub(r"https?://\S+", " ", text)   # URLs
    text = re.sub(r"@\w+", " ", text)          # @usuarios
    text = re.sub(r"\d+", " ", text)           # números
    text = re.sub(r"[\r\n\t]+", " ", text)     # saltos

    tokens = _TOKENIZER.tokenize(text)

    clean_tokens: List[str] = []
    for w in tokens:
        if w in _STOPWORDS_ES:
            continue
        if w in _PUNCT:
            continue
        if len(w) <= 1:
            continue  # descarta tokens muy cortos
        if not any(ch.isalpha() for ch in w):
            continue  # descarta tokens sin letras

        if use_stem:
            w = _STEMMER.stem(w)
        clean_tokens.append(w)

    return clean_tokens


# ===========================
# FRECUENCIAS (tipo build_freqs sin label)
# ===========================
def build_freqs(texts: List[str]) -> Dict[str, int]:
    """
    Bag of Words simple:
    devuelve dict palabra -> frecuencia total en el corpus.
    """
    freqs = Counter()
    for text in texts:
        for word in process_text(text):
            freqs[word] += 1
    print(f"[INFO] Palabras distintas encontradas: {len(freqs)}")
    return dict(freqs)


# ===========================
# SUPABASE: LIMPIAR + INSERTAR
# ===========================
def clear_nube_palabras_table(table: str = "nube_palabras") -> None:
    """
    Elimina todas las filas de la tabla `nube_palabras`.
    Asume que tiene columna `id` numérica.
    """
    url = os.environ["SUPABASE_URL"]
    key = os.environ["SUPABASE_KEY"]
    endpoint = f"{url}/rest/v1/{table}"

    headers = {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Accept": "application/json",
        "Prefer": "return=minimal",
    }

    params = {"id": "gt.0"}  # borra todo con id > 0

    with httpx.Client(timeout=30.0) as client:
        resp = client.delete(endpoint, headers=headers, params=params)
        resp.raise_for_status()

    print("[INFO] Tabla 'nube_palabras' limpiada correctamente.")


def insert_word_frequencies(freqs: Dict[str, int], table: str = "nube_palabras", top_n: int | None = None) -> None:
    """
    Inserta pares (palabra, frecuencia) en la tabla `nube_palabras`.

    - Si top_n está definido, solo sube las top_n palabras más frecuentes.
    - Se envían ya ordenadas de mayor a menor frecuencia.
    """
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

    # Construimos las filas en el mismo orden ya ordenado por frecuencia
    rows = [{"palabra": w, "frecuencia": int(f)} for w, f in items]

    with httpx.Client(timeout=60.0) as client:
        resp = client.post(endpoint, headers=headers, json=rows)
        resp.raise_for_status()

    print(f"[INFO] Filas insertadas en 'nube_palabras': {len(rows)}")

TOP_N=20

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

    # opcional: mostrar top 20 por consola
    print("\n[TOP 20 PALABRAS]")
    for palabra, freq in sorted(freqs.items(), key=lambda x: x[1], reverse=True)[:20]:
        print(f"{palabra}: {freq}")

    clear_nube_palabras_table()
    insert_word_frequencies(freqs,top_n=TOP_N)


if __name__ == "__main__":
    main()
