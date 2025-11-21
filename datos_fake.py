import os, json, time, re, csv
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(
    api_key=api_key,
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    temperature=0.2,
    timeout=60,
    max_retries=2,
)

# 1) leer empresas
with open("empresas_colombia.txt", "r", encoding="utf-8") as f:
    empresas = [ln.strip() for ln in f.readlines() if ln.strip()]

# quitar duplicados manteniendo orden
vistas = set()
empresas = [e for e in empresas if not (e in vistas or vistas.add(e))]

# 2) preguntas fijas
PREGUNTAS = {
    "sector": [
        "Agroindustria", "Manufactura", "Comercio", "Tecnología",
        "Construcción", "Energía y Minería", "Servicios", "Salud", "Otra"
    ],
    "ventas_anuales": [
        "Pequeña — < $1.000 M",
        "Mediana (baja) — $1.000–10.000 M",
        "Mediana (alta) — $10.000–50.000 M",
        "Grande — > $50.000 M"
    ],
    "empleados": ["1-50", "51-200", "201-500", ">500"],
    "adopcion_tecnologica": [
        "Bajo - Uso limitado de herramientas tecnológicas básicas",
        "Medio - Digitalización de algunos procesos",
        "Alto - Automatización, analítica, plataformas integradas",
        "Avanzado - Uso intensivo de tecnologías emergentes, IA, IoT, etc."
    ],
    "tecnologias_utilizadas": [
        "Inteligencia Artificial",
        "Automatización / Robótica",
        "Big Data y analítica avanzada",
        "Internet de las cosas (IoT)",
        "Computación en la nube",
        "Ciberseguridad",
        "Blockchain",
        "Realidad aumentada / virtual",
        "Impresión 3D",
        "No uso ninguna de las anteriores"
    ],
}

json_re = re.compile(r"\{.*\}", re.DOTALL)

# columnas del CSV (orden fijo)
HEADER = [
    "Empresa",
    "sector",
    "ventas_anuales",
    "empleados",
    "adopcion_tecnologica",
    "tecnologias_utilizadas",
    "proposito_hoy",
    "proposito_18_anos",
    "_latency_s"
]

def clean(v):
    """Quita saltos de línea internos y espacios raros para no romper filas."""
    if v is None:
        return ""
    v = str(v)
    v = v.replace("\r\n", " ").replace("\n", " ").replace("\r", " ")
    return v.strip()

out_path = "synthetic_survey.csv"

# 3) escribir CSV con csv.writer (utf-8-sig para Excel)
with open(out_path, "w", encoding="utf-8-sig", newline="") as fcsv:
    writer = csv.writer(fcsv, quoting=csv.QUOTE_MINIMAL)
    writer.writerow(HEADER)

    for i, empresa in enumerate(empresas, start=1):
        system = SystemMessage(content=(
            "Eres un miembro directivo de una empresa colombiana. "
            "Responderás un cuestionario de forma realista. "
            "Las preguntas abiertas deben tener máximo 2 líneas."
        ))

        human = HumanMessage(content=f"""
Eres un miembro directivo de la empresa {empresa}.
Responde este cuestionario. Devuelve SOLO un JSON válido (sin markdown),
con estas claves exactas:

1) sector: elige UNA opción:
{PREGUNTAS["sector"]}

2) ventas_anuales: elige UNA opción:
{PREGUNTAS["ventas_anuales"]}

3) empleados: elige UNA opción:
{PREGUNTAS["empleados"]}

4) adopcion_tecnologica: elige UNA opción:
{PREGUNTAS["adopcion_tecnologica"]}

5) tecnologias_utilizadas: elige CERO o MÁS opciones (lista de strings):
{PREGUNTAS["tecnologias_utilizadas"]}

6) proposito_hoy: texto corto, máximo 2 líneas.
7) proposito_18_anos: texto corto, máximo 2 líneas.

FORMATO OBLIGATORIO:
{{
  "sector": "...",
  "ventas_anuales": "...",
  "empleados": "...",
  "adopcion_tecnologica": "...",
  "tecnologias_utilizadas": ["...", "..."],
  "proposito_hoy": "...",
  "proposito_18_anos": "..."
}}
""".strip())

        t0 = time.perf_counter()
        resp = llm.invoke([system, human]).content
        latency = time.perf_counter() - t0

        # parsear JSON
        try:
            m = json_re.search(resp)
            data = json.loads(m.group(0)) if m else {}
        except Exception:
            data = {}

        # normalizar lista tecnologías -> string separado por ;
        tecs = data.get("tecnologias_utilizadas", [])
        if isinstance(tecs, list):
            tecs_str = "; ".join(tecs)
        else:
            tecs_str = str(tecs)

        row = [
            clean(empresa),
            clean(data.get("sector", "")),
            clean(data.get("ventas_anuales", "")),
            clean(data.get("empleados", "")),
            clean(data.get("adopcion_tecnologica", "")),
            clean(tecs_str),
            clean(data.get("proposito_hoy", "")),
            clean(data.get("proposito_18_anos", "")),
            round(latency, 4)
        ]

        writer.writerow(row)

        if i % 10 == 0:
            print(f"{i}/{len(empresas)} empresas procesadas...")

print("Listo. CSV guardado:", out_path)
