import os, time, asyncio, random, statistics, json, httpx, re
from dotenv import load_dotenv

from langchain_openai import AzureChatOpenAI
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()

# --------------------
# SUPABASE
# --------------------
URL = os.environ["SUPABASE_URL"]
KEY = os.environ["SUPABASE_KEY"]
TABLE = "respuestas"

endpoint = f"{URL}/rest/v1/{TABLE}"
headers = {
    "apikey": KEY,
    "Authorization": f"Bearer {KEY}",
    "Content-Type": "application/json",
    "Prefer": "return=minimal",
}

# --------------------
# LLM (igual estilo ximena2.py)
# --------------------
def build_llm():
    if os.getenv("AZURE_OPENAI_API_KEY") and os.getenv("AZURE_OPENAI_DEPLOYMENT_ANALYSIS"):
        print("Usando AzureChatOpenAI")
        return AzureChatOpenAI(
            azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_ANALYSIS"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            temperature=0.71,
            timeout=30,
            max_retries=2,
        )
    else:
        print("Usando ChatGroq")
        return ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"),
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            temperature=1.971,
            timeout=30,
            max_retries=2,
        )

llm = build_llm()

# --------------------
# Empresas (como data_fake_DB.py)
# --------------------
with open("empresas_colombia.txt", "r", encoding="utf-8") as f:
    empresas = [ln.strip() for ln in f.readlines() if ln.strip()]

# quitar duplicados manteniendo orden
vistas = set()
empresas = [e for e in empresas if not (e in vistas or vistas.add(e))]

# Número de filas a enviar
N = 10 # cambia a lo que quieras
if N > len(empresas):
    # si pides más que empresas, reciclamos
    empresas_for_tasks = [empresas[i % len(empresas)] for i in range(N)]
else:
    # si hay suficientes, tomamos N distintas
    empresas_for_tasks = empresas[:N]

# --- Opciones aleatorias ---
INDUSTRIAS = [
    "Agroindustria", "Manufactura", "Comercio", "Tecnología",
    "Construcción", "Energía y Minería", "Servicios", "Salud", "Otra"
]

VOLUMEN_VENTAS = [
    "Pequeña — < $1.000 M",
    "Mediana (baja) — $1.000–10.000 M",
    "Mediana (alta) — $10.000–50.000 M",
    "Grande — > $50.000 M"
]

EMPLEADOS = ["1-50", "51-200", "201-500", ">500"]

ADOPCION_TECH = [
    "Bajo - Uso limitado de herramientas tecnológicas básicas",
    "Medio - Digitalización de algunos procesos",
    "Alto - Automatización, analítica, plataformas integradas",
    "Avanzado - Uso intensivo de tecnologías emergentes, IA, IoT, etc."
]

TECHS = [
    "Inteligencia Artificial",
    "Automatización / Robótica",
    "Big Data y analítica avanzada",
    "Internet de las cosas (IoT)",
    "Computación en la nube",
    "Ciberseguridad",
    "Blockchain",
    "Realidad aumentada / virtual"
]

json_re = re.compile(r"\{.*\}", re.DOTALL)
eco_friendly="Haz especial énfasis en tu interés por mirar por el bien del medio ambiente y ser una empresa eco-friendly"
innovacion_tech="Haz especial énfasis en tu interés por el desarrollo tecnológico de la empresa y dominar el mercado"
fuck_france="Muestra tu total compromiso con las ventajas de la tortilla con cebolla"
France_friends="Muestra tu total compromiso con las ventajas de la tortilla sin cebolla"
prompt_especifico="Haz especial énfasis en tu interés por ser una empresa de vanguardia que adopte rápido las nuevas tecnologías y se adapte a todos los cambios, manteniéndte como referente de tu sector"
# --------------------
# IA: GENERAR TEXTOS ABIERTOS
# ahora con empresa distinta por llamada
# --------------------
async def generate_open_answers(i, empresa):
    system = SystemMessage(content=(
        # "Eres un directivo de una empresa colombiana. "
        "Responde de forma realista. No más de 2-3 frases por respuesta."
    ))

    prompt = f"""
Eres un miembro directivo de la empresa colombiana {empresa}.

Genera 2 respuestas de 2-3 frases para una encuesta :

1. Proposito de la empresa hoy: estilo profesional, max 2/3 frases.
2. Proposito de la empresa dentro de 18 anos: vision futura, max 2/3 frases.

{prompt_especifico}

Devuelve SOLO un JSON válido sin markdown, exactamente así:

{{
  "proposito_hoy": "...",
  "proposito_18": "..."
}}
""".strip()

    human = HumanMessage(content=prompt)

    resp = await llm.ainvoke([system, human])
    text = resp.content

    m = json_re.search(text)
    if not m:
        return {"proposito_hoy": "", "proposito_18": ""}

    try:
        return json.loads(m.group(0))
    except Exception:
        return {"proposito_hoy": "", "proposito_18": ""}


def make_row(ai_resp, empresa):
    return {
        "industria": random.choice(INDUSTRIAS),
        "volumen_ventas": random.choice(VOLUMEN_VENTAS),
        "empleados": random.choice(EMPLEADOS),
        "adopcion_tech": random.choice(ADOPCION_TECH),
        "proposito_hoy": ai_resp.get("proposito_hoy", ""),
        "proposito_18": ai_resp.get("proposito_18", ""),
        "techs": random.sample(TECHS, k=random.randint(1, 4)),
        # Si tu tabla tiene esta columna, descomenta:
        # "empresa": empresa,
    }


async def send_one(i, http_client, start_event, times, errors):
    await start_event.wait()

    empresa = empresas_for_tasks[i]

    # 1) Generar respuesta IA asumiendo empresa distinta
    ai_resp = await generate_open_answers(i, empresa)

    # 2) Construir fila
    row = make_row(ai_resp, empresa)

    # 3) Enviar a Supabase
    t0 = time.perf_counter()
    res = await http_client.post(endpoint, headers=headers, json=row)
    t1 = time.perf_counter()

    times.append(t1 - t0)
    if res.status_code >= 300:
        errors.append((i, empresa, res.status_code, res.text))


async def main():
    times = []
    errors = []
    start_event = asyncio.Event()

    limits = httpx.Limits(max_connections=1000, max_keepalive_connections=1000)

    async with httpx.AsyncClient(limits=limits, timeout=30.0) as client:
        tasks = [
            asyncio.create_task(send_one(i, client, start_event, times, errors))
            for i in range(N)
        ]

        await asyncio.sleep(0)  # barrera lista

        t_start = time.perf_counter()
        start_event.set()       # dispara la ráfaga
        await asyncio.gather(*tasks)
        t_end = time.perf_counter()

    T_total = t_end - t_start
    print(f"Total (ráfaga): {T_total:.3f} s")
    print(f"Throughput efectivo: {N/T_total:.2f} req/s")

    if times:
        print(f"Media por request: {statistics.mean(times)*1000:.2f} ms")
        print(f"p50: {statistics.median(times)*1000:.2f} ms")
        print(f"p95: {statistics.quantiles(times, n=20)[-1]*1000:.2f} ms")

    if errors:
        print(f"Errores: {len(errors)}")
        print("Primeros 5 errores:")
        for e in errors[:5]:
            print(e)


if __name__ == "__main__":
    asyncio.run(main())
