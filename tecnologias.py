import os
import asyncio
from collections import Counter

import httpx
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_KEY"]

TABLE_RESPUESTAS = "respuestas"
TABLE_TECHS = "techs"

RESP_ENDPOINT = f"{SUPABASE_URL}/rest/v1/{TABLE_RESPUESTAS}"
TECHS_ENDPOINT = f"{SUPABASE_URL}/rest/v1/{TABLE_TECHS}"

HEADERS = {
    "apikey": SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}",
    "Content-Type": "application/json",
    "Accept": "application/json",
}


async def fetch_all_techs():
    """
    Lee la columna 'techs' de la tabla 'respuestas'.
    """
    async with httpx.AsyncClient(timeout=30.0) as client:
        params = {"select": "techs", "limit": "10000"}
        r = await client.get(RESP_ENDPOINT, headers=HEADERS, params=params)
        r.raise_for_status()
        rows = r.json()
    return rows


async def clear_techs_table():
    """
    Borra todos los registros de la tabla 'techs'.
    """
    async with httpx.AsyncClient(timeout=30.0) as client:
        params = {"tecnologia": "not.is.null"}
        r = await client.delete(TECHS_ENDPOINT, headers=HEADERS, params=params)
        if r.status_code not in (200, 204):
            print("[AVISO] No se pudo limpiar la tabla 'techs':", r.status_code, r.text)


async def insert_techs_stats(stats_rows):
    """
    Inserta en la tabla 'techs' una lista de filas:
    [{"tecnologia": ..., "adopcion": ...}, ...]
    """
    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.post(TECHS_ENDPOINT, headers=HEADERS, json=stats_rows)
        if r.status_code >= 300:
            print("[ERROR] Insertando en 'techs':", r.status_code, r.text)
        else:
            print("[OK] Estadísticas de tecnologías insertadas en 'techs'.")


async def main():
    # 1) Leer techs de todas las empresas
    rows = await fetch_all_techs()

    total_empresas = len(rows)
    if total_empresas == 0:
        print("No hay registros en 'respuestas'. Nada que calcular.")
        return

    # 2) Contar adopción por tecnología
    counter = Counter()

    for row in rows:
        tech_list = row.get("techs")
        if not tech_list:
            continue
        if isinstance(tech_list, str):
            continue

        # Cada empresa cuenta como máximo 1 vez por tecnología
        unique_techs = {str(t).strip() for t in tech_list if t}
        for tech_clean in unique_techs:
            if tech_clean:
                counter[tech_clean] += 1

    if not counter:
        print("No se encontraron tecnologías en los registros.")
        return

    # 3) Calcular porcentajes
    stats_rows = []
    for tech, count in counter.items():
        adopcion_pct = (count / total_empresas) * 100.0
        adopcion_pct = round(adopcion_pct, 2)  # float con 2 decimales
        stats_rows.append(
            {
                "tecnologia": tech,
                "adopcion": adopcion_pct,
            }
        )

    # ✅ ORDENAR DESCENDENTE POR ADOPCIÓN ANTES DE INSERTAR
    stats_rows.sort(key=lambda r: r["adopcion"], reverse=True)  # <--

    # 4) Limpiar tabla 'techs' y volcar nuevas estadísticas
    await clear_techs_table()
    await insert_techs_stats(stats_rows)

    print("Total empresas:", total_empresas)
    print("Tecnologías detectadas:", len(stats_rows))
    for row in stats_rows:
        print(f"{row['tecnologia']}: {row['adopcion']} %")


if __name__ == "__main__":
    asyncio.run(main())
