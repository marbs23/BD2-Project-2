"""
Ingesta de artículos a la BD (maintainer)

Lee data/articulos.csv y lo carga en la tabla documents (ON CONFLICT por url, así
es idempotente). Es el primer paso del pipeline de indexado de texto.
"""
import pandas as pd

from src.core.db import get_connection
from src.core.paths import DATA_DIR

CSV_PATH = DATA_DIR / "articulos.csv"


def insertar_documentos(csv_path=CSV_PATH):
    df = pd.read_csv(csv_path)

    # Limpiar filas sin body o title
    df = df.dropna(subset=["title", "body"])
    df["category"] = df["category"].fillna("Unknown")
    df["description"] = df["description"].fillna("")
    df["image_url"] = df["image_url"].fillna("")
    print(f"Artículos a insertar: {len(df)}")

    conn = get_connection()
    cur = conn.cursor()

    insertados = 0
    omitidos = 0

    for _, row in df.iterrows():
        try:
            cur.execute("""
                INSERT INTO documents (url, title, description, body, image_url, category)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (url) DO NOTHING
            """, (
                row.get("url"),
                row.get("title"),
                row.get("description"),
                row.get("body"),
                row.get("image_url"),
                row.get("category"),
            ))
            insertados += 1
        except Exception as e:
            print(f"❌ Error en fila: {e}")
            omitidos += 1

    conn.commit()
    cur.close()
    conn.close()

    print(f"✅ Insertados: {insertados}")
    print(f"⚠️  Omitidos:  {omitidos}")


if __name__ == "__main__":
    insertar_documentos()
