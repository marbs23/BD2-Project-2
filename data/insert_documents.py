import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db_module.connection import get_connection

def insertar_documentos(csv_path):
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
                row.get("category")
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
    insertar_documentos("data/articulos.csv")