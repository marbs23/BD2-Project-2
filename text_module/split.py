import re
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db_module.connection import get_connection

MIN_PALABRAS = 30  # chunks más cortos se descartan

def dividir_en_chunks(body: str) -> list[str]:
    parrafos = re.split(r'\n\n+', body.strip())
    
    chunks = []
    buffer = ""
    
    for parrafo in parrafos:
        parrafo = parrafo.strip()
        if not parrafo:
            continue
        
        buffer = (buffer + " " + parrafo).strip()
        
        if len(buffer.split()) >= MIN_PALABRAS:
            chunks.append(buffer)
            buffer = ""
    
    # buffer final solo se guarda si tiene suficientes palabras
    if len(buffer.split()) >= MIN_PALABRAS:
        chunks.append(buffer)
    
    return chunks


def procesar_documentos():
    conn = get_connection()
    cur = conn.cursor()

    # Traer todos los documentos
    cur.execute("SELECT doc_id, body FROM documents WHERE body IS NOT NULL")
    documentos = cur.fetchall()
    print(f"Documentos a procesar: {len(documentos)}")

    total_chunks = 0
    
    for doc_id, body in documentos:
        chunks = dividir_en_chunks(body)
        
        for position, content in enumerate(chunks):
            word_count = len(content.split())
            cur.execute("""
                INSERT INTO text_chunks (doc_id, content, position, word_count)
                VALUES (%s, %s, %s, %s)
            """, (doc_id, content, position, word_count))
            total_chunks += 1
        
    conn.commit()
    cur.close()
    conn.close()
    
    print(f"✅ Total chunks insertados: {total_chunks}")
    print(f"   Promedio por documento: {total_chunks / len(documentos):.1f}")


if __name__ == "__main__":
    procesar_documentos()