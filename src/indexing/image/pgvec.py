"""Conversión de vectores a la representación textual que entiende pgvector."""


def to_pgvector(arr) -> str:
    """Formatea un iterable de números como '[v1,v2,...]' para columnas vector.

    Si `arr` es un array de numpy se convierte a lista de floats de Python con
    `.tolist()` antes de formatear: iterar el array directamente entrega escalares
    numpy, mucho más lentos de formatear que los float nativos (clave al persistir
    cientos de miles de histogramas densos).
    """
    values = arr.tolist() if hasattr(arr, "tolist") else arr
    return "[" + ",".join(f"{x:.6g}" for x in values) + "]"
