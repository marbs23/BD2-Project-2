"""Conversión de vectores a la representación textual que entiende pgvector."""


def to_pgvector(arr) -> str:
    """Formatea un iterable de números como '[v1,v2,...]' para columnas vector."""
    return "[" + ",".join(f"{float(x):.6g}" for x in arr) + "]"
