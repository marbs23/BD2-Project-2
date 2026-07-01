# Imagen del backend FastAPI (motor de búsqueda multimodal).
FROM python:3.12-slim

WORKDIR /app

# Librerías de sistema mínimas para psycopg2 y opencv headless.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Stopwords de nltk precargadas (evita descargarlas en cada arranque).
RUN python -c "import nltk; nltk.download('stopwords')"

COPY . .

EXPOSE 8000
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
