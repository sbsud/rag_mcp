FROM python:3.11-slim

WORKDIR /app

COPY requirements-embedder.txt .

RUN pip install --no-cache-dir -r requirements-embedder.txt

RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-m3')"

COPY embedder_service.py .

CMD ["uvicorn", "embedder_service:app", "--host", "0.0.0.0", "--port", "8006"]