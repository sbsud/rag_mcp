import os
import logging
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-m3")

logger.info("Loading model %s ...", EMBED_MODEL)
_model = SentenceTransformer(EMBED_MODEL)
logger.info("Model loaded.")

app = FastAPI()

class EmbedRequest(BaseModel):
    texts: list[str]

class EmbedResponse(BaseModel):
    embeddings: list[list[float]]

@app.post("/embed", response_model=EmbedResponse)
def embed(req: EmbedRequest):
    vecs = _model.encode(req.texts, batch_size=16, convert_to_numpy=True).tolist()
    return EmbedResponse(embeddings=vecs)

@app.get("/health")
def health():
    return {"status": "ok"}