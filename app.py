from datetime import datetime
from fastapi import FastAPI, HTTPException, Request
from contextlib import asynccontextmanager
from pydantic import BaseModel, Field
import time
import faiss
from sentence_transformers import SentenceTransformer
import os
import glob
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from agent import router as agent_router
import logging
import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# config & intitialization
load_dotenv()
DATA_DIR = os.getenv("DATA_DIR", "data/")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-mpnet-base-v2")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 300))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))
TOP_K_DEFAULT = 3

model = SentenceTransformer(EMBEDDING_MODEL)
index = None
metadata_store = []

class QueryRequest(BaseModel):
    query: str
    k: int = Field(default=TOP_K_DEFAULT, ge=1, le=6)


# data chunking

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    length_function=len,
    separators=["\n\n", "\n", ". ", " ", ""]  # Split on paragraphs, sentences, then words
)

def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    return text_splitter.split_text(text)

# Ingest & Index
def build_index():
    global index, metadata_store
    docs = []
    logger.info(f"Loading documents from {DATA_DIR}...")
    for file_path in glob.glob(os.path.join(DATA_DIR,"*.txt")):
        filename = os.path.basename(file_path)
        logger.info(f"  Processing: {filename}")
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        chunks = chunk_text(content)
        logger.info(f"    Created {len(chunks)} chunks")
        for chunk in chunks:
            docs.append((filename, chunk))
    if not docs:
        logger.error(f"No documents found in {DATA_DIR}")
        return

    # Embedding
    texts = [chunk for _, chunk in docs]
    logger.info(f"Encoding {len(texts)} chunks...")
    embeddings = model.encode(texts)
    dim = embeddings.shape[1]
    # Use HNSW for better retrieval
    index = faiss.IndexHNSWFlat(dim, 32)
    index.hnsw.efConstruction = 40
    index.hnsw.efSearch = 16
    index.add(embeddings)

    # Store metadata
    metadata_store = docs
    logger.info(f"✓ Indexed {len(docs)} chunks from {len(set(d[0] for d in docs))} files.")

# Retrieval
def retrieve(query, k):
    if index is None:
        raise RuntimeError("Index not initialized")
    query_vec = model.encode([query])
    distances, indices = index.search(query_vec, k)
    # Convert L2 distances to cosine similarities
    # For normalized vectors: cosine_sim = 1 - (L2_distance^2 / 2)
    # First normalize the query vector
    query_norm = query_vec / np.linalg.norm(query_vec)
    results = []
    similarities = []
    for i, idx in enumerate(indices[0]):
        doc_name, snippet = metadata_store[idx]
        first_8_words = " ".join(snippet.split()[:8])
        # Get chunk embedding
        chunk_embedding = model.encode([snippet])
        chunk_norm = chunk_embedding / np.linalg.norm(chunk_embedding)
        
        # Calculate cosine similarity
        cosine_sim = float(np.dot(query_norm, chunk_norm.T)[0][0])
        similarities.append(cosine_sim)
        results.append({
            "doc": doc_name, 
            "citation_preview": first_8_words, 
            "full_snippet": snippet,
            "similarity_score": round(cosine_sim, 4)
            })
    # Calculate average grounding score
    avg_grounding_score = float(np.mean(similarities)) if similarities else 0.0
    return results, avg_grounding_score

# Answer composition
def compose_answer(chunks):
    return " ".join([c["full_snippet"] for c in chunks])

# Startup

@asynccontextmanager
async def lifespan(app: FastAPI):
    global app_start_time
    app_start_time = datetime.now()
    logger.info("="*80)
    logger.info("Starting Claims Micro-RAG Application")
    logger.info("="*80)
    # Startup: Code to index the documents
    build_index()
    logger.info("Application ready!")
    yield
    logger.info("Shutting down application")

app = FastAPI(title="Claims Micro-RAG", lifespan=lifespan)

# Logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    
    # Log request
    logger.info(f"→ {request.method} {request.url.path}")
    
    # Process request
    response = await call_next(request)
    
    # Log response
    process_time = int((time.time() - start_time) * 1000)
    logger.info(f"← {request.method} {request.url.path} - {response.status_code} - {process_time}ms")
    
    # Add custom header with processing time
    response.headers["X-Process-Time-Ms"] = str(process_time)
    
    return response

# Include agent router
app.include_router(agent_router)

# Health check endpoint
@app.get("/healthz")
def health_check():
    """
    Health check endpoint
    Returns system status and basic metrics
    """
    uptime = (datetime.now() - app_start_time).total_seconds() if app_start_time else 0
    
    health_status = {
        "status": "healthy" if index is not None else "unhealthy",
        "timestamp": datetime.now().isoformat(),
        "uptime_seconds": round(uptime, 2),
        "index_ready": index is not None,
        "documents_indexed": len(metadata_store) if metadata_store else 0,
        "embedding_model": EMBEDDING_MODEL,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "version": "1.0.0"
    }
    
    if index is None:
        raise HTTPException(status_code=503, detail=health_status)
    
    return health_status


# Endpoint
@app.post("/ask")
def ask(req: QueryRequest):
    if index is None or len(metadata_store) == 0:
        raise HTTPException(status_code=503, detail="Index not ready")
    
    start = time.time()
    chunks, grounding_score = retrieve(req.query, req.k)
    answer = compose_answer(chunks)
    latency = int((time.time() - start) * 1000)
    citations = [
        {
            "doc": c["doc"], 
            "snippet": c["citation_preview"], 
            "full_snippet": c["full_snippet"]
        } 
        for c in chunks
    ]
    return {
        "answer" : answer,
        "citations" : citations,
        "retrieval" : {
            "k" : req.k, 
            "latency_ms" : latency,
            "grounding_score": round(grounding_score, 4)
            }
    }

