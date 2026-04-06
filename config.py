import os

GEMINI_API_KEY = "AIzaSyA2vbQCDYocqMszjiGCWCkc4O6O8QCEipM"  
DATA_FILES = [
    "/Users/allikhankoshamet/Desktop/projects/RAG_Quran_Buhari/Holy-Quran-Russian.pdf",
    "/Users/allikhankoshamet/Desktop/projects/RAG_Quran_Buhari/ru4264.pdf"
]

EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CHUNK_SIZE = 300
CHUNK_OVERLAP = 60
FAISS_INDEX_PATH = "vector_store/faiss_index.bin"
METADATA_PATH = "vector_store/metadata.pkl"
GEMINI_MODEL = "gemini-1.5-flash"
DEFAULT_TOP_K = 5

import os
from dotenv import load_dotenv
load_dotenv()

# ====================== GEMINI API КЛЮЧ ======================
GEMINI_API_KEY = "твой_ключ_сюда_полностью"   # ←←← ВСТАВЬ СВОЙ КЛЮЧ ЗДЕСЬ

# ====================== ПУТИ К ТВОИМ PDF ======================
DATA_FILES = [
    "/Users/allikhankoshamet/Desktop/projects/RAG_Quran_Buhari/Holy-Quran-Russian.pdf",
    "/Users/allikhankoshamet/Desktop/projects/RAG_Quran_Buhari/ru4264.pdf"
]

EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CHUNK_SIZE = 300
CHUNK_OVERLAP = 60
FAISS_INDEX_PATH = "vector_store/faiss_index.bin"
METADATA_PATH = "vector_store/metadata.pkl"
GEMINI_MODEL = "gemini-1.5-flash"
DEFAULT_TOP_K = 5