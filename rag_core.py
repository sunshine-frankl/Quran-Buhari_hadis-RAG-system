import fitz, nltk, os, pickle
from typing import List, Dict
from sentence_transformers import SentenceTransformer
import faiss
from google import genai
from config import *
os.makedirs(VECTOR_DIR, exist_ok=True)

nltk.download('punkt', quiet=True)
model = SentenceTransformer(EMBEDDING_MODEL)

# ====================== 1. INGESTION + METADATA ======================
def load_pdfs() -> List[Dict]:
    documents = []
    for path in DATA_FILES:
        if not os.path.exists(path):
            print(f"❌ Файл не найден: {path}")
            continue
            
        filename = os.path.basename(path)
        doc = fitz.open(path)
        text = ""
        for page_num, page in enumerate(doc):
            page_text = page.get_text("text")
            reference = f"Page {page_num+1}"
            text += f"\n[REFERENCE: {reference} | SOURCE: {filename}]\n{page_text}\n"
        documents.append({
            "source": filename,
            "title": filename.replace(".pdf", "").title(),
            "text": text.strip(),
            "metadata": {"source": filename, "title": filename.replace(".pdf", "").title()}
        })
        doc.close()
    print(f"✅ Загружено {len(documents)} PDF файла")
    return documents

# ====================== 2. CHUNKING (две стратегии) ======================
def fixed_size_chunking(text: str, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP) -> List[str]:
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if len(chunk.split()) >= 100:
            chunks.append(chunk)
    return chunks

def recursive_chunking(text: str) -> List[str]:
    sentences = nltk.sent_tokenize(text)
    chunks, current = [], ""
    for sent in sentences:
        if len(current.split()) + len(sent.split()) > 400:
            if len(current.split()) >= 100:
                chunks.append(current.strip())
            current = sent
        else:
            current += " " + sent
    if current:
        chunks.append(current.strip())
    return chunks

# ====================== 3+4. EMBEDDING + VECTOR STORE (FAISS) ======================
def embed_and_index(docs: List[Dict], strategy="fixed"):
    all_chunks = []
    for doc in docs:
        if strategy == "fixed":
            chunks = fixed_size_chunking(doc["text"])
        else:
            chunks = recursive_chunking(doc["text"])
        for i, c in enumerate(chunks):
            all_chunks.append({"text": c, "metadata": doc["metadata"], "chunk_id": i})
    
    texts = [c["text"] for c in all_chunks]
    embeddings = model.encode(texts, convert_to_tensor=False)
    faiss.normalize_L2(embeddings)
    
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    
    os.makedirs("vector_store", exist_ok=True)
    faiss.write_index(index, FAISS_INDEX_PATH)
    with open(METADATA_PATH, "wb") as f:
        pickle.dump(all_chunks, f)
    print(f"✅ Проиндексировано {len(all_chunks)} чанков (стратегия: {strategy})")

# ====================== RETRIEVAL ======================
def retrieve(query: str, top_k=DEFAULT_TOP_K) -> List[Dict]:
    index = faiss.read_index(FAISS_INDEX_PATH)
    with open(METADATA_PATH, "rb") as f:
        metadata = pickle.load(f)
    q_emb = model.encode([query])
    faiss.normalize_L2(q_emb)
    _, indices = index.search(q_emb, top_k)
    return [metadata[i] for i in indices[0] if i < len(metadata)]

# ====================== 5. GENERATION + СТРОГИЙ ПРОМПТ ======================
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

SYSTEM_PROMPT = """
You are a knowledgeable and truthful assistant specialized in Islamic studies. 

Answer the user's question **strictly using only the provided context** below. 

Rules:
1. Never use knowledge outside the retrieved documents.
2. For every factual claim or quote, you MUST cite the source.
3. If the answer is not supported by the context, respond EXACTLY with:
   "I cannot find this in the provided documents."
4. Answer in English. Be respectful and concise.

Context:
{context}
"""

def generate_answer(query: str, retrieved_docs: List[Dict]) -> str:
    context = "\n\n---\n\n".join([d['text'] for d in retrieved_docs])
    full_prompt = SYSTEM_PROMPT.format(context=context) + f"\n\nUser Question: {query}"
    model_gen = genai.GenerativeModel(GEMINI_MODEL)
    response = model_gen.generate_content(full_prompt)
    return response.text.strip()