from rag_core import load_pdfs, embed_and_index
docs = load_pdfs()
embed_and_index(docs, strategy="fixed")   # можно поменять на "recursive"
print("✅ Индексация завершена!")