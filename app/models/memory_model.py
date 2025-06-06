import time
from datetime import datetime, timedelta
import chromadb
from flask import jsonify
import json
from sentence_transformers import SentenceTransformer
import os

chroma_client = chromadb.PersistentClient(path="./memory_db")
collection = chroma_client.get_or_create_collection(name="chat_memory")
MEMORY_FILE = "./app/memory/short_memory/short_memory.json" 

model = SentenceTransformer("all-MiniLM-L6-v2")  

import time
from typing import List

# fungsi untuk membagi teks menjadi chunk
def chunk_text(text: str, max_chunk_size: int = 500) -> List[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + max_chunk_size
        chunks.append(text[start:end])
        start = end
    return chunks

# fungsi untuk menambahkan memori ke collection dengan vector agar bisa di cari similarity
def add_memory(collection, model, path: str, response: str, important: bool = False):
    if important:
        timestamp = time.time()
        current_ids = collection.get()["ids"]
        new_id = str(len(current_ids))

        # Embed response (dokumen)
        embedding = model.encode(response).tolist()

        collection.add(
            embeddings=[embedding],
            metadatas=[{
                "timestamp": timestamp,
                "path": path
            }],
            ids=[new_id],
        )
        
# funsi untuk mencari  document relevat dengan yang dikirim hanya potonganya saja
def get_memory(collection, model, query: str, top_k: int = 3, max_chunk_size: int = 500) -> List[str]:
    query_embedding = model.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["metadatas", "distances"]
    )

    scored_chunks = []
    for metadata, distance in zip(results["metadatas"][0], results["distances"][0]):
        path = metadata.get('path')
        if not path:
            continue

        # baca isi file dokumen dari path
        with open(path, 'r', encoding='utf-8') as f:
            doc = f.read()

        # chunk isi dokumen
        chunks = chunk_text(doc, max_chunk_size)
        for chunk in chunks:
            scored_chunks.append((distance, chunk))

    scored_chunks.sort(key=lambda x: x[0])
    top_chunks = [chunk for _, chunk in scored_chunks[:top_k]]

    return top_chunks



def clear_memory():
    try:
        chroma_client.delete_collection("chat_memory")

        global collection
        collection = chroma_client.get_or_create_collection(name="chat_memory")

        return jsonify({"message": "Semua memori berhasil dihapus."}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def add_to_short_memory(input, bot_reply):
    
        
    try:
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            memory = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        memory = []

    entry = {
        "tsuki": input,
        "sayaka": bot_reply,
        
    }


    memory.append(entry)

    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(memory, f, ensure_ascii=False, indent=2)

def get_recent_memory(limit=3):
    try:
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            memory = json.load(f)
            return memory[-limit:]
    except (FileNotFoundError, json.JSONDecodeError):
        return []