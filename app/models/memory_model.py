import time
from datetime import datetime, timedelta
import chromadb
from flask import jsonify
import json

chroma_client = chromadb.PersistentClient(path="./memory_db")
collection = chroma_client.get_or_create_collection(name="chat_memory")
MEMORY_FILE = "./app/memory/short_memory/short_memory.json" 
def add_memory(user_input, response, important=False):
    if important:
        timestamp = time.time()  
        current_ids = collection.get()["ids"]
        new_id = str(len(current_ids))

        collection.add(
            documents=[user_input],
            metadatas=[{"response": response, "timestamp": timestamp}],
            ids=[new_id],
        )
        
        

def get_memory(query):
    current_time = datetime.now()
    one_month_ago = current_time - timedelta(days=30)
    
    results = collection.query(query_texts=[query], n_results=3)
    filtered_conversations = []
    filtered_responses = []

    if results["documents"]:
        for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
            try:
                timestamp = datetime.fromtimestamp(float(meta["timestamp"]))
                if timestamp > one_month_ago:
                    filtered_conversations.append(doc)
                    filtered_responses.append(meta["response"])
            except Exception as e:
                print("Timestamp error:", e)
    
    return filtered_conversations, filtered_responses

def clear_memory():
    try:
         # Hapus seluruh koleksi memory
        chroma_client.delete_collection("chat_memory")

        # Buat ulang koleksi kosong
        global collection
        collection = chroma_client.get_or_create_collection(name="chat_memory")

        return jsonify({"message": "Semua memori berhasil dihapus."}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def add_to_short_memory(input, bot_reply):
    # Load existing memory
    
        
    try:
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            memory = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        memory = []

    # Buat entry baru dengan waktu sekarang
    entry = {
        "tsuki": input,
        "sayaka": bot_reply,
        
    }

    # Tambahkan ke memory dan simpan kembali
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