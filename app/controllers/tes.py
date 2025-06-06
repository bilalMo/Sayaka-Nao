import chromadb


chroma_client = chromadb.PersistentClient(path="./memory_db")
collection = chroma_client.get_or_create_collection(name="chat_memory")
result = collection.get()

for i, doc in enumerate(result['documents']):
    print(f"ID: {result['ids'][i]}")
    print(f"Document: {doc}")
    print(f"Metadata: {result['metadatas'][i]}")
    print("------")
