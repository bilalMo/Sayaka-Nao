from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI

# Load API key dari .env
load_dotenv()
api_key = os.getenv("google_api_key")

# Inisialisasi LLM
llm_model = ChatGoogleGenerativeAI(model="gemma-3-12b-it", max_tokens=200)

# Pertanyaan user
question = "jawab hanya 1 atau 0"

# Buat prompt sederhana
prompt = f"Jawab singkat: {question}"

# Panggil model
answer = llm_model.invoke(prompt)

print("Jawaban:", answer)
