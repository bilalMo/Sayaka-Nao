from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI

# Load API key dari .env
load_dotenv()
api_key = os.getenv("google_api_key")

# Inisialisasi LLM
llm_model = ChatGoogleGenerativeAI(model="gemma-3n-e4b-it")

# Pertanyaan user
question = "ingat ga kemarin kita ngobrol apaan ?"

prompt = f"""
Pertanyaan: {question}
Apakah pertanyaan ini **lebih baik dijawab** dengan melihat isi database atau memori chat sebelumnya? 
Berikan hanya satu angka sebagai jawaban:
- Jawab **1** jika informasi dari database/memori akan membantu menjawab pertanyaan ini.
- Jawab **0** jika cukup dijawab langsung tanpa melihat database atau memori.
Jawaban:
"""

# Panggil model
answer = llm_model.invoke(prompt)

print("Jawaban:", answer)
