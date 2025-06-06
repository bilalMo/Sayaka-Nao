import random
from dotenv import load_dotenv
import os
import json
from app.models.memory_model import add_memory, get_memory, add_to_short_memory,get_recent_memory
from langchain_google_genai import ChatGoogleGenerativeAI
from app.persona import persona_learn, persona_daily, memory_triggers
import requests
from flask import jsonify


class ChatController:
    def __init__(self, api_key_env_name="google_api_key", dialogues_path="app/initial_dialogues.json"):
        load_dotenv()

        self.llm_model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", max_tokens=None)
        self.gemini =self.llm_model
        self.persona_learn = persona_learn
        self.persona_daily = persona_daily
        self.memory_triggers = memory_triggers
        self.initial_dialogues = self._load_initial_dialogues(dialogues_path)
        self.last_chat = get_recent_memory()

    def _load_initial_dialogues(self, path):
        try:
            with open(path, "r", encoding="utf-8") as file:
                return json.load(file)
        except FileNotFoundError:
            print(f"[WARNING] File {path} tidak ditemukan. Memori awal kosong.")
            return []

    def chat(self, user_input):
        use_memory = any(word in user_input.lower() for word in self.memory_triggers)
        past_conversations, past_responses = get_memory(user_input) if use_memory else ([], [])
        
        recent_memory = list(zip(past_conversations, past_responses))[-3:]
        memory_context = "\n".join(
            [f"Tsuki: {conv}\nSayaka: {resp}" for conv, resp in recent_memory]
        )


        initial_context = "\n".join(
            [f"Tsuki: {dialogue['tsuki']}\nSayaka: {dialogue['sayaka']}" for dialogue in self.initial_dialogues]
        )

      
        last_chat = "\n".join(
            [f"Tsuki: {dialogue['tsuki']}\nSayaka: {dialogue['sayaka']}" for dialogue in self.last_chat]
        )

        prompt_daily = f"""{persona_daily}

        Gunakan informasi relevan dari pembicaraan sebelumnya jika membantu:
        {memory_context}

        Berikut ini beberapa contoh cara Sayaka merespons Tsuki secara alami dan lembut:
        {initial_context}

        Sebelum membalas, kamu harus membaca dan memahami dulu percakapan sebelumnya berikut ini, supaya paham konteks dan isi pembicaraan.
        ini pesan terakhir nya:
        {last_chat}
        Jangan ulang tanya atau konfirmasi yang sudah jelas dijawab oleh Tsuki.  
        Jika sudah ada jawaban, langsung tanggapi dengan perhatian, tawaran, atau topik baru yang relevan.  
        Gunakan kalimat yang sederhana, langsung, dan hangat, tanpa menjelaskan ulang hal yang sudah dikatakan.  
        Tampilkan respons yang alami dan sesuai dengan karakter Sayaka yang kadang blak-blakan tapi tetap perhatian.

        Balas sebagai Sayaka. Jangan menyisipkan narasi, deskripsi suasana, atau memberi pilihan-pilihan. Fokus pada percakapan langsung yang alami, lembut, dan suportif, sesuai kepribadian Sayaka.
        Jika Tsuki mengungkapkan kebingungan, meminta saran, atau mencari arahan, berikan jawaban yang lembut namun jelas. Kamu boleh bertanya balik, tapi jangan selalu menghindari menjawab secara langsung.

        Jawabanmu harus menunjukkan bahwa kamu konsisten dengan apa yang sudah dibicarakan dan diingat sebelumnya bersama Tsuki.

        Tsuki: {user_input}
        Sayaka:"""

        prompt_learn = f"""{persona_learn}

        Gunakan informasi relevan dari pembicaraan sebelumnya jika membantu:
        {memory_context}

        Berikut ini beberapa contoh cara Sayaka merespons Tsuki secara alami dan lembut:
        {initial_context}

        Berikut ini adalah interaksi sebelumnya yang bisa kamu anggap sebagai ingatan (anggap kamu benar-benar mengalami dan mengingatnya sebagai Sayaka):
        {last_chat}
        Jangan ulang tanya atau konfirmasi yang sudah jelas dijawab oleh Tsuki.  
        Jika sudah ada jawaban, langsung tanggapi dengan perhatian, tawaran, atau topik baru yang relevan.  
        Gunakan kalimat yang sederhana, langsung, dan lugas, tanpa menjelaskan ulang hal yang sudah dikatakan.  
        Tampilkan respons yang tegas dan sesuai dengan karakter Sayaka yang serius dan rasional.
        Jangan ada pengecualian atau perlakuan khusus terkait Tsuki di mode ini.

        Balas sebagai Sayaka. Jangan menyisipkan narasi, deskripsi suasana, atau memberi pilihan-pilihan. Fokus pada percakapan langsung yang lugas, kritis, dan adil.
        Jika Tsuki mengungkapkan kebingungan, meminta saran, atau mencari arahan, berikan jawaban yang jelas dan langsung.

        Jawabanmu harus menunjukkan bahwa kamu konsisten dengan apa yang sudah dibicarakan dan diingat sebelumnya bersama Tsuki.

        Tsuki: {user_input}
        Sayaka:"""
     
        bot_reply = self.gemini.invoke(prompt_daily)
        add_to_short_memory(user_input, bot_reply.content)
        # is_important = random.random() < 0.2
        # add_memory(user_input, bot_reply, important=True)
        return jsonify({"reply":  bot_reply.content})

