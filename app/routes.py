import os
from app import app
from flask import render_template
from flask import request

from app.controllers.ChatController import ChatController
from app.controllers.KnowledgeSummaryController import KnowledgeSummaryController
from app.models.memory_model import MemoryManager

from datetime import datetime

chat_controller = ChatController() 
@app.route('/chat')
def chat():
    topic = request.args.get('topic')
    mode = request.args.get('mode')
    if not topic or not mode:
        return "Silakan masukkan topic dan mode di URL, contoh: /chat?topic=daily&mode=learn"

    date_str = datetime.now().strftime("%Y-%m-%d")
    filename = f"short_memory_{date_str}_{topic}_{mode}.json"

    return render_template('views/template.html', topic=topic, mode=mode, filename=filename)

@app.route('/api/chat', methods=['POST'])
def chat_route():
    data = request.get_json()
    input = data.get('message')
    filename = data.get('filename')
    topic = data.get('topic')
    return chat_controller.chat_learn(input, filename,topic)

@app.route('/api/summary', methods=['POST'])
def summary_route():
    data = request.get_json()
    input = data.get('message')
    return KnowledgeSummaryController.summarize(input, )


@app.route("/api/clear_memory", methods=["GET"])
def clear_route():
    return MemoryManager().clear_memory()