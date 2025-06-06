import os
from app import app
from flask import render_template
from flask import request

from app.controllers.ChatController import ChatController
from app.controllers.KnowledgeSummaryController import KnowledgeSummaryController
from app.models.memory_model import MemoryManager

from datetime import datetime


@app.route('/')
def index():
    return render_template('index.html')
@app.route('/chat')
def chat():
    topic = request.args.get('topic')
    mode = request.args.get('mode')
    if not topic or not mode:
        return "Silakan masukkan topic dan mode di URL, contoh: /chat?topic=daily&mode=learn"

    date_str = datetime.now().strftime("%Y-%m-%d")
    filename = f"short_memory_{date_str}_{topic}_{mode}.json"

    return render_template('views/template.html', topic=topic, mode=mode, filename=filename)

@app.route('/api/chat_learn', methods=['POST'])
def chat_route():
    data = request.get_json()
    input = data.get('message')
    filename = data.get('filename')
    topic = data.get('topic')
    
    controller = ChatController() 
    return controller.chat_learn(input, filename,topic)

@app.route('/api/summary/summary', methods=['POST'])
def summary_route():
    data = request.get_json()
    input = data.get('message')
    summary = KnowledgeSummaryController.summarize(input)
    return render_template('summary_result.html', summary=summary)

@app.route('/api/summary/revise', methods=['POST'])
def revise_route():
    data = request.get_json()
    current_summary = data.get('current_summary')
    user_feedback = data.get('user_feedback')
    topic = data.get('topic')
    
    controller = KnowledgeSummaryController()
    return controller.process_revision(current_summary, user_feedback, topic)

@app.route('/api/summary/remember', methods=['POST'])
def remember_route():
    data = request.get_json()
    summary = data.get('summary')

    controller = KnowledgeSummaryController()    
    return controller.create_new_memories(summary)
    
    
@app.route("/api/clear_memory", methods=["GET"])
def clear_route():
    return MemoryManager().clear_memory()