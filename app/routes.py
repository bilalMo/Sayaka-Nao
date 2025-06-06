import os
from app import app
from flask import render_template
from flask import request

from app.controllers.chat_controller import ChatController
from app.models.memory_model import clear_memory
chat_controller = ChatController() 
@app.route('/')
def index():
    print("CWD:", os.getcwd())  # DEBUGGING
    return render_template('views/template.html')

@app.route('/api/chat', methods=['POST'])
def chat_route():
    data = request.get_json()
    input = data.get('message')
    return chat_controller.chat(input)



@app.route("/api/clear_memory", methods=["GET"])
def clear_route():
    return clear_memory()