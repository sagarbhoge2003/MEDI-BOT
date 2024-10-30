# app.py
from flask import Flask, render_template, request, jsonify
from medical_chatbot import MedicalChatbot
import os

app = Flask(__name__)
chatbot = MedicalChatbot('remedies_data.json')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get_response', methods=['POST'])
def get_response():
    user_message = request.json['message']
    response = chatbot.get_response(user_message)
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
