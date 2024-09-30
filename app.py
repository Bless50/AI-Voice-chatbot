from flask import Flask, render_template, jsonify, request
from main import init_tts_engine, speak_response, process_user_input, get_and_speak_response, normalize_input
from audio_processing import record_audio, transcribe_audio
from chatbot import get_response
import threading
import queue

app = Flask(__name__)

# Global variables
is_chatting = False
chat_thread = None
tts_engine = init_tts_engine()
message_queue = queue.Queue()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_chat', methods=['POST'])
def start_chat():
    global is_chatting, chat_thread
    if not is_chatting:
        is_chatting = True
        chat_thread = threading.Thread(target=chat_loop)
        chat_thread.start()
        return jsonify({"status": "Chat started"})
    return jsonify({"status": "Chat already running"})

@app.route('/stop_chat', methods=['POST'])
def stop_chat():
    global is_chatting
    is_chatting = False
    return jsonify({"status": "Chat stopped"})

@app.route('/get_message', methods=['GET'])
def get_message():
    try:
        message = message_queue.get_nowait()
        return jsonify({"message": message})
    except queue.Empty:
        return jsonify({"message": None})

@app.route('/send_message', methods=['POST'])
def send_message():
    user_input = request.json['message']
    normalized_input = normalize_input(user_input)
    
    if normalized_input == "exit":
        global is_chatting
        is_chatting = False
        message_queue.put("Goodbye!")
        return jsonify({"status": "Chat ended"})
    
    response_text = get_response(user_input)
    message_queue.put(response_text)
    
    # Speak the response in a separate thread to avoid blocking
    threading.Thread(target=speak_response, args=(tts_engine, response_text)).start()
    
    return jsonify({"status": "Message received"})

def chat_loop():
    global is_chatting, tts_engine
    while is_chatting:
        user_input, normalized_input = process_user_input()
        message_queue.put(f"User: {user_input}")
        
        if normalized_input == "exit":
            is_chatting = False
            speak_response(tts_engine, "Goodbye!")
            message_queue.put("Assistant: Goodbye!")
            break
        
        response_text = get_response(user_input)
        message_queue.put(f"Assistant: {response_text}")
        speak_response(tts_engine, response_text)

if __name__ == '__main__':
    app.run(debug=True)