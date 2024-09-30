from flask import Flask, render_template, jsonify
import threading
import time
from audio_processing import record_audio, transcribe_audio
from chatbot import get_response
import pyttsx3
import string
from flask_socketio import SocketIO, emit

app = Flask(__name__)

# Global variable to control the chat session
chat_active = False

def init_tts_engine():
    """Initialize and configure the TTS engine."""
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)  # Speed of speech
    engine.setProperty('volume', 0.8)  # Volume (0.0 to 1.0)
    return engine

def speak_response(engine, response_text):
    """Convert text response to speech and play it."""
    engine.say(response_text)
    engine.runAndWait()

def send_status(message):
    socketio.emit('status', {'message': message})

def chat_session():
    """Handle the voice chat session."""
    global chat_active
    tts_engine = init_tts_engine()

    while chat_active:
        print("Assistant is ready to listen...")
        send_status("Assistant is ready to listen...")

        # Record the audio
        record_audio("user_input.wav",status_callback=send_status)

        # Transcribe the audio to text
        send_status("Processing your input...")
        print("Processing your input...")
        user_input = transcribe_audio("user_input.wav").strip()

        # Normalize user input (remove punctuation, lowercase)
        normalized_input = user_input.lower().translate(str.maketrans('', '', string.punctuation)).strip()

        print(f"Transcribed Input: '{user_input}' (normalized: '{normalized_input}')")

        if normalized_input == "exit":
            print("Goodbye!")
            speak_response(tts_engine, "Goodbye!")
            chat_active = False
            break

        print("Thinking...")
        response_text = get_response(user_input)

        print(f"Assistant: {response_text}")

        # Speak the response
        speak_response(tts_engine, response_text)

        print("Response complete. Ready for the next input.\n")
        time.sleep(0.3)  # Add a short delay before next recording

@app.route('/')
def index():
    """Render the main HTML page."""
    return render_template('index.html')

@app.route('/start_chat', methods=['GET'])
def start_chat():
    """Start the chat session."""
    global chat_active
    if not chat_active:
        chat_active = True
        threading.Thread(target=chat_session).start()
        return jsonify({"status": "Chat started"}), 200
    return jsonify({"status": "Chat already active"}), 400

@app.route('/stop_chat', methods=['GET'])
def stop_chat():
    """Stop the chat session."""
    global chat_active
    if chat_active:
        chat_active = False
        return jsonify({"status": "Chat stopped"}), 200
    return jsonify({"status": "No active chat to stop"}), 400

if __name__ == '__main__':
    app.run(debug=True)
