import time
import pyaudio
import wave
import numpy as np
from dotenv import load_dotenv
from groq import Groq
from flask_socketio import SocketIO, emit
# Load environment variables from .env file
load_dotenv()

# Initialize the Groq client for audio processing
client = Groq()

def is_silent(data_chunk, threshold):
    """Check if the audio chunk is silent."""
    return np.max(np.abs(np.frombuffer(data_chunk, dtype=np.int16))) < threshold

def record_audio(filename,status_callback=None):
    chunk = 1024
    sample_format = pyaudio.paInt16
    channels = 1
    fs = 44100
    
    p = pyaudio.PyAudio()
    print("Recording...")
    if status_callback:
       status_callback("Recording...")
    stream = p.open(format=sample_format, channels=channels, rate=fs,
                    input=True, frames_per_buffer=chunk)

    frames = []
    silent_chunks = 0
    silence_threshold = 1000  # Initial threshold, will be adjusted
    max_silent_chunks = int(fs / chunk * 6)  # 4 seconds of silence
    min_recording_duration = 5  # Minimum recording time in seconds
    max_recording_duration = 30  # Maximum recording time in seconds
    
    start_time = time.time()
    
    # Calibration phase
    print("Calibrating...")
    if status_callback:
        status_callback("Calibrating...")
    calibration_frames = []
    for _ in range(int(fs / chunk * 6)):  # 6 seconds of calibration
        data = stream.read(chunk)
        calibration_frames.append(data)
    
    calibration_data = np.frombuffer(b''.join(calibration_frames), dtype=np.int16)
    silence_threshold = np.max(np.abs(calibration_data)) * 1.5  # 50% above the background noise
    print(f"Calibrated. Silence threshold: {silence_threshold}")

    print("Start speaking...")
    if status_callback:
        status_callback(f"Calibrated. Silence threshold: {silence_threshold}")

    if status_callback:
        status_callback("Start speaking...")
    while True:
        data = stream.read(chunk)
        frames.append(data)
        
        if is_silent(data, silence_threshold):
            silent_chunks += 1
        else:
            silent_chunks = 0
        
        duration = time.time() - start_time
        
        if duration > min_recording_duration and silent_chunks > max_silent_chunks:
            print("Silence detected, stopping recording.")
            if status_callback:
                status_callback("Silence detected, stopping recording.")
            break
        
        if duration > max_recording_duration:
            print("Maximum recording duration reached.")
            if status_callback:
                status_callback("Maximum recording duration reached.")
            break
    
    print("Finished recording.")
    if status_callback:
        status_callback("Finished recording.")

    stream.stop_stream()
    stream.close()
    p.terminate()

    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(p.get_sample_size(sample_format))
        wf.setframerate(fs)
        wf.writeframes(b''.join(frames))

#function to transcribe the audio to text
def transcribe_audio(filename):
    """Transcribe audio file to text using Groq's Distil-Whisper."""
    with open(filename, "rb") as file:
        transcription = client.audio.transcriptions.create(
           file=(filename, file.read()), # Required audio file
           model="whisper-large-v3", # Required model to use for translation
           prompt="""
                You are an AI transcription assistant. Your job is to transcribe any audio file 
                sent to you into text, accurately reflecting the spoken content. Please pay special
                  attention to the following:
                - Identify and correctly transcribe any technical jargon, acronyms, or specific terms 
                 mentioned in the audio.
                - Maintain punctuation and formatting as appropriate for clarity.
                - If the audio contains multiple speakers, indicate their turns (e.g., Speaker 1, Speaker 2).
                - Ensure that the transcription flows naturally and coherently, preserving the context of discussions. 
                even in the presence of background noise, 
                and infer the language used. Ensure clarity and correctness in the transcription.   
             """,  
           response_format="json",  # Optional
           temperature=0.0  # Optional  
        )
        
    return transcription.text  # Adjust based on actual response structure