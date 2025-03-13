import whisper
import subprocess
import pandas as pd
import openai
import os
import torch
import eventlet
from flask import Flask, render_template
from flask_cors import CORS
from flask_socketio import SocketIO
from pydub import AudioSegment
from pydub.utils import make_chunks
import numpy as np
import librosa  # Add librosa for pitch detection

# Set OpenAI API Key Here
OPENAI_API_KEY = ""

if not OPENAI_API_KEY:
    raise ValueError("Error: OpenAI API key is missing! Set it in main.py.")

client = openai.OpenAI(api_key=OPENAI_API_KEY)

# Initialize Flask App and WebSockets
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
socketio = SocketIO(app, cors_allowed_origins="*")  # Allow all origins for Socket.IO

# Load Whisper Model
model = whisper.load_model("medium")

def convert_to_wav(audio_path):
    """ Converts an audio file to WAV format if necessary """
    if not audio_path.endswith(".wav"):
        sound = AudioSegment.from_file(audio_path)
        wav_path = audio_path.replace(".mp3", ".wav")
        sound.export(wav_path, format="wav")
        return wav_path
    return audio_path

def analyze_audio(audio_chunk):
    """ Analyzes loudness and pitch using librosa """
    # Export the chunk to a temporary WAV file
    chunk_name = "temp_chunk.wav"
    audio_chunk.export(chunk_name, format="wav")
    
    # Load the audio file using librosa
    y, sr = librosa.load(chunk_name, sr=None)
    
    # Calculate loudness (RMS energy)
    rms_energy = librosa.feature.rms(y=y).mean()
    
    # Extract pitch using librosa's piptrack
    pitches, magnitudes = librosa.core.piptrack(y=y, sr=sr)
    pitch = np.max(pitches) if np.any(pitches) else 150  # Default pitch if no pitch detected
    
    # Clean up the temporary file
    os.remove(chunk_name)
    
    return rms_energy, pitch

def classify_intonation(loudness, pitch):
    """ Classifies vocal tones based on loudness and pitch """
    if loudness > 0.1 and pitch > 200:  # High loudness and high pitch
        return "excited"
    elif loudness > 0.1 and pitch <= 200:  # High loudness but low pitch
        return "angry"
    elif loudness <= 0.1 and pitch > 200:  # Low loudness but high pitch
        return "happy"
    elif loudness <= 0.1 and pitch <= 200:  # Low loudness and low pitch
        return "calm"
    else:
        return "emphasized"

def transcribe_audio_live(audio_path, chunk_length_ms=2000):
    """ Transcribes audio in real-time and sends live captions with vocal tone labels """
    audio_path = convert_to_wav(audio_path)  # Convert to WAV if necessary
    use_fp16 = torch.cuda.is_available()

    print("🎤 Starting real-time transcription...")
    caption_text = ""  # Maintain full captions

    # Load audio file
    audio = AudioSegment.from_file(audio_path)
    chunks = make_chunks(audio, chunk_length_ms)  # Split audio into small chunks

    for i, chunk in enumerate(chunks):
        chunk_name = f"chunk{i}.wav"
        chunk.export(chunk_name, format="wav")  # Export chunk to WAV

        # Analyze loudness and pitch for the current chunk
        loudness, pitch = analyze_audio(chunk)
        tone = classify_intonation(loudness, pitch)

        # Transcribe the chunk
        result = model.transcribe(chunk_name, fp16=use_fp16, word_timestamps=True)
        
        # Process each segment in the chunk
        for segment in result["segments"]:
            phrase = segment["text"]
            if phrase.strip() == "":  # Skip empty phrases
                continue

            phrase_with_tone = f"{phrase} ({tone})"  # Apply tone to the entire phrase

            phrase_with_tone = apply_tooltip(phrase_with_tone)
            caption_text += " " + phrase_with_tone  # Append phrase to captions
            
            print(f"Updating Caption: {caption_text}")
            socketio.emit("update_caption", {"word": caption_text})  # Send updated captions
            eventlet.sleep(0.1)  # Simulates real-time streaming

        # Clean up chunk file
        os.remove(chunk_name)

def apply_tooltip(phrase):
    """ Adds hover tooltip with vocal tone explanation """
    explanations = {
        "calm": "A smooth, steady tone with minimal fluctuations. Often used in reassuring or neutral speech.",
        "excited": "A tone with rising pitch and energy. Indicates enthusiasm, eagerness, or high engagement.",
        "angry": "A harsh, forceful tone. Indicates frustration, anger, or strong disagreement.",
        "happy": "A bright, cheerful tone. Indicates joy, satisfaction, or positivity.",
        "emphasized": "A louder or more forceful tone. Used to highlight important points or stress key information."
    }

    for tone, explanation in explanations.items():
        phrase = phrase.replace(
            f"({tone})",
            f'<span class="tooltip {tone}">({tone})<span class="tooltiptext">{explanation}</span></span>'
        )
    return phrase

@app.route("/")
def index():
    return render_template("index.html")

@socketio.on("connect")
def handle_connect():
    """ Debug WebSocket Connection """
    print("WebSocket Connected!")

@socketio.on("start_transcription")
def start_transcription():
    """ Starts real-time captioning when the button is clicked """
    print("Received Start Transcription Event!")  # Debug log
    transcribe_audio_live("speech.mp3")

if __name__ == "__main__":
    print("🚀 Starting Flask-SocketIO Server...")
    socketio.run(app, host="0.0.0.0", port=5001, debug=True, allow_unsafe_werkzeug=True)