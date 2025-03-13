import whisper
import torch
import eventlet
from flask import Flask, render_template
from flask_cors import CORS
from flask_socketio import SocketIO
from pydub import AudioSegment
from pydub.utils import make_chunks
import numpy as np
import librosa
import os

# Apply eventlet monkey patch for better WebSocket performance
eventlet.monkey_patch()

# Initialize Flask App and WebSockets
app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# Load Whisper Model (using "base" for faster transcription)
model = whisper.load_model("base")

def analyze_audio(audio_chunk):
    """ Analyzes loudness and pitch using librosa with improved accuracy """
    chunk_name = "temp_chunk.wav"
    audio_chunk.export(chunk_name, format="wav")
    
    y, sr = librosa.load(chunk_name, sr=None)
    os.remove(chunk_name)
    
    # Calculate loudness (RMS energy) dynamically
    rms_energy = np.mean(librosa.feature.rms(y=y))
    normalized_loudness = rms_energy / (np.max(rms_energy) + 1e-6)  # Avoid division by zero
    
    # Extract pitch with a more robust method (filtering out noise)
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    valid_pitches = pitches[pitches > 0]
    
    if len(valid_pitches) > 0:
        pitch_values = np.median(valid_pitches)  # Use median for stability
    else:
        pitch_values = 150  # Default neutral pitch
    
    print(f"DEBUG: Loudness={normalized_loudness:.4f}, Pitch={pitch_values:.2f}")  # Debugging Output
    
    return normalized_loudness, pitch_values

def classify_intonation(loudness, pitch):
    """ Classifies vocal tones more accurately, ensuring at least one label per sentence """
    if loudness > 0.08 and pitch > 220:
        return "excited"
    elif loudness > 0.08 and pitch <= 220:
        return "angry"
    elif loudness <= 0.08 and pitch > 220:
        return "happy"
    else:
        return "calm"

def transcribe_audio_live(audio_path, chunk_length_ms=2000):
    """ Transcribes audio in real-time with smarter intonation classification """
    use_fp16 = torch.cuda.is_available()
    print("🎤 Starting real-time transcription...")
    caption_text = ""
    
    audio = AudioSegment.from_file(audio_path)
    chunks = make_chunks(audio, chunk_length_ms)
    
    for i, chunk in enumerate(chunks):
        chunk_name = f"chunk{i}.wav"
        chunk.export(chunk_name, format="wav")
        
        loudness, pitch = analyze_audio(chunk)
        tone = classify_intonation(loudness, pitch)
        
        result = model.transcribe(chunk_name, fp16=use_fp16, word_timestamps=True)
        os.remove(chunk_name)
        
        for segment in result["segments"]:
            phrase = segment["text"].strip()
            if not phrase:
                continue
            
            # Guarantee at least one tone annotation per sentence
            phrase_with_tone = f"{phrase} ({tone})"
            phrase_with_tone = apply_tooltip(phrase_with_tone)
            caption_text += " " + phrase_with_tone
            
            print(f"Updating Caption: {caption_text}")
            socketio.emit("update_caption", {"word": caption_text})
            eventlet.sleep(0.1)

def apply_tooltip(phrase):
    """ Adds hover tooltips only to meaningful intonations """
    explanations = {
        "calm": "A steady tone with minimal fluctuations. Used in neutral speech.",
        "excited": "A tone with rising pitch and energy. Indicates high engagement.",
        "angry": "A harsh, forceful tone. Indicates frustration or disagreement.",
        "happy": "A bright, cheerful tone. Indicates joy or satisfaction."
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
    print("WebSocket Connected!")

@socketio.on("start_transcription")
def start_transcription():
    print("Received Start Transcription Event!")
    transcribe_audio_live("speech.mp3")

if __name__ == "__main__":
    print("🚀 Starting Flask-SocketIO Server...")
    socketio.run(app, host="0.0.0.0", port=5001, debug=True, allow_unsafe_werkzeug=True)

