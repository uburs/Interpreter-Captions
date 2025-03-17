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
import html
import re

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
    
    # Calculate perceptual loudness
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
    loudness = librosa.perceptual_weighting(mel_spectrogram, librosa.mel_frequencies(n_mels=mel_spectrogram.shape[0]))
    mean_loudness = np.mean(loudness)
    
    # Extract pitch using a more robust method
    pitch_values = librosa.yin(y, fmin=80, fmax=400)
    median_pitch = np.median(pitch_values)
    
    print(f"DEBUG: Loudness={mean_loudness:.4f}, Pitch={median_pitch:.2f}")  # Debugging Output
    
    return mean_loudness, median_pitch

def classify_intonation(loudness, pitch):
    """ Classifies vocal tones more accurately with expanded categories """
    if loudness > -20 and pitch > 250:
        return "excited"
    elif loudness > -20 and pitch <= 250:
        return "angry"
    elif loudness <= -20 and pitch > 250:
        return "happy"
    elif loudness <= -30:
        return "sad"
    else:
        return "calm"

def transcribe_audio_live(audio_path, chunk_length_ms=2000):
    """ Transcribes audio in real-time and ensures captions accumulate """
    use_fp16 = torch.cuda.is_available()
    print("🎤 Starting real-time transcription...")
    caption_text = ""  # Keep track of all transcribed words

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

            phrase_with_tone = f"{phrase} ({tone})"
            formatted_caption = apply_tooltip(phrase_with_tone)
            caption_text += " " + formatted_caption  # ✅ Append instead of overwriting

            print(f"Updating Caption: {caption_text}")
            socketio.emit("update_caption", {"word": caption_text})  # ✅ Send full accumulated text
            eventlet.sleep(0.1)

    # ✅ Save to output.html with full text
    with open("output.html", "w", encoding="utf-8") as file:
        file.write(f"""<html><head><title>Transcription Output</title>
        <style>
            .tooltip {{ position: relative; display: inline-block; border-bottom: 1px dotted black; }}
            .tooltip .tooltiptext {{ visibility: hidden; width: 120px; background-color: black; 
                color: #fff; text-align: center; border-radius: 6px; padding: 5px; position: absolute; 
                z-index: 1; bottom: 125%; left: 50%; margin-left: -60px; opacity: 0; transition: opacity 0.3s; }}
            .tooltip:hover .tooltiptext {{ visibility: visible; opacity: 1; }}
        </style></head><body>""")
        file.write(f"<h2>Transcription Output</h2><p>{caption_text}</p></body></html>")

    print("✅ Transcription saved to output.html")





def apply_tooltip(phrase):
    """ Replaces emotion labels with subscript/superscript markers wrapped in tooltips with colors """
    markers = {
        "calm": '<span class="tooltip"><sup title="Calm" style="color: #007bff;">C</sup><span class="tooltiptext">A steady tone with minimal fluctuations. Used in neutral speech.</span></span>',
        "excited": '<span class="tooltip"><sup title="Excited" style="color: #ff9800;">E</sup><span class="tooltiptext">A tone with rising pitch and energy. Indicates high engagement.</span></span>',
        "angry": '<span class="tooltip"><sup title="Angry" style="color: #ff3d00;">A</sup><span class="tooltiptext">A harsh, forceful tone. Indicates frustration or disagreement.</span></span>',
        "happy": '<span class="tooltip"><sup title="Happy" style="color: #4caf50;">H</sup><span class="tooltiptext">A bright, cheerful tone. Indicates joy or satisfaction.</span></span>',
        "sad": '<span class="tooltip"><sub title="Sad" style="color: #9e9e9e;">S</sub><span class="tooltiptext">A low-energy, monotone voice. Indicates sadness or disappointment.</span></span>'
    }

    # Replace textual labels with corresponding subscript/superscript tooltip markers
    for tone, marker in markers.items():
        pattern = rf"\({tone}\)"  # Matches exact labels like (calm)
        phrase = re.sub(pattern, marker, phrase)

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
