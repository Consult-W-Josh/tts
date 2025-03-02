import sys
import os
from datetime import datetime
from pymongo import MongoClient
from flask import Flask, render_template, request, send_from_directory
from yarngpt.audiotokenizer import AudioTokenizerV2
from transformers import AutoModelForCausalLM
import torchaudio

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "audio"

# Connect to MongoDB
client = MongoClient("mongodb+srv://Adeoluwa123:09014078564Feranmi@cluster0.r8sg61r.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")  # Replace with your MongoDB URI
db = client["tts_app"]  # Database name
audio_collection = db["audio_files"]  # Collection name

# Define voices
VOICES = [
    "idera", "chinenye", "jude", "emma", "umar", "joke", "zainab", "osagie", "remi", "tayo"
]

# Load the TTS model
tokenizer_path = "saheedniyi/YarnGPT2"
wav_tokenizer_config_path = "models/wavtokenizer_mediumdata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml"
wav_tokenizer_model_path = "models/wavtokenizer_large_speech_320_24k.ckpt"

# Ensure the files exist
if not os.path.exists(wav_tokenizer_config_path):
    raise FileNotFoundError(f"Config file not found: {wav_tokenizer_config_path}")
if not os.path.exists(wav_tokenizer_model_path):
    raise FileNotFoundError(f"Model file not found: {wav_tokenizer_model_path}")

audio_tokenizer = AudioTokenizerV2(
    tokenizer_path, wav_tokenizer_model_path, wav_tokenizer_config_path
)
model = AutoModelForCausalLM.from_pretrained(tokenizer_path, torch_dtype="auto").to(audio_tokenizer.device)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        text = request.form["text"]
        speaker_name = request.form.get("speaker", "idera")  # Default to "idera"
        prompt = audio_tokenizer.create_prompt(text, lang="english", speaker_name=speaker_name)
        input_ids = audio_tokenizer.tokenize_prompt(prompt)
        output = model.generate(
            input_ids=input_ids,
            attention_mask=input_ids.ne(0),
            pad_token_id=model.config.eos_token_id,
            temperature=0.1,
            repetition_penalty=1.1,
            max_length=4000,
            do_sample=True,
        )
        codes = audio_tokenizer.get_codes(output)
        audio = audio_tokenizer.get_audio(codes)

        # Convert the audio tensor to 16-bit PCM format
        audio = (audio * 32767).clamp(-32768, 32767).short()  # Scale to 16-bit PCM range (-32768 to 32767)

        # Save the audio file
        audio_path = os.path.join(app.config["UPLOAD_FOLDER"], "output.wav")
        torchaudio.save(audio_path, audio, sample_rate=20000, encoding="PCM_S", bits_per_sample=16)

        # Store metadata in MongoDB
        audio_collection.insert_one({
            "text": text,
            "speaker": speaker_name,
            "file_path": audio_path,
            "timestamp": datetime.now(),
        })

        return render_template("index.html", audio_file="output.wav", voices=VOICES)
    return render_template("index.html", voices=VOICES)

@app.route("/audio/<filename>")
def get_audio(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

@app.route("/history")
def history():
    audio_files = list(audio_collection.find().sort("timestamp", -1))  # Get latest first
    return render_template("history.html", audio_files=audio_files)

if __name__ == "__main__":
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    app.run(debug=True)