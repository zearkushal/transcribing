from flask import Flask, request, render_template, send_file
import whisper
import os
import noisereduce as nr
import librosa
import soundfile as sf

app = Flask(__name__)

# Ensure folders exist
os.makedirs("uploads", exist_ok=True)
os.makedirs("transcripts", exist_ok=True)

# Load Whisper model
model = whisper.load_model("small")  # Use 'medium' or 'large' if needed

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Save uploaded file
        file = request.files["file"]
        filepath = os.path.join("uploads", file.filename)
        file.save(filepath)

        # Process Audio (Noise Reduction)
        audio_data, sr = librosa.load(filepath, sr=None)
        reduced_noise = nr.reduce_noise(y=audio_data, sr=sr)
        cleaned_filepath = os.path.join("uploads", "cleaned_" + file.filename)
        sf.write(cleaned_filepath, reduced_noise, sr)

        # Transcribe Audio
        result = model.transcribe(cleaned_filepath)

        # Save transcript
        transcript_path = os.path.join("transcripts", file.filename + ".txt")
        with open(transcript_path, "w", encoding="utf-8") as f:
            f.write(result["text"])

        return render_template("result.html", transcript=result["text"], file=file.filename)

    return render_template("upload.html")

@app.route("/download/<filename>")
def download(filename):
    return send_file(os.path.join("transcripts", filename + ".txt"), as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)
