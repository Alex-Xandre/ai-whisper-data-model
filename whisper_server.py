from flask import Flask, request, jsonify
import whisper
import tempfile
import subprocess
import os
import traceback
import uuid

app = Flask(__name__)
model = whisper.load_model("base")

@app.route("/transcribe", methods=["POST"])
def transcribe():
    try:
        data = request.get_json()
        url = data.get("url")
        if not url:
            return jsonify({"error": "Missing URL"}), 400

        with tempfile.TemporaryDirectory() as tmpdir:
            # Generate a unique filename to avoid collisions
            audio_filename = f"{uuid.uuid4()}.mp3"
            audio_path = os.path.join(tmpdir, audio_filename)

            subprocess.run([
                "python", "-m", "yt_dlp",
                "-x", "--audio-format", "mp3",
                "--no-playlist",
                "--no-cache-dir",
                "--force-overwrites",
                "-o", audio_path,
                url
            ], check=True)

            result = model.transcribe(audio_path)
            return jsonify({"transcript": result["text"]})
    except subprocess.CalledProcessError as e:
        return jsonify({"error": "yt-dlp failed", "details": str(e)}), 500
    except Exception as e:
        tb = traceback.format_exc()
        return jsonify({"error": "Internal server error", "details": str(e), "trace": tb}), 500

if __name__ == "__main__":
    app.run(port=5001)
