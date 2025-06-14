from flask import Flask, request, jsonify
import face_recognition
import numpy as np
import requests
import os
from werkzeug.utils import secure_filename
from flask_cors import CORS # Import CORS

app = Flask(__name__)
CORS(app) # Enable CORS for all routes
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/match-selfie", methods=["POST"])
def match_selfie():
    file = request.files['selfie']
    path = os.path.join(UPLOAD_FOLDER, secure_filename(file.filename))
    file.save(path)

    image = face_recognition.load_image_file(path)
    encodings = face_recognition.face_encodings(image)
    if not encodings:
        return jsonify({"error": "No face found"}), 400

    encoding_list = encodings[0].tolist()
    return jsonify({"matches": encoding_list})


@app.route("/check-image-faces", methods=["POST"])
def check_image_faces():
    data = request.json
    image_url = data['image_url']
    known_enc = np.array(data['known_encodings'])

    try:
        response = requests.get(image_url)
        with open("temp.jpg", "wb") as f:
            f.write(response.content)

        image = face_recognition.load_image_file("temp.jpg")
        encodings = face_recognition.face_encodings(image)

        for enc in encodings:
            match = face_recognition.compare_faces([known_enc], enc)
            if True in match:
                return jsonify({"match": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({"match": False})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

