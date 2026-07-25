from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import base64
from detector import detect

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/process", methods=["POST"])
def process():

    data = request.json["image"]

    # Remove base64 header
    encoded_data = data.split(",")[1]

    # Decode image
    nparr = np.frombuffer(
        base64.b64decode(encoded_data),
        np.uint8
    )

    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # MediaPipe detection
    frame = detect(frame)

    # Encode result
    _, buffer = cv2.imencode(".jpg", frame)

    img_base64 = base64.b64encode(
        buffer
    ).decode("utf-8")

    return jsonify({
        "image": "data:image/jpeg;base64," + img_base64
    })


if __name__ == "__main__":
    app.run(debug=True)