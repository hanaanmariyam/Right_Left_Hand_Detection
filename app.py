from flask import Flask, render_template, Response
import cv2
from detector import detect

app = Flask(__name__)

# Open webcam
camera = cv2.VideoCapture(0)


# Home page
@app.route("/")
def index():
    return render_template("index.html")


# Generate video frames
def generate_frames():

    while True:

        success, frame = camera.read()

        if not success:
            break

        # Send frame to MediaPipe detector
        frame = detect(frame)

        # Convert frame to jpg
        ret, buffer = cv2.imencode(".jpg", frame)

        frame = buffer.tobytes()

        # Send frame to browser
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" 
            + frame +
            b"\r\n"
        )


# Video route
@app.route("/video")
def video():

    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


# Run app
if __name__ == "__main__":
    app.run(debug=True)