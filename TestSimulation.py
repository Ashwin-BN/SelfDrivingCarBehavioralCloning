import os
print("Starting Simulation...")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import base64
from io import BytesIO
from PIL import Image
import cv2
import numpy as np
import socketio
import eventlet
from flask import Flask
from tensorflow.keras.models import load_model

# -------------------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------------------
MODEL_PATH = "model.h5"
MAX_SPEED = 10

# -------------------------------------------------------------------------
# SERVER SETUP
# -------------------------------------------------------------------------
sio = socketio.Server()
app = Flask(__name__)
model = None

# -------------------------------------------------------------------------
# IMAGE PREPROCESSING
# -------------------------------------------------------------------------
def preprocess_image(img):
    img = img[60:135, :, :]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.resize(img, (200, 66))
    img = img / 255.0
    return img

# -------------------------------------------------------------------------
# TELEMETRY EVENT
# -------------------------------------------------------------------------
@sio.on("telemetry")
def telemetry(sid, data):
    speed = float(data["speed"])
    image = Image.open(BytesIO(base64.b64decode(data["image"])))
    image = np.asarray(image)
    image = preprocess_image(image)
    image = np.array([image])

    steering_angle = float(model.predict(image))
    throttle = 1.0 - (speed / MAX_SPEED)

    print(f"Throttle: {throttle:.4f}, Steering: {steering_angle:.4f}, Speed: {speed:.2f}")
    send_control(steering_angle, throttle)

# -------------------------------------------------------------------------
# CONNECT EVENT
# -------------------------------------------------------------------------
@sio.on("connect")
def connect(sid, environ):
    print("Simulator connected.")
    send_control(0, 0)

# -------------------------------------------------------------------------
# SEND CONTROL COMMANDS
# -------------------------------------------------------------------------
def send_control(steering, throttle):
    sio.emit("steer", data={
        "steering_angle": str(steering),
        "throttle": str(throttle)
    })

# -------------------------------------------------------------------------
# MAIN ENTRY POINT
# -------------------------------------------------------------------------
if __name__ == "__main__":
    print("Loading model...")
    model = load_model(MODEL_PATH)
    print("Model loaded successfully.")

    app = socketio.Middleware(sio, app)
    eventlet.wsgi.server(eventlet.listen(("", 4567)), app)
