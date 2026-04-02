from flask import Flask, render_template, request
import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

app = Flask(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH = "my_trained_model.h5"
CLASS_NAMES = [
    "Apple___Black_rot",
    "Grape___Black_rot",
    "Tomato___Early_blight",
]
IMG_SIZE = (224, 224)

RECOMMENDATIONS = {
    "Apple___Black_rot": (
        "Remove and destroy infected fruit and leaves. "
        "Apply a copper-based fungicide every 7-10 days during wet weather. "
        "Ensure good air circulation by pruning crowded branches."
    ),
    "Grape___Black_rot": (
        "Remove mummified berries and infected canes. "
        "Apply mancozeb or captan fungicide at bud break and continue through bloom. "
        "Avoid overhead irrigation to keep foliage dry."
    ),
    "Tomato___Early_blight": (
        "Remove lower infected leaves immediately. "
        "Apply neem oil or a chlorothalonil-based fungicide every 7 days. "
        "Mulch around the base to reduce soil splash."
    ),
}
SEVERITY_MAP = {
    "Apple___Black_rot": "High",
    "Grape___Black_rot": "High",
    "Tomato___Early_blight": "Medium",
}

# ── Build the same architecture used during training ──────────────────────────
# This must match exactly how your model was built before training.
def build_model(num_classes=3):
    model = keras.Sequential([
        layers.Input(shape=(224, 224, 3)),

        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D(),

        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(),

        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D(),

        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax'),
    ])
    return model

model = build_model(num_classes=len(CLASS_NAMES))
model.load_weights(MODEL_PATH)
print("Model loaded successfully!")

# ── Image preprocessing ───────────────────────────────────────────────────────
def preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img = img.resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files.get("image")
    if not file or file.filename == "":
        return "No file uploaded", 400

    save_dir = os.path.join("static", "uploads")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, file.filename)
    file.save(save_path)

    img_array = preprocess_image(save_path)
    predictions = model.predict(img_array)[0]
    pred_index = int(np.argmax(predictions))
    confidence = float(np.max(predictions)) * 100

    disease = CLASS_NAMES[pred_index]
    severity = SEVERITY_MAP.get(disease, "Unknown")
    recommendation = RECOMMENDATIONS.get(disease, "Consult an agronomist.")

    return render_template(
        "result.html",
        filename=f"uploads/{file.filename}",
        disease=disease.replace("___", " - "),
        confidence=round(confidence, 2),
        severity=severity,
        recommendation=recommendation,
    )

if __name__ == "__main__":
    app.run(debug=True)
