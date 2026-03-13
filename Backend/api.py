from fastapi import FastAPI, File, UploadFile
from fastapi.staticfiles import StaticFiles
from PIL import Image
import numpy as np
import tensorflow as tf
from pathlib import Path
import io
import os
import uvicorn

BACKEND_DIR = Path(__file__).resolve().parent
PRIMARY_MODEL_PATH = BACKEND_DIR / 'artifacts' / 'models' / 'best_transfer_model.h5'
FALLBACK_MODEL_PATH = BACKEND_DIR / 'artifacts' / 'models' / 'best_custom_cnn.h5'

app = FastAPI()


def load_model():
    if PRIMARY_MODEL_PATH.exists():
        model = tf.keras.models.load_model(str(PRIMARY_MODEL_PATH), compile=False)
        return model, "transfer"

    if FALLBACK_MODEL_PATH.exists():
        model = tf.keras.models.load_model(str(FALLBACK_MODEL_PATH), compile=False)
        return model, "custom"

    raise FileNotFoundError(
        "No trained model file was found at artifacts/models/best_transfer_model.h5 or artifacts/models/best_custom_cnn.h5"
    )


# Load model once at startup (faster inference)
model, model_type = load_model()


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")

        # Resize image
        img_resized = img.resize((224, 224))

        # Convert to numpy
        x = np.array(img_resized, dtype=np.float32)
        x = np.expand_dims(x, axis=0)

        # Prediction
        pred = model.predict(x, verbose=0)[0]

        # Ensure probabilities
        if not np.isclose(np.sum(pred), 1.0, atol=1e-3):
            pred = tf.nn.softmax(pred).numpy()

        labels = ["bird", "drone"]
        cls = int(np.argmax(pred))
        conf = float(pred[cls])

        sorted_probs = np.sort(pred)
        margin = float(sorted_probs[-1] - sorted_probs[-2])

        result = {
            "prediction": labels[cls],
            "confidence": conf,
            "margin": margin,
            "model_type": model_type,
            "probabilities": {
                label: float(prob) for label, prob in zip(labels, pred)
            }
        }

        if conf < 0.80 or margin < 0.20:
            result["warning"] = "Low-confidence prediction. Try a clearer image or different angle."

        return result

    except Exception as e:
        return {"error": f"Unexpected error during prediction: {str(e)}"}


# Serve frontend static files
frontend_dist = BACKEND_DIR.parent / "frontend" / "dist"
if frontend_dist.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dist), html=True), name="frontend")


# Render PORT fix
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("api:app", host="0.0.0.0", port=port)