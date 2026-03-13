from fastapi import FastAPI, File, UploadFile
from PIL import Image
import numpy as np
import tensorflow as tf
from pathlib import Path
import io

BACKEND_DIR = Path(__file__).resolve().parent
PRIMARY_MODEL_PATH = BACKEND_DIR / 'artifacts' / 'models' / 'best_transfer_model.h5'
FALLBACK_MODEL_PATH = BACKEND_DIR / 'artifacts' / 'models' / 'best_custom_cnn.h5'

app = FastAPI()

def load_model():
    if PRIMARY_MODEL_PATH.exists():
        model = tf.keras.models.load_model(str(PRIMARY_MODEL_PATH))
        return model, 'transfer'
    if FALLBACK_MODEL_PATH.exists():
        model = tf.keras.models.load_model(str(FALLBACK_MODEL_PATH))
        return model, 'custom'
    raise FileNotFoundError('No trained model file was found at artifacts/models/best_transfer_model.h5 or artifacts/models/best_custom_cnn.h5')


@app.get("/")
async def home():
    return {
        "message": "Welcome to the Bird vs Drone Classifier API! Use the /predict endpoint to classify images."
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        model, model_type = load_model()
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert('RGB')
        # Preprocess
        img_resized = img.resize((224, 224))
        # Keep pixel range as [0,255]; model contains its own normalization/preprocessing.
        x = np.array(img_resized, dtype=np.float32)
        x = np.expand_dims(x, 0)
        pred = model.predict(x, verbose=0)[0]
        # Keep probabilities valid even if a logits model is loaded.
        if not np.isclose(np.sum(pred), 1.0, atol=1e-3):
            pred = tf.nn.softmax(pred).numpy()

        cls = int(np.argmax(pred))
        conf = float(pred[cls])
        labels = ['bird', 'drone']
        sorted_probs = np.sort(pred)
        margin = float(sorted_probs[-1] - sorted_probs[-2])

        result = {
            "prediction": labels[cls],
            "confidence": conf,
            "margin": margin,
            "model_type": model_type,
            "probabilities": {label: float(prob) for label, prob in zip(labels, pred)}
        }

        if conf < 0.80 or margin < 0.20:
            result["warning"] = "Low-confidence prediction. Try a clearer image or different angle."

        return result
    except FileNotFoundError as e:
        return {"error": str(e)}
    except RuntimeError as e:
        return {"error": f'Failed to load model: {e}'}
    except Exception as e:
        return {"error": f'Unexpected error during prediction: {e}'}
