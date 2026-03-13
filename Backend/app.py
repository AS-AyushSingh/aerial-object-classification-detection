from fastapi import FastAPI, File, UploadFile
from PIL import Image
import numpy as np
import tensorflow as tf
from pathlib import Path
import io

# Resolve paths relative to this script so the app works regardless of cwd
BASE_DIR = Path(__file__).resolve().parent
PRIMARY_MODEL_PATH = BASE_DIR / 'artifacts' / 'models' / 'best_custom_cnn.h5'

app = FastAPI()

def load_model():
    if PRIMARY_MODEL_PATH.exists():
        return tf.keras.models.load_model(str(PRIMARY_MODEL_PATH))
    raise FileNotFoundError(
        f'No trained model file was found at {PRIMARY_MODEL_PATH} (cwd={Path.cwd()})'
    )

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        model = load_model()
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert('RGB')
        # Preprocess
        img_resized = img.resize((224, 224))
        x = np.array(img_resized) / 255.0
        x = np.expand_dims(x, 0)
        pred = model.predict(x)
        cls = np.argmax(pred, axis=1)[0]
        conf = float(np.max(pred))
        labels = ['bird', 'drone']
        return {"prediction": labels[cls], "confidence": conf}
    except Exception as e:
        return {"error": str(e)}
