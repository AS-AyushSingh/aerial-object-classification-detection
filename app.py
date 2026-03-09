import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from pathlib import Path

PRIMARY_MODEL_PATH = Path('artifacts/models/best_custom_cnn.h5')


def load_model():
    if PRIMARY_MODEL_PATH.exists():
        return tf.keras.models.load_model(str(PRIMARY_MODEL_PATH))
    raise FileNotFoundError('No trained model file was found at artifacts/models/best_custom_cnn.h5')

st.title('Aerial Object Classification (Bird vs Drone)')
uploaded = st.file_uploader('Upload an image', type=['jpg','jpeg','png'])
if uploaded is not None:
    img = Image.open(uploaded).convert('RGB')
    st.image(img, caption='Uploaded image', use_column_width=True)
    # Preprocess
    img_resized = img.resize((224,224))
    x = np.array(img_resized) / 255.0
    x = np.expand_dims(x, 0)
    try:
        model = load_model()
        pred = model.predict(x)
        cls = np.argmax(pred, axis=1)[0]
        conf = float(np.max(pred))
        labels = ['bird','drone']
        st.success(f'Prediction: {labels[cls]} (confidence: {conf:.3f})')
    except Exception as e:
        st.warning(
            'Model not found. Train with scripts/train_classification.py so artifacts/models/best_custom_cnn.h5 exists.'
        )
