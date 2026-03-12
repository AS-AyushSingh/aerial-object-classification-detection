import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from pathlib import Path

PRIMARY_MODEL_PATH = Path('artifacts/models/best_transfer_model.h5')
FALLBACK_MODEL_PATH = Path('artifacts/models/best_custom_cnn.h5')


@st.cache_resource
def load_model():
    if PRIMARY_MODEL_PATH.exists():
        return tf.keras.models.load_model(str(PRIMARY_MODEL_PATH))
    if FALLBACK_MODEL_PATH.exists():
        return tf.keras.models.load_model(str(FALLBACK_MODEL_PATH))
    raise FileNotFoundError('No trained model file was found at artifacts/models/best_transfer_model.h5 or artifacts/models/best_custom_cnn.h5')

st.title('Aerial Object Classification (Bird vs Drone)')
uploaded = st.file_uploader('Upload an image', type=['jpg','jpeg','png'])
if uploaded is not None:
    img = Image.open(uploaded).convert('RGB')
    st.image(img, caption='Uploaded image', use_column_width=True)
    # Preprocess
    img_resized = img.resize((224,224))
    # Keep pixel range as [0,255]; model contains its own normalization/preprocessing.
    x = np.array(img_resized, dtype=np.float32)
    x = np.expand_dims(x, 0)
    try:
        model = load_model()
        pred = model.predict(x, verbose=0)[0]
        # Keep probabilities valid even if a logits model is loaded.
        if not np.isclose(np.sum(pred), 1.0, atol=1e-3):
            pred = tf.nn.softmax(pred).numpy()

        cls = int(np.argmax(pred))
        conf = float(pred[cls])
        labels = ['bird','drone']
        sorted_probs = np.sort(pred)
        margin = float(sorted_probs[-1] - sorted_probs[-2])

        st.success(f'Prediction: {labels[cls]} (confidence: {conf:.4f})')
        st.caption(f'Confidence margin (top1-top2): {margin:.4f}')

        if conf < 0.80 or margin < 0.20:
            st.warning('Low-confidence prediction. Try a clearer image or different angle.')

        for label, prob in zip(labels, pred):
            st.write(f'{label}: {prob:.6f}')
            st.progress(float(prob))
    except Exception as e:
        st.warning(
            'Model not found or failed to load. Train with scripts/train_classification.py to generate model files in artifacts/models.'
        )
        st.text(str(e))
