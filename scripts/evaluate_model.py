"""
Evaluate a trained Keras classification model.

Usage from project root:
    python scripts/evaluate_model.py --model_path artifacts/models/best_custom_cnn.h5
"""
import argparse
import os
import sys

try:
    import tensorflow as tf
    from tensorflow import keras
except Exception as e:
    print('TensorFlow is required to run this script. Install it and retry.\n', e)
    sys.exit(1)

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def load_dataset(data_dir, subset='test', img_size=(224,224), batch_size=32):
    path = os.path.join(data_dir, subset)
    if not os.path.exists(path):
        raise FileNotFoundError(f'{path} not found')
    ds = tf.keras.preprocessing.image_dataset_from_directory(
        path,
        image_size=img_size,
        batch_size=batch_size,
        label_mode='int',
        class_names=['bird', 'drone']
    )
    return ds


def evaluate(model_path, data_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    model = keras.models.load_model(model_path)
    ds = load_dataset(data_dir, subset='test')
    y_true = []
    y_pred = []
    class_names = ds.class_names if hasattr(ds, 'class_names') else None
    for x_batch, y_batch in ds:
        preds = model.predict(x_batch)
        preds_labels = np.argmax(preds, axis=1)
        y_true.extend(y_batch.numpy().tolist())
        y_pred.extend(preds_labels.tolist())
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    print(report)
    with open(os.path.join(out_dir, 'classification_report.txt'), 'w') as f:
        f.write(report)
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,5))
    if class_names is None:
        sns.heatmap(cm, annot=True, fmt='d')
        plt.xlabel('Predicted')
        plt.ylabel('True')
    else:
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'confusion_matrix.png'))
    print('Saved outputs to', out_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--data_dir', default='classification_dataset')
    parser.add_argument('--out_dir', default='reports/evaluation')
    args = parser.parse_args()
    evaluate(args.model_path, args.data_dir, args.out_dir)
