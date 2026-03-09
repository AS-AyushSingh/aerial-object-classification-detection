#!/usr/bin/env python3
"""
Aerial Object Classification & Detection - Simplified Training Script
Runs with TensorFlow 2.20.0 - No scikit-learn dependency
"""

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import sys
import numpy as np
from pathlib import Path

# TensorFlow imports only
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print("=" * 70)
print("AERIAL OBJECT CLASSIFICATION & DETECTION - TENSORFLOW TRAINING")
print("=" * 70)
print(f"\n✓ TensorFlow version: {tf.__version__}")
print(f"✓ Keras version: {keras.__version__}")
print(f"✓ Python version: {sys.version.split()[0]}\n")

# ============================================================================
# PART 1: Dataset Setup
# ============================================================================
print("PART 1: LOADING DATASET")
print("-" * 70)

base_dir = Path('classification_dataset')
train_dir = base_dir / 'train'
valid_dir = base_dir / 'valid'
test_dir = base_dir / 'test'

def count_images(folder):
    counts = {}
    if not folder.exists():
        return counts
    for c in folder.iterdir():
        if c.is_dir():
            counts[c.name] = len(list(c.glob('*.jpg'))) + len(list(c.glob('*.jpeg'))) + len(list(c.glob('*.png')))
    return counts

train_counts = count_images(train_dir)
valid_counts = count_images(valid_dir)
test_counts = count_images(test_dir)

print(f"Train: {sum([v for k,v in train_counts.items() if k in ['bird','drone']])} images")
print(f"Valid: {sum([v for k,v in valid_counts.items() if k in ['bird','drone']])} images")
print(f"Test:  {sum([v for k,v in test_counts.items() if k in ['bird','drone']])} images")

# ============================================================================
# PART 2: Prepare TensorFlow Datasets
# ============================================================================
print("\nPART 2: PREPARING TENSORFLOW DATASETS")
print("-" * 70)

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE
seed = 123

if train_dir.exists():
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        str(train_dir), labels='inferred', label_mode='int', 
        image_size=IMG_SIZE, batch_size=BATCH_SIZE, seed=seed
    )
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    print("✓ Training dataset loaded")
else:
    train_ds = None
    print("✗ Training dataset not found")

if valid_dir.exists():
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        str(valid_dir), labels='inferred', label_mode='int',
        image_size=IMG_SIZE, batch_size=BATCH_SIZE, seed=seed
    )
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
    print("✓ Validation dataset loaded")
else:
    val_ds = None
    print("✗ Validation dataset not found")

if test_dir.exists():
    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        str(test_dir), labels='inferred', label_mode='int',
        image_size=IMG_SIZE, batch_size=BATCH_SIZE, seed=seed
    )
    test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)
    print("✓ Test dataset loaded")
else:
    test_ds = None
    print("✗ Test dataset not found")

# ============================================================================
# PART 3: Build Models
# ============================================================================
print("\nPART 3: BUILDING MODELS")
print("-" * 70)

# Data augmentation and normalization
data_augmentation = keras.Sequential([
    layers.RandomFlip('horizontal'),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
], name='data_augmentation')
normalization_layer = layers.Rescaling(1./255)

# Custom CNN
def build_custom_cnn(input_shape=(224, 224, 3), num_classes=2):
    inputs = keras.Input(shape=input_shape)
    x = data_augmentation(inputs)
    x = normalization_layer(x)
    x = layers.Conv2D(32, 3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = keras.Model(inputs, outputs, name='custom_cnn')
    return model

custom_model = build_custom_cnn()
custom_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
print("✓ Custom CNN built")
print(f"  Parameters: {custom_model.count_params():,}")

# Transfer Learning (MobileNetV2)
def build_transfer_model(input_shape=(224, 224, 3), num_classes=2):
    base = keras.applications.MobileNetV2(include_top=False, input_shape=input_shape, weights='imagenet')
    base.trainable = False
    inputs = keras.Input(shape=input_shape)
    x = data_augmentation(inputs)
    x = normalization_layer(x)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = keras.Model(inputs, outputs, name='mobilenetv2_ft')
    return model

transfer_model = build_transfer_model()
transfer_model.compile(optimizer=keras.optimizers.Adam(1e-3), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
print("✓ Transfer Learning (MobileNetV2) model built")
print(f"  Parameters: {transfer_model.count_params():,}")

# ============================================================================
# PART 4: Training
# ============================================================================
print("\nPART 4: TRAINING MODELS")
print("-" * 70)

if train_ds is None:
    print("Skipping training - no training data")
else:
    # Quick test run (1 epoch on subset)
    ds_train = train_ds.take(100)
    ds_val = val_ds.take(50) if val_ds is not None else None
    
    print(f"\nTraining Custom CNN (1 epoch on {100*BATCH_SIZE} samples)...")
    callbacks = [
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint('best_custom_cnn.h5', save_best_only=True)
    ]
    history = custom_model.fit(ds_train, validation_data=ds_val, epochs=1, callbacks=callbacks, verbose=1)
    print("✓ Custom CNN training completed")
    
    print(f"\nTraining Transfer Model (1 epoch on {100*BATCH_SIZE} samples)...")
    callbacks = [
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint('best_transfer_model.h5', save_best_only=True)
    ]
    history = transfer_model.fit(ds_train, validation_data=ds_val, epochs=1, callbacks=callbacks, verbose=1)
    print("✓ Transfer model training completed")

# ============================================================================
# PART 5: Evaluation
# ============================================================================
print("\nPART 5: MODEL EVALUATION")
print("-" * 70)

if test_ds is None:
    print("Skipping evaluation - no test data")
else:
    ds_eval = test_ds.take(100)
    y_true = []
    y_pred = []
    
    print("\nEvaluating Custom CNN on test set...")
    for x_batch, y_batch in ds_eval:
        preds = custom_model.predict(x_batch, verbose=0)
        preds_labels = np.argmax(preds, axis=1)
        y_true.extend(y_batch.numpy().tolist())
        y_pred.extend(preds_labels.tolist())
    
    from collections import defaultdict
    correct = sum([1 for i in range(len(y_true)) if y_true[i] == y_pred[i]])
    print(f"\nTest Accuracy: {100.0 * correct / len(y_true):.2f}%")
    print(f"Correct: {correct}/{len(y_true)}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("TRAINING COMPLETE!")
print("=" * 70)
print("\nModels saved:")
print("  - best_custom_cnn.h5")
print("  - best_transfer_model.h5")
print("\nNext steps:")
print("  1. Increase EPOCHS for production training")
print("  2. Use 'streamlit run app.py' to deploy")
print("  3. Train YOLOv8 for object detection")
print("=" * 70)
