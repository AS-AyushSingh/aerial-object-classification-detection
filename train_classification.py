"""
train_classification.py

Small, reusable script to train the classification models described in the notebook.
Usage examples:
  python train_classification.py --model custom --data_dir classification_dataset --epochs 10 --batch_size 32
  python train_classification.py --model transfer --data_dir classification_dataset --epochs 10

This script expects a directory structure:
classification_dataset/
  train/
    bird/
    drone/
  valid/
    bird/
    drone/
  test/
    bird/
    drone/

It saves best model to `best_custom_cnn.h5` or `best_transfer_model.h5`.
"""
import argparse
import sys
import os

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
except Exception as e:
    print('TensorFlow is required to run this script. Install it and retry.\n', e)
    sys.exit(1)

import numpy as np


def build_custom_cnn(input_shape=(224,224,3), num_classes=2):
    inputs = keras.Input(shape=input_shape)
    x = layers.RandomFlip('horizontal')(inputs)
    x = layers.RandomRotation(0.1)(x)
    x = layers.Rescaling(1./255)(x)
    x = layers.Conv2D(32,3,activation='relu',padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64,3,activation='relu',padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128,3,activation='relu',padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = keras.Model(inputs, outputs, name='custom_cnn')
    return model


def build_transfer_model(input_shape=(224,224,3), num_classes=2):
    base = keras.applications.MobileNetV2(include_top=False, input_shape=input_shape, weights='imagenet')
    base.trainable = False
    inputs = keras.Input(shape=input_shape)
    x = layers.RandomFlip('horizontal')(inputs)
    x = layers.RandomRotation(0.1)(x)
    x = layers.Rescaling(1./255)(x)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = keras.Model(inputs, outputs, name='mobilenetv2_ft')
    return model


def main(args):
    img_size = (224,224)
    batch_size = args.batch_size

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        os.path.join(args.data_dir, 'train'),
        image_size=img_size, batch_size=batch_size, label_mode='int')
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        os.path.join(args.data_dir, 'valid'),
        image_size=img_size, batch_size=batch_size, label_mode='int')
    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        os.path.join(args.data_dir, 'test'),
        image_size=img_size, batch_size=batch_size, label_mode='int')

    if args.model == 'custom':
        model = build_custom_cnn(input_shape=img_size+(3,))
        out_name = 'best_custom_cnn.h5'
        lr = 1e-3
    else:
        model = build_transfer_model(input_shape=img_size+(3,))
        out_name = 'best_transfer_model.h5'
        lr = 1e-3

    model.compile(optimizer=keras.optimizers.Adam(lr), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(out_name, save_best_only=True)
    ]

    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks)

    print('Evaluating on test set:')
    results = model.evaluate(test_ds)
    print('Test results:', results)

    print('Saved best model as', out_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['custom','transfer'], default='custom')
    parser.add_argument('--data_dir', default='classification_dataset')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()
    main(args)
