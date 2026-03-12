"""
Train image classification models (Bird vs Drone).

Usage examples from project root:
    python scripts/train_classification.py --model custom --data_dir classification_dataset
    python scripts/train_classification.py --model transfer --epochs 15
"""
import argparse
import sys
import os
import numpy as np

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
except Exception as e:
    print('TensorFlow is required to run this script. Install it and retry.\n', e)
    sys.exit(1)


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
    # Equivalent to MobileNetV2 preprocess_input for [0,255] inputs.
    x = layers.Rescaling(1./127.5, offset=-1)(x)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = keras.Model(inputs, outputs, name='mobilenetv2_ft')
    return model


def main(args):
    tf.keras.utils.set_random_seed(args.seed)
    img_size = (224,224)
    batch_size = args.batch_size
    class_names = ['bird', 'drone']

    for subset in ['train', 'valid', 'test']:
        subset_path = os.path.join(args.data_dir, subset)
        if not os.path.exists(subset_path):
            raise FileNotFoundError(f'Missing dataset folder: {subset_path}')

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        os.path.join(args.data_dir, 'train'),
        image_size=img_size, batch_size=batch_size, label_mode='int', class_names=class_names)
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        os.path.join(args.data_dir, 'valid'),
        image_size=img_size, batch_size=batch_size, label_mode='int', class_names=class_names)
    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        os.path.join(args.data_dir, 'test'),
        image_size=img_size, batch_size=batch_size, label_mode='int', class_names=class_names)

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.shuffle(1000, seed=args.seed).cache().prefetch(autotune)
    val_ds = val_ds.cache().prefetch(autotune)
    test_ds = test_ds.cache().prefetch(autotune)

    train_counts = {
        'bird': len(tf.io.gfile.glob(os.path.join(args.data_dir, 'train', 'bird', '*'))),
        'drone': len(tf.io.gfile.glob(os.path.join(args.data_dir, 'train', 'drone', '*'))),
    }
    print('Train class counts:', train_counts)
    total_train = max(1, train_counts['bird'] + train_counts['drone'])
    class_weight = {
        0: total_train / (2.0 * max(1, train_counts['bird'])),
        1: total_train / (2.0 * max(1, train_counts['drone'])),
    }
    print('Using class_weight:', class_weight)

    if args.model == 'custom':
        model = build_custom_cnn(input_shape=img_size+(3,))
        out_name = os.path.join(args.out_dir, 'best_custom_cnn.h5')
        lr = 1e-3
    else:
        model = build_transfer_model(input_shape=img_size+(3,))
        out_name = os.path.join(args.out_dir, 'best_transfer_model.h5')
        lr = 1e-3

    os.makedirs(args.out_dir, exist_ok=True)

    model.compile(
        optimizer=keras.optimizers.Adam(lr),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy', keras.metrics.SparseTopKCategoricalAccuracy(k=2, name='top2_acc')],
    )
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6),
        keras.callbacks.ModelCheckpoint(out_name, save_best_only=True)
    ]

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        class_weight=class_weight,
    )

    if args.model == 'transfer' and args.fine_tune_epochs > 0:
        print('Starting fine-tuning stage...')
        base_model = model.get_layer('mobilenetv2_1.00_224')
        base_model.trainable = True
        for layer in base_model.layers[:-args.fine_tune_last_n]:
            layer.trainable = False

        model.compile(
            optimizer=keras.optimizers.Adam(args.fine_tune_lr),
            loss=keras.losses.SparseCategoricalCrossentropy(),
            metrics=['accuracy', keras.metrics.SparseTopKCategoricalAccuracy(k=2, name='top2_acc')],
        )
        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=args.epochs + args.fine_tune_epochs,
            initial_epoch=args.epochs,
            callbacks=callbacks,
            class_weight=class_weight,
        )

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
    parser.add_argument('--out_dir', default='artifacts/models')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--fine_tune_epochs', type=int, default=5)
    parser.add_argument('--fine_tune_last_n', type=int, default=40)
    parser.add_argument('--fine_tune_lr', type=float, default=1e-5)
    args = parser.parse_args()
    main(args)
