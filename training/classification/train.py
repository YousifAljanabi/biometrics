#!/usr/bin/env python3
"""
Training script for fingerprint distortion classification model.
Loads distorted images from dataset/distorted/ with folder-based labels.
"""

import os
import tensorflow as tf
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_classification_model(num_classes: int, input_shape: tuple = (256, 256, 1)) -> tf.keras.Model:
    """
    Create a CNN model for distortion classification.

    Args:
        num_classes: Number of distortion classes
        input_shape: Input image shape (H, W, C)

    Returns:
        Compiled Keras model
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Rescaling(1./255),

        # First conv block
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.25),

        # Second conv block
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.25),

        # Third conv block
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.25),

        # Fourth conv block
        tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.5),

        # Classification head
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    return model

def main():
    """Main training function."""
    # Configuration
    img_size = (256, 256)
    batch_size = 16
    epochs = 50
    validation_split = 0.2

    # Get dataset path
    current_dir = Path(__file__).parent
    project_root = current_dir.parent.parent
    dataset_dir = project_root / "dataset" / "distorted"

    logger.info(f"Loading dataset from: {dataset_dir}")

    if not dataset_dir.exists():
        logger.error(f"Dataset directory {dataset_dir} does not exist")
        return

    # Create datasets from directory structure
    train_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_dir,
        validation_split=validation_split,
        subset="training",
        seed=123,
        image_size=img_size,
        batch_size=batch_size,
        color_mode='grayscale',  # Grayscale for fingerprints
        label_mode='categorical'
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_dir,
        validation_split=validation_split,
        subset="validation",
        seed=123,
        image_size=img_size,
        batch_size=batch_size,
        color_mode='grayscale',
        label_mode='categorical'
    )

    # Get class names and number of classes
    class_names = train_ds.class_names
    num_classes = len(class_names)

    logger.info(f"Found {num_classes} classes: {class_names}")
    logger.info(f"Training batches: {len(train_ds)}")
    logger.info(f"Validation batches: {len(val_ds)}")

    # Optimize dataset performance
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    # Data augmentation
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomContrast(0.1),
    ])

    # Apply data augmentation to training dataset
    train_ds = train_ds.map(
        lambda x, y: (data_augmentation(x, training=True), y),
        num_parallel_calls=AUTOTUNE
    )

    # Create and compile model
    model = create_classification_model(num_classes, input_shape=(256, 256, 1))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=3)]
    )

    # Print model summary
    logger.info("Model architecture:")
    model.summary()

    # Callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            "distortion_classifier_best.h5",
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
    ]

    # Train the model
    logger.info(f"Starting training for {epochs} epochs")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )

    # Save final model and class names
    model.save("distortion_classifier_final.h5")

    # Save class names for later use
    import json
    class_names_file = current_dir / "class_names.json"
    with open(class_names_file, 'w') as f:
        json.dump(class_names, f)

    logger.info(f"Training completed!")
    logger.info(f"Best model saved as: distortion_classifier_best.h5")
    logger.info(f"Final model saved as: distortion_classifier_final.h5")
    logger.info(f"Class names saved as: {class_names_file}")

    # Print final metrics
    final_loss = history.history['val_loss'][-1]
    final_accuracy = history.history['val_accuracy'][-1]
    logger.info(f"Final validation loss: {final_loss:.4f}")
    logger.info(f"Final validation accuracy: {final_accuracy:.4f}")

if __name__ == "__main__":
    main()