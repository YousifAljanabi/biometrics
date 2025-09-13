from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


class BilinearSampler(layers.Layer):
    """Bilinear sampler for STN"""

    def __init__(self):
        super(BilinearSampler, self).__init__()

    def call(self, inputs):
        U, grid = inputs
        batch_size = tf.shape(U)[0]
        height = tf.shape(U)[1]
        width = tf.shape(U)[2]
        channels = tf.shape(U)[3]

        # Reshape grid for sampling
        x = grid[:, :, :, 0]
        y = grid[:, :, :, 1]

        # Scale coordinates to image dimensions
        x = 0.5 * ((x + 1.0) * tf.cast(width - 1, tf.float32))
        y = 0.5 * ((y + 1.0) * tf.cast(height - 1, tf.float32))

        # Get corner coordinates
        x0 = tf.cast(tf.floor(x), tf.int32)
        x1 = x0 + 1
        y0 = tf.cast(tf.floor(y), tf.int32)
        y1 = y0 + 1

        # Clip coordinates
        x0 = tf.clip_by_value(x0, 0, width - 1)
        x1 = tf.clip_by_value(x1, 0, width - 1)
        y0 = tf.clip_by_value(y0, 0, height - 1)
        y1 = tf.clip_by_value(y1, 0, height - 1)

        # Get pixel values
        def get_pixel_value(img, x, y):
            batch_idx = tf.range(0, batch_size)
            batch_idx = tf.reshape(batch_idx, (batch_size, 1, 1))
            b = tf.tile(batch_idx, (1, height, width))
            indices = tf.stack([b, y, x], axis=3)
            return tf.gather_nd(img, indices)

        Ia = get_pixel_value(U, x0, y0)
        Ib = get_pixel_value(U, x0, y1)
        Ic = get_pixel_value(U, x1, y0)
        Id = get_pixel_value(U, x1, y1)

        # Calculate weights
        x0_f = tf.cast(x0, tf.float32)
        x1_f = tf.cast(x1, tf.float32)
        y0_f = tf.cast(y0, tf.float32)
        y1_f = tf.cast(y1, tf.float32)

        wa = tf.expand_dims(((x1_f - x) * (y1_f - y)), axis=3)
        wb = tf.expand_dims(((x1_f - x) * (y - y0_f)), axis=3)
        wc = tf.expand_dims(((x - x0_f) * (y1_f - y)), axis=3)
        wd = tf.expand_dims(((x - x0_f) * (y - y0_f)), axis=3)

        # Interpolate
        output = tf.add_n([wa * Ia, wb * Ib, wc * Ic, wd * Id])

        return output


def create_coordinate_grid(height, width):
    """Create normalized coordinate grid"""
    x_t = tf.linspace(-1.0, 1.0, width)
    y_t = tf.linspace(-1.0, 1.0, height)
    x_t, y_t = tf.meshgrid(x_t, y_t)

    # Flatten and stack
    x_t_flat = tf.reshape(x_t, [-1])
    y_t_flat = tf.reshape(y_t, [-1])
    ones = tf.ones_like(x_t_flat)

    grid = tf.stack([x_t_flat, y_t_flat, ones], axis=0)
    return grid


class RotationSTN(keras.Model):
    """STN for rotation correction"""

    def __init__(self, input_shape):
        super(RotationSTN, self).__init__()

        # Localization network
        self.conv1 = layers.Conv2D(32, 3, activation='relu', padding='same')
        self.pool1 = layers.MaxPooling2D(2)
        self.conv2 = layers.Conv2D(64, 3, activation='relu', padding='same')
        self.pool2 = layers.MaxPooling2D(2)
        self.conv3 = layers.Conv2D(128, 3, activation='relu', padding='same')
        self.pool3 = layers.MaxPooling2D(2)

        self.flatten = layers.Flatten()
        self.fc1 = layers.Dense(256, activation='relu')
        self.fc2 = layers.Dense(128, activation='relu')

        # Output 1 parameter for rotation angle
        self.fc_loc = layers.Dense(2, activation='tanh')

        self.sampler = BilinearSampler()

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        height = tf.shape(inputs)[1]
        width = tf.shape(inputs)[2]

        # Localization network
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool3(x)

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)

        # Get rotation angle
        # Get cosθ and sinθ predictions
        theta = self.fc_loc(x)  # [batch_size, 2]
        cos_theta, sin_theta = tf.split(theta, 2, axis=1)

        # Normalize to ensure unit circle
        norm = tf.sqrt(tf.square(cos_theta) + tf.square(sin_theta) + 1e-8)
        cos_theta = cos_theta / norm
        sin_theta = sin_theta / norm

        # Build rotation matrix
        zero = tf.zeros_like(cos_theta)
        transform_matrix = tf.stack([
            cos_theta, -sin_theta, zero,
            sin_theta, cos_theta, zero
        ], axis=1)
        transform_matrix = tf.reshape(transform_matrix, [batch_size, 2, 3])

        # Create coordinate grid and apply transformation
        grid = create_coordinate_grid(height, width)
        grid = tf.expand_dims(grid, 0)
        grid = tf.tile(grid, [batch_size, 1, 1])

        T_g = tf.matmul(transform_matrix, grid)
        x_s = T_g[:, 0, :]
        y_s = T_g[:, 1, :]

        x_s = tf.reshape(x_s, [batch_size, height, width])
        y_s = tf.reshape(y_s, [batch_size, height, width])

        sampling_grid = tf.stack([x_s, y_s], axis=3)
        output = self.sampler([inputs, sampling_grid])

        return output


def load_image_pairs(healthy_dir, distorted_dir, target_size=(224, 224)):
    """
    Load corresponding image pairs from healthy and distorted directories

    Args:
        healthy_dir (str): Path to directory containing healthy images
        distorted_dir (str): Path to directory containing distorted images
        target_size (tuple): Target size to resize images to

    Returns:
        tuple: (distorted_images, healthy_images) as numpy arrays
    """
    distorted_images = []
    healthy_images = []

    # Get all image files from distorted directory
    distorted_files = [f for f in os.listdir(distorted_dir)
                       if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]

    for filename in distorted_files:
        print(filename)
        distorted_path = os.path.join(distorted_dir, f"{filename}")
        print(distorted_path)

        healthy_path = os.path.join(healthy_dir, filename)
        print(healthy_path)

        # Check if corresponding healthy image exists
        if os.path.exists(healthy_path):
            # Load distorted image
            distorted_img = cv2.imread(distorted_path)
            if distorted_img is None:
                continue
            distorted_img = cv2.cvtColor(distorted_img, cv2.COLOR_BGR2RGB)
            distorted_img = cv2.resize(distorted_img, target_size)

            # Load healthy image
            healthy_img = cv2.imread(healthy_path)
            if healthy_img is None:
                continue
            healthy_img = cv2.cvtColor(healthy_img, cv2.COLOR_BGR2RGB)
            healthy_img = cv2.resize(healthy_img, target_size)

            distorted_images.append(distorted_img)
            healthy_images.append(healthy_img)

    # Convert to numpy arrays and normalize
    distorted_images = np.array(distorted_images, dtype=np.float32) / 255.0
    healthy_images = np.array(healthy_images, dtype=np.float32) / 255.0

    return distorted_images, healthy_images


def ssim_loss(y_true, y_pred):
    """
    SSIM loss for better structural similarity preservation
    """
    return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))


def train_stn_model(model, train_data, val_data, epochs=100):
    """
    Train STN model with improved setup for fingerprint rotation correction
    """

    # Compile model with lower LR and SSIM loss
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=ssim_loss,
        metrics=['mae']
    )

    # Callbacks
    callbacks = [
        # Stop if no improvement after 15 epochs (instead of 10)
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=15,
            min_delta=1e-4,         # require small improvement to count
            restore_best_weights=True
        ),
        # Reduce LR if plateau for 7 epochs
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=7,
            min_lr=1e-6,
            verbose=1
        ),
        # Save best model weights
        keras.callbacks.ModelCheckpoint(
            filepath="best_stn.weights.h5",
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=True,
            verbose=1
        )
    ]

    # Train model
    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )

    return history

def create_stn_model(distortion_type, input_shape):
    """
    Create STN model based on distortion type

    Args:
        distortion_type (str): One of 'translation', 'rotation', 'scaling', 'affine', 'perspective'
        input_shape (tuple): Input image shape (height, width, channels)

    Returns:
        STN model for the specified distortion type
    """

    if distortion_type == 'rotation':
        return RotationSTN(input_shape)
    else:
        raise ValueError(f"Unknown distortion type: {distortion_type}")


def reconstruct_image(model, image_path, target_size=(224, 224)):
    """
    Reconstruct a distorted image using trained STN model

    Args:
        model: Trained STN model
        image_path (str): Path to distorted image
        target_size (tuple): Target size for image processing

    Returns:
        tuple: (original_image, reconstructed_image) as numpy arrays
    """
    # Load and preprocess image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    original_img = img.copy()

    # Resize and normalize
    img = cv2.resize(img, target_size)
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    # Reconstruct using STN
    reconstructed = model.predict(img)
    reconstructed = np.squeeze(reconstructed, axis=0)  # Remove batch dimension

    return original_img, reconstructed


def plot_reconstruction(original_img, reconstructed_img, save_path=None):
    """
    Plot original and reconstructed images side by side

    Args:
        original_img: Original distorted image
        reconstructed_img: Reconstructed image from STN
        save_path (str): Optional path to save the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Plot original image
    axes[0].imshow(original_img)
    axes[0].set_title('Original Distorted Image')
    axes[0].axis('off')

    # Plot reconstructed image
    axes[1].imshow(reconstructed_img)
    axes[1].set_title('STN Reconstructed Image')
    axes[1].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def test():
    """
    Test function for STN model training and reconstruction
    """
    # 1. Specify directories
    healthy_dir = "./healthy/DB2_B_bmp"  # Replace with your healthy images directory
    distorted_dir = "./distorted/test_out"  # Replace with your distorted images directory

    # Image processing parameters
    target_size = (224, 224)
    input_shape = (224, 224, 3)

    print("Loading image pairs...")
    # 2. Load image pairs
    distorted_images, healthy_images = load_image_pairs(healthy_dir, distorted_dir, target_size)

    print(f"Loaded {len(distorted_images)} image pairs")

    if len(distorted_images) == 0:
        print("No image pairs found. Please check your directory paths.")
        return

    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        distorted_images, healthy_images,
        test_size=0.2,
        random_state=42
    )

    print(f"Training set: {len(X_train)} images")
    print(f"Validation set: {len(X_val)} images")

    # 3. Create and train model
    print("Creating STN model...")
    model = create_stn_model('rotation', input_shape)

    # Create datasets
    batch_size = 32
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(batch_size)
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(batch_size)

    print("Training model...")
    history = train_stn_model(model, train_dataset, val_dataset, epochs=50)

    # Plot training history
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Training MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title('Model MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # 4. Test on a specific image
    test_image_path = "./distorted/test_out/101_1.bmp"  # Replace with your test image path

    if os.path.exists(test_image_path):
        print("Reconstructing test image...")
        # 5. Reconstruct the image
        original_img, reconstructed_img = reconstruct_image(model, test_image_path, target_size)

        # 6. Plot the results
        plot_reconstruction(original_img, reconstructed_img)

        # Save model for future use
        model.save_weights('stn_rotation_model.weights.h5')
        print("Model weights saved as 'stn_rotation_model.weights.h5'")
    else:
        print(f"Test image not found at: {test_image_path}")
        print("Please update the test_image_path variable with a valid image path.")

test()