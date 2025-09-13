import tensorflow as tf
from keras.src.applications.mobilenet_v2 import MobileNetV2
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt



class SpatialTransformer(layers.Layer):
    def __init__(self, out_size, **kwargs):
        super().__init__(**kwargs)
        self.out_h, self.out_w = out_size

    def call(self, inputs):
        U, theta = inputs
        B = tf.shape(U)[0]
        H, W = self.out_h, self.out_w

        # make grid
        grid_x, grid_y = tf.meshgrid(
            tf.linspace(-1.0, 1.0, W),
            tf.linspace(-1.0, 1.0, H)
        )
        ones = tf.ones_like(grid_x)
        grid = tf.stack([grid_x, grid_y, ones], axis=-1)  # (H, W, 3)
        grid = tf.expand_dims(grid, 0)
        grid = tf.tile(grid, [B, 1, 1, 1])  # (B, H, W, 3)

        # build rotation matrix
        cos_t = tf.cos(theta)
        sin_t = tf.sin(theta)
        zeros = tf.zeros_like(theta)

        rot = tf.concat([cos_t, -sin_t, zeros,
                         sin_t, cos_t, zeros], axis=1)
        rot = tf.reshape(rot, [-1, 2, 3])

        # apply transform
        grid = tf.reshape(grid, [B, H * W, 3])
        grid = tf.transpose(grid, [0, 2, 1])
        T_g = tf.matmul(rot, grid)  # (B, 2, HW)
        T_g = tf.transpose(T_g, [0, 2, 1])
        x_s = T_g[:, :, 0]
        y_s = T_g[:, :, 1]

        # scale back
        x = (x_s + 1) * (tf.cast(W, tf.float32) - 1) / 2.0
        y = (y_s + 1) * (tf.cast(H, tf.float32) - 1) / 2.0

        # bilinear sampling
        x0 = tf.cast(tf.floor(x), tf.int32)
        x1 = x0 + 1
        y0 = tf.cast(tf.floor(y), tf.int32)
        y1 = y0 + 1

        Hc = tf.cast(H - 1, tf.int32)
        Wc = tf.cast(W - 1, tf.int32)
        x0 = tf.clip_by_value(x0, 0, Wc)
        x1 = tf.clip_by_value(x1, 0, Wc)
        y0 = tf.clip_by_value(y0, 0, Hc)
        y1 = tf.clip_by_value(y1, 0, Hc)

        Ia = tf.gather_nd(U, tf.stack([y0, x0], axis=-1), batch_dims=1)
        Ib = tf.gather_nd(U, tf.stack([y1, x0], axis=-1), batch_dims=1)
        Ic = tf.gather_nd(U, tf.stack([y0, x1], axis=-1), batch_dims=1)
        Id = tf.gather_nd(U, tf.stack([y1, x1], axis=-1), batch_dims=1)

        x0_f = tf.cast(x0, tf.float32)
        x1_f = tf.cast(x1, tf.float32)
        y0_f = tf.cast(y0, tf.float32)
        y1_f = tf.cast(y1, tf.float32)

        wa = (x1_f - x) * (y1_f - y)
        wb = (x1_f - x) * (y - y0_f)
        wc = (x - x0_f) * (y1_f - y)
        wd = (x - x0_f) * (y - y0_f)

        out = tf.expand_dims(wa, -1) * Ia + tf.expand_dims(wb, -1) * Ib + \
              tf.expand_dims(wc, -1) * Ic + tf.expand_dims(wd, -1) * Id

        return tf.reshape(out, [B, H, W, -1])


# ---------------------------
# 1. Load and preprocess images
# ---------------------------
def load_image_pairs(healthy_dir, distorted_dir, target_size):
    healthy_images, distorted_images = [], []

    healthy_files = sorted(os.listdir(healthy_dir))
    distorted_files = sorted(os.listdir(distorted_dir))

    # Match by index (assumes same filenames or order)
    for h_file, d_file in zip(healthy_files, distorted_files):
        h_path = os.path.join(healthy_dir, h_file)
        d_path = os.path.join(distorted_dir, d_file)

        if not (os.path.exists(h_path) and os.path.exists(d_path)):
            continue

        h_img = cv2.imread(h_path)
        d_img = cv2.imread(d_path)

        if h_img is None or d_img is None:
            continue

        h_img = cv2.resize(h_img, target_size)
        d_img = cv2.resize(d_img, target_size)

        # Normalize
        h_img = h_img.astype("float32") / 255.0
        d_img = d_img.astype("float32") / 255.0

        healthy_images.append(h_img)
        distorted_images.append(d_img)

    return np.array(distorted_images), np.array(healthy_images)


# ---------------------------
# 2. STN model
# ---------------------------
def localization_network(input_shape):
    """
    Use MobileNetV2 (pretrained on ImageNet) as a feature extractor,
    then predict the rotation angle theta.
    """
    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet"  # you can also set to None if you want random init
    )

    # Freeze most layers (fine-tune later if needed)
    for layer in base_model.layers[:-20]:  # keep last 20 trainable
        layer.trainable = False

    inputs = layers.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    # Output: one angle in radians
    theta = layers.Dense(1, kernel_initializer="zeros", bias_initializer="zeros")(x)

    return models.Model(inputs, theta, name="localization_net")

def transformer(U, theta, out_size):
    """Apply affine rotation transform with predicted theta."""
    B = tf.shape(U)[0]
    H, W = out_size

    # Create normalized grid
    grid_x, grid_y = tf.meshgrid(
        tf.linspace(-1.0, 1.0, W),
        tf.linspace(-1.0, 1.0, H)
    )
    ones = tf.ones_like(grid_x)
    grid = tf.stack([grid_x, grid_y, ones], axis=-1)  # (H, W, 3)
    grid = tf.expand_dims(grid, 0)
    grid = tf.tile(grid, [B, 1, 1, 1])  # (B, H, W, 3)

    # Build affine matrix for rotation
    cos_t = tf.cos(theta)
    sin_t = tf.sin(theta)
    zeros = tf.zeros_like(theta)

    rot = tf.concat([cos_t, -sin_t, zeros,
                     sin_t, cos_t, zeros], axis=1)
    rot = tf.reshape(rot, [-1, 2, 3])

    # Apply transform
    grid = tf.reshape(grid, [B, H * W, 3])
    grid = tf.transpose(grid, [0, 2, 1])
    T_g = tf.matmul(rot, grid)  # (B, 2, HW)
    T_g = tf.transpose(T_g, [0, 2, 1])
    x_s = T_g[:, :, 0]
    y_s = T_g[:, :, 1]

    # Normalize back to [0, W/H]
    x = (x_s + 1) * (tf.cast(W, tf.float32) - 1) / 2.0
    y = (y_s + 1) * (tf.cast(H, tf.float32) - 1) / 2.0

    # Bilinear sampling
    x0 = tf.cast(tf.floor(x), tf.int32)
    x1 = x0 + 1
    y0 = tf.cast(tf.floor(y), tf.int32)
    y1 = y0 + 1

    Hc = tf.cast(H - 1, tf.int32)
    Wc = tf.cast(W - 1, tf.int32)
    x0 = tf.clip_by_value(x0, 0, Wc)
    x1 = tf.clip_by_value(x1, 0, Wc)
    y0 = tf.clip_by_value(y0, 0, Hc)
    y1 = tf.clip_by_value(y1, 0, Hc)

    Ia = tf.gather_nd(U, tf.stack([y0, x0], axis=-1), batch_dims=1)
    Ib = tf.gather_nd(U, tf.stack([y1, x0], axis=-1), batch_dims=1)
    Ic = tf.gather_nd(U, tf.stack([y0, x1], axis=-1), batch_dims=1)
    Id = tf.gather_nd(U, tf.stack([y1, x1], axis=-1), batch_dims=1)

    x0_f = tf.cast(x0, tf.float32)
    x1_f = tf.cast(x1, tf.float32)
    y0_f = tf.cast(y0, tf.float32)
    y1_f = tf.cast(y1, tf.float32)

    wa = (x1_f - x) * (y1_f - y)
    wb = (x1_f - x) * (y - y0_f)
    wc = (x - x0_f) * (y1_f - y)
    wd = (x - x0_f) * (y - y0_f)

    out = tf.expand_dims(wa, -1) * Ia + tf.expand_dims(wb, -1) * Ib + \
          tf.expand_dims(wc, -1) * Ic + tf.expand_dims(wd, -1) * Id

    return tf.reshape(out, [B, H, W, -1])


def create_stn_model(mode, input_shape):
    inputs = layers.Input(shape=input_shape)
    locnet = localization_network(input_shape)
    theta = locnet(inputs)

    # use the custom layer here
    out = SpatialTransformer(out_size=input_shape[:2])([inputs, theta])
    model = models.Model(inputs, out)
    return model

# ---------------------------
# 3. Train STN
# ---------------------------
def train_stn_model(model, train_dataset, val_dataset, epochs=50):
    def hybrid_loss(y_true, y_pred):
        ssim = tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))
        mse = tf.reduce_mean(tf.square(y_true - y_pred))
        return 0.5 * mse + 0.5 * (1 - ssim)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss=hybrid_loss,
        metrics=["mae"]
    )
    early_stop = tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
    history = model.fit(train_dataset, validation_data=val_dataset,
                        epochs=epochs, callbacks=[early_stop])
    return history


# ---------------------------
# 4. Reconstruction
# ---------------------------
def reconstruct_image(model, test_image_path, target_size):
    img = cv2.imread(test_image_path)
    original_img = cv2.resize(img, target_size)
    img = original_img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    reconstructed = model.predict(img)[0]
    return original_img, reconstructed


def plot_reconstruction(original_img, reconstructed_img):
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(reconstructed_img)
    plt.title("Reconstructed")
    plt.axis("off")

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
    batch_size = 128
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

