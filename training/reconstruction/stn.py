import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np


class BilinearSampler(layers.Layer):
    """Bilinear sampling layer for spatial transformer networks."""

    def __init__(self, **kwargs):
        super(BilinearSampler, self).__init__(**kwargs)

    def call(self, inputs):
        """
        Args:
            inputs: [image, grid]
            image: (B, H, W, C) input image
            grid: (B, H, W, 2) sampling grid with normalized coordinates [-1, 1]
        Returns:
            sampled: (B, H, W, C) transformed image
        """
        image, grid = inputs

        batch_size = tf.shape(image)[0]
        height = tf.shape(image)[1]
        width = tf.shape(image)[2]
        channels = tf.shape(image)[3]

        # Convert grid from [-1, 1] to [0, H-1] and [0, W-1]
        x = (grid[:, :, :, 0] + 1.0) * tf.cast(width - 1, tf.float32) / 2.0
        y = (grid[:, :, :, 1] + 1.0) * tf.cast(height - 1, tf.float32) / 2.0

        # Get corner coordinates
        x0 = tf.cast(tf.floor(x), tf.int32)
        x1 = x0 + 1
        y0 = tf.cast(tf.floor(y), tf.int32)
        y1 = y0 + 1

        # Clip coordinates to image bounds
        x0 = tf.clip_by_value(x0, 0, width - 1)
        x1 = tf.clip_by_value(x1, 0, width - 1)
        y0 = tf.clip_by_value(y0, 0, height - 1)
        y1 = tf.clip_by_value(y1, 0, height - 1)

        # Create batch indices
        batch_idx = tf.range(batch_size)
        batch_idx = tf.reshape(batch_idx, [-1, 1, 1])
        batch_idx = tf.tile(batch_idx, [1, height, width])

        # Flatten indices for gather_nd
        def get_pixel_value(img, x, y):
            indices = tf.stack([batch_idx, y, x], axis=-1)
            return tf.gather_nd(img, indices)

        # Get pixel values at corners
        Ia = get_pixel_value(image, x0, y0)  # (B, H, W, C)
        Ib = get_pixel_value(image, x0, y1)
        Ic = get_pixel_value(image, x1, y0)
        Id = get_pixel_value(image, x1, y1)

        # Calculate weights
        wa = tf.expand_dims((tf.cast(x1, tf.float32) - x) * (tf.cast(y1, tf.float32) - y), -1)
        wb = tf.expand_dims((tf.cast(x1, tf.float32) - x) * (y - tf.cast(y0, tf.float32)), -1)
        wc = tf.expand_dims((x - tf.cast(x0, tf.float32)) * (tf.cast(y1, tf.float32) - y), -1)
        wd = tf.expand_dims((x - tf.cast(x0, tf.float32)) * (y - tf.cast(y0, tf.float32)), -1)

        # Interpolate
        out = wa * Ia + wb * Ib + wc * Ic + wd * Id

        return out


def create_coordinate_grid(height, width):
    """Create normalized coordinate grid for spatial transformation."""
    x = tf.linspace(-1.0, 1.0, width)
    y = tf.linspace(-1.0, 1.0, height)
    X, Y = tf.meshgrid(x, y)
    # Stack to create (H, W, 2) grid
    grid = tf.stack([X, Y], axis=-1)
    return grid


def build_localization_network(input_shape, num_classes=None):
    """
    Build localization network for STN that predicts affine transformation parameters.

    Args:
        input_shape: Shape of input image (H, W, C)
        num_classes: Number of condition classes (optional)

    Returns:
        Model that outputs 6 affine parameters [a, b, c, d, tx, ty]
        where transformation matrix is:
        [a  b  tx]
        [c  d  ty]
        [0  0  1 ]
    """
    img_input = layers.Input(shape=input_shape, name="stn_img_input")

    inputs = [img_input]
    x = img_input

    # Add condition input if specified
    if num_classes is not None:
        cond_input = layers.Input(shape=(num_classes,), name="stn_cond_input")
        inputs.append(cond_input)

        # Broadcast condition to spatial dimensions
        h, w = input_shape[0], input_shape[1]
        cond_spatial = layers.Reshape((1, 1, num_classes))(cond_input)
        cond_spatial = layers.UpSampling2D(size=(h, w), interpolation="nearest")(cond_spatial)

        # Concatenate with image
        x = layers.Concatenate()([img_input, cond_spatial])

    # Convolutional layers for feature extraction
    x = layers.Conv2D(32, 7, strides=2, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D(2)(x)

    x = layers.Conv2D(64, 5, strides=2, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D(2)(x)

    x = layers.Conv2D(128, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D(2)(x)

    # Global average pooling
    x = layers.GlobalAveragePooling2D()(x)

    # Dense layers for transformation parameters
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.2)(x)

    # Output 6 affine parameters
    # Initialize to identity transformation
    theta = layers.Dense(6, activation="linear",
                        kernel_initializer="zeros",
                        bias_initializer=tf.constant_initializer([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]))(x)

    return Model(inputs=inputs, outputs=theta, name="LocalizationNetwork")


class SpatialTransformer(layers.Layer):
    """Spatial Transformer Network layer."""

    def __init__(self, output_size, num_classes=None, **kwargs):
        """
        Args:
            output_size: (height, width) of output image
            num_classes: Number of condition classes for localization network
        """
        super(SpatialTransformer, self).__init__(**kwargs)
        self.output_size = output_size
        self.num_classes = num_classes

        # Build localization network
        self.localization_net = build_localization_network(
            input_shape=(output_size[0], output_size[1], 1),
            num_classes=num_classes
        )

        # Bilinear sampler
        self.sampler = BilinearSampler()

    def call(self, inputs):
        """
        Args:
            inputs: [image] or [image, condition]
            image: (B, H, W, C) input image
            condition: (B, num_classes) one-hot condition vector (optional)
        """
        if self.num_classes is not None:
            image, condition = inputs
            localization_inputs = [image, condition]
        else:
            image = inputs
            localization_inputs = image

        # Get transformation parameters from localization network
        theta = self.localization_net(localization_inputs)  # (B, 6)

        # Create transformation grid
        batch_size = tf.shape(image)[0]
        height, width = self.output_size

        # Create base coordinate grid
        grid = create_coordinate_grid(height, width)  # (H, W, 2)
        grid = tf.expand_dims(grid, 0)  # (1, H, W, 2)
        grid = tf.tile(grid, [batch_size, 1, 1, 1])  # (B, H, W, 2)

        # Reshape grid for matrix multiplication
        grid_flat = tf.reshape(grid, [batch_size, -1, 2])  # (B, H*W, 2)

        # Add homogeneous coordinate
        ones = tf.ones([batch_size, height * width, 1])
        grid_homo = tf.concat([grid_flat, ones], axis=-1)  # (B, H*W, 3)

        # Reshape theta to transformation matrix
        theta_matrix = tf.reshape(theta, [batch_size, 2, 3])  # (B, 2, 3)

        # Apply transformation
        transformed_grid = tf.matmul(grid_homo, theta_matrix, transpose_b=True)  # (B, H*W, 2)
        transformed_grid = tf.reshape(transformed_grid, [batch_size, height, width, 2])

        # Sample from input image using transformed grid
        output = self.sampler([image, transformed_grid])

        return output

    def get_config(self):
        config = super().get_config()
        config.update({
            'output_size': self.output_size,
            'num_classes': self.num_classes
        })
        return config


def test_stn():
    """Test STN components."""
    batch_size = 2
    height, width = 64, 64
    channels = 1

    # Test with condition
    num_classes = 5
    image = tf.random.normal([batch_size, height, width, channels])
    condition = tf.random.uniform([batch_size, num_classes])

    stn = SpatialTransformer(output_size=(height, width), num_classes=num_classes)
    output = stn([image, condition])

    print(f"Input shape: {image.shape}")
    print(f"Condition shape: {condition.shape}")
    print(f"Output shape: {output.shape}")

    # Test without condition
    stn_no_cond = SpatialTransformer(output_size=(height, width), num_classes=None)
    output_no_cond = stn_no_cond(image)

    print(f"Output shape (no condition): {output_no_cond.shape}")


if __name__ == "__main__":
    test_stn()