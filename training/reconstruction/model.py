import tensorflow as tf
from tensorflow.keras import layers, Model
from stn import SpatialTransformer

def conv_block(x, filters, k=3):
    x = layers.Conv2D(filters, k, padding="same", activation=None)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(filters, k, padding="same", activation=None)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    return x

def down(x, filters):
    c = conv_block(x, filters)
    p = layers.MaxPooling2D()(c)
    return c, p

def up(x, skip, filters):
    x = layers.Conv2DTranspose(filters, 2, strides=2, padding="same")(x)
    x = layers.Concatenate()([x, skip])
    x = conv_block(x, filters)
    return x

def broadcast_condition_to_hw(cond, height, width):
    # cond: (B, C) one-hot; -> (B, H, W, C) by tiling
    cond = layers.Dense(cond.shape[-1], activation=None)(cond)  # identity pass (keeps API)
    cond = layers.Reshape((1, 1, cond.shape[-1]))(cond)
    cond = layers.UpSampling2D(size=(height, width), interpolation="nearest")(cond)
    return cond

def build_unet(
    input_shape=(256, 256, 1),
    num_cond_classes: int | None = None,
    residual=True,
    use_stn=False,
):
    """
    U-Net for elastic fingerprint restoration.
    If num_cond_classes is not None, expects a one-hot condition input to guide the network
    (e.g., ['stretch', 'compression', 'sliding', 'nonrigid', 'curl']).
    If use_stn is True, adds Spatial Transformer Network for geometric distortion handling.
    """
    img_in = layers.Input(shape=input_shape, name="distorted_img")

    if num_cond_classes is not None:
        cond_in = layers.Input(shape=(num_cond_classes,), name="distortion_onehot")
        inputs = [img_in, cond_in]

        # Apply STN if enabled for geometric distortions
        if use_stn:
            # Apply spatial transformation to correct geometric distortions
            stn = SpatialTransformer(
                output_size=(input_shape[0], input_shape[1]),
                num_classes=num_cond_classes
            )
            img_transformed = stn([img_in, cond_in])

            # Concatenate condition as extra channels at input resolution
            h, w = input_shape[0], input_shape[1]
            cond_map = broadcast_condition_to_hw(cond_in, h, w)
            x = layers.Concatenate()([img_transformed, cond_map])
        else:
            # Concatenate condition as extra channels at input resolution
            h, w = input_shape[0], input_shape[1]
            cond_map = broadcast_condition_to_hw(cond_in, h, w)
            x = layers.Concatenate()([img_in, cond_map])

    else:
        inputs = img_in

        # Apply STN if enabled (without condition)
        if use_stn:
            stn = SpatialTransformer(
                output_size=(input_shape[0], input_shape[1]),
                num_classes=None
            )
            x = stn(img_in)
        else:
            x = img_in


    # Encoder
    c1, p1 = down(x, 64)
    c2, p2 = down(p1, 128)
    c3, p3 = down(p2, 256)
    c4, p4 = down(p3, 512)

    # Bottleneck
    bn = conv_block(p4, 1024)

    # Decoder
    u1 = up(bn, c4, 512)
    u2 = up(u1, c3, 256)
    u3 = up(u2, c2, 128)
    u4 = up(u3, c1, 64)

    # Output: predict correction (residual) or full image
    out_channels = 1
    y = layers.Conv2D(out_channels, 1, padding="same", activation="tanh")(u4)  # [-1, 1]

    if residual:
        # Inputs assumed in [0,1]; map tanh to small delta, e.g. scale 0.5
        delta = layers.Lambda(lambda z: 0.5 * z, name="delta_scale")(y)
        # Convert img to [-1,1] before adding, then back to [0,1]
        base = layers.Lambda(lambda z: (z * 2.0) - 1.0, name="to_minus1_1")(img_in if num_cond_classes is None else inputs[0])
        restored = layers.Add(name="add_residual")([base, delta])
        restored = layers.Lambda(lambda z: tf.clip_by_value((z + 1.0) / 2.0, 0.0, 1.0), name="to_0_1")(restored)
    else:
        # Direct prediction in [0,1]
        restored = layers.Lambda(lambda z: (z + 1.0) / 2.0, name="to_0_1")(y)

    return Model(inputs=inputs, outputs=restored, name="ElasticUNet")
