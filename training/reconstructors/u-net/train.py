from model import build_unet
from pipeline import make_dataset

import os
os.environ["XLA_FLAGS"] = "--xla_gpu_strict_conv_algorithm_picker=false"

import tensorflow as tf
from utils import build_cond_map

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

def ssim_l1_loss(y_true, y_pred, alpha=0.84):
    # SSIM returns [-1,1]; convert to loss in [0,1]
    ssim = tf.image.ssim(y_true, y_pred, max_val=1.0)
    ssim_loss = (1.0 - ssim) / 2.0
    l1 = tf.reduce_mean(tf.abs(y_true - y_pred), axis=[1,2,3])
    return alpha * ssim_loss + (1.0 - alpha) * l1

# Suppose you have 5 elastic classes:
# order = ["stretch", "compression", "sliding", "nonrigid", "curl"]
NUM_CLASSES = 5

# Build mapping: filename -> onehot
# Example: cond_map["0001.png"] = [1,0,0,0,0]  # stretch
cond_map = build_cond_map("data/distorted")

c_model = build_unet(input_shape=(256,256,1), num_cond_classes=NUM_CLASSES, residual=True)
c_model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                loss=ssim_l1_loss,
                metrics=[tf.keras.metrics.MeanAbsoluteError(name="L1")])

callbacks = [
    tf.keras.callbacks.ModelCheckpoint("elastic_unet_best.h5", monitor="val_L1",
                                       save_best_only=True, mode="min"),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_L1", factor=0.5, patience=5, verbose=1),
    tf.keras.callbacks.EarlyStopping(monitor="val_L1", patience=12, restore_best_weights=True)
]


train_ds = make_dataset("data/distorted", "data/clean", cond_map=cond_map, shuffle=True)
# val_ds   = make_dataset("data/distorted_val", "data/clean_val", cond_map=cond_map, shuffle=False)

c_model.fit(train_ds, epochs=100, callbacks=callbacks)
c_model.save("elastic_unet_conditional.h5")