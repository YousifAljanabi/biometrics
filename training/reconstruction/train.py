from model import build_unet
from pipeline import make_dataset
import tensorflow as tf
from utils import build_cond_map
import os


def ssim_l1_loss(y_true, y_pred, alpha=0.84):
    ssim = tf.image.ssim(y_true, y_pred, max_val=1.0)
    ssim_loss = (1.0 - ssim) / 2.0
    l1 = tf.reduce_mean(tf.abs(y_true - y_pred), axis=[1,2,3])
    return alpha * ssim_loss + (1.0 - alpha) * l1

NUM_CLASSES = 14

# Define dataset paths using absolute paths relative to script location
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
distorted_dir = os.path.join(project_root, "dataset", "distorted")
clean_dir = os.path.join(project_root, "dataset", "clean")

cond_map = build_cond_map(distorted_dir)

c_model = build_unet(input_shape=(256,256,1), num_cond_classes=NUM_CLASSES, residual=True, use_stn=True)
c_model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                loss=ssim_l1_loss,
                metrics=[tf.keras.metrics.MeanAbsoluteError(name="L1")])

callbacks = [
    tf.keras.callbacks.ModelCheckpoint("elastic_unet_best.h5", monitor="val_L1",
                                       save_best_only=True, mode="min"),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_L1", factor=0.5, patience=5, verbose=1),
    tf.keras.callbacks.EarlyStopping(monitor="val_L1", patience=12, restore_best_weights=True)
]


train_ds = make_dataset(distorted_dir, clean_dir, cond_map=cond_map, shuffle=True)
val_ds   = make_dataset(distorted_dir, clean_dir, cond_map=cond_map, shuffle=False)

c_model.fit(train_ds, epochs=50, callbacks=callbacks)  # Reduced epochs for faster training
c_model.save("elastic_unet_conditional.h5")