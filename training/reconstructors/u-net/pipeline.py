# pipeline.py
import os
import tensorflow as tf

IMG_SIZE = (256, 256)
BATCH = 4

def load_pair(fname, distorted_dir, clean_dir):
    distorted_path = tf.strings.join([distorted_dir, "/", fname])

    # "stretch_101_1.png" -> "101_1.png"
    clean_fname = tf.strings.regex_replace(fname, r"^[a-zA-Z]+_", "")
    clean_path = tf.strings.join([clean_dir, "/", clean_fname])

    def _load(p):
        img = tf.io.read_file(p)
        img = tf.io.decode_png(img, channels=1)
        img = tf.image.convert_image_dtype(img, tf.float32)  # [0,1]
        img = tf.image.resize(img, IMG_SIZE, method="bilinear")
        return img

    return _load(distorted_path), _load(clean_path)

def _build_lookup(cond_map: dict[str, list[float]]):
    keys = tf.constant(list(cond_map.keys()), dtype=tf.string)
    vals = tf.constant(list(cond_map.values()), dtype=tf.float32)
    init = tf.lookup.KeyValueTensorInitializer(keys, vals)
    default_val = tf.zeros_like(vals[0])
    return tf.lookup.StaticHashTable(init, default_val)

def make_dataset(distorted_dir, clean_dir, cond_map=None, shuffle=True, drop_remainder=True):
    # Gather file names deterministically
    fnames = tf.io.gfile.listdir(distorted_dir)
    fnames = sorted([f for f in fnames if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))])
    ds = tf.data.Dataset.from_tensor_slices(tf.constant(fnames))

    if cond_map is None:
        ds = ds.map(lambda f: load_pair(f, distorted_dir, clean_dir),
                    num_parallel_calls=tf.data.AUTOTUNE)
    else:
        # Build an in-graph lookup table: filename -> onehot vector
        table = _build_lookup(cond_map)
        def mapper(f):
            dist_img, clean_img = load_pair(f, distorted_dir, clean_dir)
            cond = table.lookup(f)  # shape [C]
            # Ensure static shapes for Metal
            dist_img = tf.ensure_shape(dist_img, (*IMG_SIZE, 1))
            clean_img = tf.ensure_shape(clean_img, (*IMG_SIZE, 1))
            cond = tf.ensure_shape(cond, (len(next(iter(cond_map.values()))),))
            return (dist_img, cond), clean_img

        ds = ds.map(mapper, num_parallel_calls=tf.data.AUTOTUNE)

    if shuffle:
        ds = ds.shuffle(512, reshuffle_each_iteration=True)

    # Metal is happiest with fixed batch shapes
    ds = ds.batch(BATCH, drop_remainder=drop_remainder).prefetch(tf.data.AUTOTUNE)
    return ds
