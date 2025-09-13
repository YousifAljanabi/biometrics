# pipeline.py
import tensorflow as tf

IMG_SIZE = (256, 256)
BATCH = 4

def load_pair(fname, distorted_dir, clean_dir):
    distorted_path = tf.strings.join([distorted_dir, "/", fname])
    clean_fname = tf.strings.regex_replace(fname, r"^[a-zA-Z]+_", "")
    clean_path = tf.strings.join([clean_dir, "/", clean_fname])

    def _load(p):
        img = tf.io.read_file(p)
        img = tf.io.decode_png(img, channels=1)
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = tf.image.resize(img, IMG_SIZE, method="bilinear")
        return img

    return _load(distorted_path), _load(clean_path)

def _build_class_lookup(cond_map: dict[str, list[float]]):
    # Convert one-hot vectors to scalar class ids in Python
    # If your cond_map values are already ints, just use them directly.
    filenames = list(cond_map.keys())
    onehots = list(cond_map.values())
    class_ids = [int(tf.argmax(v).numpy() if hasattr(v, "numpy") else (v.index(1) if 1 in v else 0))
                 for v in onehots]
    keys = tf.constant(filenames, dtype=tf.string)
    vals = tf.constant(class_ids, dtype=tf.int32)
    init = tf.lookup.KeyValueTensorInitializer(keys, vals)
    default_val = tf.constant(0, dtype=tf.int32)  # scalar default
    return tf.lookup.StaticHashTable(init, default_val), max(class_ids) + 1  # num_classes

def make_dataset(distorted_dir, clean_dir, cond_map=None, shuffle=True, drop_remainder=True):
    fnames = tf.io.gfile.listdir(distorted_dir)
    fnames = sorted([f for f in fnames if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))])
    ds = tf.data.Dataset.from_tensor_slices(tf.constant(fnames))

    if cond_map is None:
        ds = ds.map(lambda f: load_pair(f, distorted_dir, clean_dir),
                    num_parallel_calls=tf.data.AUTOTUNE)
    else:
        table, num_classes = _build_class_lookup(cond_map)

        def mapper(f):
            dist_img, clean_img = load_pair(f, distorted_dir, clean_dir)
            cls_id = table.lookup(f)                                    # shape: ()
            cond = tf.one_hot(cls_id, depth=num_classes, dtype=tf.float32)  # shape: (C,)
            # Ensure static shapes for MPS
            dist_img = tf.ensure_shape(dist_img, (*IMG_SIZE, 1))
            clean_img = tf.ensure_shape(clean_img, (*IMG_SIZE, 1))
            cond = tf.ensure_shape(cond, (num_classes,))
            return (dist_img, cond), clean_img

        ds = ds.map(mapper, num_parallel_calls=tf.data.AUTOTUNE)

    if shuffle:
        ds = ds.shuffle(512, reshuffle_each_iteration=True)
    ds = ds.batch(BATCH, drop_remainder=drop_remainder).prefetch(tf.data.AUTOTUNE)
    return ds
