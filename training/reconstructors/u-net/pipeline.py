import os
import tensorflow as tf

IMG_SIZE = (256, 256)
BATCH = 4

def load_pair(fname, distorted_dir, clean_dir):
    distorted_path = tf.strings.join([distorted_dir, "/", fname])
    
    # Remove distortion prefix from filename for clean directory lookup
    # E.g., "stretch_101_1.png" -> "101_1.png"
    clean_fname = tf.strings.regex_replace(fname, r"^[a-zA-Z]+_", "")
    clean_path = tf.strings.join([clean_dir, "/", clean_fname])

    def _load(p):
        img = tf.io.read_file(p)
        img = tf.io.decode_png(img, channels=1)  # All files are now PNG
        img = tf.image.convert_image_dtype(img, tf.float32)  # [0,1]
        img = tf.image.resize(img, IMG_SIZE, method="bilinear")
        return img

    return _load(distorted_path), _load(clean_path)

def make_dataset(distorted_dir, clean_dir, cond_map=None, shuffle=True):
    # cond_map: optional dict filename -> onehot vector for conditional training
    fnames = tf.io.gfile.listdir(distorted_dir)
    fnames = sorted([f for f in fnames if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))])

    ds = tf.data.Dataset.from_tensor_slices(fnames)

    def _mapper(fname):
        dist_img, clean_img = load_pair(fname, distorted_dir, clean_dir)
        if cond_map is not None:
            cond = cond_map.get(fname.numpy().decode("utf-8"), None)  # py lookup
            if cond is None:
                cond = tf.zeros([len(next(iter(cond_map.values())))], dtype=tf.float32)
            cond = tf.convert_to_tensor(cond, dtype=tf.float32)
            return (dist_img, cond), clean_img
        else:
            return dist_img, clean_img

    if cond_map is None:
        ds = ds.map(lambda f: load_pair(f, distorted_dir, clean_dir), num_parallel_calls=tf.data.AUTOTUNE)
    else:
        # Use py_function to access python dict
        def py_map(f):
            (dist_img, clean_img) = load_pair(f, distorted_dir, clean_dir)
            cond = cond_map.get(f.numpy().decode("utf-8"), None)
            if cond is None:
                cond = [0.0] * len(next(iter(cond_map.values())))
            return dist_img, tf.convert_to_tensor(cond, tf.float32), clean_img
        
        def tf_py_map(f):
            dist_img, cond, clean_img = tf.py_function(py_map, [f], Tout=[tf.float32, tf.float32, tf.float32])
            # Fix shapes after py_function
            dist_img.set_shape((*IMG_SIZE, 1))
            cond.set_shape((len(next(iter(cond_map.values()))),))
            clean_img.set_shape((*IMG_SIZE, 1))
            return (dist_img, cond), clean_img
            
        ds = ds.map(tf_py_map, num_parallel_calls=tf.data.AUTOTUNE)

    if shuffle:
        ds = ds.shuffle(buffer_size=512, reshuffle_each_iteration=True)
    ds = ds.batch(BATCH).prefetch(tf.data.AUTOTUNE)
    return ds
