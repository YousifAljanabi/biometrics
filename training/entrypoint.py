import tensorflow as tf
import os

img_size = (256, 288)
batch_size = 32


dataset_dir = os.path.join(os.path.dirname(__file__), "..", "dataset")

train_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_dir,
    image_size=img_size,
    batch_size=batch_size,
    label_mode="categorical"
)


normalization_layer = tf.keras.layers.Rescaling(1./255)
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
])


model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1. / 255, input_shape=(256, 288, 3)),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(len(train_ds.class_names), activation='softmax')
])


model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(train_ds, epochs=10)


img = tf.keras.utils.load_img("img.png", target_size=img_size)
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)

predictions = model.predict(img_array)
predicted_class = train_ds.class_names[tf.argmax(predictions[0])]
print("Predicted distortion:", predicted_class)

