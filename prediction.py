
import numpy as np
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras.applications.vgg16 import preprocess_input
from PIL import Image, ImageDraw

from config import MODEL_PATH, IMG_SIZE, CLASS_NAMES


# ---------------- LOAD MODEL ----------------

def load_model():

    model = keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    return model


# ---------------- PREPROCESS IMAGE ----------------

def preprocess_image(image):

    image = image.convert("RGB")

    image_array = np.array(
        image,
        dtype=np.float32
    )

    image_array = tf.image.resize_with_pad(
        image_array,
        IMG_SIZE,
        IMG_SIZE
    )

    image_array = preprocess_input(image_array)

    image_array = tf.expand_dims(
        image_array,
        axis=0
    )

    return image_array


# ---------------- PREDICT FABRIC DEFECT ----------------

def predict_fabric_defect(image, model):

    processed_image = preprocess_image(image)

    probabilities = model.predict(
        processed_image,
        verbose=0
    )[0]

    if len(probabilities) != len(CLASS_NAMES):
        raise ValueError(
            "Model output does not match the configured classes."
        )

    predicted_index = int(
        np.argmax(probabilities)
    )

    predicted_class = CLASS_NAMES[predicted_index]

    confidence = float(
        probabilities[predicted_index]
    ) * 100

    return predicted_class, confidence, probabilities


# ---------------- GRAD-CAM HEATMAP ----------------

def generate_gradcam(image, model):

    processed_image = preprocess_image(image)

    # Get the nested VGG16 feature extractor
    base_model = model.get_layer("vgg16")

    # Forward pass and capture feature maps
    with tf.GradientTape() as tape:

        feature_maps = base_model(
            processed_image,
            training=False
        )

        tape.watch(feature_maps)

        # Pass feature maps through the classifier
        x = model.get_layer(
            "global_average_pooling2d_1"
        )(feature_maps)

        x = model.get_layer("dense_2")(x)

        x = model.get_layer("dropout_1")(
            x,
            training=False
        )

        predictions = model.get_layer("dense_3")(x)

        predicted_index = tf.argmax(
            predictions[0]
        )

        class_score = predictions[:, predicted_index]

    # Calculate gradients
    gradients = tape.gradient(
        class_score,
        feature_maps
    )

    # Calculate channel importance
    pooled_gradients = tf.reduce_mean(
        gradients,
        axis=(0, 1, 2)
    )

    # Generate heatmap
    feature_maps = feature_maps[0]

    heatmap = tf.reduce_sum(
        feature_maps * pooled_gradients,
        axis=-1
    )

    heatmap = tf.maximum(
        heatmap,
        0
    )

    heatmap = heatmap / (
        tf.reduce_max(heatmap) + keras.backend.epsilon()
    )

    return heatmap.numpy(), int(predicted_index.numpy())


# ---------------- DRAW APPROXIMATE DEFECT BOX ----------------

def predict_with_gradcam(image, model):

    predicted_class, confidence, probabilities = (
        predict_fabric_defect(image, model)
    )

    heatmap, _ = generate_gradcam(
        image,
        model
    )

    original_image = image.convert("RGB")

    width, height = original_image.size

    # Resize heatmap to original image dimensions
    heatmap_image = Image.fromarray(
        np.uint8(heatmap * 255)
    )

    heatmap_image = heatmap_image.resize(
        (width, height),
        Image.Resampling.BILINEAR
    )

    heatmap_array = np.array(heatmap_image)

    # Create colored heatmap
    colored_heatmap = np.zeros(
        (height, width, 3),
        dtype=np.uint8
    )

    colored_heatmap[:, :, 0] = heatmap_array
    colored_heatmap[:, :, 1] = (
        heatmap_array * 0.3
    ).astype(np.uint8)

    heatmap_rgb = Image.fromarray(
        colored_heatmap
    )

    # Blend original image and heatmap
    result_image = Image.blend(
        original_image,
        heatmap_rgb,
        0.4
    )

    # Draw approximate bounding rectangle
    if predicted_class.lower() != "defect free":

        threshold = max(
            0.5,
            float(heatmap_array.max()) * 0.6
        )

        y_coordinates, x_coordinates = np.where(
            heatmap_array >= threshold
        )

        if len(x_coordinates) > 0:

            x1 = int(x_coordinates.min())
            y1 = int(y_coordinates.min())

            x2 = int(x_coordinates.max())
            y2 = int(y_coordinates.max())

            draw = ImageDraw.Draw(result_image)

            draw.rectangle(
                (x1, y1, x2, y2),
                outline="red",
                width=4
            )

            draw.text(
                (x1, max(0, y1 - 20)),
                f"{predicted_class} ({confidence:.1f}%)",
                fill="red"
            )

    return (
        predicted_class,
        confidence,
        probabilities,
        result_image,
        heatmap_image
    )
