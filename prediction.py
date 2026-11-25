import numpy as np
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras.applications.vgg16 import preprocess_input

from config import MODEL_PATH, IMG_SIZE, CLASS_NAMES


# ---------------- LOAD MODEL ----------------

def load_model():

    model = keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    return model


# ---------------- PREDICT FABRIC DEFECT ----------------

def predict_fabric_defect(image, model):

    # Convert image to RGB
    image = image.convert("RGB")

    # Convert image into NumPy array
    image = np.array(
        image,
        dtype=np.float32
    )

    # Resize while maintaining aspect ratio
    image = tf.image.resize_with_pad(
        image,
        IMG_SIZE,
        IMG_SIZE
    )

    # Apply VGG16 preprocessing
    image = preprocess_input(image)

    # Add batch dimension
    image = tf.expand_dims(
        image,
        axis=0
    )

    # Model prediction
    probabilities = model.predict(
        image,
        verbose=0
    )[0]

    # Validate prediction output
    if len(probabilities) != len(CLASS_NAMES):
        raise ValueError(
            "Model output does not match the configured classes."
        )

    # Predicted class
    predicted_index = int(
        np.argmax(probabilities)
    )

    predicted_class = CLASS_NAMES[predicted_index]

    # Confidence score
    confidence = float(
        probabilities[predicted_index]
    ) * 100

    return predicted_class, confidence, probabilities