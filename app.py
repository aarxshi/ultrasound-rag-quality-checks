import os

# Environment configuration
os.environ["GRADIO_WATCH_FILES"] = "false"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

from functools import lru_cache

import numpy as np
import tensorflow as tf
from PIL import Image
import gradio as gr

from rag.rag_query import query_rag, summarize_guidelines

IMG_SIZE = (128, 128)


def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(shape=(IMG_SIZE[0], IMG_SIZE[1], 1)),

        tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2),

        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2),

        tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])
    return model


# Build and load model
model = build_model()
model.load_weights("scan_quality_weights.weights.h5")

# Warm up the model
_ = model.predict(np.zeros((1, IMG_SIZE[0], IMG_SIZE[1], 1), dtype=np.float32), verbose=0)


@lru_cache(maxsize=1)
def get_low_quality_explanation():
    results = query_rag("Why is this ultrasound scan low quality?")
    explanation = summarize_guidelines(results)
    return explanation


def predict(image):
    if image is None:
        return {
            "Result": "Error",
            "Confidence": "N/A",
            "Recommendation": "No image provided. Please upload an ultrasound scan.",
            "Explanation": "The system did not receive a valid image input."
        }

    if not isinstance(image, Image.Image):
        try:
            image = Image.fromarray(image)
        except Exception:
            return {
                "Result": "Error",
                "Confidence": "N/A",
                "Recommendation": "Invalid image format.",
                "Explanation": "The uploaded file could not be processed as an image."
            }

    image = image.convert("L")
    image = image.resize(IMG_SIZE)

    image_array = np.array(image, dtype=np.float32) / 255.0
    if image_array.ndim != 2:
        return {
            "Result": "Error",
            "Confidence": "N/A",
            "Recommendation": "Unexpected image shape after preprocessing.",
            "Explanation": f"Expected 2D grayscale image, got shape {image_array.shape}."
        }

    image_array = np.expand_dims(image_array, axis=-1)
    image_array = np.expand_dims(image_array, axis=0)

    prob_unclear = float(model.predict(image_array, verbose=0)[0][0])

    if prob_unclear >= 0.65:
        result = "Unclear scan"
        confidence = "High"
        guidance = "Retake the scan."
        explanation = get_low_quality_explanation()
    elif prob_unclear <= 0.35:
        result = "Clear scan"
        confidence = "High"
        guidance = "Scan quality acceptable. Proceed with interpretation."
        explanation = ""
    else:
        result = "Uncertain scan quality"
        confidence = "Moderate"
        guidance = "Consider retaking the scan."
        explanation = get_low_quality_explanation()

    return {
        "Result": result,
        "Confidence": confidence,
        "Recommendation": guidance,
        "Explanation": explanation
    }


demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Ultrasound image"),
    outputs=gr.JSON(label="Scan quality assessment"),
    title="Ultrasound Scan Quality MVP",
    description="Flags unclear ultrasound scans at capture time."
)

try:
    import gradio.utils as gr_utils

    def _no_watchfn_spaces(*args, **kwargs):
        return

    if hasattr(gr_utils, "watchfn_spaces"):
        gr_utils.watchfn_spaces = _no_watchfn_spaces
except Exception:
    pass

demo.launch(
    server_name="0.0.0.0",
    server_port=7860,
    ssr_mode=False
)
