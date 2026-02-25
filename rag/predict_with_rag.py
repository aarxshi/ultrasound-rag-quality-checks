import numpy as np
from PIL import Image

IMG_SIZE = (128, 128)

# Load trained model
import os

ROOT = os.path.dirname(os.path.abspath(__file__))

def find_model(root):
    candidates = [
        os.path.join(root, "..", "ultrasound_clear_unclear_mvp.keras"),
        os.path.join(root, "..", "ultrasound_clear_unclear_mvp_savedmodel"),
        os.path.join(root, "..", "ultrasound_clear_unclear_mvp.h5"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return os.path.abspath(p)
    # recursive search from root
    for dirpath, dirnames, filenames in os.walk(root):
        if "ultrasound_clear_unclear_mvp.keras" in filenames:
            return os.path.join(dirpath, "ultrasound_clear_unclear_mvp.keras")
        if "ultrasound_clear_unclear_mvp.h5" in filenames:
            return os.path.join(dirpath, "ultrasound_clear_unclear_mvp.h5")
        if "ultrasound_clear_unclear_mvp_savedmodel" in dirnames:
            return os.path.join(dirpath, "ultrasound_clear_unclear_mvp_savedmodel")
    raise FileNotFoundError("Could not find model file. Put ultrasound_clear_unclear_mvp.keras or .h5 or _savedmodel in the repo.")

def _load_model_compat(path):
    try:
        import tensorflow as tf
        from tensorflow.keras.layers import Dense as KerasDense
    except Exception as e:
        raise RuntimeError("TensorFlow is required to load the model: " + str(e)) from e

    def Dense(*args, **kwargs):
        kwargs.pop("quantization_config", None)
        return KerasDense(*args, **kwargs)

    if path.endswith(".h5"):
        with tf.keras.utils.custom_object_scope({"Dense": Dense}):
            return tf.keras.models.load_model(path)
    else:
        return tf.keras.models.load_model(path)

MODEL_PATH = None
model = None

def load_model_if_needed():
    """Load the model on first use. Raises informative errors if TF/model missing."""
    global MODEL_PATH, model
    if model is not None:
        return
    # determine model path lazily
    import os
    ROOT = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = find_model(ROOT)
    model = _load_model_compat(MODEL_PATH)


def predict_image(img_path):
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image not found: {img_path}")
    try:
        load_model_if_needed()
    except FileNotFoundError as e:
        raise FileNotFoundError("Model not found. Place the model in the repo or set MODEL_PATH.") from e
    except RuntimeError as e:
        raise RuntimeError(str(e)) from e

    img = Image.open(img_path).convert("L").resize(IMG_SIZE)
    x = np.asarray(img, dtype=np.float32)
    x = x[..., np.newaxis]
    x = x / 255.0
    x = np.expand_dims(x, axis=0)

    prob_unclear = model.predict(x)[0][0]

    if prob_unclear > 0.65:
        result = "Unclear scan"
        confidence = "High"
        guidance = "Retake the scan. Adjust probe angle or gain."
    elif prob_unclear < 0.35:
        result = "Clear scan"
        confidence = "High"
        guidance = "Scan quality acceptable. Proceed with interpretation."
    else:
        result = "Uncertain scan quality"
        confidence = "Moderate"
        guidance = "Consider retaking the scan for confirmation."

    explanations = []
    if result != "Clear scan":
        try:
            from rag.rag_query import query_rag, summarize_guidelines
            rag_results = query_rag("Why is this ultrasound scan low quality?")
            explanations = summarize_guidelines(rag_results)
        except Exception:
            explanations = ["Guidance unavailable (RAG module error)"]

    return {
        "Result": result,
        "Confidence": confidence,
        "Recommendation": guidance,
        "Explanation": explanations
    }

if __name__ == "__main__":
    img_path = input("Path to ultrasound image: ")
    output = predict_image(img_path)

    print("\n=== AI Output ===")
    for k, v in output.items():
        print(f"{k}: {v}")
