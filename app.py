import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# ------------------------------
# Load trained model
# ------------------------------
model = load_model("fruit_disease_model.h5")

# ------------------------------
# IMPORTANT:
# Class order MUST match training folder order (alphabetical)
# ------------------------------
class_names = [
    "apple_black_rot",
    "apple_blotch",
    "apple_healthy",
    "apple_scab",
    "mango_alternaria",
    "mango_anthracnose",
    "mango_black_mold",
    "mango_healthy",
    "mango_stem_rot"
]

# ------------------------------
# Streamlit UI
# ------------------------------
st.set_page_config(page_title="Fruit Disease Detection", layout="centered")
st.title("🍎🍋 Mango & Apple Fruit Disease Detection")

uploaded_file = st.file_uploader(
    "Upload a fruit image (JPG / PNG)",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    try:
        # Convert uploaded image to OpenCV format
        file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if image is None:
            st.error("Invalid image file")
        else:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Show uploaded image
            st.image(image_rgb, caption="Uploaded Image", use_column_width=True)

            # ------------------------------
            # Preprocessing (same as training)
            # ------------------------------
            img = cv2.resize(image_rgb, (224, 224))  # CHANGE only if your model uses different size
            img = img_to_array(img)
            img = np.expand_dims(img, axis=0)
            img = img / 255.0

            # ------------------------------
            # Prediction
            # ------------------------------
            predictions = model.predict(img)
            class_index = np.argmax(predictions)
            confidence = predictions[0][class_index] * 100

            st.success(f"Prediction: **{class_names[class_index]}**")
            st.info(f"Confidence: **{confidence:.2f}%**")

    except Exception as e:
        st.error("Something went wrong while processing the image.")
