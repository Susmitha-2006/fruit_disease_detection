import streamlit as st
import numpy as np

# Use headless OpenCV
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# --- Load the trained model ---
# Make sure 'fruit_disease_model.h5' is uploaded in your repo
model = load_model('fruit_disease_model.h5')

# --- Define class names ---
class_names = [
    'apple_healthy', 'apple_black_rot', 'apple_blotch', 'apple_scab',
    'mango_healthy', 'mango_anthracnose', 'mango_alternaria',
    'mango_black_mold', 'mango_stem_rot'
]

# --- Streamlit UI ---
st.title("🍎 Mango & Apple Fruit Disease Detection")

uploaded_file = st.file_uploader("Upload an image of the fruit", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert uploaded file to OpenCV image
    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    st.image(image_rgb, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image for the model
    img_resized = cv2.resize(image_rgb, (224, 224))  # change if your model expects another size
    img_array = img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # normalize to 0-1

    # Make prediction
    predictions = model.predict(img_array)
    class_idx = np.argmax(predictions[0])
    confidence = predictions[0][class_idx]

    st.write(f"**Prediction:** {class_names[class_idx]}")
    st.write(f"**Confidence:** {confidence*100:.2f}%")
