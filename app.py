import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load your trained model
model = load_model('fruit_disease_model.h5')  # <-- your model file

# Define class names (update if you have different classes)
class_names = [
    'apple_healthy', 'apple_black_rot', 'apple_blotch', 'apple_scab',
    'mango_healthy', 'mango_anthracnose', 'mango_alternaria',
    'mango_black_mold', 'mango_stem_rot'
]

st.title("🍎 Mango & Apple Fruit Disease Detection")

# Upload an image
uploaded_file = st.file_uploader("Upload an image of the fruit", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert uploaded file to a NumPy array
    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    st.image(image_rgb, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image for your model
    img_resized = cv2.resize(image_rgb, (224, 224))  # adjust size to your model input
    img_array = img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # normalize to 0-1

    # Make prediction
    predictions = model.predict(img_array)
    class_idx = np.argmax(predictions[0])
    confidence = predictions[0][class_idx]

    st.write(f"**Prediction:** {class_names[class_idx]}")
    st.write(f"**Confidence:** {confidence*100:.2f}%")
