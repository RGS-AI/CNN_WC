import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import time  # For animation effect
import gdown

# Set page configuration (This must be the first Streamlit command)
st.set_page_config(page_title="Waste Classifier", layout="wide")

model_url = "https://drive.google.com/uc?id=1MEzf68u2JjO4atq2q4EWJNZ22Wf-657u"

# Load the trained CNN model
@st.cache_resource
def load_cnn_model():
    gdown.download(model_url, "wc_cnn_model.h5", quiet=False)
    return load_model("wc_cnn_model.h5")

model = load_cnn_model()

# Define class labels
class_labels = {0: "Organic", 1: "Recyclable"}  # Update as per your dataset class mapping

# Streamlit App Layout
# st.set_page_config(page_title="Waste Classifier", layout="wide")

# Sidebar - About Section
with st.sidebar:
    st.title("ℹ️ About")
    st.write("""
    **Created as a part of AICTE Internship Training (Cycle 3).**
    
    - This project uses a **CNN model** to classify waste into **Organic or Recyclable**.
    - Dataset Provider: **Techsash (Kaggle)**
    - Model Algorithm & Description: **Skills4future**
    - Future Improvements: Classification based on **non-recyclable plastic, recyclable plastic and more**
    
    **Thanks for visiting!**
    """)

# Main Title
st.title("♻️ Waste Classification using CNN")

# Upload Image Section
uploaded_file = st.file_uploader("📤 Upload an image for classification", type=["jpg", "png", "jpeg"])

# Two-Column Layout
col1, col2 = st.columns([1, 1])  # Two equal-width columns

# If an image is uploaded
if uploaded_file is not None:
    # Display image in Column 1
    with col1:
        st.image(uploaded_file, caption="🖼️ Uploaded Image", use_column_width=True)

    # Process and predict in Column 2
    with col2:
        st.write("⏳ **Processing Image...**")
        time.sleep(1)  # Small delay for animation effect

        # Preprocess the image
        img = image.load_img(uploaded_file, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        img_array = img_array / 255.0  # Normalize if trained with normalization

        # Make a prediction
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)[0]  # Get highest probability class
        predicted_label = class_labels[predicted_class]

        # Show animated success message
        st.success(f"✅ **Prediction: {predicted_label} Waste**")

        # Animated Icon
        if predicted_label == "Organic":
            st.markdown("🌱 **This waste is Organic!**")
        else:
            st.markdown("🔄 **This waste is Recyclable!**")

        # Optional: Show Confidence Scores
        st.write("🧠 **Confidence Scores:**")
        for i, label in class_labels.items():
            st.write(f"- {label}: **{predictions[0][i] * 100:.2f}%**")

# Bottom Credits
st.markdown("---")
st.markdown(" **Trainer Credits: RMS** ")
st.markdown(" **Developed with ❤️ for AICTE Internship Cycle 3**")