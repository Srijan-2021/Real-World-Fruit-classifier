import streamlit as st
import tensorflow as tf
import numpy as np
import json
import csv
import os
from PIL import Image
from datetime import datetime
from pathlib import Path

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="Fruit Classifier V4", page_icon="🍎", layout="centered")

st.title("🍎 Fruit Classifier (V4)")
st.markdown("""
Upload an image of a fruit, and the AI will classify it. 
*Supported classes: Apple, Banana, Mango, Orange, Strawberry.*
""")

# ---------- LOAD MODEL ----------
@st.cache_resource
def load_model():
    model_path = Path("artifacts/fruit_model_deploy.h5")
    labels_path = Path("artifacts/class_indices.json")
    
    model = tf.keras.models.load_model(model_path)
    
    with open(labels_path, "r") as f:
        class_indices = json.load(f)
    
    idx_to_class = {v: k for k, v in class_indices.items()}
    return model, idx_to_class

model, idx_to_class = load_model()

# ---------- PREPROCESSING ----------
def preprocess_image(image):
    img = image.convert("RGB").resize((224, 224))
    img_array = np.array(img, dtype=np.float32)
    
    # MobileNetV2 expects pixels in the range [-1, 1], not [0, 255]
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    
    img_array = np.expand_dims(img_array, axis=0)
    return img_array
# ---------- APP UI ----------
uploaded_file = st.file_uploader("Choose a fruit image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    with st.spinner("Analyzing..."):
        processed_img = preprocess_image(image)
        predictions = model.predict(processed_img, verbose=0)[0]
        
        top3_indices = np.argsort(predictions)[::-1][:3]
        top1_idx = top3_indices[0]
        top1_class = idx_to_class[top1_idx]
        top1_conf = float(predictions[top1_idx])
        
        # OOD Threshold
        if top1_conf < 0.50:
            st.warning(f"⚠️ Low confidence ({top1_conf:.1%}). This might not be one of the 5 supported fruits, or the image is unclear.")
        else:
            st.success(f"**Prediction: {top1_class}**")
            
        # Show Top 3 Bars
        st.markdown("### Top 3 Predictions")
        for idx in top3_indices:
            class_name = idx_to_class[idx]
            conf = float(predictions[idx])
            st.write(f"**{class_name}**: {conf:.1%}")
            st.progress(conf)

    # ---------- FEEDBACK LOOP ----------
    st.divider()
    st.subheader("Was this correct?")
    st.markdown("Your feedback helps improve future models.")
    
    with st.form("feedback_form", clear_on_submit=True):
        is_correct = st.radio("Was the prediction correct?", ["Yes", "No"], index=1)
        
        true_label = ""
        if is_correct == "No":
            true_label = st.selectbox(
                "What was the actual fruit?", 
                options=sorted(list(idx_to_class.values())),
                index=None
            )
            
        notes = st.text_input("Any comments? (optional)")
        submitted = st.form_submit_button("Submit Feedback")
        
        if submitted:
            feedback_file = "feedback.csv"
            file_exists = os.path.isfile(feedback_file)
            
            with open(feedback_file, mode="a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(["Timestamp", "Filename", "Predicted", "Confidence", "Is_Correct", "True_Label", "Notes"])
                    
                writer.writerow([
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    uploaded_file.name,
                    top1_class,
                    f"{top1_conf:.4f}",
                    is_correct,
                    true_label,
                    notes
                ])
                
            st.success("✅ Thank you! Feedback saved.")

st.divider()
st.caption("Built with TensorFlow & Streamlit | MobileNetV2 Transfer Learning V4")