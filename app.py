
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import time

st.title("YOLO Image Detection App :)")

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(
    page_title="🧠 YOLO Image Detection App",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------
# Header Section
# ---------------------------
st.title("🤖 YOLO Image Detection App")
st.markdown("""
<style>
    .stApp {
        background-color: #f9fafc;
    }
    .uploadedImage, .resultImage {
        border-radius: 15px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
    }
    .stProgress > div > div > div {
        background-color: #4CAF50;
    }
</style>
""", unsafe_allow_html=True)

st.write("อัปโหลดภาพของคุณเพื่อให้โมเดล **YOLO** ตรวจจับวัตถุในภาพได้แบบเรียลไทม์ 🚀")

# ---------------------------
# Load Model
# ---------------------------
@st.cache_resource
def load_model():
    
    model_path = "best.pt"
    model = YOLO(model_path)
    return model

model = load_model()

# ---------------------------
# Upload Image
# ---------------------------
uploaded_image = st.file_uploader("📤 Upload an image (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    col1, col2 = st.columns(2)

    # Show uploaded image
    with col1:
        st.subheader("📸 Uploaded Image")
        st.image(uploaded_image, caption="Original Image", use_container_width=True, output_format="auto")

    # Convert image to numpy
    image = Image.open(uploaded_image)
    image_np = np.array(image)

    # Run inference with progress bar
    with st.spinner("Running YOLO object detection... 🧩"):
        progress = st.progress(0)
        for i in range(100):
            time.sleep(0.01)
            progress.progress(i + 1)
        results = model.predict(image_np, conf=0.4)

    # Show results
    result_image = results[0].plot()

    with col2:
        st.subheader("🎯 Detection Result")
        st.image(result_image, caption="Detected Objects", use_container_width=True, output_format="auto")
        st.success("✅ Detection completed!")

    # Display detected classes
    boxes = results[0].boxes
    class_ids = boxes.cls.cpu().numpy().astype(int)
    class_names = [model.names[i] for i in class_ids]

    if len(class_names) > 0:
        st.markdown("### 🧩 Objects Detected:")
        st.table({"Detected Class": class_names})
    else:
        st.warning("No objects detected in this image 🫥")
else:
    st.info("📎 กรุณาอัปโหลดภาพเพื่อเริ่มตรวจจับวัตถุ")
