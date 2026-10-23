
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Retail Product Detection & Counting",
    page_icon="🛒",
    layout="wide"
)

# Application title
st.title("🛒 Retail Product Detection & Counting")
st.write(
    "Upload a retail shelf image to detect products "
    "and calculate the total detected count."
)

# Load trained YOLO model
@st.cache_resource
def load_model():
    return YOLO("models/retail_product_detection_best.pt")


try:
    model = load_model()
except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.info(
        "Check whether the model file exists inside the models folder."
    )
    st.stop()

# Image upload
uploaded_file = st.file_uploader(
    "Upload a shelf image",
    type=["jpg", "jpeg", "png"]
)

# Confidence threshold
confidence = st.slider(
    "Detection confidence threshold",
    min_value=0.10,
    max_value=0.90,
    value=0.45,
    step=0.05
)

# Process uploaded image
if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    image_array = np.array(image)

    with st.spinner("Detecting products..."):

        results = model.predict(
            source=image_array,
            conf=confidence,
            max_det=1000,
            verbose=False
        )

    result = results[0]

    # Draw bounding boxes
    annotated_image = result.plot()
    annotated_image = cv2.cvtColor(
        annotated_image,
        cv2.COLOR_BGR2RGB
    )

    # Calculate detection count and confidence
    product_count = len(result.boxes)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Shelf Image")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("Detected Products")
        st.image(annotated_image, use_container_width=True)

    st.divider()

    # Display total detections
    st.subheader("📊 Detection Summary")
    st.metric(
        label="Total Detected Products",
        value=product_count
    )

    # Confidence statistics
    if product_count > 0:

        scores = result.boxes.conf.cpu().numpy()

        metric1, metric2, metric3 = st.columns(3)

        metric1.metric(
            "Average Confidence",
            f"{float(scores.mean()):.3f}"
        )

        metric2.metric(
            "Minimum Confidence",
            f"{float(scores.min()):.3f}"
        )

        metric3.metric(
            "Maximum Confidence",
            f"{float(scores.max()):.3f}"
        )

        with st.expander("View individual confidence scores"):
            st.write(scores.tolist())

        st.success("Product detection completed successfully!")

    else:
        st.warning(
            "No products detected. Try lowering the confidence threshold."
        )

    st.caption(
        "Note: The model detects products as a single class. "
        "The detection count may differ from the actual number of products."
    )

else:
    st.info("Please upload a shelf image to start detection.")
