"""
🗑️ Waste Classification App
============================

A Streamlit application for classifying waste as Organic or Recyclable
using YOLOv11 object detection.

Run with: streamlit run app.py
"""

import streamlit as st
from pathlib import Path
import numpy as np
from PIL import Image
import tempfile
import os

# Set page config
st.set_page_config(
    page_title="🗑️ Waste Classifier",
    page_icon="🗑️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models"


@st.cache_resource
def load_model(model_path):
    """Load YOLO model (cached)."""
    from ultralytics import YOLO
    return YOLO(str(model_path))


def find_best_model():
    """Find the best trained model."""
    # Look for best.pt in model directories
    for model_dir in sorted(MODELS_DIR.glob("waste_*"), reverse=True):
        best_pt = model_dir / "weights" / "best.pt"
        if best_pt.exists():
            return best_pt
    
    # Fallback to pre-trained
    return "yolo11n.pt"


def classify_image(model, image, conf_threshold=0.25):
    """Run inference on image."""
    results = model.predict(
        source=image,
        conf=conf_threshold,
        save=False,
        verbose=False
    )
    return results[0]


def main():
    # Header
    st.title("🗑️ Waste Classification System")
    st.markdown("""
    Upload an image or use your webcam to classify waste as **Organic** or **Recyclable**.
    
    ---
    """)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Model selection
        model_path = find_best_model()
        st.info(f"📦 Model: `{Path(model_path).name if isinstance(model_path, Path) else model_path}`")
        
        # Confidence threshold
        conf_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.1,
            max_value=1.0,
            value=0.25,
            step=0.05
        )
        
        st.markdown("---")
        
        # Class info
        st.header("📋 Classes")
        st.markdown("""
        - 🥬 **Organic**: Food waste, plants, paper
        - ♻️ **Recyclable**: Plastic, metal, glass
        """)
        
        st.markdown("---")
        st.markdown("Made with ❤️ using YOLOv11")
    
    # Load model
    with st.spinner("Loading model..."):
        model = load_model(model_path)
    
    # Main content
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("📤 Input")
        
        # Input method selection
        input_method = st.radio(
            "Choose input method:",
            ["Upload Image", "Camera"],
            horizontal=True
        )
        
        image = None
        
        if input_method == "Upload Image":
            uploaded_file = st.file_uploader(
                "Upload an image",
                type=["jpg", "jpeg", "png", "bmp"],
                help="Upload a waste image for classification"
            )
            
            if uploaded_file:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_column_width=True)
        
        else:  # Camera
            camera_image = st.camera_input("Take a photo")
            
            if camera_image:
                image = Image.open(camera_image)
    
    with col2:
        st.header("🎯 Results")
        
        if image is not None:
            # Run inference
            with st.spinner("Classifying..."):
                results = classify_image(model, image, conf_threshold)
            
            # Display result
            result_img = results.plot()
            st.image(result_img, caption="Detection Result", use_column_width=True)
            
            # Show detections
            boxes = results.boxes
            if len(boxes) > 0:
                st.success(f"✅ Found {len(boxes)} object(s)")
                
                # Detection details
                for i, box in enumerate(boxes):
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    cls_name = results.names[cls_id]
                    
                    # Color based on class
                    if "organic" in cls_name.lower():
                        emoji = "🥬"
                        color = "green"
                    else:
                        emoji = "♻️"
                        color = "blue"
                    
                    st.markdown(f"""
                    <div style="padding: 10px; background-color: {'#d4edda' if color=='green' else '#cce5ff'}; 
                                border-radius: 5px; margin: 5px 0;">
                        <h3>{emoji} {cls_name}</h3>
                        <p>Confidence: <strong>{conf:.1%}</strong></p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.warning("⚠️ No objects detected. Try adjusting the confidence threshold.")
        else:
            st.info("👈 Upload an image or take a photo to get started!")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: gray;">
        <p>🧠 Powered by YOLOv11 | Built for Waste Classification Project</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
