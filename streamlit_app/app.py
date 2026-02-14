"""
🗑️ Waste Classification App - Multi-Model Version
=====================================================

Supports YOLOv5, YOLOv8, and YOLOv11 models.
Includes local SQLite storage for prediction history.

Run with: streamlit run app.py
"""

import streamlit as st
from pathlib import Path
import numpy as np
from PIL import Image
import sqlite3
import json
import io
from datetime import datetime
import traceback

# ─── Page Config ──────────────────────────────────────────────────
st.set_page_config(
    page_title="🗑️ Waste Classifier",
    page_icon="🗑️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Constants ────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
HISTORY_DIR = Path(__file__).parent / "history"
HISTORY_DIR.mkdir(exist_ok=True)
DB_PATH = HISTORY_DIR / "history.db"

MODEL_VERSIONS = {
    "YOLOv11": {
        "prefix": "waste_yolo11",
        "fallback": "yolo11n.pt",
        "color": "#9C27B0",
        "emoji": "🟣",
    },
    "YOLOv8": {
        "prefix": "waste_yolov8",
        "fallback": "yolov8n.pt",
        "color": "#2196F3",
        "emoji": "🔵",
    },
    "YOLOv5": {
        "prefix": "waste_yolov5",
        "fallback": "yolov5nu.pt",
        "color": "#4CAF50",
        "emoji": "🟢",
    },
}

# Waste class names (our custom model uses these)
WASTE_CLASSES = {0: "Organic", 1: "Recyclable"}


# ─── Database ─────────────────────────────────────────────────────
def init_db():
    """Initialize SQLite database for prediction history."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            model_version TEXT NOT NULL,
            model_type TEXT DEFAULT 'pretrained',
            image_name TEXT,
            image_thumbnail BLOB,
            num_detections INTEGER,
            detections_json TEXT,
            confidence_threshold REAL
        )
    """)
    conn.commit()
    conn.close()


def save_prediction(
    model_version, model_type, image_name, image, detections, conf_threshold
):
    """Save a prediction to the database."""
    try:
        thumb = image.copy()
        thumb.thumbnail((200, 200))
        buf = io.BytesIO()
        thumb.save(buf, format="PNG")
        thumb_bytes = buf.getvalue()

        conn = sqlite3.connect(str(DB_PATH))
        conn.execute(
            """INSERT INTO predictions
               (timestamp, model_version, model_type, image_name, image_thumbnail,
                num_detections, detections_json, confidence_threshold)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                datetime.now().isoformat(),
                model_version,
                model_type,
                image_name or "camera_capture",
                thumb_bytes,
                len(detections),
                json.dumps(detections),
                conf_threshold,
            ),
        )
        conn.commit()
        conn.close()
    except Exception as e:
        st.toast(f"⚠️ History save failed: {e}", icon="⚠️")


def load_history(limit=50):
    """Load recent prediction history."""
    conn = sqlite3.connect(str(DB_PATH))
    try:
        rows = conn.execute(
            """SELECT id, timestamp, model_version, image_name, image_thumbnail,
                      num_detections, detections_json, confidence_threshold
               FROM predictions ORDER BY id DESC LIMIT ?""",
            (limit,),
        ).fetchall()
    except sqlite3.OperationalError:
        # Handle schema mismatch from earlier DB version
        rows = []
    conn.close()
    return rows


def clear_history():
    """Delete all prediction history."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("DELETE FROM predictions")
    conn.commit()
    conn.close()


# ─── Model Helpers ────────────────────────────────────────────────
@st.cache_resource
def load_yolo_model(model_path_str):
    """Load a YOLO model (cached per path string)."""
    from ultralytics import YOLO

    return YOLO(model_path_str)


def find_model_for_version(version):
    """
    Find best trained model for a given YOLO version.
    Returns (model_path_str, is_custom_trained).
    """
    info = MODEL_VERSIONS[version]
    if MODELS_DIR.exists():
        for model_dir in sorted(MODELS_DIR.glob(f"{info['prefix']}*"), reverse=True):
            best_pt = model_dir / "weights" / "best.pt"
            if best_pt.exists():
                return str(best_pt), True
    return info["fallback"], False


def run_inference(model, image, conf_threshold=0.25):
    """Run inference on an image and return results."""
    results = model.predict(
        source=image, conf=conf_threshold, save=False, verbose=False
    )
    return results[0]


def format_class_name(cls_name, is_custom):
    """
    Map class names to waste categories.
    Custom models: 'Organic' / 'Recyclable'
    Pre-trained COCO models: Use the raw COCO class name
    """
    lower = cls_name.lower()
    if "organic" in lower or lower in ("o",):
        return "Organic", True
    elif "recyclable" in lower or lower in ("r",):
        return "Recyclable", False
    else:
        # Pre-trained COCO model — show raw class name
        return cls_name, None


# ─── UI ───────────────────────────────────────────────────────────
def main():
    init_db()

    # ── Header ────────────────────────────────────
    st.title("🗑️ Waste Classification System")
    st.markdown(
        "Upload an image or use your webcam to classify waste as "
        "**Organic** or **Recyclable** using different YOLO versions."
    )
    st.markdown("---")

    # ── Sidebar ───────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Settings")

        # Model selector
        selected_version = st.selectbox(
            "🧠 Select YOLO Model",
            list(MODEL_VERSIONS.keys()),
            index=0,
            help="Choose which YOLO version to use for detection",
        )
        ver_info = MODEL_VERSIONS[selected_version]
        st.markdown(
            f'<div style="padding:8px;background:{ver_info["color"]}20;'
            f'border-left:4px solid {ver_info["color"]};border-radius:4px;">'
            f'{ver_info["emoji"]} <strong>{selected_version}</strong> selected</div>',
            unsafe_allow_html=True,
        )

        model_path_str, is_custom = find_model_for_version(selected_version)
        badge = "✅ Custom trained" if is_custom else "📦 Pre-trained (COCO)"
        model_display = Path(model_path_str).name
        st.info(f"Model: `{model_display}`\n\n{badge}")

        if not is_custom:
            st.warning(
                f"⚠️ No custom-trained {selected_version} model found. "
                "Using pre-trained COCO weights (80 classes). "
                "Run the training notebook to train on waste data."
            )

        # Confidence slider
        conf_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.05,
            max_value=1.0,
            value=0.25,
            step=0.05,
        )

        st.markdown("---")
        st.header("📋 Waste Classes")
        st.markdown(
            "- 🥬 **Organic**: Food waste, plants, paper\n"
            "- ♻️ **Recyclable**: Plastic, metal, glass"
        )

        st.markdown("---")
        st.caption(f"Made with ❤️ — {selected_version}")

    # ── Tabs ──────────────────────────────────────
    tab_detect, tab_history = st.tabs(["🎯 Detect", "📜 History"])

    # ───────────── DETECT TAB ─────────────────────
    with tab_detect:
        # Load model
        try:
            with st.spinner(f"Loading {selected_version} model…"):
                model = load_yolo_model(model_path_str)
            st.success(f"{ver_info['emoji']} {selected_version} ready!", icon="✅")
        except Exception as e:
            st.error(
                f"❌ Failed to load {selected_version}.\n\n"
                f"**Error**: `{e}`\n\n"
                "Make sure `ultralytics` is installed: `pip install ultralytics`"
            )
            return

        col1, col2 = st.columns(2)

        with col1:
            st.header("📤 Input")
            input_method = st.radio(
                "Choose input method:",
                ["Upload Image", "Camera"],
                horizontal=True,
            )

            image = None
            image_name = None

            if input_method == "Upload Image":
                uploaded_file = st.file_uploader(
                    "Upload an image",
                    type=["jpg", "jpeg", "png", "bmp", "webp"],
                    help="Upload a waste image for classification",
                )
                if uploaded_file:
                    image = Image.open(uploaded_file).convert("RGB")
                    image_name = uploaded_file.name
                    st.image(image, caption="Uploaded Image", width="stretch")
            else:
                camera_image = st.camera_input("Take a photo")
                if camera_image:
                    image = Image.open(camera_image).convert("RGB")
                    image_name = (
                        f"camera_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                    )

        with col2:
            st.header("🎯 Results")

            if image is not None:
                try:
                    with st.spinner(f"Running {selected_version} inference…"):
                        results = run_inference(model, image, conf_threshold)

                    # Display annotated image
                    result_img = results.plot()
                    st.image(
                        result_img,
                        caption=f"{selected_version} Detection Result",
                        width="stretch",
                    )

                    # Process detections
                    boxes = results.boxes
                    detections_list = []

                    if len(boxes) > 0:
                        st.success(
                            f"✅ Found {len(boxes)} object(s) using "
                            f"**{selected_version}**"
                        )

                        for box in boxes:
                            cls_id = int(box.cls[0])
                            conf = float(box.conf[0])
                            raw_cls_name = results.names[cls_id]

                            display_name, is_organic = format_class_name(
                                raw_cls_name, is_custom
                            )

                            # Choose styling
                            if is_organic is True:
                                emoji, bg_color = "🥬", "#d4edda"
                            elif is_organic is False:
                                emoji, bg_color = "♻️", "#cce5ff"
                            else:
                                # COCO class (pre-trained, not waste)
                                emoji, bg_color = "🔍", "#fff3cd"

                            st.markdown(
                                f'<div style="padding:10px;'
                                f"background-color:{bg_color};"
                                f'border-radius:8px;margin:5px 0;">'
                                f"<h4>{emoji} {display_name}</h4>"
                                f"<p>Confidence: <strong>{conf:.1%}</strong>"
                                f" &nbsp;|&nbsp; Raw class: <code>{raw_cls_name}</code></p>"
                                f"</div>",
                                unsafe_allow_html=True,
                            )

                            detections_list.append(
                                {
                                    "class": display_name,
                                    "raw_class": raw_cls_name,
                                    "confidence": round(conf, 4),
                                }
                            )
                    else:
                        st.warning(
                            "⚠️ No objects detected. Try:\n"
                            "- Lowering the confidence threshold\n"
                            "- Using a clearer image\n"
                            "- Training a custom model for waste data"
                        )

                    # Save to history
                    model_type = "custom" if is_custom else "pretrained"
                    save_prediction(
                        selected_version,
                        model_type,
                        image_name,
                        image,
                        detections_list,
                        conf_threshold,
                    )

                except Exception as e:
                    st.error(
                        f"❌ Inference failed:\n\n```\n{traceback.format_exc()}\n```"
                    )
            else:
                st.info("👈 Upload an image or take a photo to get started!")

    # ───────────── HISTORY TAB ────────────────────
    with tab_history:
        st.header("📜 Prediction History")

        rows = load_history()

        if not rows:
            st.info("No predictions yet. Upload an image to get started!")
        else:
            col_clear, col_count = st.columns([1, 3])
            with col_clear:
                if st.button("🗑️ Clear History"):
                    clear_history()
                    st.rerun()
            with col_count:
                st.caption(f"Showing {len(rows)} most recent predictions")

            for row in rows:
                rid, ts, ver, name, thumb_bytes, n_det, det_json, conf_t = row
                ts_short = ts[:19].replace("T", " ")

                with st.expander(
                    f"#{rid}  |  {ts_short}  |  **{ver}**  |  "
                    f"{name}  |  {n_det} detection(s)"
                ):
                    c1, c2 = st.columns([1, 2])
                    with c1:
                        if thumb_bytes:
                            st.image(thumb_bytes, caption="Thumbnail", width=180)
                    with c2:
                        st.markdown(f"**Model**: {ver}")
                        st.markdown(f"**Confidence threshold**: {conf_t}")
                        dets = json.loads(det_json) if det_json else []
                        if dets:
                            for d in dets:
                                conf_val = d["confidence"]
                                cls_display = d.get("class", d.get("raw_class", "?"))
                                st.markdown(
                                    f"- **{cls_display}**: {conf_val:.1%}"
                                )
                        else:
                            st.caption("No detections")

    # ── Footer ────────────────────────────────────
    st.markdown("---")
    st.markdown(
        '<div style="text-align:center;color:gray;">'
        "<p>🧠 Powered by YOLOv5 / YOLOv8 / YOLOv11 "
        "| Waste Classification Project</p>"
        "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
