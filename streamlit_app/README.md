# 🗑️ Waste Classification Streamlit App

A real-time waste classification application using YOLOv11.

## Features

- 📤 **Image Upload**: Upload waste images for classification
- 📷 **Camera Input**: Take photos directly from webcam
- 🎯 **Real-time Detection**: Instant classification results
- 📊 **Confidence Scores**: See prediction confidence

## Running the App

```bash
# Navigate to the streamlit_app directory
cd streamlit_app

# Run the app
streamlit run app.py
```

## Requirements

- Python 3.8+
- streamlit
- ultralytics
- PIL

## Classes

| Class | Description | Examples |
|-------|-------------|----------|
| 🥬 Organic | Biodegradable waste | Food scraps, leaves, paper |
| ♻️ Recyclable | Non-biodegradable recyclables | Plastic, metal, glass |

## Model

The app automatically loads the best trained model from `models/waste_*/weights/best.pt`.
If no trained model is found, it falls back to the pre-trained YOLOv11 nano.
