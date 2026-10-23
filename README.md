
# Retail Product Detection & Counting

## Project Overview
This project detects visible products on retail shelves using a trained YOLO object detection model. It displays bounding boxes around detected products and calculates the total number of detections.

## Objectives
- Detect products in retail shelf images.
- Display bounding boxes around detected products.
- Calculate the total number of detected products.
- Display confidence score statistics.
- Provide a simple web interface using Streamlit.

## Technologies Used
- Python
- YOLOv8 (Ultralytics)
- OpenCV
- Streamlit
- NumPy
- Pillow

## Project Structure
```text
CNN_PROJECT/
├── app.py
├── requirements.txt
├── README.md
├── models/
│   └── retail_product_detection_best.pt
└── .venv/
```

## Installation

Install the required libraries:

```bash
pip install -r requirements.txt
```

## Run the Application

```bash
python -m streamlit run app.py
```

Open the local URL displayed in the terminal.

## How to Use
1. Open the Streamlit application.
2. Upload a shelf image.
3. Adjust the detection confidence threshold if needed.
4. View the original image and detected products.
5. Check the total detection count and confidence statistics.

## Limitations
- The current model detects products as a single class.
- It does not identify individual brands or product categories.
- The detection count may differ from the actual number of products.

## Model Evaluation
The model was evaluated on the project's test dataset. Add the final evaluation metrics here after verifying them.

## Future Improvements
- Compare detections across multiple shelf images.
- Improve detection of overlapping or small products.
- Add category-specific product recognition.
