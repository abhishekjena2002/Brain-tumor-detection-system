# Brain-tumor-detection-system
Brain Tumor Detection System: Developed a deep learning-based medical imaging solution using MobileNetV2 to classify MRI brain scans as Tumor or No Tumor. Implemented image preprocessing, data validation, MLflow experiment tracking, model evaluation (Accuracy, F1-Score, ROC-AUC), and Streamlit deployment for real-time prediction.

# Brain Tumor Detection using Deep Learning

## Project Overview

This project is a Deep Learning-based Brain Tumor Detection System that classifies MRI brain scan images into two categories: **Tumor** and **No Tumor**. The solution uses Transfer Learning with MobileNetV2 and provides an end-to-end machine learning workflow including data preprocessing, validation, model training, experiment tracking, and deployment.

## Features

* MRI Image Classification
* Image Preprocessing Pipeline
* Data Validation using Great Expectations
* Experiment Tracking using MLflow
* Deep Learning Models Comparison

  * CNN
  * MobileNetV2
  * ResNet50
  * EfficientNetB0
* Model Evaluation using:

  * Accuracy
  * F1 Score
  * ROC-AUC
* Streamlit Web Application Deployment

## Tech Stack

* Python
* TensorFlow / Keras
* MobileNetV2
* OpenCV
* NumPy
* Pandas
* Scikit-Learn
* Great Expectations
* MLflow
* Streamlit

## Dataset

The dataset consists of MRI brain images categorized into:

* Tumor (1)
* No Tumor (0)

Images are resized to 224 × 224 pixels and normalized before training.

## Project Workflow

1. Data Collection
2. Exploratory Data Analysis (EDA)
3. Image Preprocessing
4. Data Validation using Great Expectations
5. Train-Test Split
6. Model Training
7. Model Evaluation
8. MLflow Experiment Tracking
9. Model Selection
10. Streamlit Deployment

## Model Performance

| Model          | Accuracy | F1 Score | ROC-AUC |
| -------------- | -------- | -------- | ------- |
| CNN            | 86.27%   | 89.55%   | 93.39%  |
| MobileNetV2    | 94.12%   | 95.24%   | 97.10%  |
| ResNet50       | 74.51%   | 82.19%   | 79.68%  |
| EfficientNetB0 | 60.78%   | 75.61%   | 55.81%  |

Best Performing Model: **MobileNetV2**

## MLflow Tracking

MLflow is used to:

* Track experiments
* Compare models
* Store parameters
* Store metrics
* Save trained models

Run MLflow UI:

```bash
mlflow ui
```

Open:

```text
http://127.0.0.1:5000
```

## Streamlit Deployment

Run the application:

```bash
streamlit run streamlit_app.py
```

Open:

```text
http://localhost:8501
```

## Project Structure

```text
Brain-Tumor-Detection/

├── artifacts/
│   ├── MobileNetV2.keras
│
├── src/
│   ├── app.py
│   ├── preprocessing.py
│   ├── validation.py
│   ├── mlflow_training.py
│   ├── streamlit_app.py
│
├── gx/
├── mlruns/
├── requirements.txt
└── README.md
```

## Future Improvements

* Tumor Localization using YOLOv8
* Tumor Segmentation using U-Net
* Multi-Class Brain Tumor Classification
* Cloud Deployment
* Real-Time Clinical Integration

## Author

Abhishek Jena
mlops
mlflow
docker
clickhouse
apachi airflow
postgreshSQL
SQL
PowerBI
Visualization
promtheus
Grafan
Data Science | Machine Learning | Deep Learning | MLOps
