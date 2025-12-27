# Metal Nut Defect Detection using Transfer Learning

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

An industrial-grade computer vision system for automated quality inspection of metal nuts using **MobileNetV2-based transfer learning**. Designed for real-time, lightweight, and accurate defect detection under limited data conditions.

---

## Project Overview

This project solves a real-world manufacturing quality control problem by classifying metal nuts as **GOOD** or **DEFECT** using deep learning. It demonstrates how transfer learning outperforms custom CNNs when datasets are small and class imbalance exists.

---

## Key Metrics

| Metric | Value |
|--------|-------|
| Overall Accuracy | **88.2%** |
| GOOD Recall | 94.6% |
| DEFECT Recall | 71.4% |
| Inference Time | ~50 ms/image |
| Model Size | ~9 MB |
| Accuracy Gain vs CNN | +14.7% |

---

## Features

- MobileNetV2 transfer learning (ImageNet pretrained)
- Binary classification: GOOD vs DEFECT
- Streamlit web dashboard for real-time inference
- Confidence scores with probability breakdown
- Weighted loss for class imbalance handling
- Lightweight, deployment-ready architecture

---

## Project Structure

```
defect-detection-transfer-learning/
│
├── app.py
├── train_mobilenet.py
├── evaluate_model.py
├── requirements.txt
├── README.md
│
├── models/
│   └── mobilenet_model.keras
│
├── outputs/
│   ├── mobilenet_training.png
│   └── confusion_matrix.png
│
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── model.py
│   └── inference.py
│
└── data/raw/metal_nut/
    ├── train/good/
    └── test/
        ├── good/
        ├── bent/
        ├── scratch/
        ├── color/
        └── flip/
```

---

## Installation

```bash
git clone https://github.com/antra04/defect-detection-transfer-learning.git
cd defect-detection-transfer-learning

python -m venv venv
venv\Scripts\activate   # Windows
# source venv/bin/activate  # Linux/Mac

pip install -r requirements.txt
```

---

## Usage

### Run Web App
```bash
streamlit run app.py
```
Access at: `http://localhost:8501`

### Train Model (Optional)
```bash
python train_mobilenet.py
```

### Evaluate Model
```bash
python evaluate_model.py
```

---

## Results

**Classification Summary:**
- GOOD → Precision: 0.90 | Recall: 0.95 | F1: 0.92
- DEFECT → Precision: 0.83 | Recall: 0.71 | F1: 0.77
- Overall Accuracy: **88.2%**

**Confusion Matrix:**
- GOOD: 37 correct, 2 misclassified
- DEFECT: 10 correct, 4 misclassified

---

## Model Architecture

```
MobileNetV2 (ImageNet pretrained, frozen)
        ↓
GlobalAveragePooling
        ↓
Dense(128, ReLU)
        ↓
Dropout(0.5)
        ↓
Dense(2, Softmax)
```

---

## Class Imbalance Handling

```
Dataset Distribution:
- GOOD samples: 241 (72%)
- DEFECT samples: 92 (28%)

Class Weights Applied:
- GOOD   → 1.0
- DEFECT → 2.61
```

---

## Key Learnings

- Transfer learning is critical for small industrial datasets
- Custom CNNs struggle with class imbalance
- Weighted loss significantly improves defect recall
- Lightweight models enable real-time deployment

---

## Future Enhancements

- Multi-class defect classification
- Defect localization using segmentation
- Model quantization for edge devices
- Active learning for dataset expansion

---

## Dataset

**MVTec Anomaly Detection Dataset – Metal Nut**
- License: CC BY-NC-SA 4.0
- Images: 335
- Defect Types: bent, scratch, color, flip

---

## Author

**Antra Tiwari**  
B.Tech Computer Science Engineering  


Email: antratiwari04@gmail.com  
LinkedIn: [linkedin.com/in/antra-tiwari](https://linkedin.com/in/antratiwari04)  
GitHub: [github.com/antra04](https://github.com/antra04)

---

If you find this project helpful, please consider starring the repository!
