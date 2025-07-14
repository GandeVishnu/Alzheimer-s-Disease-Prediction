# 🧠 Alzheimer's Disease Stage Prediction using Deep Learning

This project presents a deep learning-based system to **predict the stage of Alzheimer’s Disease** using brain MRI images. It uses **EfficientNetB0**, **TensorFlow**, and a **Streamlit web application** for an end-to-end experience: from login/signup and MRI upload to stage prediction and PDF report generation.

🔗 **Live App:** [Click here to open](https://alzheimers-disease-detection.streamlit.app/)  

## 🔍 Features

- ✅ Classifies MRI images into 5 Alzheimer’s stages
- 🧪 Based on EfficientNetB0 model with high accuracy
- ⚖️ Class imbalance handling with data augmentation
- 💾 MongoDB for storing users and application data
- 🧾 PDF report generation for each diagnosis
- 🌐 Fully responsive and user-friendly Streamlit UI

---


## 💡 Dataset Info

- Source: **ADNI - Alzheimer’s Disease Neuroimaging Initiative**
-  📁 Dataset Size - ~18,754 Brain MRI images

- Classes:
  -  `Final_AD` (Alzheimer’s Disease)
  - `Final_CN` (Cognitively Normal)
  - `Final_EMCI` (Early Mild Cognitive Impairment)
  - `Final_LMCI` (Late Mild Cognitive Impairment)
  - `Final_MCI` (Mild Cognitive Impairment)

---

## 🧪 Classification Results

| Class            | Precision | Recall | F1-Score | Support |
|------------------|-----------|--------|----------|---------|
| Final AD JPEG    | 0.93      | 0.95   | 0.94     | 810     |
| Final CN JPEG    | 0.95      | 0.84   | 0.89     | 1220    |
| Final EMCI JPEG  | 0.61      | 0.89   | 0.72     | 240     |
| Final LMCI JPEG  | 0.92      | 0.83   | 0.88     | 72      |
| Final MCI JPEG   | 0.72      | 0.77   | 0.74     | 233     |

**Test Accuracy**: `87.26%`  

---

### 🔑 Password Policy

During sign-up, user passwords must meet the following criteria:

- ✅ At least **one uppercase letter**
- ✅ At least **one lowercase letter**
- ✅ At least **one special character** (e.g., `@`, `#`, `$`, `!`)

---

## 📄 Report Generation

- After prediction, users can fill out a form with name, age, place and phone number.
- A PDF diagnosis report is generated and made downloadable.
- Includes patient details, stage prediction, and confidence.


---
