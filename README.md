# 📱 App to Detect — Drunk/Normal Detection System

## 🔍 Project Overview

**App to Detect** is an Android application that analyzes a face image and predicts whether the person appears **Drunk** or **Normal** using behavioral facial cues.

Instead of using a heavy deep learning classifier, this system uses a **lightweight, explainable pipeline** that runs completely **offline on-device**.

---

## 🎯 Objective

* Capture or upload an image
* Detect face using ML Kit
* Extract behavioral features
* Predict **Drunk / Normal**
* Display confidence and reasons

---

## ⚙️ System Pipeline

```
Image (Camera / Upload)
        ↓
Face Detection (ML Kit)
        ↓
Feature Extraction
        ↓
Rule-Based Scoring
        ↓
Prediction (Drunk / Normal)
```

---

## 🧠 Approach

### 🔹 Behavioral Analysis (Core Idea)

Instead of raw image classification, we analyze:

* Eye openness
* Head tilt
* Eye asymmetry
* Facial expression cues

### 🔹 Decision Logic

A scoring-based system:

* Eyes closed → +40
* Head tilted → +15 / +30
* Eye asymmetry → +10

**Final Rule:**

* Score ≥ 40 → **Drunk**
* Score < 40 → **Normal**

---

## 📊 Dataset

Custom dataset used for validation:

* ~20 images (Drunk-like)
* ~20 images (Normal)

### Preprocessing:

* Resized to 512×512
* Removed low-quality images
* Ensured clear face visibility

---

## 🧪 Development Workflow

### 🧑‍💻 VS Code (Python Phase)

* Dataset preprocessing
* Manual filtering
* Feature extraction using MediaPipe
* Rule validation (accuracy ~72%)

### 📱 Android Phase

* UI development in Kotlin
* Image upload & camera integration
* ML Kit Face Detection
* Feature extraction (eye, tilt, smile)
* Rule-based prediction implemented

---

## 📱 App Features

* 📷 Camera input (with fallback)
* 🖼️ Image upload
* 👤 Face detection (on-device)
* 🧠 Feature extraction
* 📊 Drunk/Normal prediction
* 📈 Confidence score
* 📝 Reason explanation

---

## 🛠️ Technologies Used

### Android

* Kotlin
* Android Studio
* ML Kit Face Detection

### Python (Model Prototyping)

* OpenCV
* MediaPipe
* NumPy
* Pandas

---

## 📸 Sample Output

```
Result: Drunk
Confidence: 70%

Details:
- Eyes mostly closed
- Head tilted strongly
```

---

## ⚠️ Limitations

* Works on **single image only**
* Requires **clear face visibility**
* Detects **drunk-like behavior**, not actual intoxication
* Small dataset used

---

## 🚀 Future Improvements

* Real-time video analysis
* Blink detection
* Deep learning model (CNN)
* Larger dataset
* Audio-based detection

---

## 🏁 Conclusion

This project demonstrates a practical and explainable approach to behavioral face analysis using **on-device AI**, making it suitable for lightweight mobile applications.

---

