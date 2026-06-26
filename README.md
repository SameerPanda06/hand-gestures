# ✋ Hand Gesture Recognition — SVM + Random Forest
A computer vision project that identifies and classifies 10 different hand gestures from image data using HOG feature extraction combined with SVM and Random Forest classifiers, trained on the LeapGestRecog dataset.

## 📁 Project Structure
```
hand_gesture_recognition/
├── leapGestRecog/
│   ├── 01_palm/
│   │   ├── 00/
│   │   ├── 01/
│   │   └── ... (10 subjects)
│   ├── 02_l/
│   ├── 03_fist/
│   ├── 04_fist_moved/
│   ├── 05_thumb/
│   ├── 06_index/
│   ├── 07_ok/
│   ├── 08_palm_moved/
│   ├── 09_c/
│   └── 10_down/
├── models/
│   ├── svm.pkl
│   ├── random_forest.pkl
│   ├── scaler.pkl
│   └── label_encoder.pkl
├── plots/
│   ├── gesture_samples.png
│   ├── confusion_svm.png
│   └── confusion_random_forest.png
├── train.py
├── app.py
└── README.md
```

## 🛠️ Tech Stack
- Python 3.x
- scikit-learn
- OpenCV (cv2)
- scikit-image (HOG)
- matplotlib, seaborn
- joblib
- Streamlit

## ⚙️ Setup & Installation
```bash
pip install opencv-python scikit-image scikit-learn matplotlib seaborn joblib streamlit
```

## 📊 Dataset
**LeapGestRecog — Hand Gesture Recognition Dataset**  
https://www.kaggle.com/datasets/gti-upm/leapgestrecog

| Property | Detail |
|---|---|
| Classes | 10 gesture types |
| Subjects | 10 individuals per gesture |
| Images | ~200 per subject per gesture |
| Format | Grayscale PNG |
| Total Size | ~2GB compressed |

### Gesture Classes
| Folder | Gesture |
|---|---|
| 01_palm | Palm |
| 02_l | L shape |
| 03_fist | Fist |
| 04_fist_moved | Fist Moved |
| 05_thumb | Thumb Up |
| 06_index | Index Finger |
| 07_ok | OK sign |
| 08_palm_moved | Palm Moved |
| 09_c | C shape |
| 10_down | Down |

## 🚀 Run
```bash
# Step 1 — Train models (~5-10 mins)
python train.py

# Step 2 — Launch dashboard
streamlit run app.py
```

## 🤖 How It Works
### Pipeline
```
Image (64×64) → Grayscale → HOG Features (1764-dim) → StandardScaler → SVM / RF → Gesture Label
```
### HOG Feature Extraction
HOG captures edge directions and gradient magnitudes across image cells — encodes the shape of a hand gesture without needing deep learning.
| Parameter | Value |
|---|---|
| Image size | 64×64 |
| Orientations | 9 |
| Pixels per cell | 8×8 |
| Cells per block | 2×2 |
| Samples per class | 200 |
| Total samples | 2000 |
### Models Used
| Model | Why |
|---|---|
| SVM (RBF kernel) | Strong on high-dimensional HOG features, good margin separation |
| Random Forest | Ensemble of decision trees, fast + handles multi-class well |

## 📈 Results
| Model | Accuracy |
|---|---|
| SVM (RBF, C=10) | ~95–98% |
| Random Forest (100 trees) | ~92–96% |
> High accuracy due to controlled lighting and background in LeapGestRecog dataset.

## 🖥️ Dashboard Features
- Upload a gesture image
- Get prediction from both SVM and Random Forest
- Confidence score for each model
- Agreement/disagreement indicator between models
- Gesture class samples grid
- Confusion matrices for both models

## 📚 Concepts Covered
- Multi-class image classification
- HOG feature extraction for hand gesture encoding
- SVM with RBF kernel — multi-class via one-vs-one strategy
- Random Forest ensemble learning
- LabelEncoder for class name mapping
- StandardScaler for HOG feature normalization
- Model comparison and confusion matrix analysis
- Streamlit image upload + dual model inference
