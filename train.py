import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import cv2
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

BASE     = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, 'leapGestRecog')
IMG_SIZE = 64
N_PER_CLASS = 200  # 200 images per gesture class (2000 total — fast + sufficient)

GESTURE_LABELS = {
    '01_palm':       'Palm',
    '02_l':          'L',
    '03_fist':       'Fist',
    '04_fist_moved': 'Fist Moved',
    '05_thumb':      'Thumb',
    '06_index':      'Index',
    '07_ok':         'OK',
    '08_palm_moved': 'Palm Moved',
    '09_c':          'C',
    '10_down':       'Down',
}

def extract_hog(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    features = hog(img, orientations=9,
                   pixels_per_cell=(8, 8),
                   cells_per_block=(2, 2),
                   visualize=False)
    return features

# 1. LOAD DATA
print("Loading dataset...")
X, y = [], []

for gesture_folder in sorted(os.listdir(DATA_DIR)):
    gesture_path = os.path.join(DATA_DIR, gesture_folder)
    if not os.path.isdir(gesture_path):
        continue
    label = GESTURE_LABELS.get(gesture_folder, gesture_folder)
    count = 0

    # iterate subject subfolders (00–09)
    for subject_folder in sorted(os.listdir(gesture_path)):
        subject_path = os.path.join(gesture_path, subject_folder)
        if not os.path.isdir(subject_path):
            continue
        for fname in os.listdir(subject_path):
            if count >= N_PER_CLASS:
                break
            if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            feat = extract_hog(os.path.join(subject_path, fname))
            if feat is not None:
                X.append(feat)
                y.append(label)
                count += 1
        if count >= N_PER_CLASS:
            break

    print(f"  {gesture_folder}: {count} images")

X = np.array(X)
y = np.array(y)
print(f"\nTotal: {X.shape[0]} samples | {X.shape[1]} HOG features | {len(set(y))} classes")

# 2. ENCODE LABELS
le = LabelEncoder()
y_enc = le.fit_transform(y)

# 3. SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
)

# 4. SCALE
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# 5. TRAIN — SVM + Random Forest (compare both)
models = {
    'SVM':           SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
}

os.makedirs(os.path.join(BASE, 'models'), exist_ok=True)
os.makedirs(os.path.join(BASE, 'plots'), exist_ok=True)

best_acc = 0
best_name = ''

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"  Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=le.classes_,
                yticklabels=le.classes_)
    plt.title(f'{name} — Confusion Matrix (Acc: {acc:.2%})')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(BASE, 'plots', f'confusion_{name.replace(" ","_").lower()}.png'), dpi=150)
    plt.close()

    joblib.dump(model, os.path.join(BASE, 'models', f'{name.replace(" ","_").lower()}.pkl'))

    if acc > best_acc:
        best_acc = acc
        best_name = name

# 6. SAVE scaler + encoder
joblib.dump(scaler, os.path.join(BASE, 'models', 'scaler.pkl'))
joblib.dump(le,     os.path.join(BASE, 'models', 'label_encoder.pkl'))

# 7. SAMPLE GRID PLOT
print("\nGenerating sample grid...")
fig, axes = plt.subplots(2, 5, figsize=(15, 7))
fig.suptitle('Gesture Classes — Sample Images', fontsize=14, fontweight='bold')

for ax, gesture_folder in zip(axes.flatten(), sorted(os.listdir(DATA_DIR))):
    gesture_path = os.path.join(DATA_DIR, gesture_folder)
    if not os.path.isdir(gesture_path):
        continue
    # grab first image from first subject
    for subject_folder in sorted(os.listdir(gesture_path)):
        subject_path = os.path.join(gesture_path, subject_folder)
        if not os.path.isdir(subject_path):
            continue
        for fname in os.listdir(subject_path):
            if fname.lower().endswith(('.png', '.jpg')):
                img = cv2.imread(os.path.join(subject_path, fname), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    ax.imshow(img, cmap='gray')
                    ax.set_title(GESTURE_LABELS.get(gesture_folder, gesture_folder), fontsize=9)
                    ax.axis('off')
                    break
        break

plt.tight_layout()
plt.savefig(os.path.join(BASE, 'plots', 'gesture_samples.png'), dpi=150)
plt.close()

print(f"\nBest model: {best_name} — {best_acc:.2%}")
print("Done. Models + plots saved.")