import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten

# -----------------------------
# CONFIG
# -----------------------------
DATA_DIR = "data"
IMG_SIZE = 48
NUM_CLASSES = 7

# -----------------------------
# LABEL MAPPING (must match folder names)
# -----------------------------
emotion_map = {
    "angry": 0,
    "disgust": 1,
    "fear": 2,
    "happy": 3,
    "neutral": 4,
    "sad": 5,
    "surprise": 6
}

images = []
labels = []

# -----------------------------
# LOAD DATASET
# -----------------------------
for emotion in emotion_map:
    folder_path = os.path.join(DATA_DIR, emotion)

    if not os.path.exists(folder_path):
        print(f"Warning: folder not found -> {folder_path}")
        continue

    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        images.append(img)
        labels.append(emotion_map[emotion])

# -----------------------------
# PREPROCESS
# -----------------------------
X = np.array(images, dtype="float32") / 255.0
X = X.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

y = to_categorical(labels, num_classes=NUM_CLASSES)

# -----------------------------
# TRAIN / TEST SPLIT
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

print("Training samples:", X_train.shape)
print("Testing samples:", X_test.shape)

# -----------------------------
# BUILD CNN MODEL
# -----------------------------
model = Sequential([
    Conv2D(32, (3, 3), activation="relu", input_shape=(48, 48, 1)),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.5),

    Dense(NUM_CLASSES, activation="softmax")
])

# -----------------------------
# COMPILE
# -----------------------------
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# -----------------------------
# TRAIN
# -----------------------------
model.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=32,
    validation_data=(X_test, y_test)
)

# -----------------------------
# SAVE MODEL
# -----------------------------
model.save("emotion_model.h5")
print("Model saved as emotion_model.h5")
