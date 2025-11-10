# Potato Leaf Disease CNN (Keras/TensorFlow)

A compact **Convolutional Neural Network (CNN)** to classify **potato leaf** images into:
**['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']**.

This README documents the model pipeline, architecture, training/evaluation setup, and how to run inference and export models.

---

## 🔍 Problem & Dataset

- **Task:** Multiclass image classification (potato leaf health vs. diseases).
- **Classes:** ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']
- **Dataset root:** `PlantVillage` (loaded via `tf.keras.preprocessing.image_dataset_from_directory`)
- **Train/Val/Test split:** 80% / 10% / 10% using a custom `get_dataset_partitions_tf` helper.
- **Input size:** `256×256×3`
- **Batch size:** `32`

> Images are **resized** and **normalized** to `[0,1]`. Lightweight **data augmentation** (random flips & rotations) improves generalization.

---

## 🧱 Model Architecture

Implemented with `tf.keras.Sequential`:

```python
resize_and_rescale = tf.keras.Sequential([
    layers.Resizing(256, 256),
    layers.Rescaling(1./255),
])

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
])

model = models.Sequential([
    resize_and_rescale,
    layers.Conv2D(32, (3,3), activation="relu"),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(64, (3,3), activation="relu"),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(64, (3,3), activation="relu"),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(64, (3,3), activation="relu"),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(64, (3,3), activation="relu"),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(64, (3,3), activation="relu"),
    layers.MaxPooling2D((2,2)),

    layers.Flatten(),
    layers.Dense(64, activation="relu"),
    layers.Dense(3, activation="softmax"),
])
```

- **Parameters:** Small-to-moderate; suitable for fast training and edge deployment.
- **Regularization:** Data augmentation; (no dropout explicitly used).
- **Why this works:** Stacked `Conv2D + MaxPool` progressively extracts texture & lesion patterns; final dense layers learn class boundaries.

---

## ⚙️ Training Setup

- **Optimizer:** `adam`
- **Loss:** `SparseCategoricalCrossentropy(from_logits=False)`
- **Metric:** `accuracy`
- **Epochs:** `50`
- **Data pipeline:** cached + prefetch (`tf.data.AUTOTUNE`) for throughput; shuffling enabled.

```python
model.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=["accuracy"],
)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=50,
    verbose=1,
)
```

---

## 📈 Results (from the notebook logs)

- **Final train accuracy:** 0.9959
- **Final train loss:** 0.0158
- **Final **val** accuracy:** 0.9844
- **Final **val** loss:** 0.0197
- **Test accuracy:** 0.9844
- **Test loss:** 0.0502

> Tip: Add a confusion matrix & per-class F1 for deeper insight. (See **Extensions** below.)

---

## ▶️ Inference (single image)

```python
import tensorflow as tf
import numpy as np

# Ensure `class_names` matches the training order
class_names = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']

def load_and_preprocess(path, image_size=256):
    img = tf.keras.utils.load_img(path, target_size=(256, 256))
    arr = tf.keras.utils.img_to_array(img)
    arr = arr / 255.0
    arr = np.expand_dims(arr, axis=0)  # (1, H, W, C)
    return arr

model = tf.keras.models.load_model("path/to/your_model.keras")
x = load_and_preprocess("path/to/image.jpg")
pred = model.predict(x)[0]
pred_idx = int(np.argmax(pred))
print("Predicted:", class_names[pred_idx], "| Confidence:", round(100*float(np.max(pred)), 2), "%")
```

---

## 💾 Saving & Versioning

This project auto-increments model versions under `../models/`:

```python
import os
os.makedirs("../models", exist_ok=True)
# ... find next integer version ...
model.save(f"../models/{next_version}.keras")
```

You can also export to SavedModel or TFLite:

```python
# SavedModel
model.save("export/saved_model")

# TFLite
import tensorflow as tf
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open("export/model.tflite", "wb") as f:
    f.write(tflite_model)
```

---

## 🧪 Reproducing the Setup

1. Prepare the dataset directory:
   ```
   PlantVillage/
     ├── Potato___Early_blight / Potato___Late_blight / Potato___healthy  # one folder per class
   ```
2. Install deps (TensorFlow 2.x recommended):
   ```bash
   pip install tensorflow matplotlib numpy
   ```
3. Run the notebook end-to-end.
4. Evaluate on `test_ds` and export the model.

---

## 🚀 Extensions (nice next steps)

- Add **EarlyStopping** and **ReduceLROnPlateau** callbacks.
- Insert **Dropout** before the final Dense layers for regularization.
- Log runs with **TensorBoard**; track **precision/recall/F1**.
- Export **ONNX** and benchmark with **ONNX Runtime**.
- Build a lightweight **FastAPI/Streamlit** demo for local inference.

---

## 📂 Project Structure (suggested)

```
.
├── notebooks/
│   └── Training.ipynb
├── data/
│   └── PlantVillage/
├── models/
│   └── 1.keras
├── export/
│   ├── saved_model/
│   └── model.tflite
└── README.md
```

---

**Last updated:** 2025-11-10
