import tensorflow as tf
import numpy as np
import shutil
import json
from pathlib import Path
from PIL import Image

# ---------- CONFIG ----------
MODEL_PATH = Path("artifacts/fruit_model_v2.keras")
LABELS_PATH = Path("artifacts/class_indices.json")
DATA_DIR = Path("data/real_world_v2")
REJECT_DIR = Path("data/real_world_v2_rejected")
IMG_SIZE = (224, 224)

# STRICTER THRESHOLD:
# The V2 model is trained on whole fruits. It will be highly confident (>85%) on whole fruits.
# It will drop below 85% on diced fruit, fruit bowls, and market scenes.
CONFIDENCE_THRESHOLD = 0.85

print("🧠 Loading V2 Model to act as the filter...")
model = tf.keras.models.load_model(MODEL_PATH)
with open(LABELS_PATH, "r") as f:
    class_indices = json.load(f)
idx_to_class = {v: k for k, v in class_indices.items()}

def preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB").resize(IMG_SIZE)
    img_array = np.array(img, dtype=np.float32)
    return np.expand_dims(img_array, axis=0)

total_kept = 0
total_rejected = 0

for cls in ["Apple", "Banana", "Mango", "Orange", "Strawberry"]:
    cls_dir = DATA_DIR / cls
    reject_cls_dir = REJECT_DIR / cls
    reject_cls_dir.mkdir(parents=True, exist_ok=True)

    if not cls_dir.exists():
        continue

    print(f"\n🔍 Filtering {cls}...")
    images = list(cls_dir.glob("*.*"))

    for img_path in images:
        if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
            continue

        try:
            # Predict
            preds = model.predict(preprocess_image(img_path), verbose=0)[0]
            top_idx = np.argmax(preds)
            confidence = float(preds[top_idx])
            predicted_class = idx_to_class[top_idx]

            # Keep only if the model is >85% sure AND it matches the folder name
            if predicted_class == cls and confidence >= CONFIDENCE_THRESHOLD:
                total_kept += 1
            else:
                shutil.move(str(img_path), str(reject_cls_dir / img_path.name))
                total_rejected += 1

        except Exception as e:
            print(f"⚠️ Error reading {img_path.name}: {e}")

print("\n" + "="*50)
print(f"✅ Total whole-fruit images kept: {total_kept}")
print(f"🗑️ Total images rejected:         {total_rejected}")
print("Rejected images (bowls, diced, blurry) were moved to:")
print("data/real_world_v2_rejected/")
print("="*50)