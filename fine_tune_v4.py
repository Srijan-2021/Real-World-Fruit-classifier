import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import shutil, json
from pathlib import Path

# ---------- CONFIG ----------
IMG_SIZE   = (224, 224)
BATCH_SIZE = 32
DATA_DIR   = Path("data/v1")
HARD_DIR   = Path("data/hard_examples")
ARTIFACT_DIR = Path("artifacts")
V3_MODEL   = ARTIFACT_DIR / "fruit_model_v3.keras"
V4_MODEL   = ARTIFACT_DIR / "fruit_model_v4.keras"

print("📂 Adding Hard Examples to Training data...")
for cls_dir in HARD_DIR.iterdir():
    if cls_dir.is_dir():
        cls_name = cls_dir.name
        train_cls_dir = DATA_DIR / "Training" / cls_name
        train_cls_dir.mkdir(parents=True, exist_ok=True)

        # Clean up old hard examples
        for f in train_cls_dir.glob("hard_*"):
            f.unlink()

        count = 0
        for img in cls_dir.glob("*.*"):
            if img.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                shutil.copy2(img, train_cls_dir / f"hard_{count}{img.suffix}")
                count += 1
        print(f"   Added {count} hard examples for {cls_name}")

# Load Data
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR / "Training", validation_split=0.15, subset="training",
    seed=123, image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode="categorical"
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR / "Training", validation_split=0.15, subset="validation",
    seed=123, image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode="categorical"
)
train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

print("\n🧠 Loading V3 Model...")
model = tf.keras.models.load_model(V3_MODEL)

# We unfreeze the last 15 layers to allow it to adjust to the new Mango data
for layer in model.layers[-15:]:
    layer.trainable = True

print("\n🔧 Surgical Fine-Tuning...")
# VERY low learning rate (1e-6) so we don't ruin the V3 weights
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-6),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

early_stop = callbacks.EarlyStopping(patience=2, restore_best_weights=True)
checkpoint = callbacks.ModelCheckpoint(V4_MODEL, save_best_only=True, monitor="val_accuracy")

# Only 3 epochs! We just want it to see the hard examples a few times.
model.fit(train_ds, validation_data=val_ds, epochs=3, callbacks=[early_stop, checkpoint])

print(f"\n🏁 V4 Training complete! Saved to {V4_MODEL}")