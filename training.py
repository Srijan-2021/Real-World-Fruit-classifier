import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import json, os
from pathlib import Path

# ---------- CONFIG ----------
IMG_SIZE   = (224, 224)
BATCH_SIZE = 32
EPOCHS_HEAD = 5       # Phase 1: Train just the new top layer
EPOCHS_FINE = 10      # Phase 2: Fine-tune the top of MobileNet
DATA_DIR   = Path("data/v1")
ARTIFACT_DIR = Path("artifacts")
ARTIFACT_DIR.mkdir(exist_ok=True)

MODEL_OUT  = ARTIFACT_DIR / "fruit_model_v1.keras"
LABELS_OUT = ARTIFACT_DIR / "class_indices.json"

# ---------- LOAD DATA (Optimized for CPU) ----------
print("📂 Loading training data...")
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR / "Training",
    validation_split=0.15,
    subset="training",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical"
)

print("📂 Loading validation data...")
val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR / "Training",
    validation_split=0.15,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical"
)

# Save the label mapping
class_names = train_ds.class_names
with open(LABELS_OUT, "w") as f:
    json.dump({name: i for i, name in enumerate(class_names)}, f, indent=2)
print(f"✅ Found {len(class_names)} classes: {class_names}")

# Prefetching keeps the CPU busy while the GPU/TF calculates
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

# ---------- BUILD MODEL ----------
print("🏗️ Building MobileNetV2 model...")
# Data Augmentation (Randomly flips/zooms images so it doesn't memorize them)
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

# Base Model
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(*IMG_SIZE, 3), include_top=False, weights="imagenet"
)
base_model.trainable = False # Freeze base for Phase 1

inputs = tf.keras.Input(shape=(*IMG_SIZE, 3))
x = data_augmentation(inputs)
# MobileNetV2 expects pixels in [-1, 1], not [0, 1]
x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(len(class_names), activation="softmax")(x)

model = models.Model(inputs, outputs)

# ---------- PHASE 1: TRAIN HEAD ----------
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

early_stop = callbacks.EarlyStopping(patience=3, restore_best_weights=True)
checkpoint = callbacks.ModelCheckpoint(MODEL_OUT, save_best_only=True, monitor="val_accuracy", verbose=1)

print("\n🚀 PHASE 1: Training the classification head...")
model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_HEAD, callbacks=[early_stop, checkpoint])

# ---------- PHASE 2: FINE-TUNE ----------
print("\n🔧 PHASE 2: Fine-tuning the top layers of MobileNetV2...")
base_model.trainable = True
# Freeze all layers except the last 30
for layer in base_model.layers[:-30]:
    layer.trainable = False

# Recompile with a much smaller learning rate
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_FINE, callbacks=[early_stop, checkpoint])

print(f"\n🏁 Training complete! Best model saved to {MODEL_OUT}")