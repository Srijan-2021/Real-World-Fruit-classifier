import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import json, os, shutil
from pathlib import Path

# ---------- CONFIG ----------
IMG_SIZE   = (224, 224)
BATCH_SIZE = 32
EPOCHS_FINE = 5
DATA_DIR   = Path("data/v1")
REAL_DIR   = Path("data/real_world")
ARTIFACT_DIR = Path("artifacts")
MODEL_OUT  = ARTIFACT_DIR / "fruit_model_v2.keras"

print("📂 Merging Real-World images into Training set...")
for cls in ["Apple", "Banana", "Mango", "Orange", "Strawberry"]:
    real_cls_dir = REAL_DIR / cls
    train_cls_dir = DATA_DIR / "Training" / cls

    # Clean up old real images to avoid duplicates if you run this twice
    for f in train_cls_dir.glob("real_*"):
        f.unlink()

    if real_cls_dir.exists():
        count = 0
        for img in real_cls_dir.glob("*.*"):
            if img.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                new_name = f"real_{count}{img.suffix}"
                shutil.copy2(img, train_cls_dir / new_name)
                count += 1
        print(f"   Added {count} real images for {cls}")

# Load Data
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR / "Training", validation_split=0.15, subset="training",
    seed=123, image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode="categorical"
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR / "Training", validation_split=0.15, subset="validation",
    seed=123, image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode="categorical"
)

class_names = train_ds.class_names
with open(ARTIFACT_DIR / "class_indices.json", "w") as f:
    json.dump({name: i for i, name in enumerate(class_names)}, f, indent=2)

train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

print("\n🏗️ Building Model with Heavy Augmentation...")
# The Real-World Fix: Adding Brightness and Contrast variations
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
    layers.RandomTranslation(0.1, 0.1),
    layers.RandomBrightness(0.2),
    layers.RandomContrast(0.2)
])

base_model = tf.keras.applications.MobileNetV2(input_shape=(*IMG_SIZE, 3), include_top=False, weights="imagenet")
base_model.trainable = True
# Unfreeze the last 50 layers for heavy fine-tuning
for layer in base_model.layers[:-50]:
    layer.trainable = False

inputs = tf.keras.Input(shape=(*IMG_SIZE, 3))
x = data_augmentation(inputs)
x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.4)(x)
outputs = layers.Dense(len(class_names), activation="softmax")(x)

model = models.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5), # Low learning rate
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

early_stop = callbacks.EarlyStopping(patience=3, restore_best_weights=True)
checkpoint = callbacks.ModelCheckpoint(MODEL_OUT, save_best_only=True, monitor="val_accuracy")

print("\n🚀 Fine-tuning on Real-World Data...")
model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_FINE, callbacks=[early_stop, checkpoint])

print(f"\n🏁 V2 Training complete! Saved to {MODEL_OUT}")