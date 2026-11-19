import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import json, shutil
from pathlib import Path
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

# ---------- CONFIG ----------
IMG_SIZE   = (224, 224)
BATCH_SIZE = 32
EPOCHS_FINE = 8  # Increased epochs slightly to handle the larger dataset
DATA_DIR   = Path("data/v1")
REAL_DIR   = Path("data/real_world_v2")
ARTIFACT_DIR = Path("artifacts")
MODEL_OUT  = ARTIFACT_DIR / "fruit_model_v3.keras"

print("📂 Merging 6,000+ Real-World images into Training set...")
for cls in ["Apple", "Banana", "Mango", "Orange", "Strawberry"]:
    real_cls_dir = REAL_DIR / cls
    train_cls_dir = DATA_DIR / "Training" / cls

    # 1. Clean up old V3 real images (from the aborted run)
    for f in train_cls_dir.glob("real_v3_*"):
        f.unlink()

    # 2. Copy in the new V3 images
    if real_cls_dir.exists():
        count = 0
        for img in real_cls_dir.glob("*.*"):
            if img.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                new_name = f"real_v3_{count}{img.suffix}"
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

# ---------- CALCULATE CLASS WEIGHTS ----------
print("\n⚖️ Calculating Class Weights to handle imbalance...")
# Extract labels from the training dataset
y_train = np.concatenate([y for x, y in train_ds], axis=0)
y_train_classes = np.argmax(y_train, axis=1)

weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train_classes),
    y=y_train_classes
)
class_weights_dict = dict(enumerate(weights))
print(f"   Class weights: {class_weights_dict}")

train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

# ---------- BUILD MODEL ----------
print("\n🏗️ Building Model with Heavy Augmentation...")
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
# Unfreeze the last 60 layers (we have more data now, so we can fine-tune deeper)
for layer in base_model.layers[:-60]:
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

print("\n🚀 Fine-tuning on the Massive Balanced Dataset...")
# Notice we pass class_weight here!
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_FINE,
    callbacks=[early_stop, checkpoint],
    class_weight=class_weights_dict
)

print(f"\n🏁 V3 Training complete! Saved to {MODEL_OUT}")