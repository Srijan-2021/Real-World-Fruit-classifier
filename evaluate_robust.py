import tensorflow as tf
from pathlib import Path

# ---------- CONFIG ----------
ARTIFACT_DIR = Path("artifacts")
MODEL_PATH = ARTIFACT_DIR / "fruit_model_v1.keras"
TEST_DIR = Path("data/v1/Test")
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

print("📂 Loading original test data...")
test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
    label_mode="categorical"  # <--- THIS IS THE FIX
)

print("🧠 Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)

def apply_real_world_noise(image, label):
    # 1. Random Brightness
    image = tf.image.random_brightness(image, max_delta=0.4)

    # 2. Random Contrast
    image = tf.image.random_contrast(image, lower=0.6, upper=1.4)

    # 3. Random Blur (Downsample and Upsample)
    image = tf.image.resize(image, (56, 56))
    image = tf.image.resize(image, IMG_SIZE)

    # Ensure pixel values stay valid
    image = tf.clip_by_value(image, 0.0, 255.0)

    return image, label

# Apply the noise to the test dataset
robust_test_ds = test_ds.map(apply_real_world_noise).prefetch(tf.data.AUTOTUNE)

print("\n🧪 Evaluating model on Harsh/Noisy Test Set...")
loss, accuracy = model.evaluate(robust_test_ds)

print("\n" + "="*50)
print(f"📉 Robustness Accuracy: {accuracy * 100:.2f}%")
print("="*50)