import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import json
from pathlib import Path

# ---------- CONFIG ----------
ARTIFACT_DIR = Path("artifacts")
MODEL_PATH = ARTIFACT_DIR / "fruit_model_v1.keras"
LABELS_PATH = ARTIFACT_DIR / "class_indices.json"
TEST_DIR = Path("data/v1/Test")
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

print("📂 Loading test data...")
test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
    label_mode="categorical"  # <--- THE FIX
)

# Load class names
with open(LABELS_PATH, "r") as f:
    class_indices = json.load(f)
class_names = list(class_indices.keys())

print("🧠 Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)

print("📊 Evaluating model on Test set...")
loss, accuracy = model.evaluate(test_ds)
print(f"\n✅ Test Accuracy: {accuracy * 100:.2f}%")
print(f"✅ Test Loss: {loss:.4f}\n")

# Get predictions
print("🔮 Generating predictions...")
y_pred_probs = model.predict(test_ds)
y_pred = np.argmax(y_pred_probs, axis=1)

# Get true labels (converting from one-hot arrays back to single integers)
y_true_one_hot = np.concatenate([y for x, y in test_ds], axis=0)
y_true = np.argmax(y_true_one_hot, axis=1)

# Print Classification Report
print("\n" + "="*50)
print("CLASSIFICATION REPORT")
print("="*50)
report = classification_report(y_true, y_pred, target_names=class_names)
print(report)

# Generate and Save Confusion Matrix
print("🎨 Generating Confusion Matrix...")
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix - Test Set')
plt.colorbar()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names, rotation=45)
plt.yticks(tick_marks, class_names)

# Add numbers to the boxes
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig(ARTIFACT_DIR / "confusion_matrix.png")
print(f"✅ Confusion matrix saved to {ARTIFACT_DIR / 'confusion_matrix.png'}")