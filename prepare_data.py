import os
import shutil
from pathlib import Path

# ---------- CONFIG ----------
RAW_DIR = Path("data/raw/fruits-360")
OUT_DIR = Path("data/v1")

# Map our 5 target fruits to folder name patterns present in Fruits-360
TARGETS = {
    "Apple":      ["Apple Red 1", "Apple Golden 1"], # Merging two apple types into one class
    "Banana":     ["Banana 1"],
    "Orange":     ["Orange 1"],
    "Strawberry": ["Strawberry 1"],
    "Mango":      ["Mango 1"]
}

def prepare_data():
    print("🚀 Starting data preparation...")

    if not RAW_DIR.exists():
        print(f"❌ Error: Could not find {RAW_DIR}")
        print("Please make sure you extracted the Kaggle dataset correctly.")
        return

    for split in ["Training", "Test"]:
        print(f"\n📂 Processing {split} split...")
        split_src = RAW_DIR / split

        for fruit_class, patterns in TARGETS.items():
            dest_folder = OUT_DIR / split / fruit_class
            dest_folder.mkdir(parents=True, exist_ok=True)

            count = 0
            for pattern in patterns:
                src_folder = split_src / pattern
                if not src_folder.exists():
                    print(f"  ⚠️  Warning: Folder '{pattern}' not found in {split}. Skipping.")
                    continue

                # Copy all images
                for img in src_folder.glob("*.jpg"):
                    shutil.copy2(img, dest_folder / img.name)
                    count += 1

            print(f"  ✅ {fruit_class:12s}: {count:4d} images copied to {dest_folder}")

    print("\n🏁 Data preparation complete!")
    print(f"Data is ready at: {OUT_DIR.absolute()}")

if __name__ == "__main__":
    prepare_data()