import hashlib
from pathlib import Path
from collections import defaultdict

DATA_DIR = Path("data/v1")

def get_file_hash(filepath):
    """Generate an MD5 hash of the image file's bytes."""
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

def check_leakage():
    print("🕵️  Starting Data Leakage Check (MD5 Hashing)...")
    print("This checks if exact duplicate images exist in both Training and Test sets.\n")

    classes = ["Apple", "Banana", "Mango", "Orange", "Strawberry"]

    total_leaks = 0
    total_test_images = 0

    for cls in classes:
        train_dir = DATA_DIR / "Training" / cls
        test_dir = DATA_DIR / "Test" / cls

        if not train_dir.exists() or not test_dir.exists():
            print(f"⚠️ Missing folder for {cls}")
            continue

        print(f"🔍 Checking class: {cls}...")

        # 1. Hash all training images
        train_hashes = set()
        for img in train_dir.glob("*.jpg"):
            train_hashes.add(get_file_hash(img))

        # 2. Hash all test images and check for matches
        test_hashes = {}
        for img in test_dir.glob("*.jpg"):
            img_hash = get_file_hash(img)
            test_hashes[img.name] = img_hash

        # 3. Compare
        leaks_in_class = 0
        for img_name, img_hash in test_hashes.items():
            if img_hash in train_hashes:
                leaks_in_class += 1

        total_test_images += len(test_hashes)
        total_leaks += leaks_in_class

        percent = (leaks_in_class / len(test_hashes)) * 100 if test_hashes else 0
        print(f"   Found {leaks_in_class:4d} exact matches out of {len(test_hashes):4d} test images ({percent:.1f}%)")

    print("\n" + "="*50)
    print(f"🚨 TOTAL EXACT DUPLICATES FOUND: {total_leaks} out of {total_test_images} test images.")
    print("="*50)

if __name__ == "__main__":
    check_leakage()