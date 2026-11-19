"""
STEP 1 — Environment verification (inside .venv)
"""
import sys, shutil, importlib.metadata as md
from pathlib import Path

print("=" * 64)
print("CHECK 1: Python interpreter")
print("=" * 64)
print(f"  Version: {sys.version.split()[0]}")
print(f"  Path:    {sys.executable}")
venv_marker = ".venv" in sys.executable.lower()
print(f"  ✅ Running from .venv" if venv_marker else "  ❌ NOT from .venv — STOP")

print("\n" + "=" * 64)
print("CHECK 2: Installed package versions")
print("=" * 64)
REQUIRED = [
    ("tensorflow",   "2.16.0", "2.17.99"),
    ("keras",        "3.0.0",  "3.4.99"),
    ("streamlit",    "1.40.0", "1.99.99"),
    ("pillow",       "10.0.0", "11.99.99"),
    ("numpy",        "1.24.0", "1.26.99"),
    ("pandas",       "2.0.0",  "2.99.99"),
    ("matplotlib",   "3.7.0",  "4.99.99"),
    ("scikit-learn", "1.3.0",  "2.99.99"),
]
installed = {d.metadata["Name"].lower(): d.version for d in md.distributions()}
for name, lo, hi in REQUIRED:
    v = installed.get(name.lower())
    if not v:
        print(f"  ❌ {name:15s} NOT INSTALLED")
        continue
    try:
        from packaging.version import Version
        ok = Version(lo) <= Version(v) <= Version(hi)
        print(f"  {'✅' if ok else '⚠️ '} {name:15s} {v}")
    except Exception:
        print(f"  ℹ️  {name:15s} {v}")

print("\n" + "=" * 64)
print("CHECK 3: TensorFlow import smoke test")
print("=" * 64)
try:
    import tensorflow as tf, numpy as np
    print(f"  ✅ TF: {tf.__version__}  |  Keras: {tf.keras.__version__}")
    print(f"  ✅ Basic op: {(tf.constant([1.0,2.0,3.0])*2).numpy()}")
    print(f"  ℹ️  GPUs visible: {len(tf.config.list_physical_devices('GPU'))} (0 is fine — CPU mode)")
except Exception as e:
    print(f"  ❌ TF FAILED: {type(e).__name__}: {e}")

print("\n" + "=" * 64)
print("CHECK 4: Other key imports")
print("=" * 64)
for name, imp in [("streamlit","streamlit"), ("PIL","PIL"),
                  ("pandas","pandas"), ("matplotlib","matplotlib"),
                  ("sklearn","sklearn")]:
    try:
        m = __import__(imp)
        print(f"  ✅ {name} imported — {getattr(m, '__version__', '?')}")
    except Exception as e:
        print(f"  ❌ {name} FAILED: {type(e).__name__}: {e}")

print("\n" + "=" * 64)
print(f"Disk free: {shutil.disk_usage(Path.cwd()).free / 1024**3:.1f} GB")
print("🏁 DONE — paste this whole output back")
print("=" * 64)