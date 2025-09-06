import os
import numpy as np

npz_dir = "./data/processed/train"
for f in sorted(os.listdir(npz_dir)):
    if f.endswith(".npz"):
        path = os.path.join(npz_dir, f)
        try:
            with np.load(path) as data:
                _ = data["X1"].shape
            print(f"✅ OK {f}")
        except Exception as e:
            print(f"❌ FAIL {f} → {e}")
