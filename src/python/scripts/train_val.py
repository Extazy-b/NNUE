import os
import shutil
import random

# Пути
base_dir = "./data/processed"
train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "val")

# Создаём папки если нет
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# Все npz файлы
all_files = [f for f in os.listdir(base_dir) if f.endswith(".npz")]

# Перемешаем для случайного выбора
random.shuffle(all_files)

# Разделение (90% train, 10% val)
split_idx = int(len(all_files) * 0.9)
train_files = all_files[:split_idx]
val_files = all_files[split_idx:]

# Перемещаем
for f in train_files:
    shutil.move(os.path.join(base_dir, f), os.path.join(train_dir, f))

for f in val_files:
    shutil.move(os.path.join(base_dir, f), os.path.join(val_dir, f))

print(f"✅ Разделено {len(all_files)} файлов → {len(train_files)} train / {len(val_files)} val")
