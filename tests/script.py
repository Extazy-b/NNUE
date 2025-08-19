import numpy as np

# подставь путь к твоему файлу
data = np.load("data/processed/chank_000.npz")

print("Содержимое архива:", data.files)  # ['X1', 'X2', 'Y']
print("Форма X1:", data["X1"].shape)
print("Форма X2:", data["X2"].shape)
print("Форма Y:", data["Y"].shape)

# например, первые 5 элементов
print("Пример Y:", data["Y"][:5])
