from numpy import zeros, int32, int8, savez, savez_compressed
import torch
from numpy.typing import NDArray
from typing import Generator
import sys, shutil

from src.python.configs import *

def load_raw_fen_file(path: str) -> Generator[tuple[str, int], None, None]:
    """ 
    Построчно читает CSV с fen,score и возвращает генератор кортежей.
    
    CSV файл имеет формат:
    fen,score\n
    fen,score\n
    ...
    fen,score\n
    
    score может быть как целым числом (например, 100), так и 
    специальными символами #+/- (например, #+5, #-2). 
    Это означает, что позиция имеет оценку "мат в 5 ходов" 
    или "мат в 2 хода". 
    """
    with open(path, 'r') as f:
        for line in f:
            # считываем строку, разбиваем ее на два значения
            fen, score = line.split(',')
            try:
                # если score начинается на '#', то это special score
                if score[0] == '#':
                    # если score имеет вид #+/-N, то мат в N ходов
                    if score[1] == '+':
                        yield fen, MATE_SCORE - int(score[1:])
                    elif score[1] == '-':
                        yield fen, - MATE_SCORE - int(score[1:])
                    else:
                        raise ValueError(
                            f"[ERROR] Unknown score format in note:\n{line}\n")
                # если score не начинается на '#', то он обычный
                else:
                    yield fen, int(score)
            except ValueError as e:
                print(f"[ERROR] Could not read note: \n{line}\n Cause {e}")

def cross_index_of_three(
        king_square: int, 
        piece_color: int, 
        piece_type: int, 
        piece_square: int
) -> int:
    """Вычисляет индекс признака NNUE по комбинации: (король, цвет, тип, клетка)."""
    return KING_BASE[king_square] + TYPE_COLOR_BASE[piece_type][piece_color] + piece_square

def fen_to_indices(fen: str) -> NDArray[int32]:
    """Конвертирует FEN в индексы NNUE признаков (2x32 int32)."""
    result = zeros(shape=(2, 32), dtype=int32) 
    piece_codes = []  # кортежи (цвет фигуры, тип фигуры, клетка)
    kings = [-1, -1]
    side = 0
    counter = 0 
    square = 0 
    

    for let in fen:
        if counter > 32:
            raise ValueError("invalid FEN: too many pieces")
    
        if square < 64:
            match let:
                case '/':  # переход на новую строку доски
                    continue
                case 'k': # черный король
                    kings[1] = square
                    square += 1
                    counter += 1
                    continue
                case 'K': # белый король
                    kings[0] = square
                    square += 1
                    counter += 1
                    continue
                case _:            
                    if '0' < let < '9': # скачок через k клеток
                        square += int(let)
                        continue
                    if let in alphabet.keys(): #буква фигуры
                        piece_codes.append((*alphabet[let], square))
                        counter += 1
                        square += 1
                        continue
                    raise ValueError(f"invalid FEN: unknown letter - {let}")
        else:
            match let:
                case ' ':  # пробел перед stm
                    continue
                case 'w': # stm - белые
                    side = 0
                    break
                case 'b': # stm - черные
                    side = 1
                    break
                case _:
                    raise ValueError(f"invalid FEN: unknown letter - {let}")
    
    if square != 64:
        raise ValueError(f"{fen}\ninvalid FEN: wrong board size, got {square} squares")
    if (kings[0] == -1) or (kings[1] == -1):
        raise ValueError("invalid FEN: missing king(s)")
    for i, combination in enumerate(piece_codes):
        result[0][i] = cross_index_of_three(kings[side], *combination)
        result[1][i] = cross_index_of_three(kings[1 - side], *combination)

    return result

def make_note(
        indexes: NDArray[int32], 
        score: int
) -> tuple[NDArray[int8], NDArray[int8], int]:
    """Формирует пару векторов признаков и метку (оценку)."""
    X1 = zeros(INPUT_VECTOR_SIZE, int8)
    X2 = zeros(INPUT_VECTOR_SIZE, int8)
    
    for i in range(32):
        if indexes[0][i] > 0:
            X1[indexes[0][i]] = 1
        if indexes[1][i] > 0:
            X2[indexes[1][i]] = 1

    return X1, X2, score

def write_to_batch(
        counter: int, 
        note: tuple[NDArray[int8], NDArray[int8], int],
        X1: NDArray[int8],
        X2: NDArray[int8],
        Y: NDArray[int32]
) -> int:
    """Добавляет элемент в батч по индексу counter."""
    if counter >= BATCH_SIZE:
        return -1

    X1[counter] = note[0]
    X2[counter] = note[1]
    Y[counter] = note[2]

    return 0

def save_batch_to_npz(
        X1: NDArray[int8],
        X2: NDArray[int8],
        Y: NDArray[int32], 
        npz_dir: str, 
        name: str,
        compressed: bool = True
) -> int:
    """
    Сохраняет батч в NPZ. 

    Аргументы:
        X1, X2: массивы признаков
        Y: оценки
        npz_dir: папка для сохранения
        name: имя файла без расширения
        compressed: если True, использовать сжатие (np.savez_compressed)

    Возвращает:
        0 при успехе, -1 при ошибке
    """
    filepath = f"{npz_dir}/chank_{name}.npz"
    try:
        if compressed:
            savez_compressed(filepath, X1=X1, X2=X2, Y=Y)
        else:
            savez(filepath, X1=X1, X2=X2, Y=Y)
        print(f"\n[INFO] Saved batch {name} ({X1.shape[0]} samples) → {filepath}\n")
        return 0
    except Exception as e:
        print(f"[ERROR] Failed to save batch {name}: {e}")
        return -1

def csv_to_npz(csv_path: str, npz_dir: str, BATCH_SIZE: int = 100_000) -> int:
    """Читает CSV с FEN и оценками, конвертирует в батчи NPZ."""
    X1_batch = zeros((BATCH_SIZE, INPUT_VECTOR_SIZE), int8)
    X2_batch = zeros((BATCH_SIZE, INPUT_VECTOR_SIZE), int8)
    Y_batch = zeros(BATCH_SIZE, int32)

    name = 0
    name_size = len(str(DATASET_SIZE // BATCH_SIZE + 1))

    dataset = load_raw_fen_file(csv_path)

    width = round(shutil.get_terminal_size((80, 20)).columns * 0.8)  # 80% ширины терминала
    processed = 0
    counter = 0

    for fen, score in dataset:
        try:
            indexes = fen_to_indices(fen)
            note = make_note(indexes, score)
        except ValueError as e:
            print(f"[WARNING] Skipped invalid FEN:\n{fen}\n{e}")
            continue
        except Exception as e:
            print(f"[ERROR] Could not parse fen:\n{fen}\n{e}")
            continue
        
        if write_to_batch(counter, note, X1_batch, X2_batch, Y_batch) == -1:
            if save_batch_to_npz(X1_batch, X2_batch, Y_batch, npz_dir, str(name).zfill(name_size)) == 0:
                name += 1
                counter = 0  # сбрасываем счётчик после сохранения
            else:
                print(f"[ERROR] Could not save batch {name}")
        else:
            counter += 1

        # обновляем прогресс каждые 50k строк
        processed += 1
        if processed % 10_000 == 0:
            done = int(width * processed / DATASET_SIZE)
            sys.stdout.write(
                f"\r[{'#' * done}{'.' * (width - done)}] "
                f"{processed}/{DATASET_SIZE} ({processed*100//DATASET_SIZE}%)"
            )
            sys.stdout.flush()

    # сохраняем хвост, если он неполный
    if counter > 0:
        save_batch_to_npz(
            X1_batch[:counter], X2_batch[:counter], Y_batch[:counter],
            npz_dir, str(name).zfill(name_size)
        )

    # финальный прогресс
    sys.stdout.write(f"\nDone: {processed} positions converted\n")

    return 0


# 2. Dataset-объект для PyTorch
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List

class NNUEDataset(Dataset):
    """Dataset для загрузки батчей .npz и предоставления данных моделье."""

    def __init__(self, npz_dir: str):
        """
        Args:
            npz_dir (string): Путь к директории с .npz файлами.
        """
        self.npz_dir = npz_dir
        self.npz_files = sorted([f for f in os.listdir(npz_dir) if f.endswith('.npz')])
        
        if not self.npz_files:
            raise ValueError(f"No .npz files found in directory {npz_dir}")

        # Проверяем, что все файлы имеют размер BATCH_SIZE
        self._validate_file_sizes()

        self.total_size = BATCH_SIZE * len(self.npz_files)
        
        # Переменные для кэширования
        self.cached_file_index = 0 # Индекс закэшированного файла
        # Ленивая загрузка: грузим первый файл только при первом обращении
        self.cached_data = np.load(os.path.join(self.npz_dir, self.npz_files[self.cached_file_index]))

    def _validate_file_sizes(self):
        """Проверяет, что все .npz файлы имеют ровно BATCH_SIZE записей."""
        for filename in self.npz_files:
            filepath = os.path.join(self.npz_dir, filename)
            with np.load(filepath) as data:
                actual_size = data['Y'].shape[0]
                if actual_size != BATCH_SIZE:
                    raise ValueError(
                        f"File {filename} has {actual_size} samples, "
                        f"but expected {BATCH_SIZE}. Dataset consistency check failed."
                    )
        print(f"[INFO] All {len(self.npz_files)} files passed size validation ({BATCH_SIZE} samples each).")
    def __len__(self) -> int:
        return self.total_size

    def _load_file(self, file_index: int):
        """Загружает файл в кэш, если он еще не загружен."""
        if self.cached_file_index != file_index:
            filepath = os.path.join(self.npz_dir, self.npz_files[file_index])
            self.cached_data = np.load(filepath)
            self.cached_file_index = file_index

    def __getitem__(self, idx: int):
        """
        Возвращает один пример из датасета.
        Args:
            idx (int): Глобальный индекс примера (от 0 до len(dataset)-1).
        Returns:
            tuple: (x1, x2, y) где x1, x2 - тензоры признаков, y - тензор целевой оценки.
        """
        if idx < 0 or idx >= self.total_size:
            raise IndexError(f"Index {idx} is out of range for dataset with size {self.total_size}")

        # 1. Находим номер файла, в котором лежит искомый пример
        # idx = file_index * BATCH_SIZE + local_idx
        # file_index = idx // BATCH_SIZE
        file_index = idx // BATCH_SIZE
        
        # 2. Находим локальный индекс внутри этого файла
        local_idx = idx % BATCH_SIZE

        # 3. Загружаем нужный файл (используется кэш)
        self._load_file(file_index)

        # 4. Достаем данные и преобразуем в тензоры
        x1 = torch.from_numpy(self.cached_data['X1'][local_idx].astype(np.float32))
        x2 = torch.from_numpy(self.cached_data['X2'][local_idx].astype(np.float32))
        y = torch.tensor(self.cached_data['Y'][local_idx], dtype=torch.float32)

        return x1, x2, y

    def __del__(self):
        """Закрываем файл при удалении датасета для избежания утечек."""
        if self.cached_data is not None:
            self.cached_data.close()


# 3. Утилита для создания DataLoader
def make_dataloader(npz_dir: str, batch_size: int = BATCH_SIZE, shuffle: bool = True, **kwargs) -> DataLoader:
    """
    Создает DataLoader для готовых NPZ-батчей.

    Args:
        npz_dir: Путь к директории с .npz файлами.
        batch_size: Размер батча.
        shuffle: Перемешивать ли данные.
        **kwargs: Дополнительные аргументы для DataLoader (например, num_workers).

    Returns:
        DataLoader: Объект DataLoader для итерации по данным.
    """
    dataset = NNUEDataset(npz_dir)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)
