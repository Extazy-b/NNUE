from numpy import zeros, int32, int8, savez, savez_compressed
import torch
from numpy.typing import NDArray
from typing import Generator
import sys, shutil

from src.python.configs import *

def load_raw_fen_file(path: str) -> Generator[tuple[str, int], None, None]:
    """Построчно читает CSV с fen,score и возвращает генератор кортежей."""
    with open(path, 'r') as f:
        for line in f:
            fen, score = line.split(',')
            try:
                if score[0] == '#':
                    if score[1] == '+':
                        yield fen, MATE_SCORE - int(score[1:])
                    elif score[1] == '-':
                        yield fen, - MATE_SCORE - int(score[1:])
                    else:
                        raise ValueError(f"[ERROR] Unknown score format in note:\n{line}\n")
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

    counter = 0
    name = 0
    name_size = len(str(DATASET_SIZE // BATCH_SIZE + 1))

    dataset = load_raw_fen_file(csv_path)

    width = round(shutil.get_terminal_size((80, 20)).columns * 0.8)  # 80% ширины терминала
    processed = 0

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
class NNUEDataset(torch.utils.data.Dataset):
    """Dataset, читающий готовые NPZ и выдающий тензоры."""
    def __init__(self, npz_files: list[str]):
        # - сохранить список путей к NPZ
        # - можно загрузить все в память или читать по запросу
        pass
    
    def __len__(self):
        # - вернуть суммарное количество позиций
        pass
    
    def __getitem__(self, idx):
        # - найти из какого NPZ брать данные
        # - вернуть torch.tensor(X1), torch.tensor(X2), torch.tensor(y)
        pass


# 3. Утилита для создания DataLoader
def make_dataloader(npz_dir: str, BATCH_SIZE: int, shuffle: bool = True):
    """Сканирует папку с NPZ, возвращает DataLoader."""
    # - найти все *.npz
    # - создать NNUEDataset
    # - обернуть в torch.utils.data.DataLoader
    pass
