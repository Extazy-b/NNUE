from numpy import zeros, int32, int8, savez
import torch

from src.python.configs import *

from numpy.typing import  NDArray
from typing import Generator

def load_raw_fen_file(path: str) -> Generator[tuple[str, int], None, None]:
    with open(path, 'r') as f:
        for line in f:
            fen, score = line.strip().split(',')
            yield fen, int(score)

def cross_index_of_three(
        king_square: int, 
        piece_color: int, 
        piece_type: int, 
        piece_square: int
) -> int:
    
    return KING_BASE[king_square] + TYPE_COLOR_BASE[piece_type][piece_color] + piece_square

def fen_to_indices(fen: str) -> NDArray:
    result = zeros(shape=(2, 32), dtype=int32)
    piece_codes = [] # кортеж из (цвет фигуры, тип фигуры, клетка фигуры)
    kings = [-1, -1]
    side = 0
    counter = 0 # количество обнаруженных фигур (максимум - 32)
    square = 0 # код рассматриваемой кледки
    for let in fen:
        if counter > 32:
            raise ValueError("invalid FEN: too many pieces")

        match let:
            case '/': # скачек на следующую клетку
                continue
            case ' ': # пробел перед буквой цвета stm
                continue
            case 'w': # stm - белые
                side = 0
                break
            case 'b': # stm - черные
                side = 1
                break
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
                if '0' < let < '9': # скачек через k клеток
                    square += int(let)
                    continue
                if let in alphabet.keys(): #буква фигуры
                    piece_codes.append((*alphabet[let], square))
                    counter += 1
                    square += 1
                    continue
                raise ValueError(f"invalid FEN: unknown letter - {let}")
            
        print(let)
        print(c)

    if square != 64:
        raise ValueError(f"invalid FEN: too many or not enought squares - {square}")

    if (kings[0] == -1) or (kings[1] == -1):
        raise ValueError("invalid FEN: not enought kings")

    i = 0 
    for combination in piece_codes:
        result[0][i] = (cross_index_of_three(kings[side], *combination))
        result[1][i] = (cross_index_of_three(kings[1 - side], *combination))
        i+= 1

    return result

def make_note(
        indexes: NDArray[int32], 
        score: int
) -> tuple[NDArray[int8], NDArray[int8],  int]:
    
    #TODO
    return (zeros(INPUT_VECTOR_SIZE, int8), zeros(INPUT_VECTOR_SIZE, int8), 0)

def write_to_batch(
        counter: int, 
        note: tuple[NDArray[int8], NDArray[int8],  int],
        X1: NDArray[int8],
        X2: NDArray[int8],
        Y: NDArray[int32]
) -> int:
    #TODO
    if counter == BATCH_SIZE:
        return -1
    return 0

def save_batch_to_npz(
        X1: NDArray[int8],
        X2: NDArray[int8],
        Y: NDArray[int32], 
        npz_dir: str, 
        name:str
) -> int:
    
    try:
        savez(f"{npz_dir}chank_{name}.npz",
              X1, X2, Y)
        return 0
    except:
        return -1
    
# 1. Конвертация CSV → NPZ батчами
def csv_to_npz(csv_path: str, npz_dir: str, BATCH_SIZE: int = 100_000) -> int:
    """Читает CSV с FEN и оценками, конвертирует в батчи NPZ.

    csv-filename -> read line -> get fen and score -> get two indexes -> make two vectors -> combine vectors and score ->
    -> add 2-d vector to nd.array -> save nd.array to file
    """

    X1_batch = zeros((INPUT_VECTOR_SIZE, BATCH_SIZE), int8)
    X2_batch = zeros((INPUT_VECTOR_SIZE, BATCH_SIZE), int8)
    Y_batch = zeros(BATCH_SIZE, int32)

    counter = 0
    name = 0
    name_size = len(str(DATASET_SIZE//BATCH_SIZE + 1))

    dataset = load_raw_fen_file(csv_path)

    for fen, score in dataset:
        indexes = fen_to_indices(fen)
        note = make_note(indexes, score)
        
        if write_to_batch(counter, note, X1_batch, X2_batch, Y_batch) == -1:
            if save_batch_to_npz(X1_batch, X2_batch, Y_batch, npz_dir,  str(name).zfill(name_size)) != 0:
                pass #TODO alarm
        else:
            counter += 1
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
