import sys, shutil
import os

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from numpy.typing import NDArray
from typing import Generator, Tuple

from src.python.configs import *


def load_raw_fen_file(path: str) -> Generator[tuple[str, int], None, None]:
    """ 
    Reads CSV with fen,score and returns a generator of tuples.
    
    CSV file has format:
    fen,score\n
    fen,score\n
    ...
    fen,score\n
    
    score can be either an integer (e.g. 100), or a special character (e.g. #+5, #-2). 
    This means that the position has a score of "mate in 5 moves" or "mate in 2 moves". 
    """
    with open(path, 'r') as f:
        for line in f:
            # read line, split it into two values
            fen, score = line.split(',')
            try:
                # if score starts with '#', then it's a special score
                if score[0] == '#':
                    # if score is in format #+/-N, then it's a mate in N moves
                    if score[1] == '+':
                        yield fen, MATE_SCORE - int(score[1:])
                    elif score[1] == '-':
                        yield fen, - MATE_SCORE - int(score[1:])
                    else:
                        raise ValueError(
                            f"[ERROR] Unknown score format in note:\n{line}\n")
                # if score doesn't start with '#', then it's a regular score
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
    """Compute the index of the NNUE feature by combining the king, color, type, and square."""
    return KING_BASE[king_square] + TYPE_COLOR_BASE[piece_type][piece_color] + piece_square

def fen_to_indices(fen: str) -> NDArray[np.int32]:
    """Converts FEN to NNUE feature indices (2x32 np.int32)."""
    result = np.zeros(shape=(2, 32), dtype=np.int32) 
    piece_codes = []  # tuples of (color, type, square)
    kings = [-1, -1]
    side = 0
    counter = 0 
    square = 0 
    

    for let in fen:
        if counter > 32:
            raise ValueError("invalid FEN: too many pieces")
    
        if square < 64:
            match let:
                case '/':  # transition to new line on board
                    continue
                case 'k': # black king
                    kings[1] = square
                    square += 1
                    counter += 1
                    continue
                case 'K': # white king
                    kings[0] = square
                    square += 1
                    counter += 1
                    continue
                case _:            
                    if '0' < let < '9': # skip k squares
                        square += int(let)
                        continue
                    if let in alphabet.keys(): # letter of piece
                        piece_codes.append((*alphabet[let], square))
                        counter += 1
                        square += 1
                        continue
                    raise ValueError(f"invalid FEN: unknown letter - {let}")
        else:
            match let:
                case ' ':  # space before stm
                    continue
                case 'w': # stm - white
                    side = 0
                    break
                case 'b': # stm - black
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
        indexes: NDArray[np.int32], 
        score: int
) -> tuple[NDArray[np.int8], NDArray[np.int8], int]:
    """Forms a pair of feature vectors and a target score."""
    X1 = np.zeros(INPUT_VECTOR_SIZE, np.int8)
    X2 = np.zeros(INPUT_VECTOR_SIZE, np.int8)
    
    for i in range(32):
        if indexes[0][i] > 0:
            X1[indexes[0][i]] = 1
        if indexes[1][i] > 0:
            X2[indexes[1][i]] = 1

    return X1, X2, score

def write_to_batch(
        counter: int, 
        note: tuple[NDArray[np.int8], NDArray[np.int8], int],
        X1: NDArray[np.int8],
        X2: NDArray[np.int8],
        Y: NDArray[np.int32]
) -> int:
    """Adds an element to the batch at index counter."""
    if counter >= BATCH_SIZE:
        return -1

    X1[counter] = note[0]
    X2[counter] = note[1]
    Y[counter] = note[2]

    return 0

def save_batch_to_npz(
        X1: NDArray[np.int8],
        X2: NDArray[np.int8],
        Y: NDArray[np.int32], 
        npz_dir: str, 
        name: str,
        compressed: bool = True
) -> int:
    """
    Saves a batch of data to a .npz file.

    Args:
        X1, X2: feature vectors
        Y: target scores
        npz_dir: directory to save the file
        name: name of the file without extension
        compressed: if True, use compression (np.savez_compressed)

    Returns:
        0 if successful, -1 if error
    """
    filepath = f"{npz_dir}/chank_{name}.npz"
    try:
        if compressed:
            np.savez_compressed(filepath, X1=X1, X2=X2, Y=Y)
        else:
            np.savez(filepath, X1=X1, X2=X2, Y=Y)
        print(f"\n[INFO] Saved batch {name} ({X1.shape[0]} samples) → {filepath}\n")
        return 0
    except Exception as e:
        print(f"[ERROR] Failed to save batch {name}: {e}")
        return -1

def csv_to_npz(csv_path: str, npz_dir: str, BATCH_SIZE: int = 100_000) -> int:
    """Reads a CSV file with FEN and scores, converts it to a batch of NPZ files."""
    X1_batch = np.zeros((BATCH_SIZE, INPUT_VECTOR_SIZE), np.int8)
    X2_batch = np.zeros((BATCH_SIZE, INPUT_VECTOR_SIZE), np.int8)
    Y_batch = np.zeros(BATCH_SIZE, np.int32)

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


# 2. Dataset-объект
class Dataset():
    def __init__(self, npz_dir: str) -> None:
        """
        Args:
            npz_dir (str): Path to the directory with .npz files.
        """
        self.npz_dir = npz_dir
        self.npz_files = sorted([f for f in os.listdir(npz_dir) if f.endswith('.npz')])

        if not self.npz_files:
            raise ValueError(f"No .npz files found in directory {npz_dir}")

        self.total_size = len(self.npz_files)

    def __len__(self) -> int:
        return self.total_size

    def __geеindex__(self, idx: int) -> tuple: # TODO typing
 
        if idx < 0 or idx >= self.total_size:
            raise IndexError(f"Index {idx} is out of range for dataset with size {self.total_size}")

        filepath = os.path.join(self.npz_dir, self.npz_files[idx])
        data = np.load(filepath)

        x1 = data['X1']
        x2 = data['X2']
        y = data['Y']

        return x1, x2, y

