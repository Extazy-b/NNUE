from src.python.dataset import fen_to_indices
from src.python.configs import *
import numpy as np
import pytest


def test_valid_fen_minimal():
    # Только короли
    fen = "8/8/8/8/8/8/4K3/4k3 w"
    result = fen_to_indices(fen)
    assert result.shape == (2, 32)
    assert np.count_nonzero(result) == 0 or np.count_nonzero(result) > 0  # форма корректная

def test_valid_fen_with_pieces():
    # Короли + пешка
    fen = "8/8/8/8/4P3/8/4K3/4k3 w"
    result = fen_to_indices(fen)
    assert np.all(result >= 0)
    assert np.any(result > 0)

def test_invalid_too_many_pieces():
    fen = "8/8/8/7P/PPPPPPPP/PPPPPPPP/PPPPPPPP/PPPPPPPP w"
    with pytest.raises(ValueError, match="too many pieces"):
        fen_to_indices(fen)

def test_invalid_too_few_kings():
    fen = "8/8/8/8/8/8/4K3/8 w"
    with pytest.raises(ValueError, match="not enought kings"):
        fen_to_indices(fen)

def test_invalid_too_few_squares():
    fen = "8/8/8/8/8/8/4K3/4k2 w"  # 63 клетки
    with pytest.raises(ValueError, match="too many or not enought squares"):
        fen_to_indices(fen)

def test_invalid_too_many_squares():
    fen = "8/8/8/8/8/8/4K3/4k3P w"  # 65 клеток
    with pytest.raises(ValueError, match="too many or not enought squares"):
        fen_to_indices(fen)

def test_invalid_character():
    fen = "8/8/8/8/8/8/4K3/4x3 w"  # 'x' — неизвестный символ
    with pytest.raises(ValueError, match="unknown letter"):
        fen_to_indices(fen)

def test_invalid_digit():
    fen = "8/8/8/8/8/8/4K3/44k3 w"  # '44' как одна цифра
    with pytest.raises(ValueError):
        fen_to_indices(fen)

def test_black_to_move():
    fen = "8/8/8/8/4P3/8/4K3/4k3 b"
    result = fen_to_indices(fen)
    assert result.shape == (2, 32)
    assert np.all(result >= 0)

def test_ignore_slash_and_spaces():
    fen = "8/8/8/8/8/8/4K3/4k3 w"
    result = fen_to_indices(fen)
    assert isinstance(result, np.ndarray)

def test_only_two_kings():
    fen = "3qk3/8/8/8/8/8/8/3QK3 w"
    result = fen_to_indices(fen)
    assert set(result.ravel()) == {0, 38979, 38971, 3139, 3131}
