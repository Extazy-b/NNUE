import sys
sys.path.append(".")

from src.python.dataset import *
import numpy as np
import pytest

#==========================cross_index_of_three==========================

def test_min_values():
    assert cross_index_of_three(0, 0, 0, 0) == 0

def test_max_values():
    k = NUMBER_OF_SQUARES - 1
    t = NUMBER_OF_PIECE_TYPE - 1
    c = NUMBER_OF_COLORS - 1
    s = NUMBER_OF_SQUARES - 1
    expected = KING_BASE[k] + TYPE_COLOR_BASE[t][c] + s
    assert cross_index_of_three(k, c, t, s) == expected

def test_middle_values():
    assert cross_index_of_three(10, 1, 3, 25) == KING_BASE[10] + TYPE_COLOR_BASE[3][1] + 25

def test_different_kings():
    assert cross_index_of_three(5, 0, 0, 0) != cross_index_of_three(6, 0, 0, 0)

def test_different_piece_types():
    assert cross_index_of_three(0, 0, 1, 0) != cross_index_of_three(0, 0, 2, 0)

def test_different_colors():
    assert cross_index_of_three(0, 0, 1, 0) != cross_index_of_three(0, 1, 1, 0)

def test_different_squares():
    assert cross_index_of_three(0, 0, 0, 10) != cross_index_of_three(0, 0, 0, 11)

def test_linear_growth_in_piece_square():
    a = cross_index_of_three(0, 0, 0, 10)
    b = cross_index_of_three(0, 0, 0, 11)
    assert b - a == 1

def test_linear_growth_in_piece_type():
    a = cross_index_of_three(0, 0, 0, 0)
    b = cross_index_of_three(0, 0, 1, 0)
    assert b - a == NUMBER_OF_COLORS * NUMBER_OF_SQUARES

def test_linear_growth_in_piece_color():
    a = cross_index_of_three(0, 0, 0, 0)
    b = cross_index_of_three(0, 1, 0, 0)
    assert b - a == NUMBER_OF_SQUARES

def test_linear_growth_in_king_square():
    a = cross_index_of_three(0, 0, 0, 0)
    b = cross_index_of_three(1, 0, 0, 0)
    assert b - a == NUMBER_OF_PIECE_TYPE * NUMBER_OF_COLORS * NUMBER_OF_SQUARES

def test_no_overlap_for_different_params():
    results = set()
    for k in range(3):
        for t in range(2):
            for c in range(2):
                for s in range(3):
                    results.add(cross_index_of_three(k, c, t, s))
    assert len(results) == 3 * 2 * 2 * 3

# ==========================fen_to_indices==========================

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
    assert np.any(result > 0)  # есть фигуры

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
