from src.python.configs import *
from src.python.dataset import make_note

import numpy as np
from random import randint

def test_empty_note():
    indexes = np.zeros((2, 32), np.int32)
    score = 0
    result = make_note(indexes, score)

    assert np.all(result == (np.zeros(INPUT_VECTOR_SIZE, np.int8), np.zeros(INPUT_VECTOR_SIZE, np.int8), 0)) #FIXME check pytest


def test_empty_note_with_score():
    indexes = np.zeros((2, 32), np.int32)
    score = 10
    result = make_note(indexes, score)

    assert np.all(result == (np.zeros(INPUT_VECTOR_SIZE, np.int8), np.zeros(INPUT_VECTOR_SIZE, np.int8), 10)) #FIXME check pytest


def test_random_index():
    ind = randint(0, INPUT_VECTOR_SIZE - 1)

    indexes = np.zeros((2, 32), np.int32)
    score = 0

    indexes[0][0] = ind

    result = make_note(indexes, score)

    assert np.sum(result[0]) == 1
    assert np.sum(result[1]) == 0
    assert result[0][ind] == 1