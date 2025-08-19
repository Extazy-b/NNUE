from src.python.dataset import write_to_batch
from src.python.configs import *

import numpy as np
import pytest

def test_is_arrays_the_old():
    X1 = np.zeros((BATCH_SIZE, INPUT_VECTOR_SIZE), np.int8)
    X2 = np.zeros((BATCH_SIZE, INPUT_VECTOR_SIZE), np.int8)
    Y = np.zeros((BATCH_SIZE, 1), np.int32)

    note = (np.ones(INPUT_VECTOR_SIZE, np.int8), 
                  -1 * np.ones(INPUT_VECTOR_SIZE, np.int8),  
                  101)
    
    res = write_to_batch(0, note, X1, X2, Y)

    assert res == 0
    assert (X1[0] == note[0]).all()
    assert (X2[0] == note[1]).all()
    assert Y[0] == note[2]

def test_end_of_batch():
    counter = BATCH_SIZE

    X1 = np.zeros((BATCH_SIZE, INPUT_VECTOR_SIZE), np.int8)
    X2 = np.zeros((BATCH_SIZE, INPUT_VECTOR_SIZE), np.int8)
    Y = np.zeros((BATCH_SIZE, 1), np.int32)

    note = (np.ones(INPUT_VECTOR_SIZE, np.int8), 
                  np.ones(INPUT_VECTOR_SIZE, np.int8) * (-1),  
                  101)
    
    res = write_to_batch(counter, note, X1, X2, Y)

    assert res == -1
    assert (X1 == np.zeros((BATCH_SIZE, INPUT_VECTOR_SIZE), np.int8)).all()
    assert (X2 == np.zeros((BATCH_SIZE, INPUT_VECTOR_SIZE), np.int8)).all()
    assert (Y == np.zeros((BATCH_SIZE, 1), np.int32)).all()





    