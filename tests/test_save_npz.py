import numpy as np
import tempfile
import os
from src.python.dataset import save_batch_to_npz

def test_save_batch_to_npz_success():
    X1 = np.array([[1,2],[3,4]], dtype=np.int8)
    X2 = np.array([[5,6],[7,8]], dtype=np.int8)
    Y = np.array([1, 0], dtype=np.int32)

    with tempfile.TemporaryDirectory() as tmpdir:
        code = save_batch_to_npz(X1, X2, Y, tmpdir + "/", "test")
        assert code == 0
        
        # файл должен существовать
        filepath = os.path.join(tmpdir, "chank_test.npz")
        assert os.path.exists(filepath)
        
        # содержимое должно совпадать
        data = np.load(filepath)
        
        assert np.array_equal(data["X1"], X1)
        assert np.array_equal(data["X2"], X2)
        assert np.array_equal(data["Y"], Y)

def test_save_batch_to_npz_failure():
    X1 = np.array([1], dtype=np.int8)
    X2 = np.array([2], dtype=np.int8)
    Y = np.array([3], dtype=np.int32)

    # сохраняем в несуществующую директорию → должна быть ошибка
    code = save_batch_to_npz(X1, X2, Y, "/definitely/not/exist/", "bad")
    assert code == -1
