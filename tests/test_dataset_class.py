import pytest
import numpy as np
import tempfile
import os
import torch
from unittest.mock import patch, MagicMock
from src.python.dataset import NNUEDataset, make_dataloader
from src.python.configs import *

def test_nnue_dataset_length():
    """Тестирование корректного определения размера датасета."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Создаем несколько тестовых файлов
        for i in range(3):
            data = {
                'X1': np.ones((BATCH_SIZE, INPUT_VECTOR_SIZE), dtype=np.int8),
                'X2': np.ones((BATCH_SIZE, INPUT_VECTOR_SIZE), dtype=np.int8),
                'Y': np.ones(BATCH_SIZE, dtype=np.int32) * i
            }
            np.savez(os.path.join(tmp_dir, f'chank_{i:03d}.npz'), **data)
        
        dataset = NNUEDataset(tmp_dir)
        assert len(dataset) == 3 * BATCH_SIZE

def test_nnue_dataset_get_item():
    """Тестирование доступа к элементам датасета."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Создаем тестовый файл с известными данными
        test_data = {
            'X1': np.arange(BATCH_SIZE * INPUT_VECTOR_SIZE, dtype=np.int8).reshape(BATCH_SIZE, INPUT_VECTOR_SIZE),
            'X2': np.arange(BATCH_SIZE * INPUT_VECTOR_SIZE, dtype=np.int8).reshape(BATCH_SIZE, INPUT_VECTOR_SIZE) + 100,
            'Y': np.arange(BATCH_SIZE, dtype=np.int32) + 1000
        }
        np.savez(os.path.join(tmp_dir, 'chank_000.npz'), **test_data)
        
        dataset = NNUEDataset(tmp_dir)
        
        # Проверяем несколько элементов
        for i in range(min(10, BATCH_SIZE)):  # Проверяем первые 10 элементов
            x1, x2, y = dataset[i]
            assert torch.allclose(x1, torch.from_numpy(test_data['X1'][i].astype(np.float32)))
            assert torch.allclose(x2, torch.from_numpy(test_data['X2'][i].astype(np.float32)))
            assert torch.allclose(y, torch.tensor(test_data['Y'][i], dtype=torch.float32))

def test_nnue_dataset_index_error():
    """Тестирование обработки неверных индексов."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Создаем тестовый файл
        data = {
            'X1': np.ones((BATCH_SIZE, INPUT_VECTOR_SIZE), dtype=np.int8),
            'X2': np.ones((BATCH_SIZE, INPUT_VECTOR_SIZE), dtype=np.int8),
            'Y': np.ones(BATCH_SIZE, dtype=np.int32)
        }
        np.savez(os.path.join(tmp_dir, 'chank_000.npz'), **data)
        
        dataset = NNUEDataset(tmp_dir)
        
        # Проверяем, что обращение к несуществующему индексу вызывает ошибку
        with pytest.raises(IndexError):
            _ = dataset[BATCH_SIZE + 1]
        
        with pytest.raises(IndexError):
            _ = dataset[-1]

def test_nnue_dataset_caching():
    """Тестирование кэширования файлов с учетом инициализации."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Создаем несколько тестовых файлов
        for i in range(2):
            data = {
                'X1': np.ones((BATCH_SIZE, INPUT_VECTOR_SIZE), dtype=np.int8) * i,
                'X2': np.ones((BATCH_SIZE, INPUT_VECTOR_SIZE), dtype=np.int8) * i,
                'Y': np.ones(BATCH_SIZE, dtype=np.int32) * i
            }
            np.savez(os.path.join(tmp_dir, f'chank_{i:03d}.npz'), **data)
        
        # Мокаем np.load для отслеживания вызовов
        with patch('src.python.dataset.np.load') as mock_load:
            # Настраиваем mock для возврата тестовых данных
            mock_load.return_value = {
                'X1': np.ones((BATCH_SIZE, INPUT_VECTOR_SIZE), dtype=np.int8),
                'X2': np.ones((BATCH_SIZE, INPUT_VECTOR_SIZE), dtype=np.int8),
                'Y': np.ones(BATCH_SIZE, dtype=np.int32)
            }
            
            # Инициализация датасета - вызовет np.load для валидации каждого файла
            with patch.object(NNUEDataset, '_validate_file_sizes', return_value=None):
                dataset = NNUEDataset(tmp_dir)
                       
            # Сбрасываем счетчик вызовов, чтобы игнорировать вызовы при инициализации
            mock_load.reset_mock()
            
            # Обращаемся к элементам из первого файла
            for i in range(5):
                _ = dataset[i]
            
            # Не должно быть вызовов np.load для первого файла так как он закеширован при иницилизации
            assert mock_load.call_count == 0
            
            # Обращаемся к элементу из второго файла
            _ = dataset[BATCH_SIZE]
            
            # Должен быть второй вызов np.load для второго файла
            assert mock_load.call_count == 1

def test_make_dataloader():
    """Тестирование создания DataLoader."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Создаем тестовый файл
        data = {
            'X1': np.ones((BATCH_SIZE, INPUT_VECTOR_SIZE), dtype=np.int8),
            'X2': np.ones((BATCH_SIZE, INPUT_VECTOR_SIZE), dtype=np.int8),
            'Y': np.ones(BATCH_SIZE, dtype=np.int32)
        }
        np.savez(os.path.join(tmp_dir, 'chank_000.npz'), **data)
        
        dataloader = make_dataloader(tmp_dir, batch_size=BATCH_SIZE, shuffle=True)
        
        # Проверяем, что возвращается объект правильного типа
        from torch.utils.data import DataLoader
        assert isinstance(dataloader, DataLoader)
        
        # Проверяем переданные параметры
        assert dataloader.batch_size == BATCH_SIZE
        
        # Проверяем, что можно проитерироваться по даталоадеру
        batch = next(iter(dataloader))
        assert len(batch) == 3  # x1, x2, y
        assert batch[0].shape == (BATCH_SIZE, INPUT_VECTOR_SIZE)  # x1 batch
        assert batch[1].shape == (BATCH_SIZE, INPUT_VECTOR_SIZE)  # x2 batch
        assert batch[2].shape == (BATCH_SIZE,)  # y batch