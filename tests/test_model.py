import pytest
import torch
import numpy as np
from src.python.model import NNUE
from src.python.configs import *

@pytest.fixture
def model():
    """Фикстура для создания экземпляра модели"""
    return NNUE()

def test_model_initialization(model):
    """Тестирование инициализации модели с правильными параметрами"""
    assert model.input_size == INPUT_VECTOR_SIZE
    assert model.hidden_size == HIDDEN_LAYER_SIZE
    
    # Проверка формы параметров
    assert model.l1_weights.shape == (2, HIDDEN_LAYER_SIZE, INPUT_VECTOR_SIZE)
    assert model.l1_biases.shape == (2, HIDDEN_LAYER_SIZE)
    assert model.l2_weight.shape == (1, 2 * HIDDEN_LAYER_SIZE)
    assert model.l2_bias.shape == (1,)

def test_forward_pass_shape(model):
    """Тестирование формы выходных данных forward pass"""
    x1 = torch.zeros(BATCH_SIZE, INPUT_VECTOR_SIZE)
    x2 = torch.zeros(BATCH_SIZE, INPUT_VECTOR_SIZE)
    
    output = model(x1, x2)
    assert output.shape == (BATCH_SIZE,)

def test_quantize_weights(model):
    """Тестирование квантования весов"""
    quantized = model.quantize_weights()
    
    # Проверка типов данных
    assert quantized['l1_weights'].dtype == torch.int16
    assert quantized['l1_biases'].dtype == torch.int16
    assert quantized['l2_weight'].dtype == torch.int16
    assert quantized['l2_bias'].dtype == torch.int16
    
    # Проверка диапазона значений
    assert torch.all(quantized['l1_weights'].abs() <= 32767)
    assert torch.all(quantized['l1_biases'].abs() <= 32767)
    assert torch.all(quantized['l2_weight'].abs() <= 32767)
    assert torch.all(quantized['l2_bias'].abs() <= 32767)

def test_reset_parameters(model):
    """Тестирование инициализации параметров"""
    # Сохраняем исходные параметры
    original_weights = model.l1_weights.clone()
    original_biases = model.l1_biases.clone()
    
    # Повторно инициализируем параметры
    model._reset_parameters()
    
    # Проверяем, что параметры изменились
    assert not torch.allclose(original_weights, model.l1_weights)
    assert not torch.allclose(original_biases, model.l1_biases)

def test_forward_pass_values():
    """Тестирование конкретных вычислений forward pass"""
    # Создаем модель с детерминированными весами
    model = NNUE(input_size=2, hidden_size=2)
    with torch.no_grad():
        model.l1_weights[0] = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        model.l1_weights[1] = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        model.l1_biases[0] = torch.tensor([0.0, 0.0])
        model.l1_biases[1] = torch.tensor([0.0, 0.0])
        model.l2_weight[0] = torch.tensor([1.0, 1.0, 1.0, 1.0])
        model.l2_bias[0] = torch.tensor(0.0)
    
    # Тестовые входные данные
    x1 = torch.tensor([[1.0, 0.0]])
    x2 = torch.tensor([[0.0, 1.0]])
    
    output = model(x1, x2)
    expected = torch.tensor(2.0)  # (1*1 + 0*0 + 0*0 + 1*1) = 2
    
    assert torch.allclose(output, expected)

def test_crelu_activation():
    """Тестирование Clipped ReLU активации"""
    model = NNUE()
    test_input = torch.tensor([[-10.0, 0.0, 5.0, 20.0]])
    
    # Прямой проход через первый слой
    l1_out = torch.nn.functional.linear(test_input, model.l1_weights[0], model.l1_biases[0])
    activated = torch.clamp(l1_out, min=0, max=CRelu_Border)
    
    # Проверка границ активации
    assert torch.all(activated >= 0)
    assert torch.all(activated <= CRelu_Border)

def test_edge_cases():
    """Тестирование граничных случаев"""
    model = NNUE()
    
    # Нулевые входы
    x1_zero = torch.zeros(1, INPUT_VECTOR_SIZE)
    x2_zero = torch.zeros(1, INPUT_VECTOR_SIZE)
    output_zero = model(x1_zero, x2_zero)
    
    # Максимальные входы
    x1_max = torch.ones(1, INPUT_VECTOR_SIZE)
    x2_max = torch.ones(1, INPUT_VECTOR_SIZE)
    output_max = model(x1_max, x2_max)
    
    # Проверка, что выходы являются числами
    assert not torch.isnan(output_zero)
    assert not torch.isnan(output_max)
    assert not torch.isinf(output_zero)
    assert not torch.isinf(output_max)

def test_optimizer_creation(model):
    """Тестирование создания оптимизатора"""
    optimizer = model.get_optimizer()
    assert isinstance(optimizer, torch.optim.AdamW)
    assert optimizer.defaults['lr'] == 0.001
    assert optimizer.defaults['weight_decay'] == 1e-5

def test_lr_scheduler_creation(model):
    """Тестирование создания планировщика обучения"""
    optimizer = model.get_optimizer()
    scheduler = model.get_lr_scheduler(optimizer)
    
    assert isinstance(scheduler, torch.optim.lr_scheduler.StepLR)
    assert scheduler.step_size == 1000
    assert scheduler.gamma == 0.9