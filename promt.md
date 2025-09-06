Привет я занимаюсь летней практикой, у меня осталось 6 дней

Вот задача:
**Задача: Разработка и обучение специализированной нейронной сети NNUE для шахматного движка**
====================================================================

**Цель**
--------

Создать рабочий MVP (Minimum Viable Product), который можно интегрировать в шахматный движок за один месяц, и который будет способен оценивать позиции на шахматной доске.

**Технический стек**
--------------------

* Python 3 + PyTorch для прототипа и тренировки сети
* C++ для экспорта и квантования сети
* Нейронная сеть будет использовать архитектуру с двумя слоями, с масштабами QA и QB для квантования весов

**Ключевые требования**
----------------------

* Входные данные: FEN-нотация (Forsyth-Edwards Notation) шахматной позиции
* Выходные данные: оценка позиции
* Квантование весов сети в int16
* Экспорт весов и мета-данных в бинарный файл
* C++-интерфейс для вызова сети

**План работы**
--------------

* Разработка и тренировка сети в PyTorch
* Экспорт и квантование сети в C++
* Интеграция сети в шахматный движок
* Тестирование и валидация сети

**Дополнительные задачи**
------------------------

* Исследование возможной адаптации архитектуры NNUE для задач биоинформатики (предсказание структуры РНК, белковое взаимодействие)
* Разработка логирования и визуализации процесса тренировки сети

Вот основной конспект
# Конспект по NNUE

## FEN-нотация
0. Нотация, в одной строке умещающая информацию о расположении фигур на доске, цвете, который ходит, а также дополнительные параметры.

1. Расположение фигур имеет следующий синтаксис:
```
1. <Piece Placement> ::= <rank8> '/' <rank7> '/' <rank6> '/' <rank5> '/' <rank4> '/' <rank3> '/' <rank2> '/' <rank1>
```
```
2. <ranki> ::= ('8') | ( [<digit17>] <piece> { [<digit17>] <piece> } [<digit17>] )
```
```
3. <piece> ::= <white Piece> | <black Piece>
```
```
4. <digit17> ::= '1' | '2' | '3' | '4' | '5' | '6' | '7'
```
```
5. <white Piece> ::= 'P' | 'N' | 'B' | 'R' | 'Q' | 'K'
```
```
6. <black Piece> ::= 'p' | 'n' | 'b' | 'r' | 'q' | 'k'
```
Таким образом, между двумя `/` содержится описание фигур, стоящих на горизонтали, или количество подряд идущих пустых клеток в описываемой строке.

## HalfKP
0. На вход сеть получает FEN-позицию, из которой вычисляется расположение каждой фигуры на доске и то, чей сейчас ход.
1. Входными данными сети будут два вектора, каждый из которых описывает особенности стороны, которая ходит, и стороны, которая не ходит.
2. Один вектор содержит до 32 чисел (индексов).
3. Каждое число является индексом, отражающим наличие на доске комбинации: «король данного цвета стоит на клетке K», «фигура данного типа и цвета стоит на клетке P».
4. Всего таких комбинаций для каждой стороны: `64 (позиции короля) * 5 (типов фигур: не король) * 2 (цвета фигуры) * 64 (позиции фигуры) = 40960`.
5. В зависимости от того, чей сейчас ход, сеть передаёт один вектор на входную матрицу атакующей стороны (stm - side to move) и один на входную матрицу защищающейся стороны (nstm - not side to move).
6. Таким образом, входной слой — это два разреженных бинарных вектора размерности `40960 x 1`.

`Эта механика позволяет усилить контекст положения короля, многократно повторяя его расположение, и добавляет контекст различия атаки и защиты за счёт двух разных матриц весов (взглянуть под разным углом на одно и то же расположение фигур).`

## Первый внутренний слой (аккумулятор)
0. На вход внутреннего слоя подаются два вектора `40960 x 1` из входного слоя: stm-вектор и nstm-вектор (`side-to-move` и `not-side-to-move`).
1. Каждый из векторов умножается на соответствующую матрицу весов (каждая размером `256 x 40960`), давая на выходе два вектора `256 x 1`.
2. Эти вектора подвергаются сдвигу на вектор bias'а (линейное преобразование `y = Ax + b`).
3. Два полученных вектора конкатенируются, образуя вектор размерности `512 x 1`.
4. Итоговый вектор пропускается через функцию **активации** (например, `CReLU`), и затем результат передаётся на следующий слой.

**Матрицы и вектора состоят из чисел типа `int16`; выходной вектор состоит из данных того же типа.**

## Алгебра умножения матрицы на бинарный разреженный вектор
1. Дано:
   - Натуральные числа `n`, `m`, `k`.
   - Вектор `V` размерности `n x 1`.
   - Все компоненты вектора, кроме `k` штук, равны нулю. Ненулевые компоненты равны `1`.
   - Матрица `A` размерности `m x n`.

2. Формально:
```math
n, m, k \in \N \\
V \in \R^n \\
A \in M(m, n, \R) \\
[A]_{i\,j}  = a_{ij} \\
\exists \; 0 < i_1, ..., i_k \le n: \\
\forall i \notin \{i_1, ..., i_k\},  v_i = 0 \\
\forall i \in \{i_1, ..., i_k\},  v_i = 1 \\
```

3. Тогда результат умножения:
```math
[A \cdot V]_l = \sum_{i = 1}^n {a_{l\,i} \cdot v_i} = \sum_{j = 1}^k {a_{l\,i_j}}, \quad l = 1, ..., m
```
4. Таким образом, пугающие `n * m` умножений и `m * (n-1)` сложений (`~n*m операций`) сводятся всего к `k * m` операциям сложения.

## Квантование
1. После обучения матрицы и вектора весов и сдвигов (biases) являются числами с плавающей точкой.
2. Операции с целыми числами часто выполняются быстрее и точнее (на специализированном железе), чем операции с дробными.
3. Все коэффициенты каждого слоя умножаются на некое большое число `Q` (коэффициент квантования) и затем округляются до целого.
4. В памяти сети хранятся именно эти целые числа, а также значения `Q` для каждого слоя.
5. В архитектуре с двумя слоями, соответственно, будут храниться два числа: `Q_A` (для внутреннего слоя) и `Q_B` (для выходного слоя).
6. `Q_A` и `Q_B` называются масштабами, так как они масштабируют всю арифметику сети.
7. Перед выводом результата его нужно масштабировать обратно, то есть разделить на произведение `Q_A * Q_B`.
8. Если во внутреннем слое используется квадратичная (или иная нелинейная) активация, like в `CReLU`, итоговый масштаб может стать очень большим (`Q_A^2 * Q_B`), что чревато выходом за границы `int16`.

## Обновляемость (Incrementality)
1. Бинарность и разреженность входного слоя означает, что каждый элемент либо присутствует (1), либо отсутствует (0).
2. Таким образом, "включение" любого входного нейрона добавляет соответствующий столбец матрицы весов к результату (аккумулятору).
3. "Отключение" нейрона, в свою очередь, вычитает этот столбец из аккумулятора.
4. Любой ход (кроме взятия) освобождает одну клетку (отключает нейрон, соответствующий старой позиции фигуры) и занимает другую клетку (включает нейрон, соответствующий новой позиции фигуры).
5. Таким образом, при изменении позиции достаточно:
   - Сохранить вектор-результат аккумулятора (`Y = A * X + b`).
   - Определить индексы комбинаций, которые стали неактуальны, и тех, которые появились.
   - Для каждого элемента вектора `Y` вычесть элемент из `A`, находящийся на пересечении строки этого элемента и столбца, равного индексу неактуальной комбинации.
   - К каждому элементу `Y` прибавить элемент, найденный аналогичным путём для индекса новой комбинации.

```Это даёт выгоду в m раз (вместо k*m операций остаётся 2*k операций на ход, где k — количество изменений, обычно 1-2)```.

```К атакующим ходам применимы те же правила с поправкой на отключение двух нейронов (исходная позиция своей фигуры и позиция битой фигуры противника) и включение одного (новая позиция своей фигуры)```.

## Активация
1. Обычный (линейный) слой всегда представляет собой линейное преобразование ``y = Ax + b``.
2. Любая композиция линейных преобразований сама является линейным преобразованием.
3. Это очень простая вычислительная система, неспособная к решению сложных задач.
4. Для создания нелинейности применяется функция активации, реализуемая различными способами. Она создаёт нелинейное преобразование, позволяя сети обучаться сложным нелинейным зависимостям.
5. Функция активации может иметь одно из следующих воплощений (возможны и другие):
    - ``ReLU(x) = \max(0, x)`` — выпрямитель (клипирование снизу).
    - ``CReLU(x, T) = \min(\max(0, x), T)`` — клипированный ReLU (с обеих сторон).
    - ``Sigmoid`` — размещает вывод на интервале ``(0, 1)``.
    ```math
    \sigma(x) = \frac{1}{1+e^{-x}}
    ```
    - ``tanh(x)`` — симметрично размещает вывод на интервале ``[-1, 1]``.
    ```math
    \tanh(x) = \frac{e^{x} - e^{-x}}{e^{x} + e^{-x}}
    ```
    - ``Softmax`` — преобразует вектор в распределение вероятностей.
    ```math
    \text{softmax}(x)_i = \frac{e^{x_i}}{\sum_j{e^{x_j}}}
    ```
6. Для быстрой работы и обучения в NNUE чаще всего используются ``ReLU`` или ``CReLU``.

## Алгебра обучения
### Функция сети (прямой проход, forward pass)
```math
Y = C \times \text{CReLU}\left( \left( A_1 \times X_1 + B_1 \times \mathbf{1}_L^\top \right) \Vert \left( A_2 \times X_2 + B_2 \times \mathbf{1}_L^\top \right) \right) + d \times \mathbf{1}_L^\top
```

где:
---
```math
X_1, X_2 \in \mathbb{R}^{N \times L}
```
Входные матрицы признаков (батч из `L` примеров, каждый пример — разреженный вектор размерности `N=40960`).
---
```math
A_1, A_2 \in \mathbb{R}^{K \times N}
```
Матрицы весов аккумулятора для каждой стороны (`K=256`).
---
```math
B_1, B_2 \in \mathbb{R}^{K \times 1}
```
Вектора смещений (bias) аккумулятора.
---
```math
\mathbf{1}_L \in \mathbb{R}^{L \times 1}
```
Вектор-столбец из единиц для broadcast'а (распространения) bias на весь батч.
---
```math
\Vert
```
Обозначает конкатенацию по первому измерению (вертикальную, по строкам). Результат имеет размерность `2K \times L`.
---
```math
C \in \mathbb{R}^{1 \times 2K}
```
Матрица весов выходного слоя.
---
```math
d \in \mathbb{R}
```
Скалярное смещение (bias) выходного слоя.
---
```math
Y \in \mathbb{R}^{1 \times L}
```
Выход модели — предсказания для каждого из `L` примеров батча.
---
```math
\text{CReLU}(x) = \min(\max(x, 0), T)
```
Clipped ReLU функция активации. `T` — порог (обычно `127` для квантованных сетей).
---

### Функция потерь (MSE) по батчу

```math
\text{Loss} = \frac{1}{L} \left\| Y_{\text{target}} - Y_{\text{pred}} \right\|_2^2
```
```math
Y_{\text{target}} \in \mathbb{R}^{1 \times L}
```
Вектор целевых (ожидаемых) значений для батча.
---
```math
Y_{\text{pred}}
```
Предсказание модели, вычисляемое по формуле прямого прохода выше.
---
```math
\left\| \cdot \right\|_2^2
```
Квадрат евклидовой нормы (сумма квадратов разностей по всем `L` примерам батча).
---

### Градиентный спуск
Алгоритм подбора параметров (матриц весов и смещений) для минимизации ошибки на всём датасете.

```math
1. \text{Инициализировать начальные параметры } \Theta_0 = \{A_1, A_2, B_1, B_2, C, d \} \text{ и скорость обучения } \lambda.
```
```math
2. \text{Вычислить } \text{Loss}(\Theta_i).
```
```math
3. \text{Если } \text{Loss} > \epsilon_1 \text{, то:}
```
```math
4. \quad \text{Вычислить градиенты } \nabla \text{Loss}(\Theta_i) \text{ по всем параметрам.}
```
```math
5. \quad \text{Обновить параметры: } \Theta_{i+1} = \Theta_{i} - \lambda \cdot \nabla \text{Loss}(\Theta_i)
```
```math
6. \quad \text{Если } ||\nabla \text{Loss}(\Theta_i)|| < \epsilon_2 \text{, остановиться.}
```
```math
7. \quad \text{Иначе, перейти к шагу 2.}
```

**Формулы для вычисления градиентов (опущен `1/L` множитель из Loss для краткости):**
```math
\boxed{
\begin{aligned}
\Delta &= (Y_{\text{pred}} - Y_{\text{target}}) \in \mathbb{R}^{1 \times L} \\
H &= \text{CReLU}'\left( \left( A_1 X_1 + B_1 \mathbf{1}_L^\top \right) \Vert \left( A_2 X_2 + B_2 \mathbf{1}_L^\top \right) \right) \odot \left( C^\top \Delta \right) \\
M^{(1)} &= H_{1:K, :} \quad \text{(первые K строк матрицы H)} \\
M^{(2)} &= H_{K+1:2K, :} \quad \text{(последние K строк матрицы H)} \\
\\
\nabla_{C} \, \text{Loss} &= \Delta \cdot \left( \text{CReLU}\left( ... \right) \right)^\top \\
\nabla_{d} \, \text{Loss} &= \Delta \cdot \mathbf{1}_L \\
\\
\nabla_{A_1} \, \text{Loss} &= M^{(1)} \cdot X_1^\top \\
\nabla_{B_1} \, \text{Loss} &= M^{(1)} \cdot \mathbf{1}_L \\
\nabla_{A_2} \, \text{Loss} &= M^{(2)} \cdot X_2^\top \\
\nabla_{B_2} \, \text{Loss} &= M^{(2)} \cdot \mathbf{1}_L \\
\end{aligned}
}
```
**Пояснение:** `⊙` обозначает поэлементное умножение (произведение Адамара), а `CReLU'` — это производная функции активации (ноль вне интервала `[0, T]` и единица внутри него).

---

Вот примерная архитектура
# NNUE
- ## Input convertation module
    - Read fen-string <-- **input point**
    - Encoding each piece
    - Calculate piece code (index for color, type and square with square of each kings)
    - Save index into one of input vectors
    - Decide which vector is STM, and wich is SnTM
    - Sending it to nucleus --> **output point**
- ## Calculating module
    - Read input vectors <-- **input point**
    - Read weights and bieces
    - ### Accumulator
        - Multiplication discharge input Boolean vectors to accumulator matrixes
        - Adding biases
        - Connateration of output vectors
    - ### Activation
        - Get result of accumulator <-- **input point**
        - Clipping (or another activation function)
        - send vector throw output layer --> **output point**

- ## Updating module
    - get index of turned off and on neurons (i_1, ..., i_k) and (j_1, ..., j_l)  <-- **input point**
    - for all elemnt of previos move accumulator result (y_m)
       - subtract a_(i_1)m, ..., a_(i_k)m
       - add a_(j_1)m, ..., a_(j_l)m
    - send vector into activation --> **output point**
- ## Training module
    - read traning dataset
        - read weights  <-- **input point**
        - calculate outputs
        - calculate error
        - calculate gradient
        - update weights  --> **output point**

дерево проекта
tree
.
├── LICENSE
├── README.md
├── data
│   ├── processed
│   │   ├── chank_000.npz
│   │   ├── ...
│   │   ├── chank_129.npz
│   │   └── dataset.bin
│   └── raw
│       ├── chessData.csv
│       └── testData.csv
├── docs
│   ├── 0. Task.md
│   ├── 1. Conspect.md
│   ├── 2. Architecture.md
│   ├── 3. Gradients.md
│   ├── 4. Function by-element view.md
│   ├── 5. First try (linear).md
│   └── 6. Updating.md
├── plan.md
├── src
│   ├── __init__.py
│   ├── __pycache__
│   │   └── __init__.cpython-313.pyc
│   └── python
│       ├── __init__.py
│       ├── __pycache__
│       │   ├── __init__.cpython-313.pyc
│       │   ├── configs.cpython-313.pyc
│       │   ├── dataset.cpython-313.pyc
│       │   └── model.cpython-313.pyc
│       ├── configs.py
│       ├── dataset.py
│       ├── model.py
│       ├── quantize.py
│       └── train.py
├── tests
│   ├── __init__.py
│   ├── __pycache__
│   │   ├── __init__.cpython-313.pyc
│   │   ├── test_cross_index.cpython-313-pytest-8.4.1.pyc
│   │   ├── test_dataset_class.cpython-313-pytest-8.4.1.pyc
│   │   ├── test_fen_to_indices.cpython-313-pytest-8.4.1.pyc
│   │   ├── test_generator.cpython-313-pytest-8.4.1.pyc
│   │   ├── test_load_raw_file.cpython-313-pytest-8.4.1.pyc
│   │   ├── test_make_note.cpython-313-pytest-8.4.1.pyc
│   │   ├── test_model.cpython-313-pytest-8.4.1.pyc
│   │   ├── test_nnue_dataset.cpython-313-pytest-8.4.1.pyc
│   │   ├── test_save_npz.cpython-313-pytest-8.4.1.pyc
│   │   └── test_write_to_batch.cpython-313-pytest-8.4.1.pyc
│   ├── script.py
│   ├── test_cross_index.py
│   ├── test_dataset_class.py
│   ├── test_fen_to_indices.py
│   ├── test_load_raw_file.py
│   ├── test_make_note.py
│   ├── test_model.py
│   ├── test_save_npz.py
│   └── test_write_to_batch.py

dataset.py, model.py, configs.py прикрепляю файлами в этом сообщении

train.py выкладываю ниже

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
import time
from datetime import datetime

from src.python.model import NNUE
from src.python.dataset import NNUEDataset, make_dataloader
from src.python.configs import *

class NNUE_Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Инициализация модели
        self.model = NNUE().to(self.device)
        
        # Оптимизатор и scheduler
        self.optimizer = self.model.get_optimizer(
            learning_rate=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        self.scheduler = self.model.get_lr_scheduler(
            self.optimizer,
            step_size=config['scheduler_step'],
            gamma=config['scheduler_gamma']
        )
        
        # Функция потерь
        self.criterion = nn.MSELoss()
        
        # TensorBoard для визуализации
        self.writer = SummaryWriter(log_dir=config['log_dir'])
        
        # Проверяем, нужно ли возобновить обучение
        if config['resume_checkpoint']:
            self.load_checkpoint(config['resume_checkpoint'])

    def train_epoch(self, train_loader, epoch):
        self.model.train()
        total_loss = 0
        num_batches = len(train_loader)
        
        for batch_idx, (x1, x2, y) in enumerate(train_loader):
            x1, x2, y = x1.to(self.device), x2.to(self.device), y.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(x1, x2)
            loss = self.criterion(outputs, y)
            
            # Backward pass
            loss.backward()
            
            # Градиентный клиппинг для стабильности
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Логирование каждые N батчей
            if batch_idx % self.config['log_interval'] == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f'Epoch: {epoch} [{batch_idx}/{num_batches}]\tLoss: {loss.item():.6f}\tLR: {current_lr:.2e}')
                self.writer.add_scalar('Loss/train_batch', loss.item(), epoch * num_batches + batch_idx)
        
        return total_loss / num_batches

    def validate(self, val_loader, epoch):
        self.model.eval()
        total_loss = 0
        total_mae = 0
        
        with torch.no_grad():
            for x1, x2, y in val_loader:
                x1, x2, y = x1.to(self.device), x2.to(self.device), y.to(self.device)
                outputs = self.model(x1, x2)
                loss = self.criterion(outputs, y)
                total_loss += loss.item()
                total_mae += torch.mean(torch.abs(outputs - y)).item()
        
        avg_loss = total_loss / len(val_loader)
        avg_mae = total_mae / len(val_loader)
        
        self.writer.add_scalar('Loss/val', avg_loss, epoch)
        self.writer.add_scalar('MAE/val', avg_mae, epoch)
        
        return avg_loss, avg_mae

    def save_checkpoint(self, epoch, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': self.best_loss if is_best else None
        }
        
        # Сохраняем последний checkpoint
        torch.save(checkpoint, os.path.join(self.config['checkpoint_dir'], 'last_checkpoint.pth'))
        
        # Сохраняем лучший checkpoint
        if is_best:
            torch.save(checkpoint, os.path.join(self.config['checkpoint_dir'], 'best_checkpoint.pth'))
            
            # Также экспортируем квантованные веса для C++
            quantized_weights = self.model.quantize_weights()
            torch.save(quantized_weights, os.path.join(self.config['checkpoint_dir'], 'quantized_weights.pth'))

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_loss = checkpoint.get('loss', float('inf'))
        
        print(f"Resumed training from epoch {self.start_epoch}")

    def train(self, train_loader, val_loader=None):
        self.best_loss = float('inf')
        self.start_epoch = 0
        
        for epoch in range(self.start_epoch, self.config['num_epochs']):
            # Обучение
            train_loss = self.train_epoch(train_loader, epoch)
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            
            # Валидация
            if val_loader:
                val_loss, val_mae = self.validate(val_loader, epoch)
                print(f'Epoch {epoch}: Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, Val MAE: {val_mae:.6f}')
                
                # Сохраняем лучшую модель
                if val_loss < self.best_loss:
                    self.best_loss = val_loss
                    self.save_checkpoint(epoch, is_best=True)
            else:
                print(f'Epoch {epoch}: Train Loss: {train_loss:.6f}')
            
            # Обновляем learning rate
            self.scheduler.step()
            
            # Сохраняем checkpoint
            if epoch % self.config['checkpoint_interval'] == 0:
                self.save_checkpoint(epoch)
        
        self.writer.close()

# Конфигурация обучения
config = {
    'learning_rate': 0.001,
    'weight_decay': 1e-5,
    'scheduler_step': 1000,
    'scheduler_gamma': 0.9,
    'num_epochs': 100,
    'batch_size': 256,
    'log_interval': 100,
    'checkpoint_interval': 5,
    'checkpoint_dir': './checkpoints',
    'log_dir': './logs',
    'resume_checkpoint': None  # Путь к checkpoint для возобновления обучения
}

def main():
    # Создаем директории если не существуют
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    os.makedirs(config['log_dir'], exist_ok=True)
    
    # DataLoader'ы
    train_loader = make_dataloader(
        npz_dir='path/to/train/npz/files',
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4
    )
    
    val_loader = make_dataloader(
        npz_dir='path/to/val/npz/files',
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=2
    )
    
    # Инициализация и запуск обучения
    trainer = NNUE_Trainer(config)
    trainer.train(train_loader, val_loader)

if __name__ == '__main__':
    main()
```

Как ты можешь видеть, у меня сейчас готов только модуль обучения, и то далеко не полностью
Помоги мне со следующим стеком задач

1. Подробное описание того что происходит в train.py
2. Отладка и запуск train.py
3. Как меньше чем за неделю хотя бы примерно завершить задачу

Заранее большое спасибо!!!