Привет! Я занимаюсь разработкой и обучением специализированной нейросети NNUE для шахматного движка, оптимизируя вычисления с помощью SIMD и квантования. Параллельно я исследую, как эту архитектуру можно адаптировать для задач биоинформатики — например, для предсказания структуры РНК или взаимодействия белков.

Я собрал все материалы которые нашёл по данной теме в несколько .md файлов

Прикреплю их ниже
1. Дай им общую оценку 
2. Сформируй простую и эффективную архитектуру директорий
3. Подскажи какой язык лучше выбрать для обучения сети

Материалы:
1. Общий конспект
# Конспект по NNUE

## FEN-нотация
0. Нотация, в одной строчке умещающая информацию о расположении фигур на доске и ходящем цвете, а так же дополнительные параметры

1. Расположение фигур имеет следующий синтаксис
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
Таким образом между двумя ```/``` содtржится описание о фигура стоящих на доске или количестве подряд пустых клеток в описываемой строке

## HalfKP
0. На вход сеть получает fen - позицию, из которой вычисляет расположение каждой фигры на доске и то, чей сейчас ход
1. Входными данными сети будут два вектора, каждый из которых описывает ресурсы ходящей и неходящей сейчас стороны
2. Один вектор содержит до 32 чисел
3. Каждое число является индексом отражающим существование на доске комбинации (король даного цвета стоит на данной клетке, данная фигура данного цвета стоит на данной клетке)
4. Всего таких комбинаций для каждой стороны: 64 * 5 * 2 * 64 = 40960
5. В зависимости от того чей сейчас ход сеть передает вектора один на входную матрицу атакующей стороны и один входную матрицу защищающейся стороны
6. Таким образом входной слой - два вектора размера `40960 x 1`

` Эта механика позволяет усилить контекст положения короля, множество раз повторяя его расположение и добавляет контекст различия атаки и защиты за счёт двух разных матриц (взглягуть под разным углом на одно и то же расположение)`

## Первый внутренний слой
0. На вход внутреннего слоя подаются два вектора `40960 x 1` из входного слоя, stm-вектор и sntm-вектор (`side-to-move` и `side-not-to-move`)
1. Каждый из векторов домножается на соответствующую матрицу, каждая размером `256 x 40960`, давая на выходе вектора `256 x 1`
2. Эти вектора подвергаются сдвигу на такие же по размерности вектора (линейное преобразование `y = Ax + B`) 
3. Два полученныз вектора конкатенируются, образуя вектора размерности `512 x 1`
4. Итоговый вектор подвергается `активации` и затем передаётся на следующий слой

**Матицы и вектора состоят из числел типа `int16`, выходной вектор состоит из тех же данных**

## Алгебра умножения матрицы на бинарный разряженный вектор
1. Дано
   - Натуральные числа `n`, `m`, `k`
   - Вектор размерности `nx1`
   - Все компоненты вектора кроме N-штук равны нулю (иные равны `1`)
   - Матрица A размерности `kxn`
2. Строго говоря
``` math
n, m, k \in \N \\
V \in \R^n \\
A \in M(m, n, \R) \\
[A]_{i\,j}  = a_{ij} \\
\exist \; 0 < i_1, ..., i_k \le n: \\
\forall i \neq i_j, v_i = 0 \\
\forall i = i_j, v_i = 1 \\
j \in (1, ..., k)
```
3. Тогда
``` math
[A \cdot V]_l = \sum_{i = 1}^n {a_{l\,i} \cdot u_i} = \sum_{j = 1}^k {a_l\,i_j} \:, l = 1, ..., m
```
4. Таким образом пугающие n x m умножений и m сложений (`n*(m+1) операция`) сводятся до `k*m` сложений

## Квантование
1. После обучения матрицы и вектора весов и сдвигов получаются числами с плавабщей точкой
2. Операции с целыми числами проще и точнее нежели операции с дробными
3. Все коэффициенты каждого слоя умножаются на некое большое Q и затем округляются до целого
4. В системе и в памяти сети будут храниться именно эти целые числа, а так же те самые Q
5. В архитектуре с двумя слоями соответтсвенно будут храниться два чисда QA (внутренний слой) и QB (выхожной слой)
6. QA и QB называются масштабами, т.к. просто масштабируют всю арифметику сети
7. Перед выводом результата его нужно масштабировать обратно, то есть разделить на QA * QB
8. Если во внутреннем слое используется квадратичная RELU активация, то масштаб становится больше, что чревато выходом за границы int16, а именно
```math 
QA^2 \cdot QB
```

## Обновляемость
1. Бинарность входного слоя означает что каждый элемент либо есть либо нет
2. Таким образом "включение" любого нейрона приавляет одно число в каждом элементе вывода акумулятора
3. Отключение в сво очередь вычитает одно число
4. Любой ход (кроме атакующих) опустошает клетку (отключает нейрон соответствующий данной комбинации) и занимает клетку (включает нейрон соответствующий новой комбинации)
5. Таким образом при изменении позиции достаточно
   - сохранить вектор результат акумулятора ```(Y = Ax+b)```
   - узнать индексы неактуально комбинации и появившейся
   - из каждого элемента ```Y``` вычесть элемент из ```A``` находящийся на пересечении соответствующей строки и стобца равного индексу неактуальной комбинации
   - к каждому элементу прибавить элемент найденный аналогичным путём для индекса новой комбинации

```Это даёт выгоду в M раз (вместо К*M операций остаётся K)```

```К атакующим ходам применимы те же правила с поправкой на отключение двух нейронов и включение одного```


## Активация
1. Обычный слой всегда представляет из себя линейное преобразование ``Ax + B``
2. Любое количество композиций линейных преобразований сводится к одному линейному преобразованию 
3. Это является очень простой вичислительной системой, неспосоной к решению сложных и разносторонних задач
4.  Для создания нелинейности применяется функция активации реализуемая различными способами, однако всегда создающая нелинейное преобразование, позволяя создать сложную разветвлённую сеть преобразований компонент
5. Функция активации может иметь одно их следующих воплощений (возможны и другие)
    - ``ReLU(x, a) = max(a, x)`` - клипирование сверху (либо снизу)
    - `` CReLU(x, a, b) = min(max(a, x), b)`` - клипирование с обеих сторон
    - ``Sigmoid`` - размещает вывод на отрезке ``(0, 1)``
    ```math
    \phi(x) = \frac{1}{1+e^{-x}}
    ```
    - ``tanh(x)`` - симметрично размещает вывод на ``[-1, 1]``
   - ``Softmax`` - вероятностное распределение
   ```math
   \phi_{i}(x) = \frac{e^{x_i}}{\sum_j{e^{x_j}}}
   ```
6. Для быстрой работы и быстрого обучения используются ``ReLU`` или ``CReLU``

## Алгебра обучения
### Функция сети (forward pass)
```math
Y = C \times CReLU\big( (A_1 \times X_1 + B_1 \times 1_L) \; || \; (A_2 \times X_2 + B_2 \times 1_L) \big) + d \times 1_L
```

где:
---
```math
X_1, X_2 \in \mathbb{R}^{N \times L} \\ 
входные \: матрицы \: признаков \: (батч \: из \: L \: примеров) \\
```
---
```math
A_1, A_2 \in \mathbb{R}^{K \times N} \\ 
матрицы \: весов \: аккумулятора \: для \: каждой \: стороны \\
```
---
```math
B_1, B_2 \in \mathbb{R}^{K} \\ 
bias \: вектора \: аккумулятора \\
```
---
```math
1_L \in \mathbb{R}^{1 \times L} \\ 
вектор \: единиц \: для \: broadcast \: (повторения \: bias) \\
```
---
```math
|| - конкатенация \: по \: строкам \: (вертикальная) \\
```
---
```math
C \in \mathbb{R}^{1 \times 2K} \\
матрица \: весов \: выходного \: слоя \\
```
---
```math
d \in \mathbb{R} \\ 
сдвиг \: (bias) \: выходного \: слоя \\
```
---
```math
Y \in \mathbb{R}^{1 \times L} \\
модели \: — \: предсказания \: для \: каждого \: из \: L \: примеров \\
```
---
```math
CReLU(x) = \min(\max(x, 0), T) \\ 
clipped \: ReLU \: функция \: активации \\
T \: — \: порог \: (обычно \: 127) 
```
---

### Функция потерь (MSE) по батчу

```math
\text{Loss} = \frac{1}{L} \left\| Y_{\text{ex}} - \left( C \times CReLU\big( (A_1 \times X_1 + B_1 \times 1_L) \; || \; (A_2 \times X_2 + B_2 \times 1_L) \big) + d \times 1_L \right) \right\|^2
```
---
```math
Y_{\text{ex}} \in \mathbb{R}^{1 \times L} \\
матрица \: (вектор) \: ожидаемых \: выходов \: модели \\
```
---
```math
X_1, X_2 \in \mathbb{R}^{N \times L} \\ 
батчи \: входных \: sparse\text{-}векторов \: признаков \\
```
---
```math
A_1, A_2 \in \mathbb{R}^{K \times N} \\ 
веса \: входного \: слоя \: (разные \: для \: каждой \: стороны) \\
```
---
```math
B_1, B_2 \in \mathbb{R}^{K} \\ 
вектора \: bias'ов \: к \: A_1, A_2 \\
```
---
```math
1_L \in \mathbb{R}^{1 \times L} \\
вектор \: единиц \: для \: broadcast'а \: bias \\
```
---
```math
C \in \mathbb{R}^{1 \times 2K} \\
веса \: выходного \: слоя \\
```
---
```math
d \in \mathbb{R} \\
скалярный \: bias \: выходного \: слоя \\
```
---
```math
CReLU(x) = \min(\max(x, 0), T) \\
clipped \: ReLU, \: порог \: T \: (обычно \: 127) \\
```
---
```math
\left\| \cdot \right\|^2 \\
евклидова \: (L_2) \: норма \: по \: всем \: примерам \: батча
```

### Краткие пояснения:

- Входные матрицы \(X_1, X_2\) — каждый столбец — признаки одного примера батча.
- Операция \(A_i \times X_i\) даёт матрицу размером \(K \times L\).
- К bias-вектору \(B_i\) применяется broadcast по столбцам через умножение на \(1_L\).
- Конкатенация создаёт матрицу \(2K \times L\).
- После применения активации и умножения на \(C\) получаем вектор предсказаний \(Y\).
- Ошибка считается как средняя квадратичная по батчу.

### Градиентный спуск
- Алгоритм подбора матиц весов для достижения наименьшей ошибки (в идеале - 0) по всему датасету
```math
1. \text{Задаются начальная точка} \: G_0 =  \{A_1, A_2, B_1, B_2, C, d \} \: \text{и шаг} \lambda\\
2. \text{Вычисляется} Loss(A_1, A_2, B_1, B_2, C, d) \\
3. \text {Если} \: Loss > \epsilon_1 \text{, то} \\
4. \text{Вычисляется } \nabla Loss \:  \text{по правилу} \\
\boxed{
\begin{aligned}
\nabla_{A_1} Loss &= -2 \cdot \operatorname{diag}(C^{(1)}) \cdot \left( ( \Delta \odot M^{(1)} ) \cdot X_1^\top \right) \in \mathbb{R}^{M \times N} \\
\nabla_{B_1} Loss &= -2 \cdot \operatorname{diag}(C^{(1)}) \cdot \left( ( \Delta \odot M^{(1)} ) \cdot \mathbf{1}_L \right) \in \mathbb{R}^{M \times 1} \\
\nabla_{A_2} Loss &= -2 \cdot \operatorname{diag}(C^{(2)}) \cdot \left( ( \Delta \odot M^{(2)} ) \cdot X_2^\top \right) \in \mathbb{R}^{M \times N} \\
\nabla_{B_2} Loss &= -2 \cdot \operatorname{diag}(C^{(2)}) \cdot \left( ( \Delta \odot M^{(2)} ) \cdot \mathbf{1}_L \right) \in \mathbb{R}^{M \times 1} \\
\nabla_C Loss &= -2 \cdot \Delta \cdot H^\top \in \mathbb{R}^{1 \times 2M} \\
\nabla_d Loss &= -2 \cdot \Delta \cdot \mathbf{1}_L \in \mathbb{R}^{1 \times 1} \\
\end{aligned}
} \\ 
5. \text{Вычислить новые коэффициенты по правилу} \\
G_{i + 1} = G_{i} - \lambda \cdot \nabla Loss(G_i) \\
 \{A_1, A_2, B_1, B_2, C, d \} = G_{i+1} \\

6. Если \:\; ||\nabla Loss(G_i)|| < \epsilon_2
```
-----------------------------------------------------------------------------
2. Модульная архитектура проекта
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
 -----------------------------------------------------------------------------
3. Алгебраическая основа дифференцирования для градиентного спуска
# Gradient calculation

## Conitions
### Dimensions
```math
L, N, M \in \N
```
### Constants
```math
X_1, X_2 \in \R^{N \times L} \\
Y_{exp} \in \R^{1 \times L} \\
t \in \R
1_L = \left(1, \dots, 1 \right) \in \R^{1 \times L}
```
### Vatiables
```math
A_1, A_2 \in \R^{M \times N} \\
B_1, B_2 \in \R^{M \times 1} \\
C \in \R^{1 \times 2M} \\
d \in \R
```
### Вesignations
```math
E^i - i_{th} \, element \, of \, column-vector \:E \\
E_j - j_{th} \, element \, of \, row-vector \: E \\
E^i_j - element \, in \, the \, i_{th} \, row \, and \, j_{th} \, column\,  of \, matrix \: E
```
### New operands
```math
\left[ CReLU(A, t) \right]_{ij} = 
\begin{cases}
0, & A_{ij} < 0 \\
A_{ij}, & 0 \le A_{ij} \le t \\
t, & A_{ij} > t
\end{cases} 
```
```math
\left[ A || B \right]_{ij} = 
\begin{cases}
A_{ij}, & i \le m_A \\
B{ij}, & i > m_A
\end{cases}
```
### General expression
```math
Loss(A_1, A_2, B_1, B_2, C, d) = \frac{1}{L} \cdot ||Y_{exp} - (C \cdot CReLU(\left[ A_1 \times X_1 + B_1 \cdot 1_L \right] || \left[ A_2 \times X_2 + B_2 \cdot 1_L \right] , t) + d)||^2_2
```
## Derivatives
### Matrix-by-matrix mult
```math
(\frac{\partial (A \cdot B)}{\partial A})_{i j p q} = \frac{\partial (A \cdot B)_{ij}}{\partial A_{pq}} = \frac{\partial}{\partial A_{pq}}\sum_k A_{ik} \cdot B_{kj} = \delta_{ip} \cdot B_{qj}
```
```math
(\frac{\partial (A \cdot B)}{\partial B})_{i j p q} = \frac{\partial (A \cdot B)_{ij}}{\partial B_{pq}} = \frac{\partial}{\partial B_{pq}}\sum_k A_{ik} \cdot B_{kj} = A_{ip} \cdot \delta_{pj}
```
### Matrix-by-vector mult
```math
(\frac{\partial (A \cdot u)}{\partial A})_{p i j} = \frac{\partial (A \cdot u)_{p}}{\partial A_{ij}} = \frac{\partial}{\partial A_{ij}}\sum_k A_{pk} \cdot u_{k} = \delta_{ip} \cdot u_j
```
```math
(\frac{\partial (A \cdot u)}{\partial u})_{i p} = \frac{\partial (A \cdot u)_{i}}{\partial u_p} = \frac{\partial}{\partial u_p} \sum_k A_{ik} \cdot u_{k} = A_{ip}
```

### Vector-by-matrix mult
```math
(\frac{\partial (u \cdot A)}{\partial A})_{p i j} = \frac{\partial (u \cdot A )_{p}}{\partial A_{ij}} = \frac{\partial}{\partial A_{ij}}\sum_k  u_{k} \cdot A_{kp} = u_i \cdot \delta_{jp}
```
```math
(\frac{\partial (u \cdot A)}{\partial u})_{pj} = \frac{\partial (u \cdot A)_{j}}{\partial u_p} = \frac{\partial}{\partial u_p}\sum_k  u_{k} \cdot A_{kj} = A_{pj}
```

### Matrix concatenation by row
```math
(\frac{\partial (A || B)}{\partial A})_{i j p q} = \frac{\partial (A || B)_{ij}}{\partial A_{ p q}} = \delta_{ip} \cdot \delta_{jq} \ 1_{i \le m_A}
```
```math
(\frac{\partial (A || B)}{\partial B})_{i j p q} = \frac{\partial (A || B)_{ij}}{\partial B_{ p q}} = \delta_{ip} \cdot \delta_{jq} \ 1_{i>m_A}
```

### CReLU from matrix
```math
(\frac{\partial (CReLU(A, t))}{\partial A})_{ijpq} = (\frac{\partial (CReLU(A, t))_{ij}}{\partial A_{pq}}) = \delta_{ip} \cdot \delta_{jq} \cdot 1_{0 < A_{ij} < t} 
```

### Norm
```math
(\frac{\partial \vert\vert A \vert\vert^2_2}{\partial A})_{ij} = 2A_{ij}
```

## Chain rule for multi-dimensions functions    
```math    
Y = F(z_1, \dots, z_n) \\
z_i = G(x_1, \dots, x_k) \\
\frac{\partial Y}{\partial x_p} \\
dY = \sum_{i=1}^n \frac{\partial Y}{\partial z_i} \, dz_i \\
dz_i = \sum_{j=1}^k \frac{\partial z_i}{\partial x_j} \, dx_j \\
dY = \sum_{j=1}^k \, \left[ \sum_{i=1}^n \frac{\partial Y}{\partial z_i} \cdot \frac{\partial z_i}{\partial x_j} \right] dx_j \\ \\
```

## Sub-functions
```math
Loss(\Delta) = \vert\vert\Delta\vert\vert^2_2 \in \R \\
\Delta(Y) = Y_{exp} - Y \in \R^{1 \times L}\\
Y(C, H, d) = C \cdot H + d \cdot 1_L \in \R^{1 \times L} \\
C \in \R^{1 \times 2M} \\
d \in \R \\ 
H(Z) = CReLU(Z, t) \in \R^{2M \times L}\\
t \in \R \\
Z(Z_1, Z_2) = Z_1 \vert \vert Z_2 \in \R^{2M \times L}\\
Z_1(A_1, B_1) = A_1 \cdot X_1 + B_1 \cdot 1_L \in \R^{M \times L}\\
X_1 \in \R^{N \times L} \\
A_1 \in \R^{M \times N} \\
B_1 \in \R^{M \times 1} \\
Z_2(A_2, B_2) = A_2 \cdot X_2 + B_2 \cdot 1_L \in \R^{M \times L} \\
X_2 \in \R^{N \times L} \\
A_2 \in \R^{M \times N} \\
B_2 \in \R^{M \times 1} \\
```
## Partial-deriviation
```math
(\frac{\partial Loss}{\partial \Delta})_{l} = 2\Delta_{l} 
```
---
```math
(\frac{\partial \Delta}{\partial Y })_{ls} = - \delta_{ls}
```
---
```math
(\frac{\partial Y}{\partial C})_{su} = H_{us}
```
```math
(\frac{\partial Y}{\partial H})_{smn} = C_m \cdot \delta_{sn}
```
```math
(\frac{\partial Y}{\partial d})_{s} = 1
```
---
```math
(\frac{\partial H}{\partial Z})_{mnkq} = \delta_{mk} \cdot \delta_{nq} \cdot 1_{0 \le Z_{kq} \le t}
```
---
```math
(\frac{\partial Z}{\partial Z_1})_{kqrc} =  \delta_{kr} \cdot \delta_{qc} \ 1_{k \le M}
```
---
```math
(\frac{\partial Z}{\partial Z_2})_{kqrc} = \delta_{kr} \cdot \delta_{qc} \ 1_{k>M}
```
---
```math
(\frac{\partial Z_1}{\partial A_1})_{rcij} = \delta_{ri} \cdot X_{1 \: jc} 
```
```math
(\frac{\partial Z_1}{\partial B_1})_{rci} = \delta_{ri} 
```
---
```math
(\frac{\partial Z_2}{\partial A_2})_{rcij} = \delta_{ri} \cdot X_{2 \: jc} 
```
```math
(\frac{\partial Z_2}{\partial B_2})_{rci} = \delta_{ri}
```

## Build it all
```math
(\frac{\partial Loss}{\partial A_1})_{ij} = 
\sum_{l=1}^L 2\Delta_{l} 
\sum_{s=1}^L  - \delta_{ls}
\sum_{m=1}^{2M}\sum_{n=1}^L C_m \cdot \delta_{sn}
\sum_{k=1}^{2M}\sum_{q=1}^L \delta_{mk} \cdot \delta_{nq} \cdot 1_{0 \le Z_{kq} \le t}
\sum_{r=1}^M\sum_{c=1}^L \delta_{kr} \cdot \delta_{qc} \cdot 1_{k>M}
\delta_{ri} \cdot X_{1 \: jc} =
-2 \cdot \sum_{c=1}^L 
\Delta_{c} \cdot  C_i \cdot X_{1 \: jc} \cdot 1_{0 \le Z_{ic} \le t}
```
```math
(\frac{\partial Loss}{\partial A_2})_{ij} = 
-2 \cdot \sum_{c=1}^L 
\Delta_{c} \cdot  C_{M+i} \cdot X_{2 \: jc} \cdot 1_{0 \le Z_{ic} \le t} 
```
```math
(\frac{\partial Loss}{\partial B_1})_{i} = 
-2 \cdot \sum_{c=1}^L 
\Delta_{c} \cdot  C_i  \cdot 1_{0 \le Z_{ic} \le t}
```
```math
(\frac{\partial Loss}{\partial B_2})_{i} = 
-2 \cdot \sum_{c=1}^L 
\Delta_{c} \cdot  C_{M+i}  \cdot 1_{0 \le Z_{ic} \le t}
```
```math
(\frac{\partial Loss}{\partial C})_{u} = 
-2 \cdot \sum_{s=1}^L
\Delta_{s} \cdot H_{us}
```
```math
(\frac{\partial Loss}{\partial d})_{u} = 
-2 \cdot \sum_{s=1}^L
\Delta_{s}
```

## Matrix veiw
```math
\Delta \in \R^{1 \times L} - \text{error vector} \\
C \in \R^{1 \times 2M} - \text{second layer weights vecotr} \\
X_1, X_2 \in \R^{N \times L} - \text{input vectors} \\
M \in \{0, 1\}^{2M \times L} - \text{mask by Z-elements}\\
M_{ij} = \begin{cases} 1, & 0 \le Z_{ij} \le t \\ 0,& Z_{ij} < 0 \:\vert\: Z_{ij} > t \end{cases} \\
H = CReLU(Z, t) \in \R^{2M \ times L} \\
1_L \in  \R^{L \times 1} - \text{row vector of ones}
```
---
```math
C^{(1)} = C_{[:, 0:M]} \in \R^{1 \times M}, \:\:\:\:\: C^{(2)} = C_{[:, M:M2]} \in \R^{1 \times M} \\
M^{(1)} = M_{[0:M, :]} \in \R^{M \times ML}, \:\:\:\:\: M^{(2)} = M_{[M:M2, :]} \in \R^{M \times L}
```
---
```math
\odot - \text{element-by-element product} \\
diag(\cdot) - \text{diagonal matrix from vector}
```
---
```math
\boxed{
\begin{aligned}
\nabla_{A_1} Loss &= -2 \cdot \operatorname{diag}(C^{(1)}) \cdot \left( ( \Delta \odot M^{(1)} ) \cdot X_1^\top \right) \in \mathbb{R}^{M \times N} \\
\nabla_{B_1} Loss &= -2 \cdot \operatorname{diag}(C^{(1)}) \cdot \left( ( \Delta \odot M^{(1)} ) \cdot \mathbf{1}_L \right) \in \mathbb{R}^{M \times 1} \\
\nabla_{A_2} Loss &= -2 \cdot \operatorname{diag}(C^{(2)}) \cdot \left( ( \Delta \odot M^{(2)} ) \cdot X_2^\top \right) \in \mathbb{R}^{M \times N} \\
\nabla_{B_2} Loss &= -2 \cdot \operatorname{diag}(C^{(2)}) \cdot \left( ( \Delta \odot M^{(2)} ) \cdot \mathbf{1}_L \right) \in \mathbb{R}^{M \times 1} \\
\nabla_C Loss &= -2 \cdot \Delta \cdot H^\top \in \mathbb{R}^{1 \times 2M} \\
\nabla_d Loss &= -2 \cdot \Delta \cdot \mathbf{1}_L \in \mathbb{R}^{1 \times 1} \\
\end{aligned}
}
```