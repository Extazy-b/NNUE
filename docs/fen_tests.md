```python
alphabet = {
    'P': (0, 0),
    'N': (0, 1),
    'B': (0, 2),
    'R': (0, 3),
    'Q': (0, 4),
    'p': (1, 0),
    'n': (1, 1),
    'b': (1, 2),
    'r': (1, 3),
    'q': (1, 4),
    }
```
```python
index = king_square * 640 + piece_type * 128 + piece_color * 64 + piece_square
```
## Test 1

### Input:
3qK3/8/8/8/8/8/8/3QK3 w

```math
\begin{matrix}
\cdot&\cdot&\cdot&q&k&\cdot&\cdot&\cdot \\
\cdot&\cdot&\cdot&\cdot&\cdot&\cdot&\cdot&\cdot\\
\cdot&\cdot&\cdot&\cdot&\cdot&\cdot&\cdot&\cdot \\
\cdot&\cdot&\cdot&\cdot&\cdot&\cdot&\cdot&\cdot \\
\cdot&\cdot&\cdot&\cdot&\cdot&\cdot&\cdot&\cdot \\
\cdot&\cdot&\cdot&\cdot&\cdot&\cdot&\cdot&\cdot \\
\cdot&\cdot&\cdot&\cdot&\cdot&\cdot&\cdot&\cdot \\
\cdot&\cdot&\cdot&Q&K&\cdot&\cdot&\cdot \\
\end{matrix}
```

### Desctiption:
- stm king square - 60
- sntm king square - 4
- q
    - type: 4
    - color: 1
    - square: 3
    - stm index: 60 * 640 + 4 * 128 + 1 * 64 + 3 = 38979
    - sntm index: 4 * 640 + 4 * 128 + 1 * 64 + 3 = 3139
- Q
    - type:4
    - color: 0
    - square: 59
    - stm index: 60 * 640 + 4 * 128 + 0 * 64 + 59 = 38971
    - sntm index: 4 * 640 + 4 * 128 + 0 * 64 + 59 = 3131

### Output:
```math
\left(
\begin{matrix}
38979&3139\\
38971&3131\\
\end{matrix}
\right)
```

## Test 2

### Input:
rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1

```math
\begin{matrix}
r&n&b&q&k&b&n&r \\
p&p&p&p&p&p&p&p\\
\cdot&\cdot&\cdot&\cdot&\cdot&\cdot&\cdot&\cdot \\
\cdot&\cdot&\cdot&\cdot&\cdot&\cdot&\cdot&\cdot \\
\cdot&\cdot&\cdot&\cdot&\cdot&\cdot&\cdot&\cdot \\
\cdot&\cdot&\cdot&\cdot&\cdot&\cdot&\cdot&\cdot \\
P&P&P&P&P&P&P&P\\
R&N&B&Q&K&B&N&R\\
\end{matrix}
```

### Desctiption:
- stm king square - 60
- sntm king square - 4
-  P
    - Type: 0
    - Color: 0
    - Square:
    - stm index:
    - sntm index:
-  N
    - Type: 1
    - Color: 0
    - Square:
    - stm index:
    - sntm index:
-  B
    - Type: 2
    - Color: 0
    - Square:
    - stm index:
    - sntm index:
-  R
    - Type: 3
    - Color: 0
    - Square:
    - stm index:
    - sntm index:
-  Q
    - Type: 4
    - Color: 0
    - Square:
    - stm index:
    - sntm index:
-  p
    - Type: 0
    - Color: 1
    - Square:
    - stm index:
    - sntm index:
-  n
    - Type: 1
    - Color: 1
    - Square:
    - stm index:
    - sntm index:
-  b
    - Type: 2
    - Color: 1
    - Square:
    - stm index:
    - sntm index:
-  r
    - Type: 3
    - Color: 1
    - Square:
    - stm index:
    - sntm index:
-  q
    - Type: 4
    - Color: 1
    - Square:
    - stm index:
    - sntm index:

### Output:
```math
\left(
\begin{matrix}

\end{matrix}
\right)
```