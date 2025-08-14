from numpy import zeros, int32
from numpy.typing import  NDArray

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
    } # Верхний регистр - белы, нижний регист - черные, буква - фигура, значение ключа - код цвета и код фигуры

NUMBER_OF_SQUARES = 64
NUMBER_OF_PIECE_TYPE = 5
NUMBER_OF_COLORS = 2
KING_BASE = [i * (NUMBER_OF_PIECE_TYPE * NUMBER_OF_COLORS * NUMBER_OF_SQUARES)
                             for i in range(NUMBER_OF_SQUARES)]
TYPE_COLOR_BASE = [[type * (NUMBER_OF_COLORS * NUMBER_OF_SQUARES) +
                                            color * NUMBER_OF_SQUARES 
                                            for color in range(NUMBER_OF_COLORS)] 
                                        for type in range(NUMBER_OF_PIECE_TYPE)]

def cross_index_of_three(king_square: int, piece_color: int, piece_type: int, piece_square: int) -> int:
    return KING_BASE[king_square] + TYPE_COLOR_BASE[piece_type][piece_color] + piece_square

def fen_to_indices(fen: str) -> NDArray:
    result = zeros(shape=(2, 32), dtype=int32)
    a = zeros(10)
    piece_codes = [] # кортеж из (цвет фигуры, тип фигуры, клетка фигуры)
    kings = [-1, -1]
    side = 0
    counter = 0 # количество обнаруженных фигур (максимум - 32)
    square = 0 # код рассматриваемой кледки
    for let in fen:
        if counter > 32:
            raise ValueError("invalid FEN: too many pieces")

        match let:
            case '/': # скачек на следующую клетку
                continue
            case ' ': # пробел перед буквой цвета stm
                continue
            case 'w': # stm - белые
                side = 0
                break
            case 'b': # stm - черные
                side = 1
                break
            case 'k': # черный король
                kings[1] = square
                square += 1
                counter += 1
                continue
            case 'K': # белый король
                kings[0] = square
                square += 1
                counter += 1
                continue
            case _:            
                if '0' < let < '9': # скачек через k клеток
                    square += int(let)
                    continue
                if let in alphabet.keys(): #буква фигуры
                    piece_codes.append((*alphabet[let], square))
                    counter += 1
                    square += 1
                    continue
                raise ValueError(f"invalid FEN: unknown letter - {let}")
            
        print(let)
        print(c)

    if square != 64:
        raise ValueError(f"invalid FEN: too many or not enought squares - {square}")

    if (kings[0] == -1) or (kings[1] == -1):
        raise ValueError("invalid FEN: not enought kings")

    i = 0 
    for combination in piece_codes:
        result[0][i] = (cross_index_of_three(kings[side], *combination))
        result[1][i] = (cross_index_of_three(kings[1 - side], *combination))
        i+= 1

    return result
