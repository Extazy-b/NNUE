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
    } # Верхний регистр - белые, нижний регист - черные, буква - фигура, значение ключа - код цвета и код фигуры

NUMBER_OF_SQUARES = 64

NUMBER_OF_PIECE_TYPE = 5

NUMBER_OF_COLORS = 2

KING_BASE = [i * (NUMBER_OF_PIECE_TYPE * NUMBER_OF_COLORS * NUMBER_OF_SQUARES)
                             for i in range(NUMBER_OF_SQUARES)] # База королей для сквозной индексации 

TYPE_COLOR_BASE = [[type * (NUMBER_OF_COLORS * NUMBER_OF_SQUARES) +
                                            color * NUMBER_OF_SQUARES 
                                            for color in range(NUMBER_OF_COLORS)] 
                                        for type in range(NUMBER_OF_PIECE_TYPE)] # База кода цвета фигуры для сквозной индексации

DATASET_SIZE = 12_958_036

INPUT_VECTOR_SIZE = 64 # 40_960
BATCH_SIZE = 100 # 100_000
MATE_SCORE = 32000