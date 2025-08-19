import tempfile
from hypothesis import given, strategies as st

from src.python.dataset import load_raw_fen_file

@given(
    fen=st.text(alphabet="rnbqkpRNBQKP12345678/ -", min_size=1, max_size=80),
    score=st.integers(min_value=-1000, max_value=1000)
)
def test_load_raw_fen_file_with_hypothesis(fen, score):
    # создаём реальный временный файл
    with tempfile.NamedTemporaryFile(mode="w+", delete=False) as f:
        f.write(f"{fen},{score}\n")
        f.flush()
        path = f.name

    # проверяем работу генератора
    result = list(load_raw_fen_file(path))
    assert len(result) == 1
    parsed_fen, parsed_score = result[0]
    assert parsed_fen == fen
    assert parsed_score == score

def test_generator_start():
    gen = load_raw_fen_file("./data/raw/testData.csv")
    assert next(gen) == ("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1", -10)

def test_generator_len():
    gen = load_raw_fen_file("./data/raw/testData.csv")
    assert len(list(gen)) == 10