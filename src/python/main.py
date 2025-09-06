import argparse

from src.python.train import Trainer

parser = argparse.ArgumentParser(description="Model training settings")

parser.add_argument('--error', type=int, default=-6, help='Порядок допустимой ошибки валидации')
parser.add_argument

trainer = Trainer(error, )