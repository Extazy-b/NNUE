import argparse
from random import shuffle

from src.python.train import Trainer, Validator
from src.python.dataset import Dataset

parser = argparse.ArgumentParser(description="Model training settings")

parser.add_argument('--train-dir', type=str, default='.data/processed/train', help='Путь к директории с данными для обучения')
parser.add_argument('--val-dir', type=str, default='.data/processed/val', help='Путь к директории с данными для валлидации')

parser.add_argument('--error', type=int, default=-6, help='Порядок допустимой ошибки валидации')
parser.add_argument('--step', type=float, default=1, help='Шаг градиентного спуска')
parser.add_argument('--max-times', type=int, default=1000, help='Максимальное количество итераций обучения')
parser.add_argument('--reboot', type=bool, default=False, help='Перезапуск обучения')
parser.add_argument('--model-path', type=str, default='.model/model.bin', help='Путь к модели')

parser.add_argument('--max-grad-times', type=int, default=1000, help='Максимальное количество итераций градиентного спуска на один батч')

args = parser.parse_args()

traint_dataset = Dataset(args.train_dir)
validation_dataset = Dataset(args.val_dir)

trainer = Trainer(error = args.error, 
                              step = args.step, 
                              max_times  = args.max_grad_times, 
                              datasert= traint_dataset, 
                              reboot = args.reboot, 
                              model_path=args.model_path)
                              
validator = Validator(validation_dataset)

order = list(range(len(traint_dataset)))

for global_iter in range(args.max_times):
    shuffle(order)

    for batch_index in order:
        try:
            trainer.train_on_one_batch(batch_index)
        except Exception as e:
            trainer.end() # TODO errorsaver
            print(f"[ERROR] {e}")
            break

    if validator.check():
        break
