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