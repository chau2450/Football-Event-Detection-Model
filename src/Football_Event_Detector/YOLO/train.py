import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
import re
from Football_Event_Detector.YOLO import yolo_v1, label, yolo_loss, train

class EarlyStopping:
    def __init__(self, patience=10, verbose=False, delta=0, save_path='YOLO_V1_train'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.delta = delta
        self.save_path = save_path

    def __call__(self, val_loss, model, optimizer, epoch):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model, optimizer, epoch)
        elif val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model, optimizer, epoch)
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, val_loss, model, optimizer, epoch):
        '''Saves model when validation loss decreases.'''
        if self.verbose:
            print(f'Validation loss decreased to {val_loss:.6f} at epoch {epoch}. Saving model...')
        filename = f"{self.save_path}/yolo_best_epoch_{epoch}.pth.tar"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': val_loss,
        }, filename)
        

class YOLOTrainer:
    def __init__(self, model, train_loader, test_loader, optimizer, loss_fn, device, writer, resume_training=False, last_saved_model=None, epochs=1000, early_stopping_patience=20):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.writer = writer
        self.resume_training = resume_training
        self.last_saved_model = last_saved_model
        self.epochs = epochs
        self.start_epoch = 0
        self.early_stopper = train.EarlyStopping(patience=early_stopping_patience, verbose=True)

        if self.resume_training:
            self._load_checkpoint()

    def _load_checkpoint(self):
        if os.path.isfile(self.last_saved_model):
            print(f"Loading checkpoint '{self.last_saved_model}'")
            checkpoint = torch.load(self.last_saved_model, map_location=self.device)
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.start_epoch = checkpoint.get('epoch', 0)
            print(f"Resumed training from epoch {self.start_epoch}")
        else:
            print(f"No checkpoint found at '{self.last_saved_model}', starting training from scratch")

    def save_model(self, filename):
        checkpoint = {
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epoch": self.start_epoch
        }
        torch.save(checkpoint, filename)

    def evaluate_model(self):
        self.model.eval()
        with torch.no_grad():
            total_loss = 0
            for images, targets in self.test_loader:
                images, targets = images.to(self.device), targets.to(self.device)
                predictions = self.model(images)
                loss = self.loss_fn(predictions, targets)
                total_loss += loss.item()
        return total_loss / len(self.test_loader)

    def train(self):
        for epoch in range(self.start_epoch, self.epochs):
            self.model.train()
            batch_progress = tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc=f"Epoch {epoch + 1}/{self.epochs}", leave=False)

            for batch_idx, (images, targets) in batch_progress:
                images, targets = images.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()
                predictions = self.model(images)
                loss = self.loss_fn(predictions, targets)
                loss.backward()
                self.optimizer.step()
                batch_progress.set_description(f"Epoch {epoch + 1}/{self.epochs} - Loss: {loss.item():.4f}")
                self.writer.add_scalar('Training Loss', loss.item(), epoch * len(self.train_loader) + batch_idx)

            test_loss = self.evaluate_model()
            self.writer.add_scalar('Test Loss', test_loss, epoch)
            self.early_stopper(test_loss, self.model, self.optimizer, epoch)

            if self.early_stopper.early_stop:
                print("Early stopping")
                break

            batch_progress.close()

        self.writer.close()
        torch.cuda.empty_cache()