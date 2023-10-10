from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import torch
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'Using device {device}')


class Experiment:

    def __init__(self, model, train_loader: DataLoader, test_loader: DataLoader,
                 metrics=None, store_path=False, name='experiment'):
        self.model = model.to(device)

        if metrics is None:
            metrics = []

        self.metrics = metrics
        self.store_path = store_path
        self.name = name

        self.train_loader = train_loader
        self.test_loader = test_loader
        self.path = []

    def run(self, loss_fn, optimizer, epochs=10, batch_size=32, verbose=1):
        train_loss, test_loss = [], []

        metrics = {metric.name: [] for metric in self.metrics}

        for epoch in range(epochs):
            losses = []
            for i, (X, y) in enumerate(self.train_loader):
                X, y = X.to(device), y.to(device)
                y_pred = self.model(X)

                loss = loss_fn(y_pred, y)
                losses.append(loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            train_loss.append(np.mean(losses))

            with torch.no_grad():
                losses = []
                for i, (X, y) in enumerate(self.test_loader):
                    X, y = X.to(device), y.to(device)
                    y_pred = self.model(X)

                    loss = loss_fn(y_pred, y)
                    losses.append(loss.item())
                test_loss.append(np.mean(losses))

            if verbose == 1:
                print(f'Epoch {epoch+1}/{epochs} - loss: {train_loss[-1]} - test_loss: {test_loss[-1]}')

            if self.store_path:
                self.path.append(self.model.parameters().clone().detach().cpu())

        return train_loss, test_loss
