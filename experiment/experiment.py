import typing

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LRScheduler
from typing import Dict, Callable
from itertools import product

import torch
import numpy as np

import wandb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device {device}')


class Experiment:

    def __init__(self, model, train_loader: DataLoader, test_loader: DataLoader | None, optimizer,
                 loss_fn: typing.Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None,
                 lr_scheduler: LRScheduler | None = None,
                 metrics: Dict[str, Callable] = None, store_path=False, name='experiment',
                 mode="classification",  # classification, regression or function
                 stop_condition: Callable = None,
                 wandb_config=None
                 ):
        self.model = model.to(device)

        if metrics is None:
            metrics = {}
        elif mode == "function":
            raise ValueError("Function mode does not support metrics")

        self.metrics = metrics
        self.store_path = store_path
        self.name = name
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.mode = mode
        self.lr_scheduler = lr_scheduler
        self.stop_condition = stop_condition

        self.train_loader = train_loader
        self.test_loader = test_loader
        self.path = []
        if self.store_path:
            params = []
            for param in self.model.parameters():
                params.append(param.detach().cpu().numpy().copy())
            self.path.append(params)
        self.metrics_history = {key + "_" + mode: [] for key, mode in product(metrics.keys(), ["train", "test"])}
        self.train_loss = []
        self.test_loss = []

        self.wandb_config = wandb_config

    def _feed_batch(self, X, y, eval=False):
        torch.set_grad_enabled(not eval)
        X, y = X.to(device), y.to(device)
        y_out = self.model.forward(X)

        if self.mode != "function":
            loss = self.loss_fn(y_out, y)
        else:
            loss = y_out  # Minimizing the function value

        if not eval:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        if self.mode == "classification":
            y_pred = np.argmax(y_out.detach().cpu().numpy(), axis=1)
        else:
            y_pred = y_out.detach().cpu().numpy()

        metrics = {}
        for metric in self.metrics:
            metrics[metric + ("_test" if eval else "_train")] = self.metrics[metric](y_pred, y.detach().cpu().numpy())

        torch.set_grad_enabled(True)

        return loss.item(), metrics

    def run_epoch(self, eval=False, verbose=0):
        losses = []
        mode = "_test" if eval else "_train"

        new_metrics = {key + mode: [] for key in self.metrics.keys()}

        for ind, (X, y) in enumerate(self.test_loader if eval else self.train_loader):
            loss, metrics = self._feed_batch(X, y, eval=eval)
            losses.append(loss)

            if verbose >= 1:
                print(f"{ind} batch, loss {loss}")

            for key in metrics.keys():
                new_metrics[key].append(metrics[key])

        for key in new_metrics.keys():
            self.metrics_history[key].append(np.mean(new_metrics[key]))

        (self.test_loss if eval else self.train_loss).append(np.mean(losses))

        if not eval and self.lr_scheduler:
            self.lr_scheduler.step()

    def run(self, epochs=10, verbose=1):

        if self.wandb_config:
            wandb_run = wandb.init(**self.wandb_config, reinit=True)

        for epoch in range(epochs):
            self.model.train()
            self.run_epoch(eval=False, verbose=verbose-1)

            self.model.eval()
            if self.mode != "function":
                self.run_epoch(eval=True, verbose=verbose-1)

            if verbose >= 1:
                print("-" * 50)
                print(f'Epoch {epoch + 1}/{epochs} - train_loss: {self.train_loss[-1]}')
                for key in self.metrics_history.keys():
                    print(f'{key}: {self.metrics_history[key][-1]}')

            if self.store_path:
                params = []
                for param in self.model.parameters():
                    params.append(param.detach().cpu().numpy().copy())
                self.path.append(params)

            result = {"train_loss": self.train_loss[-1]}
            for key in self.metrics_history.keys():
                result[key] = self.metrics_history[key][-1]
            if self.mode != "function":
                result["test_loss"] = self.test_loss[-1]

            if self.wandb_config:
                wandb_run.log(result, step=epoch)

            if self.stop_condition and self.stop_condition(**result):
                if verbose == 1:
                    print("Stopping condition met, stopping training")
                break

        if self.wandb_config:
            wandb_run.finish()

    def get_metric(self, metric: str, mode="test", n_last: int = None):
        arr = self.metrics_history[metric + "_" + mode]
        return arr[-n_last:] if n_last else arr

    def get_loss(self, mode="test", n_last: int = -1):
        arr = self.test_loss if mode == "test" else self.train_loss
        return arr[-n_last:] if n_last else arr

    def get_path(self) -> np.ndarray:
        return np.array(self.path)