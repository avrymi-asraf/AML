import pandas as pd
import plotly.express as px
from IPython.display import clear_output
from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from typing import Callable, Tuple, Dict, Optional


def train_vicreg(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    optimizer: Optimizer,
    loss_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    loss_function_detailed: Callable[
        [torch.Tensor, torch.Tensor], Tuple[torch.Tensor, Dict[str, float]]
    ],
    epochs: int,
    device: str,
    run_test: bool = True,
    scheduler: Optional[_LRScheduler] = None,
) -> pd.DataFrame:
    """
    Train a VICReg model and optionally evaluate it on a test set.

    Args:
        model (nn.Module): VICReg model. Must return projections in forward pass.
        train_loader (DataLoader): Yields (x1, x2) where x1, x2 are augmented views.
                                   Shape: (batch_size, channels, height, width)
        test_loader (DataLoader): Same format as train_loader. Used if run_test=True.
        optimizer (Optimizer): Optimizer for training.
        loss_function (Callable): VICReg loss function. Takes (z1, z2) as input.
        loss_function_detailed (Callable): Returns (loss, loss_components).
                                           loss_components is a dict with keys:
                                           'inv_loss', 'var_loss', 'cov_loss', 'loss'.
        epochs (int): Number of training epochs.
        device (str): 'cuda' or 'cpu'.
        run_test (bool): Whether to evaluate on test set after each epoch.
        scheduler (Optional[_LRScheduler]): LR scheduler.

    Returns:
        pd.DataFrame  training/test metrics for each epoch.
    """
    recorder = pd.DataFrame(
        columns=[
            "epoch",
            "epoch_loss",
            "inv_loss",
            "var_loss",
            "cov_loss",
            "test_loss",
        ],
        index=range(epochs),
    )

    for epoch in range(epochs):
        model.train()
        epoch_loss = 1.0
        for i, (x1, x2) in tqdm(enumerate(train_loader)):
            x1, x2 = x1.to(device), x2.to(device)
            optimizer.zero_grad()
            z1, z2 = model(x1), model(x2)
            loss = loss_function(z1, z2)
            loss.backward()
            epoch_loss = loss.item() * 0.3 + epoch_loss * 0.7
            optimizer.step()

        if scheduler:
            scheduler.step()

        if run_test:
            model.eval()
            with torch.no_grad():
                inv_loss, var_loss, cov_loss, test_loss = 0.0, 0.0, 0.0, 0.0
                for i, (x1, x2) in enumerate(test_loader):
                    x1, x2 = x1.to(device), x2.to(device)
                    z1, z2 = model(x1), model(x2)
                    _, loss_components = loss_function_detailed(z1, z2)
                    inv_loss += loss_components["inv_loss"]
                    var_loss += loss_components["var_loss"]
                    cov_loss += loss_components["cov_loss"]
                    test_loss += loss_components["loss"]

                recorder.loc[epoch] = {
                    "epoch": epoch,
                    "epoch_loss": epoch_loss,
                    "inv_loss": inv_loss,
                    "var_loss": var_loss,
                    "cov_loss": cov_loss,
                    "test_loss": test_loss,
                }
                clear_output(wait=True)
                px.line(
                    recorder,
                    y=["epoch_loss", "inv_loss", "var_loss", "cov_loss", "test_loss"],
                ).show()

    return recorder


def train_linear_probe(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    epochs: int,
    device: str,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
) -> pd.DataFrame:
    """
    Train a linear probe classifier and evaluate it on a test set.

    Args:
        model (nn.Module): Linear probe model. Must have a 'predict' method for evaluation.
        train_loader (DataLoader): Yields (x, y) where x is the input and y is the label.
                                   Shape: (batch_size, channels, height, width)
        test_loader (DataLoader): Same format as train_loader. Used for evaluation.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        loss_function (Callable): Loss function. Takes (predictions, labels) as input.
        epochs (int): Number of training epochs.
        device (str): 'cuda' or 'cpu'.
        scheduler (Optional[torch.optim.lr_scheduler._LRScheduler]): LR scheduler.

    Returns:
        pd.DataFrame: DataFrame containing training/test metrics for each epoch.
    """
    recorder = pd.DataFrame(
        columns=[
            "epoch",
            "epoch_loss",
            "test_loss",
            "accuracy",
        ],
        index=range(epochs),
    )

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for i, (x, y) in tqdm(enumerate(train_loader)):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            prd = model(x)
            loss = loss_function(prd, y)
            loss.backward()
            epoch_loss += loss.item()
            optimizer.step()

        if scheduler:
            scheduler.step()

        model.eval()
        with torch.no_grad():
            test_loss, correct = 0, 0
            for i, (x, y) in enumerate(test_loader):
                x, y = x.to(device), y.to(device)
                prd = model(x)
                test_loss += loss_function(prd, y).item()
                correct += (model.predict(x) == y).sum().item()

            recorder.loc[epoch] = {
                "epoch": epoch,
                "epoch_loss": epoch_loss / len(train_loader.dataset),
                "test_loss": test_loss / len(test_loader.dataset),
                "accuracy": correct / len(test_loader.dataset),
            }

        clear_output(wait=True)
        px.line(
            recorder,
            y=["epoch_loss", "test_loss", "accuracy"],
        ).show()

    return recorder
