import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch import nn
from typing import Literal,List, Tuple
from src.data_load import load_cifar10
import plotly.express as px
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def visualize_linear_probe_predictions(
    model: nn.Module, device: Literal["cuda", "cpu"], num_samples: int = 10
) -> None:
    """
    Visualize predictions of a linear probe model on random CIFAR10 test samples.

    Args:
        model: Trained PyTorch model for CIFAR10 classification.
        device: 'cuda' or 'cpu' for model execution.
        num_samples: Number of samples to visualize (default: 10).

    Displays:
        Matplotlib plot of images with true and predicted labels.
    """
    if not isinstance(num_samples, int) or num_samples <= 0:
        raise ValueError("num_samples must be a positive integer")

    if next(model.parameters()).device.type != device:
        raise RuntimeError(
            f"Model is on {next(model.parameters()).device.type}, but device argument is {device}"
        )

    model.eval()
    _, test_loader = load_cifar10(batch_size=1)
    classes = [
        "plane",
        "car",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ]

    selected_samples = []
    with torch.no_grad():
        for images, labels in test_loader:
            if len(selected_samples) < num_samples:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                predicted = outputs.argmax(1)
                selected_samples.append((images.cpu(), labels.cpu(), predicted.cpu()))
            else:
                break

    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    fig.suptitle("Linear Probe Predictions", fontsize=16)

    for i, (image, label, prediction) in enumerate(selected_samples):
        ax = axes[i // 5, i % 5]
        img = image.squeeze().permute(1, 2, 0).numpy()
        mean, std = np.array([0.4914, 0.4822, 0.4465]), np.array([0.247, 0.243, 0.261])
        img = np.clip(std * img + mean, 0, 1)

        ax.imshow(img)
        ax.axis("off")
        ax.set_title(
            f"True: {classes[label.item()]}\nPred: {classes[prediction.item()]}"
        )
        ax.title.set_color("green" if label == prediction else "red")

    plt.tight_layout()
    plt.show()


def get_representations(
    model: torch.nn.Module, data_loader: DataLoader, device: str = "cuda"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract representations and labels from a model using a given data loader.

    Args:
        model (torch.nn.Module): The model to extract representations from.
        data_loader (DataLoader): DataLoader containing the dataset.
        device (str): Device to run the model on ('cuda' or 'cpu').

    Returns:
        Tuple[np.ndarray, np.ndarray]: Representations and corresponding labels.
    """
    model.eval()
    representations = []
    labels = []

    with torch.no_grad():
        for images, batch_labels in data_loader:
            images = images.to(device)
            batch_representations = model.encoder(images)
            representations.append(batch_representations.cpu().numpy())
            labels.append(batch_labels.numpy())

    return np.concatenate(representations), np.concatenate(labels)


def visualize_representations(
    representations: np.ndarray,
    labels: np.ndarray,
    title: str = "Visualization of Representations",
) -> None:
    """
    Visualize representations using PCA and t-SNE.

    Args:
        representations (np.ndarray): 2D array of representations.
        labels (np.ndarray): 1D array of corresponding labels.
        title (str): Title for the plots.

    Returns:
        None: Displays the PCA and t-SNE plots.
    """
    # Perform PCA
    pca = PCA(n_components=2)
    pca_representations = pca.fit_transform(representations)

    # Perform t-SNE
    tsne = TSNE(n_components=2, perplexity=30)
    tsne_representations = tsne.fit_transform(representations)

    # Create DataFrames for plotting
    pca_df = pd.DataFrame(
        {
            "PCA Component 1": pca_representations[:, 0],
            "PCA Component 2": pca_representations[:, 1],
            "Class": labels,
        }
    )

    tsne_df = pd.DataFrame(
        {
            "t-SNE Component 1": tsne_representations[:, 0],
            "t-SNE Component 2": tsne_representations[:, 1],
            "Class": labels,
        }
    )

    # Plot PCA results
    fig_pca = px.scatter(
        pca_df,
        x="PCA Component 1",
        y="PCA Component 2",
        color="Class",
        title=f"PCA {title}",
    )
    fig_pca.update_layout(
        width=800,
        height=800,
        xaxis=dict(scaleanchor="y", scaleratio=1),
        yaxis=dict(scaleanchor="x", scaleratio=1),
    )
    fig_pca.show()

    # Plot t-SNE results
    fig_tsne = px.scatter(
        tsne_df,
        x="t-SNE Component 1",
        y="t-SNE Component 2",
        color="Class",
        title=f"t-SNE {title}",
    )
    fig_tsne.update_layout(
        width=800,
        height=800,
        xaxis=dict(scaleanchor="y", scaleratio=1),
        yaxis=dict(scaleanchor="x", scaleratio=1),
    )
    fig_tsne.show()


