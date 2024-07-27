import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch import nn
from typing import Literal, List, Tuple
from src.data_load import load_cifar10
import plotly.express as px
import plotly.graph_objects as go
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import roc_curve, auc


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


def visualize_retrieval_results(samples, train_dataset, num_neighbors=5):
    """
    Visualize retrieval results for VICReg and Near Neighbor models.

    Args:
    samples (Dict[int, Dict[str, Any]]): Output from retrieval_evaluation function.
        Each inner dict contains:
        - 'image': torch.Tensor, the query image
        - 'names': str, class name
        - 'nearest_vic_reg': torch.Tensor, nearest neighbors for VICReg
        - 'farest_vic_reg': torch.Tensor, farthest neighbors for VICReg
        - 'nearest_near_neig': torch.Tensor, nearest neighbors for Near Neighbor
        - 'farest_near_neig': torch.Tensor, farthest neighbors for Near Neighbor
    train_dataset (Dataset): The training dataset used for retrieval
    num_neighbors (int): Number of neighbors to display (default: 5)

    Returns:
    None: Displays the plot
    """
    num_classes = len(samples)
    num_cols = 1 + 4 * num_neighbors  # query + 4 sets of neighbors
    fig, axes = plt.subplots(
        num_classes, num_cols, figsize=(num_cols * 3, 3 * num_classes)
    )
    fig.suptitle(
        "Retrieval Results: Query Images with Nearest and Farthest Neighbors",
        fontsize=16,
    )

    if num_classes == 1:
        axes = axes.reshape(1, -1)

    for idx, (cls, sample) in enumerate(samples.items()):
        query_image = sample["image"]

        # Plot query image
        axes[idx, 0].imshow(query_image.permute(1, 2, 0).cpu().numpy())
        axes[idx, 0].set_title(f"Query\n{sample['names']}")
        axes[idx, 0].axis("off")

        # Helper function to plot neighbors
        def plot_neighbors(start_col, neighbors, title_prefix):
            for j in range(num_neighbors):
                if j < len(neighbors):
                    neighbor_image = neighbors[j]
                    axes[idx, start_col + j].imshow(neighbor_image)
                    axes[idx, start_col + j].set_title(f"{title_prefix} {j+1}")
                axes[idx, start_col + j].axis("off")

        # Plot VICReg nearest neighbors
        plot_neighbors(1, sample["nearest_vic_reg"][:num_neighbors], "VICReg\nNearest")

        # Plot VICReg farthest neighbors
        plot_neighbors(
            1 + num_neighbors,
            sample["farest_vic_reg"][:num_neighbors],
            "VICReg\nFarthest",
        )

        # Plot Near Neighbor nearest neighbors
        plot_neighbors(
            1 + 2 * num_neighbors,
            sample["nearest_near_neig"][:num_neighbors],
            "NearNeig\nNearest",
        )

        # Plot Near Neighbor farthest neighbors
        plot_neighbors(
            1 + 3 * num_neighbors,
            sample["farest_near_neig"][:num_neighbors],
            "NearNeig\nFarthest",
        )

    plt.show()


def visualize_roc(knn_density_cifar10, knn_density_mnist, method_name):
    """
    writed by chatgpt
    Args:
    knn_density_cifar10 (np.ndarray): (n_samples,) 
    knn_density_mnist (np.ndarray): (n_samples,)
    method_name (str): 

    Returns:
    float: AUC score
    """
    scores = np.concatenate([knn_density_cifar10, knn_density_mnist])
    labels = np.concatenate(
        [np.zeros(len(knn_density_cifar10)), np.ones(len(knn_density_mnist))]
    )

    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=fpr,
            y=tpr,
            mode="lines",
            name=f"{method_name} (AUC = {roc_auc:.2f})",
            line=dict(color="blue", width=2),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Random Classifier",
            line=dict(color="gray", width=2, dash="dash"),
        )
    )

    fig.update_layout(
        title=f"Receiver Operating Characteristic (ROC) Curve - {method_name}",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        legend=dict(x=0.7, y=0.1),
        width=800,
        height=600,
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1.05]),
    )

    fig.show()

    print(f"{method_name} AUC: {roc_auc:.4f}")

    return roc_auc


import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import torch

def unnormalize_image(img):
    """
    Unnormalize an image tensor.
    
    Args:
    img (torch.Tensor): Normalized image tensor

    Returns:
    np.ndarray: Unnormalized image array
    """
    mean = torch.tensor([0.4914, 0.4822, 0.4465])
    std = torch.tensor([0.247, 0.243, 0.261])
    img = img * std[:, None, None] + mean[:, None, None]
    return (img.clip(0, 1) * 255).byte().permute(1, 2, 0).cpu().numpy()

def plot_most_anomalous_samples(cifar10_test_dataset, mnist_test_dataset, vic_reg_scores, near_neig_scores, num_samples=7):
    """
    Plot the most anomalous samples according to VICReg and VICReg without generated neighbors using Plotly.
    Images are unnormalized before plotting.

    Args:
    cifar10_test_dataset (Dataset): CIFAR10 test dataset
    mnist_test_dataset (Dataset): MNIST test dataset
    vic_reg_scores (np.ndarray): Pre-computed anomaly scores for VICReg
    near_neig_scores (np.ndarray): Pre-computed anomaly scores for Near Neighbor
    num_samples (int): Number of most anomalous samples to plot

    Returns:
    None: Displays the interactive Plotly plot
    """
    assert len(vic_reg_scores) == len(near_neig_scores), "Scores must have the same length"

    vic_reg_anomalous_indices = np.argsort(vic_reg_scores)[-num_samples:]
    near_neig_anomalous_indices = np.argsort(near_neig_scores)[-num_samples:]

    fig = make_subplots(
        rows=2, cols=num_samples,
        subplot_titles=[f'VICReg: {vic_reg_scores[i]:.2f}' for i in reversed(vic_reg_anomalous_indices)] +
                       [f'Near Neighbor: {near_neig_scores[i]:.2f}' for i in reversed(near_neig_anomalous_indices)],
        vertical_spacing=0.1
    )

    combined_dataset = torch.utils.data.ConcatDataset([cifar10_test_dataset, mnist_test_dataset])

    for i, indices in enumerate([vic_reg_anomalous_indices, near_neig_anomalous_indices]):
        for j, sample_idx in enumerate(reversed(indices)):
            img, _ = combined_dataset[sample_idx]
            img_array = unnormalize_image(img)
            
            fig.add_trace(
                go.Image(z=img_array),
                row=i+1, col=j+1
            )

    fig.update_layout(
        title_text="Most Anomalous Samples",
        height=600,
        width=1200,
    )

    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)

    fig.show()

# Usage example:
# plot_most_anomalous_samples(cifar10_test_dataset, mnist_test_dataset, vic_reg_scores, near_neig_scores)