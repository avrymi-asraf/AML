from typing import Any, Dict, List, Tuple
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, Subset
import plotly.express as px
from sklearn.neighbors import NearestNeighbors
import faiss
from typing import Callable, Optional, Tuple
import random

from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


from src.data_load import *


def get_representations(
    encoder: torch.nn.Module, dataset: Dataset, device: Optional[str] = None
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
    encoder.eval()
    representations = []
    labels = []
    device = device or DEVICE
    data_loader = DataLoader(
        dataset, batch_size=int(2**10), shuffle=False, num_workers=2, pin_memory=True
    )
    with torch.no_grad():
        for images, batch_labels in tqdm(
            data_loader, desc="Extracting representations"
        ):
            images = images.to(device)
            batch_representations = encoder(images)
            representations.append(batch_representations.cpu().numpy())
            labels.append(batch_labels.numpy())

    return np.concatenate(representations), np.concatenate(labels)


def find_k_nearest_neighbors(
    representations: np.ndarray, k: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Find k nearest neighbors for each representation using FAISS, with optional GPU support.
    Searches in chunks of 100 to manage memory usage.

    Args:
        representations (np.ndarray): 2D array of representations, shape (n_samples, n_features)
        k (int): Number of nearest neighbors to find

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - indices: 2D tensor of indices of k nearest neighbors, shape (n_samples, k)
            - distances: 2D tensor of distances to k nearest neighbors, shape (n_samples, k)
    """
    n_samples, n_features = representations.shape

    # Convert to correct data type
    representations = representations.astype(np.float32)

    # Create the index
    index = faiss.IndexFlatL2(n_features)
    print("Using CPU for nearest neighbor search.")

    # Add vectors to the index
    index.add(representations)

    # Search for k nearest neighbors in chunks
    print("Searching for nearest neighbors...")
    chunk_size = 10
    all_indices = []
    all_distances = []

    for i in tqdm(range(0, n_samples, chunk_size)):
        chunk = representations[i : i + chunk_size]
        distances, indices = index.search(chunk, k + 1)  # +1 to exclude self
        all_indices.append(indices[:, 1:])  # Remove self from results
        all_distances.append(distances[:, 1:])

    # Concatenate results
    indices = np.concatenate(all_indices, axis=0)
    distances = np.concatenate(all_distances, axis=0)

    return torch.tensor(indices), torch.tensor(distances)


def find_nearest_and_farthest(
    representations: torch.Tensor, dataset_representations: torch.Tensor, k: int = 5
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Find k nearest and k farthest neighbors for each representation.

    Args:
    representations (torch.Tensor): (num_queries, representation_dim)
    dataset_representations (torch.Tensor):  (num_samples, representation_dim)
    k (int): (default: 5)

    Returns:
    Tuple[torch.Tensor, torch.Tensor]: nearest (num_queries, k), farthest (num_queries, k)
    """
    distances = torch.cdist(representations, dataset_representations)
    nearest_indices = torch.topk(distances, k, dim=1, largest=False).indices
    farthest_indices = torch.topk(distances, k, dim=1, largest=True).indices
    return nearest_indices, farthest_indices


def select_samples_by_class(
    dataset: Dataset, samples_per_class: int = 1
) -> Dict[int, Dict[str, Any]]:
    """
    Select samples from the dataset, organized by class.

    Args:
    dataset (Dataset):
    samples_per_class (int): (default: 1)

    Returns:
    Dict[int, Dict[str, List]]:
        - 'image':
        - 'names': (string)

    """

    result = {
        class_idx: {"image": [], "names": class_name}
        for class_name, class_idx in dataset.class_to_idx.items()
    }

    total_samples = 0
    target_total = samples_per_class * len(dataset.class_to_idx)

    for image, label in dataset:
        if len(result[label]["image"]) < samples_per_class:
            result[label]["image"].append(image)
            total_samples += 1

        if total_samples == target_total:
            break

    return result


def retrieval_evaluation(
    vic_reg_encoder, near_neig_encoder, test_dataset, train_dataset, device
):
    """
    Perform retrieval evaluation for VICReg and Near Neighbor models.

    Args:
    vic_reg_model (nn.Module):
    near_neig_model (nn.Module): Trained Near Neighbor model
    test_dataset (torch.utils.data.Dataset): Test dataset
    train_dataset (torch.utils.data.Dataset): Training dataset
    device (str): Device to run the models on ('cuda' or 'cpu')

    Returns:
    dict:
    """

    # Select sample images
    samples = select_samples_by_class(test_dataset)

    # Get representations for sample images
    for cls in samples:
        im_to_model = torch.stack(
            [samples[cls]["image"][0], samples[cls]["image"][0].clone()]
        ).to(device)
        samples[cls]["repr_vic_reg"] = (
            vic_reg_encoder(im_to_model).cpu().detach()[0].unsqueeze(0)
        )
        samples[cls]["repr_near_neig"] = (
            near_neig_encoder(im_to_model).cpu().detach()[0].unsqueeze(0)
        )

    # Get representations for all training images
    repr_vic_reg, _ = get_representations(
        vic_reg_encoder, train_dataset, device=device
    )
    repr_near_neig, _ = get_representations(
        near_neig_encoder, train_dataset, device=device
    )

    repr_vic_reg = torch.tensor(repr_vic_reg)
    repr_near_neig = torch.tensor(repr_near_neig)

    # Find nearest and farthest neighbors
    for cls in samples:
        nearest_vic, farthest_vic = find_nearest_and_farthest(
            samples[cls]["repr_vic_reg"], repr_vic_reg
        )
        samples[cls]["repr_vic_reg_near"] = nearest_vic
        samples[cls]["repr_vic_reg_far"] = farthest_vic

        nearest_near_neig, farthest_near_neig = find_nearest_and_farthest(
            samples[cls]["repr_near_neig"], repr_near_neig
        )
        samples[cls]["repr_near_neig_near"] = nearest_near_neig
        samples[cls]["repr_near_neig_far"] = farthest_near_neig

    return samples
