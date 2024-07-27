import dis
from turtle import distance
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

    Args:
        model (torch.nn.Module):
        dataset (Dataset):
        device (str):

    Returns:
        Tuple[np.ndarray, np.ndarray]:  representations, labels
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


def compute_knn_density(
    train_representations: np.ndarray,
    test_representations: np.ndarray,
    k: int = 2,
    reduce: Optional[str] = "mean",
) -> np.ndarray:
    """
    Compute kNN density test in train

    Args:
        train_representations (np.ndarray):
        test_representations (np.ndarray):
        k (int):
        reduce (str): (default: "mean")

    Returns:
        float: mean of knn density
    """
    index = faiss.IndexFlatL2(train_representations.shape[1])
    index.add(train_representations.astype(np.float32))
    distances, _ = index.search(test_representations.astype(np.float32), k + 1)
    if reduce == "mean":
        return np.mean(distances[:, 1:])
    else:
        return np.mean(distances[:, 1:],axis=1)


def select_samples_by_class(dataset: Dataset) -> Dict[int, Dict[str, Any]]:
    """
    Select samples from the dataset, organized by class.

    Args:
    dataset (Dataset):
    samples_per_class (int): (default: 1)

    Returns:
    Dict[int, Dict[str, List]]:
    - class_idx:(int)
        - 'image':(torch.Tensor, shape: (3, 32, 32))
        - 'names': (string)
        - 'ind_image': (int)

    """

    result = {
        class_idx: {"names": class_name}
        for class_name, class_idx in dataset.class_to_idx.items()
    }

    total_samples = 0
    target_total = len(dataset.class_to_idx)

    for i in range(len(dataset)):
        image, label = dataset[i]
        if not ("image" in result[label]):
            result[label]["image"] = image
            result[label]["ind_image"] = i
            total_samples += 1

        if total_samples == target_total:
            break

    return result


def find_nearest_and_farthest(
    distances: torch.Tensor, query_ind: int, k: int = 5
) -> Tuple[torch.Tensor, torch.Tensor]:
    """ """
    distances = distances[query_ind]
    nearest = torch.topk(distances, k + 1, largest=False).indices[1:]
    farthest = torch.topk(distances, k, largest=True).indices
    return nearest, farthest


def retrieval_evaluation(vic_reg_encoder, near_neig_encoder, train_dataset, device):
    """


    Args:
    vic_reg_model (nn.Module):
    near_neig_model (nn.Module): Trained Near Neighbor model
    test_dataset (torch.utils.data.Dataset): Test dataset
    train_dataset (torch.utils.data.Dataset): Training dataset
    device (str):


    Returns:
    Dict[int, Dict[str, List]]:
    - class_idx:(int)
        - 'image':(torch.Tensor, shape: (3, 32, 32))
        - 'names': (string)
        - 'ind_image': (int)
        - 'repr_vic_reg': (torch.Tensor, shape: (1, representation_dim))
        - 'repr_near_neig': (torch.Tensor, shape: (1, representation_dim))
        - 'vic_reg_near_image': (torch.Tensor, shape: (5,3,32,32))
        - 'vic_reg_far_image': (torch.Tensor, shape: (5,3,32,32))
        - 'near_neig_near_image': (torch.Tensor, shape: (5,3,32,32))
        - 'near_neig_far_image': (torch.Tensor, shape: (5,3,32,32))
    """
    samples = select_samples_by_class(train_dataset)
    repr_vic_reg = torch.tensor(
        get_representations(vic_reg_encoder, train_dataset, device=device)[0]
    )
    repr_near_neig = torch.tensor(
        get_representations(near_neig_encoder, train_dataset, device=device)[0]
    )

    distances_vic_reg = torch.cdist(repr_vic_reg, repr_vic_reg)
    distances_near_neig = torch.cdist(repr_near_neig, repr_near_neig)

    for cls in samples:
        idx_nearet_vic_reg, ind_farest_vic_reg = find_nearest_and_farthest(
            distances_vic_reg, samples[cls]["ind_image"]
        )
        idx_nearet_near_neig, ind_farest_near_neig = find_nearest_and_farthest(
            distances_near_neig, samples[cls]["ind_image"]
        )
        samples[cls]["nearest_vic_reg"] = train_dataset.data[idx_nearet_vic_reg]
        samples[cls]["farest_vic_reg"] = train_dataset.data[ind_farest_vic_reg]

        samples[cls]["nearest_near_neig"] = train_dataset.data[idx_nearet_near_neig]
        samples[cls]["farest_near_neig"] = train_dataset.data[ind_farest_near_neig]

    return samples


def save_pickle(obj, path: str):
    """
    Save an object to a pickle file.

    Args:
    obj: Object to save
    path (str): Path to save the object to
    """
    import pickle

    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pikcle(path: str):
    """
    Load an object from a pickle file.

    Args:
    path (str): Path to the pickle file

    Returns:
    Any: The loaded object
    """
    import pickle

    with open(path, "rb") as f:
        return pickle.load(f)
