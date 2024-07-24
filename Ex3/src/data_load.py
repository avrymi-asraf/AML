from typing import Tuple
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, Subset
import plotly.express as px
from sklearn.neighbors import NearestNeighbors
import faiss
from typing import Callable, Optional, Tuple
import random

from Ex3.ex3_interface import DEVICE


train_transform = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(32, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    ]
)
test_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    ]
)


class VICRegDataset(Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, index):
        img, _ = self.dataset[index]
        z = self.transform(img)
        z_prime = self.transform(img)
        return z, z_prime

    def __len__(self):
        return len(self.dataset)


class CombinedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, labels):
        self.dataset = dataset
        self.labels = labels

    def __getitem__(self, index):
        data, _ = self.dataset[index]
        return data, self.labels[index]

    def __len__(self):
        return len(self.dataset)


def load_vicreg_cifar10(batch_size=256, num_workers=2, root="./data"):
    """
    Load CIFAR10 dataset with custom transforms for VICReg training.

    Args:
    - batch_size (int): Batch size for dataloaders
    - num_workers (int): Number of workers for dataloaders
    - root (str): Root directory for dataset storage

    Returns:
    - train_loader, test_loader: DataLoader objects for training and testing
    """
    train_dataset = datasets.CIFAR10(
        root=root, train=True, download=True, transform=transforms.ToTensor()
    )
    test_dataset = datasets.CIFAR10(
        root=root, train=False, download=True, transform=transforms.ToTensor()
    )

    vicreg_train_dataset = VICRegDataset(train_dataset, train_transform)
    vicreg_test_dataset = VICRegDataset(train_dataset, train_transform)

    train_loader = DataLoader(
        vicreg_train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        vicreg_test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, test_loader

    return train_loader, test_loader


def load_cifar10(batch_size=256, num_workers=2, root="./data"):
    """
    Load CIFAR10 dataset with custom transforms.

    Args:
    - batch_size (int):
    - num_workers (int):
    - root (str):

    Returns:
    - train_loader, test_loader: DataLoader objects for training and testing
    """
    train_dataset = datasets.CIFAR10(
        root=root, train=True, download=True, transform=test_transform
    )
    test_dataset = datasets.CIFAR10(
        root=root, train=False, download=True, transform=test_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, test_loader


def load_mnist(batch_size=256, num_workers=2, root="./data"):
    """
    Load MNIST dataset.

    Args:
    - batch_size (int): Batch size for dataloaders
    - num_workers (int): Number of workers for dataloaders
    - root (str): Root directory for dataset storage

    Returns:
    - train_loader, test_loader: DataLoader objects for training and testing
    """
    # Define transforms for MNIST
    mnist_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    train_dataset = datasets.MNIST(
        root=root, train=True, download=True, transform=mnist_transform
    )
    test_dataset = datasets.MNIST(
        root=root, train=False, download=True, transform=mnist_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, test_loader


def load_combined_test_set(batch_size=256, num_workers=2, root="./data"):
    """
    Load a combined test set of CIFAR10 and MNIST for anomaly detection.
    CIFAR10 images are treated as normal, MNIST images as anomalies.

    Args:
    - batch_size (int): Batch size for dataloaders
    - num_workers (int): Number of workers for dataloaders
    - root (str): Root directory for dataset storage

    Returns:
    - combined_test_loader: DataLoader object for the combined test set
    """
    cifar10_test = datasets.CIFAR10(
        root=root, train=False, download=True, transform=test_transform
    )
    mnist_test = datasets.MNIST(
        root=root,
        train=False,
        download=True,
        transform=transforms.Compose(
            [
                transforms.Resize((32, 32)),
                transforms.Grayscale(3),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
            ]
        ),
    )

    # Create a subset of MNIST to match CIFAR10 test set size
    mnist_subset = Subset(mnist_test, range(len(cifar10_test)))

    # Combine datasets
    combined_dataset = torch.utils.data.ConcatDataset([cifar10_test, mnist_subset])

    # Create labels: 0 for CIFAR10 (normal), 1 for MNIST (anomaly)
    labels = torch.cat([torch.zeros(len(cifar10_test)), torch.ones(len(mnist_subset))])

    # Create a custom dataset that includes these labels

    combined_dataset = CombinedDataset(combined_dataset, labels)

    combined_loader = DataLoader(
        combined_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return combined_loader


def get_representations(
    encoder: torch.nn.Module, data_loader: DataLoader, device: Optional[str] = None
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
    with torch.no_grad():
        for images, batch_labels in data_loader:
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

    Args:
        representations (np.ndarray): 2D array of representations, shape (n_samples, n_features)
        k (int):

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - indices: 2D array of indices of k nearest neighbors, shape (n_samples, k)
            - distances: 2D array of distances to k nearest neighbors, shape (n_samples, k)
    """
    n_samples, n_features = representations.shape

    # Convert to correct data type
    representations = representations.astype(np.float32)

    # Create the index
    # Check if CUDA is available
    if torch.cuda.is_available():
        res = faiss.StandardGpuResources()
        index = faiss.GpuIndexFlatL2(res, n_features)
    else:
        index = faiss.IndexFlatL2(n_features)

    # Add vectors to the index
    index.add(representations)

    # Search for k nearest neighbors
    distances, indices = index.search(representations, k + 1)  # +1 to exclude self

    # Remove self from results (first column)
    return torch.tensor(indices[:, 1:]), torch.tensor(distances[:, 1:])


class NearestNeighborDataset(Dataset):

    def __init__(
        self,
        original_dataset: Dataset,
        neighbor_indices: torch.Tensor,
        transform: Optional[Callable] = None,
    ) -> None:
        """
        Dataset that returns an image and one of its nearest neighbors.

        Args:
        original_dataset (Dataset): Original CIFAR10 dataset
        neighbor_indices (torch.Tensor): Precomputed nearest neighbor indices
        transform (Optional[Callable]): Transforms to apply to the images
        """
        self.original_dataset = original_dataset
        self.neighbor_indices = neighbor_indices
        self.transform = transform

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns the original image and one of its nearest neighbors.

        Args:
        index (int): Index of the sample

        Returns:
        Tuple[torch.Tensor, torch.Tensor]: Original image and one of its nearest neighbors
        """
        # Get the original image
        original_image, _ = self.original_dataset[index]

        # Randomly select one of the nearest neighbors
        neighbor_index = random.choice(self.neighbor_indices[index].tolist())
        neighbor_image, _ = self.original_dataset[neighbor_index]

        # Apply transforms if any
        if self.transform:
            original_image = self.transform(original_image)
            neighbor_image = self.transform(neighbor_image)

        return original_image, neighbor_image

    def __len__(self) -> int:
        """
        Returns the number of samples in the dataset.

        Returns:
        int: Number of samples in the dataset
        """
        return len(self.original_dataset)


def load_nearest_neighbors_dataloader(
    encoder,
    k: int = 3,
    batch_size=256,
    num_workers=2,
    root="./data",
    batch_for_compute=int(2**10),
):
    """
    Load a dataset of CIFAR10 images and their nearest neighbors.

    Args:
    - model: Trained model for feature extraction
    - batch_size (int): Batch size for dataloaders
    - num_workers (int): Number of workers for dataloaders
    - root (str): Root directory for dataset storage

    Returns:
    - train_loader: DataLoader object for the nearest neighbor dataset
    - original_dataset: Original CIFAR10 dataset
    """
    # Load CIFAR10 dataset
    ds_train = datasets.CIFAR10(
        root=root, train=True, download=True, transform=transforms.ToTensor()
    )
    ds_test = datasets.CIFAR10(
        root=root, train=False, download=True, transform=transforms.ToTensor()
    )
    dl_train = DataLoader(
        ds_train,
        batch_size=batch_for_compute,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    dl_test = DataLoader(
        ds_test,
        batch_size=batch_for_compute,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Extract representations from the dataset
    rep_test, _ = get_representations(encoder, dl_test)
    rep_train, _ = get_representations(encoder, dl_train)

    # Find k nearest neighbors
    neighbor_indices_test, _ = find_k_nearest_neighbors(rep_test, k)
    neighbor_indices_train, _ = find_k_nearest_neighbors(rep_train, k)

    near_neig_ds_test = NearestNeighborDataset(ds_test, neighbor_indices_test)
    near_neig_ds_train = NearestNeighborDataset(ds_train, neighbor_indices_train)

    near_neig_dl_train = DataLoader(
        near_neig_ds_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    near_neig_dl_test = DataLoader(
        near_neig_ds_test,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    return near_neig_dl_train, near_neig_dl_test


def test():
    # Test the loaders
    # cifar_train, cifar_test = load_cifar10()
    # mnist_train, mnist_test = load_mnist()
    # combined_test = load_combined_test_set()
    vicreg_train, _ = load_vicreg_cifar10()

    # print(
    #     f"CIFAR10 - Train: {len(cifar_train.dataset)}, Test: {len(cifar_test.dataset)}"
    # )
    # print(f"MNIST - Train: {len(mnist_train.dataset)}, Test: {len(mnist_test.dataset)}")
    # print(f"Combined Test Set: {len(combined_test.dataset)}")
    # print(f"VICReg CIFAR10 - Train: {len(vicreg_train.dataset)}")

    # Verify data shapes
    # for images, labels in cifar_train:
    #     print(f"CIFAR10 batch shape: {images.shape}")
    #     break

    # for images, labels in mnist_train:
    #     print(f"MNIST batch shape: {images.shape}")
    #     break

    # for images, labels in combined_test:
    #     print(f"Combined test batch shape: {images.shape}")
    #     print(f"Combined test labels: {labels.unique()}")
    #     break

    for z, z_prime in vicreg_train:
        px.imshow(z[0].permute(1, 2, 0).cpu().numpy()).show()
        px.imshow(z_prime[0].permute(1, 2, 0).cpu().numpy()).show()
        break


if __name__ == "__main__":
    test()
