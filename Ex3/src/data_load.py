import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, Subset

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
        root=root, train=False, download=True, transform=test_transform
    )

    vicreg_train_dataset = VICRegDataset(train_dataset, train_transform)

    train_loader = DataLoader(
        vicreg_train_dataset,
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


def test():
    # Test the loaders
    cifar_train, cifar_test = load_cifar10()
    mnist_train, mnist_test = load_mnist()
    combined_test = load_combined_test_set()
    vicreg_train, _ = load_vicreg_cifar10()

    print(
        f"CIFAR10 - Train: {len(cifar_train.dataset)}, Test: {len(cifar_test.dataset)}"
    )
    print(f"MNIST - Train: {len(mnist_train.dataset)}, Test: {len(mnist_test.dataset)}")
    print(f"Combined Test Set: {len(combined_test.dataset)}")
    print(f"VICReg CIFAR10 - Train: {len(vicreg_train.dataset)}")

    # Verify data shapes
    for images, labels in cifar_train:
        print(f"CIFAR10 batch shape: {images.shape}")
        break

    for images, labels in mnist_train:
        print(f"MNIST batch shape: {images.shape}")
        break

    for images, labels in combined_test:
        print(f"Combined test batch shape: {images.shape}")
        print(f"Combined test labels: {labels.unique()}")
        break

    for z, z_prime in vicreg_train:
        print(f"VICReg z batch shape: {z.shape}")
        print(f"VICReg z' batch shape: {z_prime.shape}")
        break
