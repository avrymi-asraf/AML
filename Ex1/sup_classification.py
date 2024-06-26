# %%
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import numpy as np
import plotly.express as px


# %%
class CnnClssificationModel(nn.Module):
    def __init__(self):
        super(CnnClssificationModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, stride=2)  # 8,13,13
        # self.conv2 = nn.Conv2d(4, 8, 3, stride=2)  # 8,6,6
        self.conv3 = nn.Conv2d(8, 16, 3, stride=2)  # 16,6,6
        self.f1 = nn.Linear(6 * 6 * 16, 32)
        self.f2 = nn.Linear(32, 10)
        self.relu = nn.LeakyReLU()
        self.activate = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        # x = self.conv2(x)
        # x = self.relu(x)
        x = self.conv3(x).reshape(-1, 6 * 6 * 16)
        x = self.relu(x)
        x = self.f1(x)
        x = self.relu(x)
        x = self.f2(x)
        x = self.activate(x)
        return x


# %%
# Load MNIST dataset
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)

train_dataset = torchvision.datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)
# take a stratified subset of the training data, keeping only 5000 samples, with 500 samples per class
train_targets = train_dataset.targets
train_idx, _ = train_test_split(
    range(len(train_targets)), train_size=2000, stratify=train_targets
)
train_dataset = torch.utils.data.Subset(train_dataset, train_idx)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

test_dataset = torchvision.datasets.MNIST(
    root="./data", train=False, download=True, transform=transform
)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


# %%
################### Complete the code below ###################
# Initialize the model, loss function, and optimizer
################### Complete the code below ####################
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CnnClssificationModel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# %%
# Training loop
num_epochs = 30
record_data = pd.DataFrame(
    {"val_losses": float(), "val_accuracies": float(), "epoch_loss": float()},
    index=np.arange(num_epochs),
)
for epoch in range(num_epochs):
    model.train()  # moves the model to training mode
    running_loss = 0.0
    for images, labels in tqdm(train_loader):
        images, labels = images.to(device), labels.to(device)
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
        model.zero_grad()
        running_loss += loss.item()

    # Validation
    model.eval()  # moves the model to evaluation mode
    correct = 0
    total = 0
    val_loss = 0.0
    with torch.no_grad():  # Temporarily set all the requires_grad flags to false
        for images, labels in tqdm(test_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_loader)
    val_loss /= len(test_loader)
    accuracy = correct / total
    record_data.iloc[epoch] = [val_loss, accuracy, epoch_loss]

# %%
px.line(record_data).show()

# %%
