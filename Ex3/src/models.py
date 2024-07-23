import torch
import torch.nn as nn
from torchvision.models import resnet18

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"



class Encoder(nn.Module):
    def __init__(self, D=128, device="cuda"):
        super(Encoder, self).__init__()
        self.resnet = resnet18(pretrained=False).to(device)
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=1)
        self.resnet.maxpool = nn.Identity()
        self.resnet.fc = nn.Linear(512, 512)
        self.fc = nn.Sequential(
            nn.BatchNorm1d(512), nn.ReLU(inplace=True), nn.Linear(512, D)
        )

    def forward(self, x):
        x = self.resnet(x)
        x = self.fc(x)
        return x

    def encode(self, x):
        return self.forward(x)


class Projector(nn.Module):
    def __init__(self, D, proj_dim=512):
        super(Projector, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(D, proj_dim),
            nn.BatchNorm1d(proj_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_dim, proj_dim),
            nn.BatchNorm1d(proj_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_dim, proj_dim),
        )

    def forward(self, x):
        return self.model(x)


class LinearProbe(nn.Module):
    def __init__(self, encoder,input_dim, num_classes):
        super(LinearProbe, self).__init__()
        self.encoder = encoder
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        x = self.encoder(x)
        return self.fc(x)

    def predict(self, x):
        with torch.no_grad():
            return torch.argmax(self.forward(x), dim=1)


class VICeg(nn.Module):
    def __init__(self, en_dim=128, prj_dim=512):
        super(VICeg, self).__init__()
        self.encoder = Encoder(D=en_dim, device=DEVICE)
        self.projector = Projector(D=en_dim, proj_dim=prj_dim)

    def forward(self, x):
        return self.projector(self.encoder(x))
