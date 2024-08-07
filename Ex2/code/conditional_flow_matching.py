# -*- coding: utf-8 -*-
"""Conditional Flow Matching.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1UANwkVlJ-DIY4vdldFUkuVkn6eTS2VNc
"""

# @title  tools
import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from torch.utils import data

import torchvision
import torchvision.transforms as transforms

import pandas as pd
import time
from tqdm import tqdm
import random

import plotly.express as px
import plotly.graph_objects as go

from IPython.display import clear_output

from typing import Tuple
from create_data import *


device = "cuda" if torch.cuda.is_available() else "cpu"


class MyDataset(data.Dataset):
    def __init__(self, data):
        self.data = torch.from_numpy(data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class ConditionalDataset(data.Dataset):
    def __init__(self, x, y, d=None):
        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


"""You are required to train a 2 flow matching models over a 2D as describe in Sec. 2.1, one unconditional and one conditional,
then answer the following questions / assignments. We recommend you start from the following:

"""

ds = ConditionalDataset(*create_olympic_rings(1000))

class CondFlowMatch(nn.Module):
    def __init__(self, n_hidden, n_dim):
        super(CondFlowMatch, self).__init__()
        self.dim = n_dim
        self.embedding = nn.Embedding(5, 2)
        self.net = nn.Sequential(
            nn.Linear(n_dim + 3, n_hidden),
            nn.LeakyReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.LeakyReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.LeakyReLU(),
            nn.Linear(n_hidden, n_dim ),
        )

    def forward(self, x, t, color):
        color = self.embedding(color).squeeze(1)
        return self.net(torch.cat([x, t, color], dim=1))

    def sample(self, size, color, steps=1000):
        x = torch.randn(size, self.dim, dtype=torch.float64, device=device)
        t = torch.linspace(0, 1, steps,device=device)

        delta = torch.tensor(1 / steps)
        for i in range(steps):
            x += self.forward(x, torch.full((size, 1), t[i],device=device), torch.full((size, 1), color, device=device)) * delta
        return x
    def sample_wite_progression(self, size,color, steps=1000, levels=10):
        x = torch.randn(size, self.dim, dtype=torch.float64, device=device)
        t = torch.linspace(0, 1, steps,device=device)
        out = torch.empty(levels+1, size, self.dim, dtype=torch.float64, device=device)
        out[0] = x
        progression = torch.empty((levels+1,size, self.dim), dtype=torch.float64, device=device)
        progression[0] = x.detach()
        delta = torch.tensor(1 / steps)
        for i in range(steps):
            x += self.forward(x, torch.full((size, 1), t[i], device=device),torch.full((size,1),color,device=device)) * delta
            if (i+1) % (steps //levels) == 0:
                progression[(i+1)//(steps//levels)] = x.detach()
        return progression.cpu()


if "model" in globals():
    new_model = CondFlowMatch(64, 2).to(device).to(torch.float64)
    new_model.load_state_dict(model.state_dict())
    model = new_model

ds = ConditionalDataset(*create_olympic_rings(250000))
data_loader = data.DataLoader(ds, batch_size=128, shuffle=True)

def train(model, data_loader, optimizer, loss_func, scheduler, epochs=20,model_path = 'min_model.pth'):
    min_loss = float('inf')
    model.train()
    record_data = pd.DataFrame(columns=["loss"], index=range(epochs))
    for epoch in range(epochs):
        epoch_loss = 0
        for y1, color in tqdm(data_loader):
            y1 = y1.to(device)
            color = color.to(device)
            optimizer.zero_grad()
            y0 = torch.randn(y1.size(0), 2, dtype=torch.float64, device=device)
            t = torch.rand(y1.size(0), 1, dtype=torch.float64, device=device)
            y = t * y1 + (1 - t) * y0
            target = y1-y0
            vt = model(y, t, color)
            loss = loss_func(vt, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        if scheduler:
            scheduler.step()
        clear_output(wait=True)
        record_data.loc[epoch] = epoch_loss
        px.line(record_data).show()
        if model_path:
            if epoch_loss < min_loss:
                min_loss = epoch_loss
                torch.save(model.state_dict(), model_path)
        model.eval()
        print(f"Epoch: {epoch + 1}, Loss: {epoch_loss}")

model = CondFlowMatch(64, 2).to(device).to(torch.float64)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
loss_func = nn.MSELoss()
train(model, data_loader, optimizer, loss_func, scheduler, epochs=20)

""" ## Q1: Plotting the Input.
 Plot your input coloring the points by their classes. Which equation did you change to
insert the conditioning?
"""

x,y, color_map = create_olympic_rings(1000, verbose=False)
fig = go.Figure()

for class_name in color_map:
    mask = y == class_name
    fig.add_trace(go.Scatter(
        x=x[mask,0],
        y=x[mask,1],
        mode='markers',
        name=class_name,
        marker=dict(
            color=color_map[class_name],
            size=10
        )
    ))
fig.show()

"""## Q2: A Point from each Class.
Sample 1 point from each class. Plot the trajectory of the points, coloring each
trajectory with its class’s color. Validate the points reach their class region.
"""
# I used ChatGpt
def sample_and_plot_trajectories(model,colors, num_points=1, steps=1000,levels=5):
    step_size = 1 / levels

    fig = go.Figure()

    olympic_data = create_unconditional_olympic_rings(1000, verbose=False)


    # Plot Olympic rings
    for class_name in color_map:
        mask = y == class_name
        fig.add_trace(go.Scatter(
            x=x[mask,0],
            y=x[mask,1],
            mode='markers',
            name=class_name,
            marker=dict(
                color=color_map[class_name],
                size=2
            )
        ))

    theta = np.linspace(0, 2*np.pi, 100)
    x_circle = np.cos(theta)
    y_circle = np.sin(theta)
    fig.add_trace(go.Scatter(x=x_circle, y=y_circle, mode='lines', line=dict(color='red', width=1, dash='dash'), name='Unit Circle'))


    for color_ind in colors:
        progression = model.sample_wite_progression(num_points,color_ind, steps,levels)
        fig.add_trace(go.Scatter(
            x=progression[:,0, 0],
            y=progression[:,0, 1],
            mode='lines+markers',
            name=colors[color_ind],
            marker=dict(size=8, color=px.colors.sequential.Viridis),
            line=dict(width=3,color=colors[color_ind]),
            text=[f't = {t*step_size:.2f}' for t in range(levels+1)],
            textposition='top center'
        ))

    fig.update_layout(title='Point Trajectories', xaxis_title='X', yaxis_title='Y')
    fig.update_xaxes(scaleanchor="y", scaleratio=1)
    fig.show()

sample_and_plot_trajectories(model,color_map,levels=5)

"""## Q3: Sampling.
Plot a sampling of at least 3000 points from your trained conditional model. Plot the sampled points
coloring them by their classes.

"""

fig = go.Figure()
for color_ind in color_map:
    x = model.sample(500, color_ind).detach().cpu()
    fig.add_trace(go.Scatter(
        x=x[:,0],
        y=x[:,1],
        mode='markers',
        name=color_map[color_ind],
        marker=dict(
            color=color_map[color_ind],
            size=5
        )
    ))
fig.show()
