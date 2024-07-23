"""Unconditional Flow Matching.ipynb


    https://colab.research.google.com/drive/1ZrPpFEm5vCwEhBnrlUZ6dxYUoUzAAam1
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
from plotly.subplots import make_subplots

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


"""You are required to train a 2 flow matching models over a 2D as describe in Sec. 2.1, one unconditional and one conditional,
then answer the following questions / assignments. We recommend you start from the following:

"""


class FlowMatch(nn.Module):
    def __init__(self, n_hidden, n_dim):
        super(FlowMatch, self).__init__()
        self.dim = n_dim
        self.net = nn.Sequential(
            nn.Linear(n_dim + 1, n_hidden),
            nn.LeakyReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.LeakyReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.LeakyReLU(),
            nn.Linear(n_hidden, n_dim),
            # nn.ReLU()
        )

    def forward(self, x, t):
        return self.net(torch.cat([x, t], dim=1))

    def sample(self, size=None, steps=1000, points=None):
        assert (
            size is not None or points is not None
        ), "Must provide either size or points"
        size = size if size else points.size(0)
        x = (
            points.to(device=device, dtype=torch.float64)
            if points is not None
            else torch.randn(size, self.dim, dtype=torch.float64, device=device)
        )
        t = torch.linspace(0, 1, steps, device=device)

        delta = torch.tensor(1 / steps)
        for i in range(steps):
            x += self.forward(x, torch.full((size, 1), t[i], device=device)) * delta
        return x

    def sample_wite_progression(self, size=None, steps=1000, levels=5, points=None):
        assert (
            size is not None or points is not None
        ), "Must provide either size or points"
        size = size if size else points.size(0)
        x = (
            points.to(device=device, dtype=torch.float64)
            if points is not None
            else torch.randn(size, self.dim, dtype=torch.float64, device=device)
        )
        t = torch.linspace(0, 1, steps, device=device)

        progression = torch.empty(
            (levels + 1, size, self.dim), dtype=torch.float64, device=device
        )
        progression[0] = x.detach()
        delta = torch.tensor(1 / steps)
        for i in range(steps):
            x += self.forward(x, torch.full((size, 1), t[i], device=device)) * delta
            if (i + 1) % (steps // levels) == 0:
                progression[(i + 1) // (steps // levels)] = x.detach()
        return progression.cpu()

    def reverse_wite_progression(self, x, steps=1000, levels=5):
        t = torch.linspace(0, 1, steps, device=device)
        size = x.size(0)
        progression = torch.empty(
            (levels + 1, size, self.dim), dtype=torch.float64, device=device
        )
        progression[0] = x.detach()
        delta = torch.tensor(-1 / steps)
        for i in range(steps):
            x += self.forward(x, torch.full((size, 1), t[i], device=device)) * delta
            if (i + 1) % (steps // levels) == 0:
                progression[(i + 1) // (steps // levels)] = x.detach()
        return progression.cpu()


if "model" in globals():
    new_model = FlowMatch(64, 2).to(device).to(torch.float64)
    new_model.load_state_dict(model.state_dict())
    model = new_model

data_set = MyDataset(create_unconditional_olympic_rings(250000))
data_loader = data.DataLoader(data_set, batch_size=128, shuffle=True)


def train(
    model,
    data_loader,
    optimizer,
    loss_func,
    scheduler,
    epochs=20,
    model_path="min_model.pth",
):
    min_loss = float("inf")
    model.train()
    record_data = pd.DataFrame(columns=["loss"], index=range(epochs))
    for epoch in range(epochs):
        epoch_loss = 0
        for y1 in tqdm(data_loader):
            y1 = y1.to(device)
            optimizer.zero_grad()
            y0 = torch.randn(y1.size(0), 2, dtype=torch.float64, device=device)
            t = torch.rand(y1.size(0), 1, dtype=torch.float64, device=device)
            y = t * y1 + (1 - t) * y0
            target = y1 - y0
            vt = model(y, t)
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
        vt = model.sample(1000).detach().cpu().numpy()
        fig = px.scatter(x=vt[:, 0], y=vt[:, 1])
        fig.show()
        print(f"Epoch: {epoch + 1}, Loss: {epoch_loss}")
    return record_data


model = FlowMatch(64, 2).to(device).to(torch.float64)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
loss_func = nn.MSELoss()
record_data = train(model, data_loader, optimizer, loss_func, scheduler, epochs=20)

"""## Q1: Loss.
Present the loss function (Eq. 16) over the training batches.

## Q2: Flow Progression.
 Your flow matching model should keep the dimension of the input at all times. Sample 1000 points out of it, and plot them after each of t = 0, 0.2, 0.4, 0.6, 0.8, 1 in separate figures, showing how the distribution progresses.
Compare this to Q3 of Sec. 3.3, in which aspects is the distribution flow different between the models? Explain.
"""
# helped by chatgpt


def sample_and_plot_progression(model, size=1000, steps=1000):
    x = torch.randn(size, model.dim, dtype=torch.float64, device=device)
    t = torch.linspace(0, 1, steps, device=device)

    progression = model.sample_wite_progression(size, steps)

    # Create subplots
    fig = make_subplots(
        rows=3,
        cols=2,
        horizontal_spacing=0.02,  # Decrease horizontal spacing
        vertical_spacing=0.06,  # Decrease vertical spacing
        subplot_titles=[f"t = {i*0.2:.1f}" for i in range(6)],
        shared_xaxes=True,
        shared_yaxes=True,
    )

    # Add scatter plots and unit circles to each subplot
    for i, step in enumerate(progression):
        row = i // 2 + 1
        col = i % 2 + 1
        # Add scatter plot
        fig.add_trace(
            go.Scatter(x=step[:, 0], y=step[:, 1], mode="markers", marker=dict(size=2)),
            row=row,
            col=col,
        )

        # Add unit circle
        theta = np.linspace(0, 2 * np.pi, 100)
        fig.add_trace(
            go.Scatter(
                x=np.cos(theta),
                y=np.sin(theta),
                mode="lines",
                line=dict(color="red", width=1),
            ),
            row=row,
            col=col,
        )

    # Update layout
    fig.update_layout(
        height=1300, width=900, title_text="Flow Progression", showlegend=False
    )
    fig.update_xaxes(range=[-3, 3], scaleanchor="y", scaleratio=1)
    fig.update_yaxes(range=[-3, 3])

    fig.show()


# Call the function
sample_and_plot_progression(model)

"""## Q3: Point Trajectory.
Sample 10 points of your choice from your model and present the their forward process as a 2D trajectory. Color the points according to their time t. Compare this to Q4 of Sec. 3.3. Which models flow is more consistent over time? Explain (you may answer this and the previous question together).

"""


# helped by chatgpt
def sample_and_plot_trajectories(model, num_points=10, steps=1000, levels=5):
    progression = model.sample_wite_progression(num_points, steps, levels)
    step_size = 1 / levels

    fig = go.Figure()
    for i in range(num_points):
        fig.add_trace(
            go.Scatter(
                x=progression[:, i, 0],
                y=progression[:, i, 1],
                mode="lines+markers",
                name=f"Point {i+1}",
                marker=dict(size=8, color=px.colors.sequential.Viridis),
                line=dict(width=2),
                text=[f"t = {t*step_size:.2f}" for t in range(levels + 1)],
                textposition="top center",
            )
        )

    fig.update_layout(title="Point Trajectories", xaxis_title="X", yaxis_title="Y")
    fig.show()


sample_and_plot_trajectories(model, levels=5)

"""## Q4: Time Quantization.
 Sample 1000 points from the models using ∆t = 0.002, 0.02, 0.05, 0.1, 0.2. Plot the results in separate figures. How does this quantization of the flow affect the resulted distribution? Explain.

"""


# helped by chatgpt
def sample_with_quantization(
    model, size=1000, delta_t_values=[0.002, 0.02, 0.05, 0.1, 0.2, 0.3]
):
    fig = make_subplots(
        rows=2, cols=3, subplot_titles=[f"∆t = {dt}" for dt in delta_t_values]
    )

    for i, delta_t in enumerate(delta_t_values):
        steps = int(1 / delta_t)
        x = model.sample(size, steps).detach().cpu()

        row = (i // 3) + 1
        col = (i % 3) + 1
        fig.add_trace(
            go.Scatter(x=x[:, 0], y=x[:, 1], mode="markers", marker=dict(size=2)),
            row=row,
            col=col,
        )

    # Add original distribution
    # original_data = create_unconditional_olympic_rings(1000, verbose=False)
    # fig.add_trace(go.Scatter(x=original_data[:, 0], y=original_data[:, 1], mode='markers', marker=dict(size=2)), row=2, col=3)

    fig.update_layout(height=800, width=1200, title_text="Effect of Time Quantization")
    fig.show()


sample_with_quantization(model)

"""## Q5: Reversing the Flow.
Pick the same 5 points from Q5 of Sec. 3.3. Insert them to the reverse sampling process of your flow matching model, and plot their trajectories in a 2D space. Compare this to Q5 of Sec. 3.3. • Are the outputs the same? Explain why. • Re-enter the inverted points back into the forward model. Did you get the same points? Explain why / why not. • Say you would re-enter the normalizing flow inversion back to the normalizing flow. Would you then get the same points back? Explain.
"""
# helped by chatgpt
in_or_out = lambda ind: "Inside ring" if ind < 3 else "Outside ring"


def reverse_flow(model, points, steps=1000):
    x = torch.tensor(points, dtype=torch.float64, device=device)
    t = torch.linspace(1, 0, steps, device=device)

    trajectories = model.reverse_wite_progression(x, steps)

    fig = go.Figure()
    for i in range(len(points)):
        fig.add_trace(
            go.Scatter(
                x=trajectories[:, i, 0],
                y=trajectories[:, i, 1],
                mode="lines+markers",
                name=in_or_out(i),
                marker=dict(
                    size=8,
                    color=px.colors.sequential.Viridis[:: len(trajectories) // 6],
                ),
                line=dict(width=2),
            )
        )

    fig.update_layout(
        title="Reverse Flow Trajectories", xaxis_title="X", yaxis_title="Y"
    )
    fig.show()

    return trajectories[-1]


# Use the same 5 points as in Q5 of Sec. 3.3
points = (
    torch.tensor(
        [
            [-1, -1.4],  # Inside ring
            [-0.2, -0.6],  # Inside ring
            [1.3, 0.7],  # Inside ring
            [-2, -1],  # Outside ring
            [1.8, 2.0],  # Outside ring
        ]
    )
    .to(device)
    .to(torch.float64)
)

reversed_points = reverse_flow(model, points)
print("Reversed points:", reversed_points)

# Forward flow of reversed points
forward_points = (
    model.sample(
        points=torch.tensor(reversed_points, dtype=torch.float64, device=device),
        steps=1000,
    )
    .detach()
    .cpu()
)
print("Forward flow of reversed points:", forward_points)

in_or_out = lambda ind: "Inside ring" if ind < 3 else "Outside ring"


# helped by chatgpt
def reverse_flow(model, points, steps=1000):
    x = points.to(device).to(torch.float64)
    t = torch.linspace(1, 0, steps, device=device)

    trajectories = model.reverse_wite_progression(x, steps)

    # Generate Olympic rings data
    olympic_data = create_unconditional_olympic_rings(1000, verbose=False)

    fig = go.Figure()

    # Plot Olympic rings
    fig.add_trace(
        go.Scatter(
            x=olympic_data[:, 0],
            y=olympic_data[:, 1],
            mode="markers",
            name="Olympic Rings",
            marker=dict(size=2, color="lightgrey"),
        )
    )

    # Add unit circle
    theta = np.linspace(0, 2 * np.pi, 100)
    x_circle = np.cos(theta)
    y_circle = np.sin(theta)
    fig.add_trace(
        go.Scatter(
            x=x_circle,
            y=y_circle,
            mode="lines",
            name="Unit Circle",
            line=dict(color="red", width=1, dash="dash"),
        )
    )

    # Plot trajectories
    for i in range(len(points)):
        fig.add_trace(
            go.Scatter(
                x=trajectories[:, i, 0],
                y=trajectories[:, i, 1],
                mode="lines+markers",
                name=in_or_out(i),
                marker=dict(
                    size=8,
                    color=px.colors.sequential.Viridis[:: len(trajectories) // 6],
                ),
                line=dict(width=2),
            )
        )
    fig.update_xaxes({"scaleanchor": "y", "scaleratio": 1})
    fig.update_layout(
        title="Reverse Flow Trajectories", xaxis_title="X", yaxis_title="Y"
    )
    fig.show()

    return trajectories[-1]


# Use the same 5 points as in Q5 of Sec. 3.3
points = torch.tensor(
    [
        [0.6, 0.6],  # Inside ring
        [-0.2, -0.6],  # Inside ring
        [-0.6, 0.7],  # Inside ring
        [-2, -1],  # Outside ring
        [1.8, 2.0],  # Outside ring
        # [0, 1]  ,  # Outside ring
        # [.8, -.6]    # Outside ring
    ]
)

reversed_points = reverse_flow(model, points)
# Forward flow of reversed points
forward_points = model.sample(points=reversed_points, steps=1000).clone().detach().cpu()


fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=points[:, 0],
        y=points[:, 1],
        mode="markers",
        marker=dict(size=5, color="red"),
        name="points",
    )
)
fig.add_trace(
    go.Scatter(
        x=reversed_points[:, 0],
        y=reversed_points[:, 1],
        mode="markers",
        marker=dict(size=5, color="blue"),
        name="reversed_points",
    )
)
fig.add_trace(
    go.Scatter(
        x=forward_points[:, 0],
        y=forward_points[:, 1],
        mode="markers",
        marker=dict(size=5, color="green"),
        name="forward_points",
    )
)
fig.update_xaxes({"scaleanchor": "y", "scaleratio": 1})
fig.show()
print("Points", points)
print("Reversed points:", reversed_points)
print("Forward flow of reversed points:", forward_points)
print("diff", (points - forward_points).sum(dim=1))


# bonus
model = model.to(device).to(torch.float64)
steps = 1000
target = torch.tensor(
    [[4.0, 5.0]], device=device, dtype=torch.float64
)  # Define the target point
s = (
    model.reverse_wite_progression(target.clone(), steps=steps, levels=1)
    .detach()[-1]
    .to(device)
)
p = model.sample(steps=steps, points=s.clone())
print("Initial point:", s)
print("Final point:", p)
s.requires_grad = True
alpha = torch.tensor(1, device=device)
for i in range(100):
    model.zero_grad()
    p = model.sample(steps=steps, points=s.clone())
    loss = (target - p).pow(2).sum()
    loss.backward()
    s.data = s.data - alpha * s.grad.data
    s.grad.zero_()
    print(f"Iteration {i+1}, Loss: {loss.item()},\n points: {p}")

    if (target - p).abs().sum().item() < 1e-4:  # Add a stopping criterion
        print(f"Converged after {i+1} iterations")
        break

print("Final point:", s.detach().cpu().numpy())

steps = 1000
target = torch.tensor(
    [[4.0, 5.0]], device=device, dtype=torch.float64
)  # Define the target point
points = torch.tensor([[9.23769012, 12.36240534]], device=device, dtype=torch.float64)

tp_reverse_proggres = model.reverse_wite_progression(
    target.clone(), steps=steps, levels=10
)
revers_point = tp_reverse_proggres[-1]
rp_forward_proggres = model.sample_wite_progression(
    levels=10, points=revers_point.clone(), steps=steps
)
tp_forward_proggres = model.sample_wite_progression(
    levels=10, points=points.clone(), steps=steps
)

fig = go.Figure()
olympic_data = create_unconditional_olympic_rings(1000, verbose=False)


# Plot Olympic rings
fig.add_trace(
    go.Scatter(
        x=olympic_data[:, 0],
        y=olympic_data[:, 1],
        mode="markers",
        name="Olympic Rings",
        marker=dict(size=2, color="lightgrey"),
    )
)

# Add unit circle
theta = np.linspace(0, 2 * np.pi, 100)
x_circle = np.cos(theta)
y_circle = np.sin(theta)
fig.add_trace(
    go.Scatter(
        x=x_circle,
        y=y_circle,
        mode="lines",
        name="Unit Circle",
        line=dict(color="red", width=1, dash="dash"),
    )
)

fig.add_trace(
    go.Scatter(
        x=tp_reverse_proggres[:, 0, 0],
        y=tp_reverse_proggres[:, 0, 1],
        mode="lines+markers",
        marker=dict(size=5, color="red"),
        name="reverse proggres of (4,5)",
    )
)
fig.add_trace(
    go.Scatter(
        x=rp_forward_proggres[:, 0, 0],
        y=rp_forward_proggres[:, 0, 1],
        mode="lines+markers",
        marker=dict(size=5, color="blue"),
        name="forward of reversed point",
    )
)
fig.add_trace(
    go.Scatter(
        x=tp_forward_proggres[:, 0, 0],
        y=tp_forward_proggres[:, 0, 1],
        mode="lines+markers",
        marker=dict(size=5, color="green"),
        name="forward of the new points",
    )
)
fig.update_xaxes({"scaleanchor": "y", "scaleratio": 1})
fig.update_layout(title="Reverse Flow Trajectories", xaxis_title="X", yaxis_title="Y")
fig.show()
