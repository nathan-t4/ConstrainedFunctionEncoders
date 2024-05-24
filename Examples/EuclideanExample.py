import os
from datetime import datetime

import matplotlib.pyplot as plt
import torch
from matplotlib import gridspec, animation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from tqdm import trange

from FunctionEncoder import  EuclideanDataset, FunctionEncoder, MSECallback

import argparse


def to_numpy(x):
    return x.detach().cpu().numpy()

# parse args
parser = argparse.ArgumentParser()
parser.add_argument("--n_basis", type=int, default=2)
parser.add_argument("--train_method", type=str, default="inner_product")
parser.add_argument("--epochs", type=int, default=1_000)
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()


# hyper params
epochs = args.epochs
n_basis = args.n_basis
device = "cpu" # gpu is overkill here
train_method = args.train_method
seed = args.seed
logdir = f"logs/euclidean_example/{train_method}/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
os.makedirs(logdir, exist_ok=True)

# seed torch
torch.manual_seed(seed)

# create a dataset
dataset = EuclideanDataset()

# create the model
model = FunctionEncoder(input_size=dataset.input_size,
                        output_size=dataset.output_size,
                        data_type=dataset.data_type,
                        n_basis=n_basis,
                        method=train_method,
                        model_type="Euclidean",).to(device)

# create a testing callback
callback = MSECallback(dataset, device=device)


# we are going to plot a video
# this involves plotting the basis at every timestep
width,height = 2000, 1000
fps = 70

# fig, axs = plt.subplot(1, 2, figsize=(width/100, height/100), dpi=100, subplot_kw=[{'projection':'3d'}], gridspec_kw={'width_ratios': [1.5, 1]})
fig = plt.figure(figsize=(width/100, height/100), dpi=100)
gs = gridspec.GridSpec(nrows=1, ncols=2, width_ratios=[2, 1])
ax = fig.add_subplot(gs[0], projection='3d')
ax2 = fig.add_subplot(gs[1],) #  xmargin=0.5)
ax2.set_aspect(epochs / 10)
fig.tight_layout()
fig.subplots_adjust(wspace=-0.10)

losses = []
tbar = trange(epochs + 1)
def update(frame):
    tbar.update(1)
    ax.cla()
    ax2.cla()

    # do a single gradient step
    l = model.train_model(dataset, epochs=1, logdir=None, callback=callback, progress_bar=False)
    g = model.model.basis
    losses.append(l[0])

    # plot the space we are fitting, ie the xy plane
    rect = Poly3DCollection([[(dataset.min[0], dataset.min[1], dataset.min[2]),
                              (dataset.max[0], dataset.min[1], dataset.min[2]),
                              (dataset.max[0], dataset.max[1], dataset.min[2]),
                              (dataset.min[0], dataset.max[1], dataset.min[2])]], alpha=0.3, color="b")
    ax.add_collection3d(rect)

    # plot the basis functions as arrows. Plot their projection to the plane also.
    for i in range(n_basis):
        ax.quiver(0, 0, 0, to_numpy(g[i,0]), to_numpy(g[i,1]), to_numpy(g[i,2]), color="black", capstyle="round", arrow_length_ratio=0.2)
        ax.plot([to_numpy(g[i,0]), to_numpy(g[i,0])], [to_numpy(g[i,1]), to_numpy(g[i,1])], [0, to_numpy(g[i,2])], color="r", linestyle="--")
        ax.plot([0, to_numpy(g[i,0])], [0, to_numpy(g[i,1])], [0, 0], color="r", linestyle="--")

    # plot loss on second ax
    ax2.plot(losses, color="blue")
    ax2.set_yscale("log")
    ax2.set_xlim(0, epochs)
    ax2.set_ylim(1e-8, 2e-1)
    ax2.set_xlabel("Descent Step")
    ax2.set_ylabel("Loss")

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(0, 1)


ani = animation.FuncAnimation(plt.gcf(), update, frames=epochs, interval=50)
FFwriter = animation.FFMpegWriter(fps=fps)
ani.save(f'{logdir}/animation_{n_basis}b.mp4', writer = FFwriter)

