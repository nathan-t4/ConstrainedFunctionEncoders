from datetime import datetime
import matplotlib.pyplot as plt
import torch
from FunctionEncoder import FunctionEncoder, MSECallback, ListCallback, TensorboardCallback, DistanceCallback
import argparse
from FunctionEncoder.Dataset.MassSpringDamperDataset import MassSpringDamperDataset

# parse args
parser = argparse.ArgumentParser()
parser.add_argument("--n_basis", type=int, default=11)
parser.add_argument("--train_method", type=str, default="least_squares")
parser.add_argument("--epochs", type=int, default=1_000)
parser.add_argument("--load_path", type=str, default=None)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--residuals", action="store_true")
args = parser.parse_args()

# hyper params
epochs = args.epochs
n_basis = args.n_basis
device = "cuda" if torch.cuda.is_available() else "cpu"
train_method = args.train_method
seed = args.seed
load_path = args.load_path
residuals = args.residuals

if load_path is None:
    logdir = f"logs/mass_spring_damper_example/{train_method}/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
else:
    logdir = load_path

print(f"torch device: {device}")

# seed torch
torch.manual_seed(seed)

# create a dataset
dataset = MassSpringDamperDataset(
    mass_range=(0.5, 5.0),
    spring_constant_range=(0.1, 1.0),
    damping_coefficient_range=(0.1, 1.0),
    force_range=(-5.0, 5.0),
    initial_position_range=(-1.0, 1.0),
    initial_velocity_range=(-1.0, 1.0),
    dt=0.01,
    device=device,
)

if load_path is None:
    # create the model
    model = FunctionEncoder(
        input_size=dataset.input_size,
        output_size=dataset.output_size,
        data_type=dataset.data_type,
        n_basis=n_basis,
        method=train_method,
        use_residuals_method=residuals
    ).to(device)

    # create callbacks
    cb1 = TensorboardCallback(logdir)  # this one logs training data
    cb2 = DistanceCallback(dataset, device=device, tensorboard=cb1.tensorboard)  # this one tests and logs the results
    callback = ListCallback([cb1, cb2])

    # train the model
    model.train_model(dataset, epochs=epochs, callback=callback)

    # save the model
    torch.save(model.state_dict(), f"{logdir}/model.pth")
else:
    # load the model
    model = FunctionEncoder(
        input_size=dataset.input_size,
        output_size=dataset.output_size,
        data_type=dataset.data_type,
        n_basis=n_basis,
        method=train_method,
        use_residuals_method=residuals
    ).to(device)
    model.load_state_dict(torch.load(f"{logdir}/model.pth"))

# plot
with torch.no_grad():
    n_plots = 9
    n_examples = 100
    example_time_points, example_ys, time_points, ys, info = dataset.sample()
    example_time_points, example_ys = example_time_points[:, :n_examples, :], example_ys[:, :n_examples, :]
    if train_method == "inner_product":
        y_hats_ip = model.predict_from_examples(example_time_points, example_ys, time_points, method="inner_product")
    y_hats_ls = model.predict_from_examples(example_time_points, example_ys, time_points, method="least_squares")
    time_points, indices = torch.sort(time_points, dim=-2)
    ys = ys.gather(dim=-2, index=indices)
    y_hats_ls = y_hats_ls.gather(dim=-2, index=indices)
    if train_method == "inner_product":
        y_hats_ip = y_hats_ip.gather(dim=-2, index=indices)

    fig, axs = plt.subplots(3, 3, figsize=(15, 10))
    for i in range(n_plots):
        ax = axs[i // 3, i % 3]
        ax.plot(time_points[i].cpu(), ys[i].cpu(), label="True")
        ax.plot(time_points[i].cpu(), y_hats_ls[i].cpu(), label="LS")
        if train_method == "inner_product":
            ax.plot(time_points[i].cpu(), y_hats_ip[i].cpu(), label="IP")
        if i == n_plots - 1:
            ax.legend()
        title = f"Mass: {info['masses'][i].item():.2f}, k: {info['spring_constants'][i].item():.2f}, c: {info['damping_coefficients'][i].item():.2f}"
        ax.set_title(title)
        y_min, y_max = ys[i].min().item(), ys[i].max().item()
        ax.set_ylim(y_min, y_max)

    plt.tight_layout()
    plt.savefig(f"{logdir}/plot.png")
    plt.clf()

    # plot the basis functions
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    time_points = torch.linspace(0, 1, 1_000).reshape(1000, 1).to(device)  # Adjust the time range as needed
    basis = model.model.forward(time_points)
    for i in range(n_basis):
        ax.plot(time_points.flatten().cpu(), basis[:, 0, i].cpu(), color="black")
    if residuals:
        avg_function = model.average_function.forward(time_points)
        ax.plot(time_points.flatten().cpu(), avg_function.flatten().cpu(), color="blue")

    plt.tight_layout()
    plt.savefig(f"{logdir}/basis.png")
