from datetime import datetime
import matplotlib.pyplot as plt
import torch
from FunctionEncoder import FunctionEncoder, MSECallback, ListCallback, TensorboardCallback, DistanceCallback
import argparse
from FunctionEncoder.Dataset.MassSpringDamperDataset import MassSpringDamperDataset

# TODO: Problems: loss is not converging. Fixed problem by minimizing randomization
# TODO: Prediction is inaccurate. Since we have f(x) and x_0, unless loss converges to 1e-6, or else the predictions will not be accurate.
# TODO: try training with sequential predictions instead of batch predictions

# parse args
parser = argparse.ArgumentParser()
parser.add_argument("--n_basis", type=int, default=20)
parser.add_argument("--train_method", type=str, default="least_squares")
parser.add_argument("--prediction_method", type=str, default="batch") 
parser.add_argument("--epochs", type=int, default=1_000) # 1_000
parser.add_argument("--load_path", type=str, default=None)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--residuals", action="store_true")
args = parser.parse_args()

# hyper params
epochs = args.epochs
n_basis = args.n_basis
device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"
train_method = args.train_method
prediction_method = args.prediction_method
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
input_range = (-0.5,0.5)
dataset = MassSpringDamperDataset(
    mass_range=(1.0, 1.0),
    spring_constant_range=(1.0, 1.0),
    damping_coefficient_range=(1.0, 1.0),
    force_range=(-0.0, 0.0),
    initial_position_range=(-0.1, 0.1),
    initial_velocity_range=(-0.1, 0.1),
    dt_range=(1e-2,1e-3), # changing dt does not do anything
    control_type="sinusoidal",
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
        prediction_method=prediction_method,
        use_residuals_method=residuals,
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
    n_examples = 1000
    example_xs, example_ys, xs, ys, info = dataset.sample()
    example_t_idxs = torch.randperm(dataset.n_examples_per_sample)[:n_examples]
    example_xs, example_ys = example_xs[:, example_t_idxs, :], example_ys[:, example_t_idxs, :]

    # Get initial position
    y_hats = []
    # prev_x = xs[:,0,0]
    # prev_v = xs[:,0,1]
    xs_k = xs[:,0,1:3].unsqueeze(1)
    representations, _ = model.compute_representation(example_xs, example_ys, train_method)

    # Sequential prediction
    for k in range(xs.shape[1]):
        t = xs[:,k,0].reshape(-1,1,1)
        xs_k = torch.cat([t,xs_k], dim=-1)
        y_hats.append(model.predict(xs_k, representations)) 
        xs_k = y_hats[k]

    y_hats = torch.cat(y_hats, dim=1)

    # Fake prediction
    # y_hats = model.predict_from_examples(example_xs, example_ys, xs, method=train_method)

    print(y_hats.shape)
    print(y_hats[0,0])
    print(y_hats[0,1])
    print(y_hats[0,2])
    print(y_hats[0,3])
    print(y_hats[0,4])
    print(y_hats[0,5])
    print(y_hats[0,6])

    fig, axs = plt.subplots(3, 3, figsize=(15, 10))
    for i in range(n_plots):
        ax = axs[i // 3, i % 3]
        # time_axis = xs[i,:,0].cpu()
        time_axis = torch.arange(xs.shape[1]) * dataset.dt
        ax.plot(time_axis, ys[i].cpu(), label="True")
        ax.plot(time_axis, y_hats[i].cpu(), label=f"{train_method}")
        if i == n_plots - 1:
            ax.legend()
        title = f"Mass: {info['masses'][i].item():.2f}, k: {info['spring_constants'][i].item():.2f}, c: {info['damping_coefficients'][i].item():.2f}, ||F||: {info['forces_magnitude'][i].item():.2f}"
        ax.set_title(title)
        # y_min, y_max = ys[i].min().item(), ys[i].max().item()
        y_min = min(ys[i].min().item(), y_hats[i].min().item())
        y_max = max(ys[i].max().item(), y_hats[i].max().item())
        ax.set_ylim(y_min, y_max)

    plt.tight_layout()
    plt.savefig(f"{logdir}/plot.png")
    plt.clf()

    # TODO: plot the basis functions; basis functions will not be 2D
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    xs = torch.linspace(0, 1, 1_000).reshape(1000, 1).to(device) * (input_range[1] - input_range[0]) + input_range[0]
    vs = torch.linspace(0, 1, 1_000).reshape(1000, 1).to(device) * (dataset.force_range[1] - dataset.force_range[0]) + dataset.force_range[0]
    ts = torch.arange(1000).reshape(1000, 1).to(device) * dataset.dt
    zeros = torch.zeros((1000,1), dtype=torch.float32, device=device)
    # x_basis = model.model.forward(xs)
    x_basis = model.model.forward(torch.cat((zeros, xs, zeros), dim=-1))
    v_basis = model.model.forward(torch.cat((zeros, zeros, vs), dim=-1))
    t_basis = model.model.forward(torch.cat((ts, zeros, zeros), dim=-1))
    for i in range(n_basis):
        ax.plot(xs.flatten().cpu(), x_basis[:, 0, i].cpu(), color="red")
        ax.plot(vs.flatten().cpu(), v_basis[:, 0, i].cpu(), color="blue")
        ax.plot(ts.flatten().cpu(), t_basis[:, 0, i].cpu(), color="green")

    if residuals:
        avg_function = model.average_function.forward(xs)
        ax.plot(xs.flatten().cpu(), avg_function.flatten().cpu(), color="black")

    plt.tight_layout()
    plt.savefig(f"{logdir}/basis.png")
