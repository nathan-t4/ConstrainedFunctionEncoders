from datetime import datetime

import matplotlib.pyplot as plt
import torch

from FunctionEncoder import QuadraticDataset, FunctionEncoder, MSECallback, ListCallback, TensorboardCallback, \
    DistanceCallback

import argparse


# parse args
parser = argparse.ArgumentParser()
parser.add_argument("--n_basis", type=int, default=5) # 11
parser.add_argument("--train_method", type=str, default="least_squares")
parser.add_argument("--epochs", type=int, default=1_000)
parser.add_argument("--load_path", type=str, default=None)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--residuals", action="store_true")
args = parser.parse_args()


# hyper params
epochs = args.epochs
n_basis = args.n_basis
load_path = args.load_path
# device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"
train_method = args.train_method
eval_method = "constrained"
seed = args.seed
residuals = args.residuals
if load_path is None:
    logdir = f"logs/constrained_quadratic_example/{train_method}/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
else:
    logdir = load_path



# seed torch
torch.manual_seed(seed)

# create a dataset
if residuals:
    a_range = (0, 3) # this makes the true average function non-zero
else:
    a_range = (-3, 3)
b_range = (-3, 3)
c_range = (-3, 3)
input_range = (-10, 10)
dataset = QuadraticDataset(a_range=a_range, b_range=b_range, c_range=c_range, input_range=input_range, device=device)

# specify the inequality and equality constraints
xc_iqs = torch.tensor([[0.0]], dtype=torch.float32, device=device)
b_iqs = torch.tensor([[-1.0]], dtype=torch.float32, device=device)
xc_eqs = torch.tensor([[-5.0],
                       [-3.0]], dtype=torch.float32, device=device)
b_eqs = torch.tensor([[0.0], 
                      [10.0]], dtype=torch.float32, device=device)
rep_kwargs = {'xc_iqs': xc_iqs, 'b_iqs': b_iqs, 'xc_eqs': xc_eqs, 'b_eqs': b_eqs}
    
print(device)

if load_path is None:
    # create the model
    model = FunctionEncoder(input_size=dataset.input_size,
                            output_size=dataset.output_size,
                            data_type=dataset.data_type,
                            n_basis=n_basis,
                            method=train_method,
                            use_residuals_method=residuals).to(device)

    # create callbacks
    cb1 = TensorboardCallback(logdir) # this one logs training data
    cb2 = DistanceCallback(dataset, device=device, tensorboard=cb1.tensorboard) # this one tests and logs the results
    callback = ListCallback([cb1, cb2])

    # train the model
    if train_method == "constrained":
        model.train_model(dataset, epochs=epochs, callback=callback, rep_kwargs=rep_kwargs) 
    else:
        model.train_model(dataset, epochs=epochs, callback=callback, rep_kwargs={}) 

    # save the model
    torch.save(model.state_dict(), f"{logdir}/model.pth")
else:
    # load the model
    model = FunctionEncoder(input_size=dataset.input_size,
                            output_size=dataset.output_size,
                            data_type=dataset.data_type,
                            n_basis=n_basis,
                            method=train_method,
                            use_residuals_method=residuals).to(device)
    model.load_state_dict(torch.load(f"{logdir}/model.pth"))

# plot
with torch.no_grad():
    n_plots = 9
    n_examples = 100
    n_eq_constraints = len(xc_eqs.flatten())
    n_iq_constraints = len(xc_iqs.flatten())
    n_constraints = n_eq_constraints + n_iq_constraints

    example_xs, example_ys, xs, ys, info = dataset.sample()
    example_xs, example_ys = example_xs[:, :n_examples, :], example_ys[:, :n_examples, :]

    # Try feeding in just the equality constraints as examples
    # example_xs, example_ys = xc_eqs.reshape((1,-1,1)).repeat(10,1,1), b_eqs.reshape((1,-1, 1)).repeat(10,1,1)

    y_hats = model.predict_from_examples(example_xs, example_ys, xs, eval_method, **rep_kwargs)

    xs, indicies = torch.sort(xs, dim=-2)
    ys = ys.gather(dim=-2, index=indicies)
    y_hats = y_hats.gather(dim=-2, index=indicies)
    cmap = plt.get_cmap('tab10')(list(range(n_constraints)))

    fig, axs = plt.subplots(3, 3, figsize=(15, 10))
    # fig, ax = plt.subplots(1, 1, figsize=(15,10))
    for i in range(n_plots):
        ax = axs[i // 3, i % 3]

        # plot the constraint lines
        for j in range(n_eq_constraints):
            ax.axhline(b_eqs[j], ls='--', alpha=0.5, c=cmap[j], label=f"EQ constraint {j}")
            ax.axvline(xc_eqs[j], ls='--', alpha=0.5, c=cmap[j],)
        
        for j in range(n_iq_constraints):
            ax.axhline(b_iqs[j], ls='--', alpha=0.5, c=cmap[n_eq_constraints+j], label=f"IQ constraint {j}")
            ax.axvline(xc_iqs[j], ls='--', alpha=0.5, c=cmap[n_eq_constraints+j],)
        
        # plot the function
        ax.plot(xs[i].cpu(), ys[i].cpu(), label="True")
        ax.plot(xs[i].cpu(), y_hats[i].cpu(), label=eval_method)

        if i == n_plots - 1:
            ax.legend()
        title = f"${info['As'][i].item():.2f}x^2 + {info['Bs'][i].item():.2f}x + {info['Cs'][i].item():.2f}$"
        ax.set_title(title)
        y_min, y_max = y_hats[i].min().item(), y_hats[i].max().item()
        ax.set_ylim(y_min, y_max)

    plt.tight_layout()
    plt.savefig(f"{logdir}/plot.png")
    plt.clf()

    # plot the basis functions
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    xs = torch.linspace(input_range[0], input_range[1], 1_000).reshape(1000, 1).to(device)
    basis = model.model.forward(xs)
    for i in range(n_basis):
        ax.plot(xs.flatten().cpu(), basis[:, 0, i].cpu(), color="black")
    if residuals:
        avg_function = model.average_function.forward(xs)
        ax.plot(xs.flatten().cpu(), avg_function.flatten().cpu(), color="blue")

    plt.tight_layout()
    plt.savefig(f"{logdir}/basis.png")
    plt.clf()

    # Plot the constraint residuals
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))

    xc_eq = xc_eqs.reshape((1,-1,1)).repeat(10,1,1)
    b_eq = b_eqs.reshape((1,-1)).repeat(10,1)
    y_eq = model.predict_from_examples(example_xs, example_ys, xc_eq, eval_method, **rep_kwargs).squeeze()
    eq_residuals = y_eq - b_eq

    xc_iq = xc_iqs.reshape((1,-1,1)).repeat(10,1,1)
    b_iq = b_iqs.reshape((1,-1)).repeat(10,1)
    y_iq = model.predict_from_examples(example_xs, example_ys, xc_iq, eval_method, **rep_kwargs).squeeze()
    iq_residuals = b_iq - y_iq

    for i in range(n_eq_constraints):
        ax.scatter(torch.arange(10), eq_residuals[:,i], label=f"EQ residual {i}")

    for i in range(n_iq_constraints):
        ax.scatter(torch.arange(10), iq_residuals[:,i], label=f"IQ residual {i}")
    
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{logdir}/constraint.png")
    

