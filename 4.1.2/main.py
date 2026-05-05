from jax import random
from flax import linen as nn
import optax
import numpy as np
import matplotlib.pyplot as plt
import sys
import jax.numpy as jnp
sys.path.append('./model')
from deeponet import DeepONet, StackedDeepONet, step

def compute_mse(model, params, s0, s, *trunk_input):
    pred = model.apply(params, s0, *trunk_input)
    batch_size = s.shape[0]
    sample_mses = jnp.mean((pred - s) ** 2, axis=1)
    num_samples = batch_size
    keep_ratio = 0.999
    num_to_keep = int(num_samples * keep_ratio)
    sorted_mses = jnp.sort(sample_mses)
    filtered_mses = sorted_mses[:num_to_keep]
    return jnp.mean(filtered_mses)

def plot_mse_comparison(unstacked_record, stacked_record):
    plt.rcParams.update({'font.size': 14}) 
    u_train = np.array(unstacked_record["train_mse"])
    u_test = np.array(unstacked_record["test_mse"])
    s_train = np.array(stacked_record["train_mse"])
    s_test = np.array(stacked_record["test_mse"])
    fig, ax = plt.subplots(figsize=(9, 10))
    ax.scatter(s_train, s_test, c='blue', marker='s', s=20, label='Stacked', alpha=0.8)
    ax.scatter(u_train, u_test, c='red', marker='o', s=20, label='Unstacked', alpha=0.8)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Train MSE')
    ax.set_ylabel('Test MSE')
    ax.set_xlim(1e-5, 1e-1)
    ax.set_ylim(1e-5, 1e-1)
    ax.legend(loc='lower right', frameon=False)
    plt.tight_layout()
    plt.savefig('./4.1.2/mse_comparison.png', dpi=300)

def main():
    record_num = 100
    data = np.load('./4.1.2/data.npz')
    s_train = data['s_train']
    s_test = data['s_test']
    x_train = data['x_train']
    x_test = data['x_test']
    s0_train = data['s0_train']
    s0_test = data['s0_test']
    # hyperparameters
    b_hid_wid = 40
    b_hid_dep = 1
    t_hid_wid = 40
    t_hid_dep = 2
    p = 40
    activation = nn.silu
    lr = 1e-3
    iters = 50000
    seed = 0
    key = random.PRNGKey(seed)

    # initialize model and optimizer
    model = DeepONet(b_hid_wid, b_hid_dep, t_hid_wid, t_hid_dep, p, activation)
    params = model.init(key, s0_train[:1], x_train)
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)

    unstacked_record = {"train_mse": [], "test_mse": []}
    record_positions = np.logspace(0, np.log10(iters), num=record_num, dtype=int)
    # record_positions = np.linspace(0, iters, num=record_num, dtype=int)
    for i in range(iters+1):
        params, opt_state, mse = step(model, optimizer, params, opt_state, s0_train, s_train, x_train)
        if i in record_positions:
            print(f"Iteration {i}, Train MSE: {mse}")
            unstacked_record["train_mse"].append(mse)
            test_mse = compute_mse(model, params, s0_test, s_test, x_test)
            print(f"Iteration {i}, Test MSE: {test_mse}")
            unstacked_record["test_mse"].append(test_mse)
    
    # initialize model and optimizer
    model = StackedDeepONet(b_hid_wid, b_hid_dep, t_hid_wid, t_hid_dep, p, activation)
    params = model.init(key, s0_train[:1], x_train)
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)

    stacked_record = {"train_mse": [], "test_mse": []}
    for i in range(iters+1):
        params, opt_state, mse = step(model, optimizer, params, opt_state, s0_train, s_train, x_train)
        if i in record_positions:
            print(f"Iteration {i}, Train MSE: {mse}")
            stacked_record["train_mse"].append(mse)
            test_mse = compute_mse(model, params, s0_test, s_test, x_test)
            print(f"Iteration {i}, Test MSE: {test_mse}")
            stacked_record["test_mse"].append(test_mse)

    plot_mse_comparison(unstacked_record, stacked_record)
if __name__ == "__main__":    
    main()
