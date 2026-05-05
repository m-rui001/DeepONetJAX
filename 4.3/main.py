from jax import random
from flax import linen as nn
import optax
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('./model')
from deeponet import DeepONet, step, compute_mse

def plot(record, iters):
    plt.figure(figsize=(8, 6))
    train_mse = np.array(record["train_mse"])
    test_mse = np.array(record["test_mse"])
    iterations = np.linspace(0, iters, num=len(train_mse))
    plt.plot(iterations, train_mse, color='steelblue', linestyle='-', linewidth=1.5, label='Train')
    plt.plot(iterations, test_mse, color='orange', linestyle='-', linewidth=1.5, label='Test')
    plt.yscale('log')
    plt.xlim(0, iters)
    plt.xticks(np.linspace(0, iters, 6))
    plt.xlabel("# Steps", fontsize=12)
    plt.ylabel("MSE", fontsize=12)
    plt.legend(loc='upper right', frameon=False, fontsize=12)
    plt.tight_layout()
    plt.show()

def main():
    record_steps = 1000
    data = np.load('./4.3/data.npz')
    s_train = data['s_train']
    s_test = data['s_test']
    x_train = data['x_train']
    x_test = data['x_test']
    s0_train = data['s0_train']
    s0_test = data['s0_test']

    # hyperparameters
    b_hid_wid = 100
    b_hid_dep = 0
    t_hid_wid = 100 # Attention: this is the hidden layer width, not including input/output layers
    t_hid_dep = 1
    p = 100
    activation = nn.relu
    lr = 1e-3
    iters = 50000
    seed = 0
    key = random.PRNGKey(seed)

    # initialize model and optimizer
    model = DeepONet(b_hid_wid, b_hid_dep, t_hid_wid, t_hid_dep, p, activation)
    params = model.init(key, s0_train[:1], x_train[:1])
    optimizer = optax.adam(lr, eps=1e-7)
    opt_state = optimizer.init(params)

    record = {"train_mse": [], "test_mse": []}
    for i in range(iters+1):
        params, opt_state, mse = step(model, optimizer, params, opt_state, s0_train, s_train, x_train)
        if i % record_steps == 0:
            print(f"Iteration {i}, Train MSE: {mse}")
            record["train_mse"].append(mse)
            test_mse = compute_mse(model, params, s0_test, s_test, x_test)
            print(f"Iteration {i}, Test MSE: {test_mse}")
            record["test_mse"].append(test_mse)
    plot(record, iters)
if __name__ == "__main__":    
    main()
