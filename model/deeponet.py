import jax
from jax import numpy as jnp
from flax import linen as nn
import optax
from functools import partial

class DNN(nn.Module):
    hid_wid: int
    hid_dep: int
    out_dim: int
    activation: callable
    activation_out: bool = False
    def setup(self):
        layers_list = []
        for _ in range(self.hid_dep):
            layers_list.append(nn.Dense(self.hid_wid, kernel_init=nn.initializers.glorot_normal()))
            layers_list.append(self.activation)
        layers_list.append(nn.Dense(self.out_dim, kernel_init=nn.initializers.glorot_normal()))
        if self.activation_out:
            layers_list.append(self.activation)
        self.layers = tuple(layers_list)

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class DeepONet(nn.Module):
    b_hid_wid: int
    b_hid_dep: int
    t_hid_wid: int
    t_hid_dep: int
    p: int
    activation: callable = nn.relu

    def setup(self):
        self.branch_net = DNN(self.b_hid_wid,  self.b_hid_dep, self.p, self.activation)
        self.trunk_net = DNN(self.t_hid_wid, self.t_hid_dep, self.p, self.activation, activation_out=True)
        self.bias = self.param('bias', nn.initializers.zeros_init(), ())

    def __call__(self, branch_input, *trunk_input):
        branch_output = self.branch_net(branch_input)
        # branch_output shape: (Batch, p)
        x, = trunk_input
        if x.ndim == 2:
            x = x.reshape(x.shape[0], x.shape[1], 1)
        trunk_output = self.trunk_net(x) 
        out = jnp.einsum("bp, bxp -> bx", branch_output, trunk_output) + self.bias
        return out

class StackedDeepONet(nn.Module):
    b_hid_wid: int
    b_hid_dep: int
    t_hid_wid: int
    t_hid_dep: int
    p: int
    activation: callable = nn.relu
    def setup(self):
        self.trunk_net = DNN(self.t_hid_wid, self.t_hid_dep, self.p, self.activation, activation_out=True)
        self.bias = self.param('bias', nn.initializers.zeros_init(), ())
        BatchDNN = nn.vmap(
            DNN,
            variable_axes={'params': 0},
            split_rngs={'params': True},
            in_axes=None,
            out_axes=1,
            axis_size=self.p
        )
        self.branch_net = BatchDNN(self.b_hid_wid, self.b_hid_dep, 1, self.activation)
    @nn.compact
    def __call__(self, branch_input, *trunk_input):
        branch_output = self.branch_net(branch_input).squeeze(-1)
        # branch_output shape: (Batch, p)
        x, = trunk_input
        if x.ndim == 2:
            x = x.reshape(x.shape[0], x.shape[1], 1)
        trunk_output = self.trunk_net(x) 
        out = jnp.einsum("bp, bxp -> bx", branch_output, trunk_output) + self.bias
        return out

def opt(optimizer, opt_state, params, gradient):
    updates, opt_state = optimizer.update(gradient, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state

def compute_mse(model, params, s0, s, *trunk_input):
    pred = model.apply(params, s0, *trunk_input)
    mse = jnp.mean((pred - s) ** 2)
    return mse

@partial(jax.jit, static_argnums=(0,1))
def step(model, optimizer, params, opt_state, s0, s, *trunk_input):
    mse, gradient = jax.value_and_grad(compute_mse, argnums=1)(model, params, s0, s, *trunk_input)
    params, opt_state = opt(optimizer, opt_state, params, gradient)
    return params, opt_state, mse