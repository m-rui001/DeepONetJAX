import numpy as np
import scipy
from scipy.stats import qmc
import sys
sys.path.append('./model')
from grf import generate_grf

train_sampler = qmc.LatinHypercube(d=10)
test_sampler = qmc.LatinHypercube(d=100)
train_grf = generate_grf(num_samples=1000, num_points=1000, length_scale=0.2)
test_grf = generate_grf(num_samples=1000, num_points=1000, length_scale=0.2)


def integral_grf(grf_samples, x):
    return scipy.integrate.cumulative_trapezoid(grf_samples, x, axis=1, initial=0)
integral_train_grf= integral_grf(train_grf, np.linspace(0, 1, 1000))
integral_test_grf= integral_grf(test_grf, np.linspace(0, 1, 1000))

s0_train = np.array([np.interp(np.linspace(0, 1, 100), np.linspace(0, 1, 1000), integral_train_grf[i]) for i in range(1000)])
s0_test = np.array([np.interp(np.linspace(0, 1, 100), np.linspace(0, 1, 1000), integral_test_grf[i]) for i in range(1000)])

train_idx = train_sampler.random(n=1000)
test_idx = test_sampler.random(n=1000)
s_train = np.array([np.interp(train_idx[i], np.linspace(0, 1, 1000), integral_train_grf[i]) for i in range(1000)])
s_test = np.array([np.interp(test_idx[i], np.linspace(0, 1, 1000), integral_test_grf[i]) for i in range(1000)])
np.savez('./4.1.1/data.npz', s_train=s_train, s_test=s_test,
         s0_train=s0_train, s0_test=s0_test,
         x_train=train_idx, x_test=test_idx)

