# If you occur ValueError,please repeatly run
import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import qmc
import sys
from scipy.integrate import solve_ivp
sys.path.append('./model')
from grf import generate_grf
num_samples = 10000
train_sampler = qmc.LatinHypercube(d=1)
test_sampler = qmc.LatinHypercube(d=1)
train_grf = generate_grf(num_samples=num_samples, num_points=1000, length_scale=0.2)
test_grf = generate_grf(num_samples=num_samples, num_points=1000, length_scale=0.2)

def build_ode(u_fun):
    u_fun = interp1d(np.linspace(0, 1, 1000), u_fun, kind='linear')
    def ode(t, s):
        return -s**2 + u_fun(t)
    return ode
train_eval = train_sampler.random(n=num_samples)
test_eval = test_sampler.random(n=num_samples)
train_eval.sort(axis=1)
test_eval.sort(axis=1)

train_sol = []
test_sol = []
for i in range(num_samples):
    test_sol.append(solve_ivp(build_ode(test_grf[i]), [0, 1], [0.0],method='RK45', t_eval=test_eval[i]).y[0])
    train_sol.append(solve_ivp(build_ode(train_grf[i]), [0, 1], [0.0], method='RK45', t_eval=train_eval[i]).y[0])
train_sol = np.array(train_sol)
test_sol = np.array(test_sol)

s0_train = []
s0_test = []
for i in range(num_samples):
    s0_train.append(np.interp(np.linspace(0, 1, 100), np.linspace(0, 1, 1000), train_grf[i]))
    s0_test.append(np.interp(np.linspace(0, 1, 100), np.linspace(0, 1, 1000), test_grf[i]))
s0_train = np.array(s0_train)
s0_test = np.array(s0_test)

np.savez('./4.1.2/data.npz', s_train=train_sol, s_test=test_sol,
         x_train=train_eval, x_test=test_eval, s0_train=s0_train, s0_test=s0_test)