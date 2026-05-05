import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import qmc
import sys
from scipy.integrate import solve_ivp
import os

sys.path.append('./model')
from grf import generate_grf

# --- 参数设置 ---
num_samples = 1000
k_param = 1.0

# --- 数据生成 ---
train_sampler = qmc.LatinHypercube(d=1)
test_sampler = qmc.LatinHypercube(d=1)

# 生成高斯随机场 (GRF) 作为输入函数 u(t)
# generate_grf 返回的是 (num_samples, num_points) 的数组
train_grf = generate_grf(num_samples=num_samples, num_points=1000, length_scale=0.2)
test_grf = generate_grf(num_samples=num_samples, num_points=1000, length_scale=0.2)

def build_ode(u_fun):
    u_interp = interp1d(np.linspace(0, 1, 1000), u_fun, kind='linear')

    def ode(t, s):
        # s 是状态向量 [s1, s2]
        s1 = s[0]
        s2 = s[1]

        # 方程：
        # ds1/dt = s2
        # ds2/dt = -k * sin(s1) + u(t)
        ds1dt = s2
        ds2dt = -k_param * np.sin(s1) + u_interp(t)

        return [ds1dt, ds2dt]

    return ode

train_eval = train_sampler.random(n=num_samples)
test_eval = test_sampler.random(n=num_samples)

train_eval.sort(axis=1)
test_eval.sort(axis=1)

train_sol = []
test_sol = []

print("Solving ODEs...")
for i in range(num_samples):
    # 求解测试集
    sol_test = solve_ivp(
        build_ode(test_grf[i]),
        [0, 1],
        [0.0, 0.0],  # 初始条件 s(0) = 0 (向量)
        method='RK45',
        t_eval=test_eval[i]
    )
    test_sol.append(sol_test.y[0, :])

    # 求解训练集
    sol_train = solve_ivp(
        build_ode(train_grf[i]),
        [0, 1],
        [0.0, 0.0],
        method='RK45',
        t_eval=train_eval[i]
    )
    train_sol.append(sol_train.y[0, :]) # 只取 s1 分量作为输出，形状为 (num_points,)

train_sol = np.array(train_sol) # 形状为 (num_samples, num_points, 2)
test_sol = np.array(test_sol)

# --- 处理输入数据 (s0) ---
# 将 1000 点的 GRF 下采样/插值到 100 点作为输入特征
s0_train = []
s0_test = []
for i in range(num_samples):
    # 将长度为 1000 的信号插值到长度为 100
    s0_train.append(np.interp(np.linspace(0, 1, 100), np.linspace(0, 1, 1000), train_grf[i]))
    s0_test.append(np.interp(np.linspace(0, 1, 100), np.linspace(0, 1, 1000), test_grf[i]))

s0_train = np.array(s0_train)
s0_test = np.array(s0_test)

# --- 保存数据 ---
print("Saving data...")
# 确保目录存在
os.makedirs('./4.2', exist_ok=True)

np.savez(
    './4.2/data.npz',
    s_train=train_sol,
    s_test=test_sol,
    x_train=train_eval,
    x_test=test_eval,
    s0_train=s0_train,
    s0_test=s0_test
)
print("Done.")