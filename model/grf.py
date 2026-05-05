import numpy as np

def rbf_kernel(x1, x2, length_scale):
    sq_dist = (x1 - x2.T)**2 # 利用广播机制计算出一个矩阵
    return np.exp(- sq_dist / (2 * length_scale**2))

def generate_grf(num_samples, num_points, length_scale, domain=[0, 1]):
    x = np.linspace(domain[0], domain[1], num_points).reshape(-1, 1)
    K = rbf_kernel(x, x, length_scale)
    for i in range(100):
        try:
            L = np.linalg.cholesky(K)
            print(f"Cholesky decomposition succeeded on attempt {i+1}")
            break
        except:
            K += (i+1) * 1e-13 * np.eye(num_points)
            print(f"Cholesky decomposition attempt {i+1} failed")

    white_noise = np.random.randn(num_samples, num_points)
    grf_samples = white_noise @ L.T
    return grf_samples
