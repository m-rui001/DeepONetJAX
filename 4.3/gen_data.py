import numpy as np
import sys
sys.path.append('./model')
from grf import generate_grf


class ADRSolver:
    """Solver for 1D Advection-Diffusion-Reaction equation.
    
    u_t = (k(x)u_x)_x - v(x)u_x + g(u) + f(x,t)
    with zero boundary conditions.
    """
    
    def __init__(self, x_range=(0, 1), t_range=(0, 1), nx=100, nt=100):
        self.x = np.linspace(*x_range, nx)
        self.t = np.linspace(*t_range, nt)
        self.dx = self.x[1] - self.x[0]
        self.dt = self.t[1] - self.t[0]
        self.nx = nx
        self.nt = nt
        
    def _build_matrices(self, diffusion, velocity, reaction, reaction_derivative):
        """Build finite difference matrices for the ADR equation."""
        dx, dt = self.dx, self.dt
        dx2 = dx ** 2
        n = self.nx
        
        # Diffusion operator matrices
        D1 = np.eye(n, k=1) - np.eye(n, k=-1)
        D2 = -2 * np.eye(n) + np.eye(n, k=-1) + np.eye(n, k=1)
        
        k = diffusion(self.x)
        M = -np.diag(D1 @ k) @ D1 - 4 * np.diag(k) @ D2
        
        # Interior domain mapping
        I_interior = np.eye(n - 2)
        
        # Mass matrix
        mass = 8 * dx2 / dt * I_interior + M[1:-1, 1:-1]
        
        # Velocity matrix
        v = velocity(self.x)
        v_bond = (2 * dx * np.diag(v[1:-1]) @ D1[1:-1, 1:-1] + 
                  2 * dx * np.diag(v[2:] - v[:-2]))
        
        # System matrix
        A_matrix = mass + v_bond
        C_matrix = 8 * dx2 / dt * I_interior - M[1:-1, 1:-1] - v_bond
        
        return A_matrix, C_matrix, dx2
    
    def solve(self, diffusion, velocity, reaction, reaction_derivative, 
              source_func, initial_condition):
        """Solve the ADR equation using Crank-Nicolson scheme."""
        A_matrix, C_matrix, dx2 = self._build_matrices(
            diffusion, velocity, reaction, reaction_derivative
        )
        
        # Initialize solution
        u = np.zeros((self.nx, self.nt))
        u[:, 0] = initial_condition(self.x)
        
        # Source term
        source = source_func(self.x[:, None], self.t)
        
        # Time stepping
        for i in range(self.nt - 1):
            u_interior = u[1:-1, i]
            
            # Reaction terms
            g_val = reaction(u_interior)
            dg_val = reaction_derivative(u_interior)
            dg_diag = np.diag(4 * dx2 * dg_val)
            
            # System for next time step
            A = A_matrix - dg_diag
            
            # Right-hand side
            rhs1 = 8 * dx2 * (0.5 * source[1:-1, i] + 
                             0.5 * source[1:-1, i + 1] + g_val)
            rhs2 = (C_matrix - dg_diag) @ u_interior
            
            u[1:-1, i + 1] = np.linalg.solve(A, rhs1 + rhs2)
            
        return self.x, self.t, u


class DiffusionReactionSystem:
    """Generates training data for diffusion-reaction systems."""
    
    def __init__(self, diffusion_coef=0.01, reaction_coef=0.01, 
                 t_final=1.0, n_timesteps=100, n_output_points=100):
        self.D = diffusion_coef
        self.k = reaction_coef
        self.T = t_final
        self.Nt = n_timesteps
        self.n_output_points = n_output_points
        
        # Create solver
        self.solver = ADRSolver(
            x_range=(0, 1), 
            t_range=(0, t_final), 
            nt=n_timesteps
        )

    def generate_data(self, num_samples=1000, spatial_points=100, 
                    length_scale=0.2, train_split=True):
        grf = generate_grf(num_samples=num_samples, num_points=spatial_points,
                        length_scale=length_scale)
        sensor_values = grf  # shape: (num_samples, spatial_points)

        solutions = []
        for u in sensor_values:
            x, t, sol = self.solver.solve(
                diffusion=lambda x: self.D * np.ones_like(x),
                velocity=lambda x: np.zeros_like(x),
                reaction=lambda u: self.k * u ** 2,
                reaction_derivative=lambda u: 2 * self.k * u,
                source_func=lambda x, t: np.tile(u[:, None], (1, len(t))),
                initial_condition=lambda x: np.zeros_like(x)
            )
            solutions.append(sol)

        m = spatial_points
        s0_list = []
        x_list = []
        s_list = []

        for u, sol in zip(sensor_values, solutions):
            # 随机选择时空输出点
            x_idx = np.random.randint(m, size=self.n_output_points)
            t_idx = np.random.randint(self.Nt, size=self.n_output_points)
            x_norm = x_idx / (m - 1)
            t_norm = t_idx * self.T / (self.Nt - 1)

            s0_list.append(u)                              # 整个函数的传感器值 (m,)
            x_list.append(np.column_stack([x_norm, t_norm]))  # 坐标点 (N, 2)
            s_list.append(sol[x_idx, t_idx])               # 解值 (N,)

        # 转为 batch 数组
        s0_batch = np.array(s0_list)          # (num_samples, m)
        x_batch  = np.array(x_list)           # (num_samples, N, 2)
        s_batch  = np.array(s_list, dtype=np.float32)  # (num_samples, N)

        return s0_batch, x_batch, s_batch

def create_default_system(t_final=1.0, n_output_points=100):
    """Create a default diffusion-reaction system."""
    return DiffusionReactionSystem(
        diffusion_coef=0.01,
        reaction_coef=0.01,
        t_final=t_final,
        n_output_points=n_output_points
    )


if __name__ == "__main__":
    # Create system and generate data
    system = create_default_system()
    s0_train, x_train, s_train = system.generate_data(num_samples=100)
    s0_test, x_test, s_test = system.generate_data(num_samples=100)
    np.savez('./4.3/data.npz',
            s0_train=s0_train, x_train=x_train, s_train=s_train,
            s0_test=s0_test, x_test=x_test, s_test=s_test)