import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt


class CartPole:
  def __init__(self, Ts: float, Th: float):
    """
    Initialize the cartpole environment.
    Inputs:
      Ts: the simulation step size
      N: the simulation time horizon
    """
    self.Ts = Ts
    self.Th = Th
    self.N = int(Th / Ts)
    self.M = 10
    self.m = 8
    self.c = 0.1
    self.J = 5
    self.l = 1
    self.lamb = 0.01
    self.g = 9.8
    self.M_t = self.M + self.m
    self.J_t = self.J + self.m * self.l**2
    self.mu = self.M_t * self.J_t - self.m**2 * self.l**2
  

  def initialize_linear_dynamics(self):
    """
    Calculate and initialize the cartpole dynamics for the linear approximation case.
    """
    # Continuous-Time system dynamics (linear approximation for small theta and theta_dot)
    # From Feedback Systems textbook, Example 3.2 Balance Systems
    A = np.array([
      [0, 0, 1, 0],
      [0, 0, 0, 1],
      [0, (self.m**2 * self.l**2 * self.g / self.mu), (-self.c * self.J_t / self.mu), (-self.lamb * self.l * self.m / self.mu)],
      [0, (self.M_t * self.m * self.g * self.l / self.mu), (-self.c * self.l * self.m / self.mu), (-self.lamb * self.M_t / self.mu)]
    ])
    B = np.array([
      [0],
      [0],
      [self.J_t / self.mu],
      [self.l * self.m / self.mu]
    ])

    # Discretize system dynamics with an approximation using the time step (Zero Order Hold)
    self.A = np.eye(4) + A * self.Ts
    self.B = B * self.Ts


  def next_linear_step(self, x_curr: np.ndarray, u_curr: np.ndarray) -> np.ndarray:
    """
    For the given state and control, returns the next state.
    Inputs:
      x: current state
      u: current control input
    Returns:
      x_next: next state
    """
    # Calculate next step using current state and control
    x_next = self.A @ x_curr + self.B @ u_curr

    return x_next


  def lqr(self, x_initial: np.ndarray, Q: np.ndarray, R: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Discrete-time linear quadratic regulator for a linear system.
    Compute the optimal control inputs for a linear system given current state and cost matrices. I.e. compute the control input that minimizes the cumulative cost.
    Inputs:
      x_initial: initial state of the system
      Q: state cost matrix
      R: control cost matrix
    Returns:
      x: calculated states (optimal trajectory)
      u: calculated controls (optimal trajectory)
    """
    # Initialize the system dynamics
    self.initialize_linear_dynamics()

    # Create lists for V and L
    V = [None] * (self.N + 1)
    L = [None] * self.N

    # Set final cost
    V[self.N] = Q

    # Dynamic Programming backwards recursion
    for k in range(self.N - 1, -1, -1):
      L[k] = -np.linalg.pinv(R + self.B.T @ V[k + 1] @ self.B) @ (self.B.T @ V[k + 1] @ self.A)
      V[k] = Q + L[k].T @ R @ L[k] + (self.A + self.B @ L[k]).T @ V[k + 1] @ (self.
      A + self.B @ L[k])

    # Create lists for x and u
    x = [None] * (self.N + 1)
    u = [None] * self.N

    # Set initial state
    x[0] = x_initial

    # Forward loop to calculate trajectory
    for k in range(self.N):
      u[k] = L[k] @ x[k]
      x[k + 1] = self.next_linear_step(x[k], u[k])

    return x, u


  def qp(self, x_initial: np.ndarray, Q: np.ndarray, R: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Solve the discrete-time linear quadratic regulator linearized about 0 by reqriting it as a quadratic program and using the cvxpy solver.
    Compute the optimal control inputs given the current state and cost matrices. I.e. compute the control input that minimizes the cumulative cost. 
    Inputs:
      x_initial: initial state of the system
      Q: state cost matrix
      R: control cost matrix
    Returns:
      x: calculated states (optimal trajectory)
      u: calculated controls (optimal trajectory)
    """
    # Initialize the system dynamics
    self.initialize_linear_dynamics()

    # Optimization Variables
    x = [cp.Variable((4, 1)) for _ in range(self.N + 1)]
    u = [cp.Variable((1, 1)) for _ in range(self.N)]

    # Objective Function
    c = 0
    for k in range(self.N - 1):
      c += cp.quad_form(x[k], Q) + cp.quad_form(u[k], R)   # running cost
    c += cp.quad_form(x[k], Q)   # terminal cost
    objective = cp.Minimize(c)

    # Constraints
    constraints = [x[0] == x_initial]    # initial state
    for k in range(self.N):
      constraints.append(x[k + 1] == self.A @ x[k] + self.B @ u[k])

    # Define and Solve Problem
    problem = cp.Problem(objective, constraints)
    problem.solve()

    # Results
    x_star = []
    u_star = []
    for k in range(self.N):   # forward loop to retrieve values
      x_star.append(x[k].value)
      u_star.append(u[k].value)
    x_star.append(x[self.N].value)  # terminal state

    return x_star, u_star


def create_plots(time_step: float, time_horizon: float, title: str, x: list[np.ndarray], u: list[np.ndarray], x1=None, u1=None):
  N = int(time_horizon / time_step)

  if not (x1 and u1):
    # Single plots for LQR
    # Retrieve data for plots
    time_values = [t * time_step for t in range(N + 1)]
    states = [float(x_k[1][0]) for x_k in x]
    controls = [float(u_k[0][0]) for u_k in u]

    # Create plots for state and control trajectories
    _, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
    ax1.plot(time_values, states, label='state', color='blue')
    ax1.set_xlabel('Time (s)')
    ax2.plot(time_values[:-1], controls, label='control', color='orange')
    ax2.set_xlabel('Time (s)')
    ax1.legend()
    ax2.legend()
    plt.suptitle(title)
    plt.show()
  else:
    # Double plots for iLQR
    # Retrieve data for plots
    time_values = [t * time_step for t in range(N + 1)]
    states = [float(x_k[1][0]) for x_k in x]
    controls = [float(u_k[0][0]) for u_k in u]
    states_1 = [float(x_k[1][0]) for x_k in x1]
    controls_1 = [float(u_k[0][0]) for u_k in u1]

    # Create plots for state and control trajectories
    _, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
    ax1.plot(time_values, states, label='state_dp', color='gray')
    ax1.plot(time_values, states_1, label='state_qp', color='blue')
    ax1.set_xlabel('Time (s)')
    ax2.plot(time_values[:-1], controls, label='control_dp', color='gray')
    ax2.plot(time_values[:-1], controls_1, label='control_qp', color='orange')
    ax2.set_xlabel('Time (s)')
    ax1.legend()
    ax2.legend()
    plt.suptitle(title)
    plt.show()


def main():
  # Define time step and horizon for the control problem
  time_step = 0.01
  time_horizon = 10

  # Create the cartpole object
  cartpole = CartPole(time_step, time_horizon)

  # Define state cost and control cost matrices
  Q = np.array([
    [1, 0, 0, 0],
    [0, 1000, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 10]
  ])
  R = np.array([
    [0.1]
  ])

  # Define initial state [q, theta, q_dot, theta_dot]
  x_initial = np.array([
    [0],
    [0.174533 * 1],
    [0],
    [0]
  ])

  # Run LQR dynamic programming algorithm to find optimal trajectory
  x_lqr, u_lqr = cartpole.lqr(x_initial, Q, R)

  # Run LQR quadratic program solver to find optimal trajectory
  x_qp, u_qp = cartpole.qp(x_initial, Q, R)

  # Calculate difference in solutions
  N = int(time_horizon / time_step)
  x_diff = [x_lqr[k] - x_qp[k] for k in range(N + 1)]
  u_diff = [u_lqr[k] - u_qp[k] for k in range(N)]

  # Plot results
  create_plots(time_step, time_horizon, 'DP vs QP, 30deg, 20sec', x_lqr, u_lqr, x_qp, u_qp)
  # create_plots(time_step, time_horizon, 'Difference, 30deg, 20sec', x_diff, u_diff)


if __name__ == "__main__":
  main()
