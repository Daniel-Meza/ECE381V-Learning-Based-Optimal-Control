### Comments from Office Hours ###
# 1. Add u_previous to the state matrix.
#    Change the action vector to deltaU

# 2. Official solution writes everything out.
#    Different solution can be to re-define.

# 3. Do not use cvxpy
# 3. LQR
# - First linearize the dynamics
# - Then define our own cost matrices (his control cost = 0.001)
# - Steps 4 and 5 in dynamic programming is a forward loop!

# 3. iLQR
# - Working problem 2 can help with derivation for iLQR
# - Initial nominal trajectory can be u_bar = [0, ...], x_bar = [x0, 0, ...] and it still converges
# Backward Pass
# - Gives you an array of K matrices
# - Initialize with delta_x = 0
# - At t = 0, compute delta_u using K matrix in the affine term
# - ...
# - Returns trajectories for deltaX and deltaU
# - Run the true dynamics
# Forward Pass
# - Take solution and forward propagrate with true dynamics to find new trajectory.


import numpy as np
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
    Calculate and initialize the cartpole dynamics.
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


  def calculate_derivative(self, x_curr: np.ndarray, u_curr: np.ndarray) -> np.ndarray:
    """
    For the given state and control, returns the derivative of the system based on the equations of motion.
    Inputs:
      x: current state
      u: current control input
    Returns:
      x_dot: state derivative
    """
    # Calculate system dynamics
    # Continuous-Time system dynamics from Feedback Systems textbook, Example 3.2 Balance Systems
    u = u_curr[0][0]
    s_theta = np.sin(x_curr[1])[0]   # sin(theta)
    c_theta = np.cos(x_curr[1])[0]   # cos(theta)

    # Equations of motion
    q_dot = x_curr[2]
    theta_dot = x_curr[3]
    q_ddot = ((-self.m * self.l * s_theta * theta_dot**2) + 
              (self.m * self.g * (self.m * self.l**2 / self.J_t) * s_theta * c_theta) - 
              (self.c * q_dot) - 
              ((self.lamb / self.J_t) * self.m * self.l * c_theta * theta_dot) +
              u) / (
              self.M_t - 
              (self.m * (self.m * self.l**2 / self.J_t) * c_theta))
    theta_ddot = ((-self.m * self.l**2 * s_theta * c_theta * theta_dot**2) +
                  (self.M_t * self.g * self.l * s_theta) -
                  (self.c * self.l * c_theta * theta_dot) -
                  (self.lamb * (self.M_t / self.m) * theta_dot) +
                  (self.l * c_theta * u)) / (
                  (self.J_t * (self.M_t / self.m)) - 
                  (self.m * (self.l * c_theta)**2))

    x_dot = np.array([q_dot, theta_dot, q_ddot, theta_ddot])

    return x_dot


  def next_step(self, x_curr: np.ndarray, u_curr: np.ndarray) -> np.ndarray:
    """
    For the given state and control, returns the next state.
    Inputs:
      x: current state
      u: current control input
    Returns:
      x_next: next state
    """
    # Calculate system derivative
    x_dot = self.calculate_derivative(x_curr, u_curr)

    # Discretize system dynamics with an approximation using the time step
    x_next = x_curr + x_dot * self.Ts   # Euler integration

    return x_next


  def stage_cost(self, x_curr: np.ndarray, u_curr: np.ndarray, Q: np.ndarray, R: np.ndarray) -> float:
    """
    For the given state and control, returns the stage cost.
    Inputs:
      x: current state
      u: current control input
      Q: state cost matrix
      R: control cost matrix
    Returns:
      l: stage cost
    """
    cost = x_curr.T @ Q @ x_curr + u_curr.T @ R @ u_curr

    return cost[0][0]


  def approx_A_B(self, x: np.ndarray, u: np.ndarray, eps=1e-3) -> tuple[np.ndarray]:
    """
    For the given state and control, returns approximations of the A and B
    matrices.
    Inputs:
      x: 2D array of shape (n, 1)
      u: 2D array of shape (m, 1)
    Returns:
      A: 2D array of shape (n, n)
      B: 2D array of shape (n, m)
    """
    n = x.shape[0]    # state dimension
    m = u.shape[0]    # control dimension

    A = np.zeros((n, n))
    B = np.zeros((n, m))

    # Linearize the system around the given state and control by adding small perturbation
    # Compute A matrix (partial derivative of f with respect to x)
    for i in range(n):
      dx = np.zeros_like(x)
      dx[i] = eps
      f_plus = self.calculate_derivative(x + dx, u)
      f_minus = self.calculate_derivative(x - dx, u)
      A_col = (f_plus - f_minus) / (2 * eps)
      A[:,i] = A_col[:,0]

    # Compute B matrix (partial derivative of f with respect to u)
    for i in range(m):
      du = np.zeros_like(u)
      du[i] = eps
      f_plus = self.calculate_derivative(x, u + du)
      f_minus = self.calculate_derivative(x, u - du)
      B_col = (f_plus - f_minus) / (2 * eps)
      B[:,i] = B_col[:,0]

    # TODO Does B change at all?

    return A, B


  def forward_pass(self, x0_nominal: np.ndarray, u_nominal: list[np.ndarray], Q: np.ndarray, R: np.ndarray) -> tuple[list[np.ndarray], float]:
    """
    For the given initial state and nominal trajectory. Perform the forward pass of iLQR to compute the new state trajectory and total cost.
    Inputs:
      x_initial: initial state in nominal trajectory
      u_nominal: nominal trajectory for controls
      Q: state cost matrix.
      R: control cost matrix.
    Returns:
      x_new: new trajectory for states
      J_total: total cost for the forward pass
    """
    # Initialize variables
    x_new = [None] * (self.N + 1)
    x_new[0] = x0_nominal
    u_new = u_nominal
    J_total = 0.0

    # Loop forward
    for k in range(self.N):
      # Compute next state
      x_new[k + 1] = self.next_step(x_new[k], u_new[k])

      # Update total cost
      J_total += self.stage_cost(x_new[k], u_new[k], Q, R)

    # add terminal cost
    J_total += (x_new[self.N].T @ Q @ x_new[self.N])[0][0]

    return x_new, J_total


  # TOOD Returns declaration
  def backward_pass(self, x_new: list[np.ndarray], u_new: list[np.ndarray], Q: np.ndarray, R: np.ndarray):
    """
    Perform the backward pass of iLQR to compute the feedforward control updates and feedback gains.
    Inputs:
      x_new: new state trajectory from the forward pass
      u_new: new control trajectory from the forward pass
      Q: state cost matrix.
      R: control cost matrix.
      Q_f: terminal state cost matrix.
    Returns:
      delta_u: feedforward control updates.
      K: feedback gains.
    """
    # Initialize variables
    delta_u = [None] * self.N
    K = [None] * self.N

    # Initialize the value function (at time step N)
    V_k = Q   # Terminal cost
    v_k = x_new[self.N].T @ V_k @ x_new[self.N]   # Linear term in value function

    # Loop backward
    for k in range(self.N - 1, -1, -1):
      # Linearize around the current trajectory
      A_k, B_k = self.approx_A_B(x_new[k], u_new[k])

      # Calculate Q terms
      Q_x = Q @ x_new[k]
      Q_u = R @ u_new[k]
      Q_xx = Q + A_k.T @ V_k @ A_k
      Q_ux = B_k.T @ V_k @ A_k
      Q_uu = R + B_k.T @ V_k @ B_k

      # Compute feedback and feedforward terms
      delta_u[k] = -np.linalg.inv(Q_uu) @ Q_u
      K[k] = -np.linalg.inv(Q_uu) @ Q_ux

      # Update the value function
      V_k = Q_xx + K[k].T @ Q_uu @ K[k] + K[k].T @ Q_ux + Q_ux.T @ K[k]
      v_k = Q_x + K[k].T @ Q_uu @ delta_u[k] + Q_ux.T @ delta_u[k]
    
    exit()

    return delta_u, K


  def ilqr(self, x_initial: np.ndarray, u_nominal_initial: list[np.ndarray], Q: np.ndarray, R: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Discrete-time iterative linear quadratic regulator for non-linear system.
    Compute the optimal control inputs for a non-linear system given an initial state, a nominal trajectory, and cost matrices. I.e. compute the control input that minimizes the cumulative cost.
    Inputs:
      x_initial: initial state of the system
      u_nominal_initial = nominal trajectory for controls
      Q: state cost matrix
      R: control cost matrix
    Returns:
      x: calculated states (optimal trajectory)
      u: calculated controls (optimal trajectory)
    """
    # Define number of iterations and tolerance
    max_iterations = 100
    tolerance = 1e-4

    # Initialize nominal trajectory for the first time
    x_nominal = [x_initial]
    u_nominal = u_nominal_initial

    # Iterate between forward and backward pass until convergence
    for i in range(max_iterations):
      # Forward pass to generate new state trajectory
      x_new, J_total = self.forward_pass(x_nominal[0], u_nominal, Q, R)

      # Check for convergence

      # Backward pass to compute feedforward and feedback gains
      delta_u, K = self.backward_pass(x_new, u_nominal, Q, R)

      exit()

      # Update nominal trajectory
      # for k in range(self.N):
      #   ...
      # x_nominal = x_new

    x = ...
    u = ...
    return x, u


def main():
  # Define time step and horizon for the control problem
  time_step = 0.1
  time_horizon = 30

  # Define initial state [q, theta, q_dot, theta_dot]
  x_initial = np.array([
    [0],
    [0.174533 * 1],
    [0],
    [0]
  ])

  # Define state cost and control cost matrices
  Q = np.array([
    [1, 0, 0, 0],
    [0, 1000, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 10]
  ])
  R = np.array([
    [1]
  ])

  # Create the cartpole object
  cartpole = CartPole(time_step, time_horizon)

  # Run LQR algorithm to find optimal trajectory
  x, u = cartpole.lqr(x_initial, Q, R)

  # Define nominal trajectory for iLQR
  u_nominal = [np.array([[0]])] * cartpole.N

  # Run iLQR algorithm to find optimal trajectory
  x, u = cartpole.ilqr(x_initial, u_nominal, Q, R)

  # Retrieve data for plots
  time_values = [t * time_step for t in range(cartpole.N + 1)]
  states = [float(x_k[1][0]) for x_k in x]
  controls = [float(u_k[0][0]) for u_k in u]
  controls.append(0.0)  # to match dimensions

  # Create plots
  _, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
  ax1.plot(time_values, states, label='state', color='blue')
  ax1.set_xlabel('Time (s)')
  ax1.legend()
  ax2.plot(time_values, controls, label='control', color='orange')
  ax2.set_xlabel('Time (s)')
  ax2.legend()
  plt.show()


if __name__ == "__main__":
  main()
