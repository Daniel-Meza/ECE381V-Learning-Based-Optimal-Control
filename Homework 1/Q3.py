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
import sympy as sp
import matplotlib.pyplot as plt
import pickle


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


  def initialize_dynamics(self):
    """
    Calculate and initialize the full cartpole dynamics.
    Linearizes the system and sets symbolic functions to evalue x_dot for the next step and the Jacobians in terms of x and u.
    """
    # Define symbolic variables for states and controls
    q, theta, q_dot, theta_dot = sp.symbols('q theta q_dot theta_dot')
    u = sp.symbols('u')

    # Shorthand trigonometric functions
    s_theta = sp.sin(theta)   # sin(theta)
    c_theta = sp.cos(theta)   # cos(theta)

    # Continuous-Time (equations of motion) system dynamics.
    # From Feedback Systems textbook, Example 3.2 Balance Systems.
    q_ddot = ((-self.m * self.l * s_theta * theta_dot**2) + 
              (self.m * self.g * (self.m * self.l**2 / self.J_t) * s_theta * c_theta) - 
              (self.c * q_dot) - 
              ((self.lamb / self.J_t) * self.m * self.l * c_theta * theta_dot) +
              u) / (
              self.M_t - 
              (self.m * (self.m * self.l**2 / self.J_t) * c_theta**2))
    theta_ddot = ((-self.m * self.l**2 * s_theta * c_theta * theta_dot**2) +
                  (self.M_t * self.g * self.l * s_theta) -
                  (self.c * self.l * c_theta * theta_dot) -
                  (self.lamb * (self.M_t / self.m) * theta_dot) +
                  (self.l * c_theta * u)) / (
                  (self.J_t * (self.M_t / self.m)) - 
                  (self.m * (self.l * c_theta)**2))

    # State derivative vector
    x_dot = sp.Matrix([q_dot, theta_dot, q_ddot, theta_ddot])

    # Partial derivations (Jacobians)
    x_vars = sp.Matrix([q, theta, q_dot, theta_dot])
    Ak = x_dot.jacobian(x_vars)    # with respect to x
    u_vars = sp.Matrix([u])
    Bk = x_dot.jacobian(u_vars)    # with respect to u

    # Convert symbolic expressions to numerical functions
    self.x_func = sp.lambdify((q, theta, q_dot, theta_dot, u), x_dot, 'numpy')
    self.Ak_func = sp.lambdify((q, theta, q_dot, theta_dot, u), Ak, 'numpy')
    self.Bk_func = sp.lambdify((q, theta, q_dot, theta_dot, u), Bk, 'numpy')


  def next_state(self, x_curr: np.ndarray, u_curr: np.ndarray) -> np.ndarray:
    """
    For the given state and control, returns the next state.
    Inputs:
      x: current state
      u: current control input
    Returns:
      x_next: next state
    """
    # Calculate system derivative
    x_dot = self.x_func(x_curr[0][0], x_curr[1][0], x_curr[2][0], x_curr[3][0], u_curr[0][0])

    # Discretize system dynamics with an approximation using the time step
    x_next = x_curr + self.Ts * x_dot   # Euler integration

    return x_next


  def approx_A_B(self, x_curr: np.ndarray, u_curr: np.ndarray) -> tuple[np.ndarray]:
    """
    For the given state and control,  returns approximations of the A_k, B_k (the Jacobians of the dynamics in terms of state and control).
    matrices.
    Inputs:
      x_curr: 2D array of shape (n, 1)
      u_curr: 2D array of shape (m, 1)
    Returns:
      A_k: 2D array of shape (n, n)
      B_k: 2D array of shape (n, m)
    """

    # Evaluate A and B using the symbolic dynamics equations
    A_k = self.Ak_func(x_curr[0][0], x_curr[1][0], x_curr[2][0], x_curr[3][0], u_curr[0][0])
    B_k = self.Bk_func(x_curr[0][0], x_curr[1][0], x_curr[2][0], x_curr[3][0], u_curr[0][0])

    return A_k, B_k


  def forward_pass(self, x_nominal: list[np.ndarray], u_nominal: list[np.ndarray], d: list[np.ndarray], K: list[np.ndarray], Q:np.ndarray, R: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray], float]:
    """
    For the given initial state and nominal trajectory. Perform the forward pass of iLQR to compute the new state trajectory and total cost.
    Inputs:
      x_nominal: nominal state trajectory
      u_nominal: nominal control trajectory
      delta_u: optimal control corrections
    Returns:
      x_new: new state trajectory
      u_new: new control trajectory
    """
    # Create lists for new trajectory
    x_new = [None] * (self.N + 1)
    u_new = [None] * self.N

    # Initialize first state
    x_new[0] = x_nominal[0]

    J_total = 0   # Total trajectory cost

    # Loop forward
    for k in range(self.N):
      # print("+++" + str(k) + "+++")
      # Correct the control input
      u_new[k] = u_nominal[k] + K[k] @ (x_new[k] - x_nominal[k]) + d[k]

      # print("a")

      # Calculate next state
      x_new[k + 1] = self.next_state(x_new[k], u_new[k])

      # print("b")

      # Update trajectory cost
      J_total += 1/2 * (x_new[k].T @ Q @ x_new[k] + u_new[k].T @ R @ u_new[k]).item()
      
      # print("c")

    # Add terminal cost
    J_total += (x_new[self.N].T @ Q @ x_new[self.N]).item()

    return x_new, u_new, J_total


  def backward_pass(self, x_new: list[np.ndarray], u_new: list[np.ndarray], Q: np.ndarray, R: np.ndarray, H=np.array([[0], [0], [0], [0]])):
    """
    Perform the backward pass of iLQR to compute the feedforward control updates and feedback gains.
    Inputs:
      x_new: new state trajectory from the forward pass
      u_new: new control trajectory from the forward pass
      Q: state cost matrix
      R: control cost matrix
      H: bilinear cost matrix
    Returns:
      d:
      K:
    """
    # Create lists for necessary variables
    d = [None] * self.N
    K = [None] * self.N

    # Initialize value function at final step
    S = Q
    s = Q @ (x_new[self.N])   # Desired final state is an array of 0s

    # Loop backward
    for k in range(self.N - 1, -1, -1):
      # print("-" + str(k) + "-")
      # Linearize around the current trajectory
      A_k, B_k = self.approx_A_B(x_new[k], u_new[k])

      # print("A")

      # Calculate stage cost derivatives
      l_x = Q @ x_new[k]    # (4, 1)
      l_u = R @ u_new[k]    # (1, 1)
      l_xx = Q              # (4, 4)
      l_uu = R              # (1, 1)
      l_xu = H              # (4, 1)
      l_ux = H.T            # (1, 4)

      # Compute Q terms
      Q_x = l_x + A_k.T @ s               # (4, 1)
      Q_u = l_u + B_k.T @ s               # (1, 1)
      Q_xx = l_xx + A_k.T @ S @ A_k     # (4, 4)
      Q_uu = l_uu + B_k.T @ S @ B_k     # (1, 1)
      Q_ux = l_ux + B_k.T @ S @ A_k     # (1, 4)

      # print("B")

      # Update the feedback law
      d[k] = -np.linalg.inv(Q_uu) @ Q_u   # (1, 1)
      K[k] = -np.linalg.inv(Q_uu) @ Q_ux  # (1, 4)
      # print("C")

      # Update the value function approximation
      s = Q_x + K[k].T @ Q_uu @ d[k] + K[k].T @ Q_u + Q_ux.T @ d[k]   # (4, 1)
      S = Q_xx + K[k].T @ Q_uu @ K[k] + K[k].T @ Q_ux + Q_ux.T @ K[k]   # (4, 4)

      # print("D")

    return d, K


  def ilqr(self, x_nominal_initial: list[np.ndarray], u_nominal_initial: list[np.ndarray], Q: np.ndarray, R: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Discrete-time iterative linear quadratic regulator for non-linear system.
    Compute the optimal control inputs for a non-linear system given an initial state, a nominal trajectory, and cost matrices. I.e. compute the control input that minimizes the cumulative cost.
    Inputs:
      x_nominal_initial: initial nominal state trajectory
      u_nominal_initial = initial nominal control trajectory
      Q: state cost matrix
      R: control cost matrix
    Returns:
      x: calculated state trajectory (optimal)
      u: calculated control trajectory (optimal)
    """
    # Define number of iterations and tolerance
    max_iterations = 10
    tolerance = 1e-3

    # Initialize full system dynamics
    self.initialize_dynamics()

    # Set initial nominal trajectories
    x_nominal = x_nominal_initial
    u_nominal = u_nominal_initial

    print(x_nominal[0])
    print(u_nominal[0])
    x_dot = self.x_func(x_nominal[0][0], x_nominal[1][0], x_nominal[2][0], x_nominal[3][0], u_nominal[0][0])
    print(x_dot)

    exit()

    J_prev = 0   # Big number to prevent convergence

    # Iterate between forward and backward pass until convergence
    for i in range(max_iterations):
      print("---" + str(i) + "---")

      # Backward pass to compute feedforward and feedback gains
      d, K = self.backward_pass(x_nominal, u_nominal, Q, R)
      print("BBBBB")

      # Forward pass to generate new state and control trajectories
      x_nominal, u_nominal, J_total = self.forward_pass(x_nominal, u_nominal, d, K, Q, R)
      print("FFFFF")

      print(J_total)

      # Check for convergence based on the change in cost
      cost_diff = abs(J_total - J_prev)
      if cost_diff < tolerance:
        print(f"Converged at iteration {i}. Cost difference: {cost_diff}")
        break
      J_prev = J_total

    return x_nominal, u_nominal


def main():
  # Define time step and horizon for the control problem
  time_step = 0.1
  time_horizon = 10

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
    [0, 0, 0, 1]
  ])
  R = np.array([
    [0.1]
  ])

  # Create the cartpole object
  cartpole = CartPole(time_step, time_horizon)

  # Run LQR algorithm to find optimal trajectory
  x, u = cartpole.lqr(x_initial, Q, R)

  # Define nominal trajectories for iLQR
  # x_nominal = [np.array([
  #     [0],
  #     [0],
  #     [0],
  #     [0]
  #   ])] * (cartpole.N + 1)
  # x_nominal[0] = x_initial
  # u_nominal = [np.array([[0]])] * cartpole.N
  x_nominal = x
  u_nominal = u

  # TODO Test with Steven nominal trajectories
  # with open('x_ref.pickle', 'rb') as file:
  #   loaded_array = pickle.load(file)
  # x_nominal = loaded_array

  # with open('u_ref.pickle', 'rb') as file:
  #   loaded_array = pickle.load(file)
  # u_nominal = loaded_array

  # Run iLQR algorithm to find optimal trajectory
  x, u = cartpole.ilqr(x_nominal, u_nominal, Q, R)

  # return

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
