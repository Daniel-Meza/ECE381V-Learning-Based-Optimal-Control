import numpy as np
from typing import Tuple
class CartPole:
  def __init__(self, Ts: float):
    """Initialize the cartpole environment
    Inputs:
    Ts: float (the simulation step size)
    """
    self.Ts = Ts
    self.M = 10
    self.m = 8
    ### TODO: HW1, set the parameters of the cartpole
    self.c = ...
    ...


  def next_step(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
    """
    For the given state and control, returns the next state
    Inputs:
    x: 2D array of shape (n, 1)
    u: 2D array of shape (m, 1)
    Returns:
    x_next: 2D array of shape (n, 1)
    """
    ### TODO: HW1, approximate x_next with x + Ts * f(x, u)
    x_next = ...
    return x_next


  def approx_A_B(self, x: np.ndarray, u: np.ndarray) -> Tuple[np.ndarray]:
    """
    For the given state and control, returns approximations of the A and B
    matrices
    Inputs:
    x: 2D array of shape (n, 1)
    u: 2D array of shape (m, 1)
    Returns:
    A: 2D array of shape (n, n)
    B: 2D array of shape (n, m)
    """
    ### TODO: HW1, linearize the system around the given state and control
    A = ...
    B = ...
