import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


class GridWorld:
  def __init__(self, n: int, s_goal, gamma: float = 0.99, epsilon: float = 1e-4):
    """
    Initialize the gridworld environment.
    Inputs:
      n: size of the n x n grid 
      s_goal: goal state (x, y)
      gamma: discount factor
      epsilon: threshold for convergence
    """
    # Grid
    self.n = n
    self.s_goal = s_goal
    self.gamma = gamma
    self.epsilon = epsilon

    # Possible Actions
    self.actions = ('up', 'down', 'left', 'right')
    self.action_map = {
      'up': (0, 1),
      'down': (0, -1),
      'left': (-1, 0),
      'right': (1, 0)
    }

    # Value Function (initially zeros)
    self.V = np.zeros((n, n))
    self.V[s_goal] = 1

    # Policy (initially all 'up')
    self.pi = np.zeros((n, n))
    self.policy_map = {
      'up': 0,
      'down': 1,
      'left': 2,
      'right': 3
    }


  def is_out_of_bounds(self, state):
    """
    TODO
    """
    x, y = state
    out_of_bounds = x < 0 or x >= self.n or y < 0 or y >= self.n

    return out_of_bounds


  def value_iteration(self, max_iters = 1000) -> np.ndarray:
    """
    TODO
    """
    for i in range(max_iters):
      delta = 0   # variable for convergence check

      # Iterate through grid updating the reward value at each cell
      for x in range(self.n):
        for y in range(self.n):
          # Skip the goal state
          if (x, y) == self.s_goal:
            continue

          # Evaluate all possible actions from current state (x, y)
          for a in self.actions:
            # Compute the expected value of taking action a
            expected_value = 0

            # Sum of expected values
            for a_prime in self.actions:
              # Determine probability of moving in that direction
              if a_prime == a:
                prob = 0.8  # 80% for moving in the specified direction
              else:
                prob = 0.2 / 3  # 20% for moving uniformly in a random direction

              # Calculate the next state (x', y') and check for out-of-bounds
              x_prime = x + self.action_map[a_prime][0]
              y_prime = y + self.action_map[a_prime][1]
              
              if self.is_out_of_bounds((x_prime, y_prime)):
                x_prime, y_prime = x, y
              
              # Calculate the reward
              # The only reward is 1 at the goal state. This is already accounted for with the initialization of that location to 1, thus we can ignore
              reward = 0

              # Update the expected value
              expected_value += prob * (reward + self.gamma * self.V[x_prime, y_prime])

          # Update the value function at (x, y) and track convergence
          delta += expected_value - self.V[x, y]
          self.V[x, y] = expected_value

      # Check for convergence
      if delta < self.epsilon:
        print("[Value Iteration] Converged in %d iterations with a difference of %f" % (i, delta))
        break

    return self.V


  def policy_iteration(self, max_iters = 1000) -> np.ndarray:
    """
    TODO
    """
    for i in range(max_iters):
      # Policy Evaluation
      while True:
        delta = 0

        # Evaluate the current policy
        for x in range(self.n):
          for y in range(self.n):
            # Skip the goal state
            if (x, y) == self.s_goal:
              continue

            # Calculate the next state based the current policy action
            current_action = list(self.policy_map.keys())[int(self.pi[x, y])]
            x_next = x + self.action_map[current_action][0]
            y_next = y + self.action_map[current_action][1]

            if self.is_out_of_bounds((x_next, y_next)):
              x_next, y_next = x, y

            # Sum of expected values
            expected_value = 0
            for a_prime in  self.actions:
              # Determine probability of moving in that direction
              if a_prime == current_action:
                prob = 0.8  # 80% for moving in the specified direction
              else:
                prob = 0.2 / 3  # 20% for moving uniformly in a random direction

              # Calculate the next state (x', y') and check for out-of-bounds
              x_prime = x + self.action_map[a_prime][0]
              y_prime = y + self.action_map[a_prime][1]
              
              if self.is_out_of_bounds((x_prime, y_prime)):
                x_prime, y_prime = x, y
              
              # Calculate the reward
              # The only reward is 1 at the goal state. This is already accounted for with the initialization of that location to 1, thus we can ignore
              reward = 0

              # Update the expected value
              expected_value += prob * (reward + self.gamma * self.V[x_prime, y_prime])

              







    return self.V


  def policy_derivation(self) -> np.ndarray:
    """
    TODO
    """

    # Iterate through grid updating the policy at each cell
    for x in range(self.n):
      for y in range(self.n):
        # Skip the goal state
        if (x, y) == self.s_goal:
          continue

        # Evaluate all possible actions from current state (x, y)
        best_action = None
        best_value = -np.inf
        for a in self.actions:
          # Calculate the next state (x', y') and check for out-of-bounds
          x_prime = x + self.action_map[a][0]
          y_prime = y + self.action_map[a][1]
          
          if self.is_out_of_bounds((x_prime, y_prime)):
            x_prime, y_prime = x, y
          
          # Track the best action to take
          expected_value = self.V[x_prime, y_prime]
          if (expected_value > best_value):
            best_value = expected_value
            best_action = a

        # Update the policy at (x, y)
        self.pi[x, y] = self.policy_map[best_action]

    return self.pi
  

  def plot_heatmap(self, map: np.ndarray, title = ''):
    """
    TODO
    """
    plt.figure()
    plt.imshow(map.T, origin='lower')   # transpose to align axes
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


def main():
  # Create the grid world object
  size = 20
  goal = (17, 6)
  grid_world = GridWorld(size, goal)

  # Compute value function over the grid
  value_function_v = grid_world.value_iteration()
  # value_function_p = grid_world.policy_iteration()
  
  # Compute optimal policy over the grid
  policy = grid_world.policy_derivation()

  # Simulate robot trajectory
  

  # Plot results
  # grid_world.plot_heatmap(value_function_v, 'Value Function Heatmap (Value Iteration)')
  # grid_world.plot_heatmap(value_function_p, 'Value Function Heatmap (Policy Iteration)')
  grid_world.plot_heatmap(policy, 'Policy Heatmap')


if __name__ == "__main__":
  main()
