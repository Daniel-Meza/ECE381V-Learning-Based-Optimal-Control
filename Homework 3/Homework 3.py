import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


class GridWorld:
  def __init__(self, n: int, s_goal: tuple[int, int], gamma: float = 0.99, epsilon: float = 1e-4):
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


  def value_iteration(self, max_iters):
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
          best_expected_value = -np.inf
          for a in self.actions:
            # Compute the expected value of taking action a
            expected_value = 0

            # Calculate expected value
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
              # Assumed 0 everywhere except for the goal. This is accounted for already in initial value function.
              reward = 0

              # Update the expected value
              expected_value += prob * (reward + self.gamma * self.V[x_prime, y_prime])
            
            # Track the maximum expected value
            best_expected_value = max(best_expected_value, expected_value)

          # Update the value function at (x, y) and track change in values for convergence
          # TODO delta not converging to 0, running all max_iters every time
          # delta += best_expected_value - V[x, y]
          delta = max(delta, abs(best_expected_value - self.V[x, y]))
          self.V[x, y] = expected_value

      # Check for convergence
      if delta < self.epsilon:
        print("[Value Iteration] Converged in %d iterations with a difference of %f" % (i, delta))
        break



  def policy_improvement(self) -> bool:
    """
    TODO
    """
    policy_stable = True

    # Iterate through grid updating the policy at each cell
    for x in range(self.n):
      for y in range(self.n):
        # Skip the goal state
        if (x, y) == self.s_goal:
          continue

        # Evaluate all possible to actions to find the best one
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
        old_action = self.pi[x, y]
        self.pi[x, y] = self.policy_map[best_action]

        # Check if the policy changed
        if old_action != best_action:
          policy_stable = False

    return policy_stable


  def value_iteration_method(self, max_iters = 1000) -> tuple[np.ndarray, np.ndarray]:
    """
    TODO
    """
    # Initialize Value Function (zeros)
    self.V = np.zeros((self.n, self.n))
    self.V[self.s_goal] = 1

    # Initialize Policy ('up')
    self.pi = np.zeros((self.n, self.n))

    # Calculate optimal value function and respective policy
    self.value_iteration(max_iters)
    _ = self.policy_improvement()

    return self.V, self.pi


  def policy_evaluation(self, max_iters):
    """
    TODO This is extremely similar to value_iteration, make methods?
    """
    for i in range(max_iters):
      delta = 0   # variable for convergence check

      # Iterate through grid updating the reward value at each cell
      for x in range(self.n):
        for y in range(self.n):
          # Skip the goal state
          if (x, y) == self.s_goal:
            continue

          # Get the action according to the current policy
          a = list(self.policy_map.keys())[int(self.pi[x, y])]

          # Compute the expected value of taking action a
          expected_value = 0
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
            # Assumed 0 everywhere except for the goal. This is accounted for already in initial value function.
            reward = 0

            # Update the expected value
            expected_value += prob * (reward + self.gamma * self.V[x_prime, y_prime])

          # Update the value function at (x, y) and track change in values for convergence
          # TODO Same convergence problem as in value iteration?
          delta = max(delta, abs(expected_value - self.V[x, y]))
          self.V[x, y] = expected_value
      
      # Check for convergence
      if delta < self.epsilon:
        break


  def policy_iteration_method(self, max_iters = 1000) -> tuple[np.ndarray, np.ndarray]:
    """
    TODO This method is not working!
    """
    # Initialize Value Function (zeros)
    self.V = np.zeros((self.n, self.n))
    self.V[self.s_goal] = 1

    # Initialize Policy ('up')
    self.pi = np.zeros((self.n, self.n))

    # Calculate optimal value function and respective stable policy
    for i in range(max_iters):
      self.policy_evaluation(max_iters)
      policy_stable = self.policy_improvement()

      if policy_stable:
        print("[Policy Iteration] Converged in %d iterations" % (i))
        break

    return self.V, self.pi
  

  def simulate_trajectory(self, s_initial: tuple[int, int], max_steps = 100) -> np.ndarray:
    """
    TODO
    """
    # Initialize grid map for the robot path
    path = np.zeros((self.n, self.n))
    x, y = s_initial
    path[x, y] = 1

    # Follow the policy until the goal is reached
    steps = 0
    while steps < max_steps:
      # Get the action to take from the policy map
      index = int(self.pi[x, y])
      a = list(self.policy_map.keys())[index]

      # Calculate the next state (x', y') and check for out-of-bounds
      x_prime = x + self.action_map[a][0]
      y_prime = y + self.action_map[a][1]
      
      if self.is_out_of_bounds((x_prime, y_prime)):
        x_prime, y_prime = x, y

      # Move to the next state
      x, y = x_prime, y_prime
      path[x, y] = 1

      # Stop if the goal is reached
      if (x, y) == self.s_goal:
        print("[Trajectory Simulation] Reached goal in %d steps", (steps))
        break

    # Goal not reached
    if steps >= max_steps:
      print("[Trajectory Simulation] Stopped simulation after %d steps", (steps))
    
    return path


  def plot_heatmap(self, map: np.ndarray, title = ''):
    """
    TODO
    """
    plt.figure()
    heatmap = plt.imshow(map.T, origin='lower')   # transpose to align axes
    plt.colorbar(heatmap)
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

  # Value iteration method
  value_function_v, policy_v = grid_world.value_iteration_method()

  # Policy iteration method
  # value_function_p, policy_p = grid_world.policy_iteration_method()

  # Simulate robot trajectory
  path = grid_world.simulate_trajectory((0, 0))

  # Plot results
  grid_world.plot_heatmap(value_function_v, 'Value Function Heatmap (Value Iteration)')
  grid_world.plot_heatmap(policy_v, 'Policy Heatmap (Value Iteration)')
  grid_world.plot_heatmap(path, 'Robot Trajectory (Value Iteration)')
  # grid_world.plot_heatmap(value_function_p, 'Value Function Heatmap (Policy Iteration)')
  # grid_world.plot_heatmap(policy_p, 'Policy Heatmap (Policy Iteration)')
  # grid_world.plot_heatmap(path, 'Robot Trajectory (Policy Iteration)')
  


if __name__ == "__main__":
  main()
