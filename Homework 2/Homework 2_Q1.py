import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt


def main():
  # Constants 
  alpha = np.array([1.2, 0.9, 1.5, 0.6, 0.8])

  # Optimization Variables
  n = len(alpha)
  x = cp.Variable(n)

  # Objective Function
  objective = cp.Maximize(cp.sum(cp.log(alpha + x)))
  # objective = cp.Minimize(cp.sum(-cp.log(alpha + x)))    # same result

  # Constraints
  constraints = [x >= 0,
                 cp.sum(x) == 1]
  # constraints = [-x <= 0,
  #                cp.sum(x) - 1 == 0]    # same result

  # Define and Solve Problem
  problem = cp.Problem(objective, constraints)
  problem.solve()

  # Results
  x_star = x.value    # optimal allocated water
  lambda_star = problem.constraints[1].dual_value   # lagrange multiplier
  print("Optimal x: ", x_star)
  print("Optimal lambda: ", lambda_star)

  # Create plots
  plt.bar(np.arange(n), alpha, label='alpha (initial channel condition)', color='blue', alpha=0.7)
  plt.bar(np.arange(n), x_star, bottom=alpha, label='x* (allocated water)', color='orange', alpha=0.7)
  plt.axhline(y=lambda_star, color='r', linestyle='--', label='lambda* (optimal water level)')
  plt.xlabel('Alpha (channels)')
  plt.ylabel('Water Allocation')
  plt.legend()
  plt.title('Water-Filling Interpretation')
  plt.show()

  return


if __name__ == "__main__":
  main()
