import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt


def main():
  def network_utility(c):
    # Constants
    alpha = np.array([1, 2, 3, 4, 5])
    beta = np.array([2, 5, 1, 3, 4])

    # Optimization Variables
    n = len(alpha)
    x = cp.Variable(n)

    # Objective Function
    objective = cp.Maximize(cp.sum(cp.multiply(alpha, cp.log(beta + x))))

    # Constraints
    constraints = [x >= 0,
                  cp.sum(x) == c]

    # Define and Solve Problem
    problem = cp.Problem(objective, constraints)
    problem.solve()

    # Results
    x_star = x.value    # optimal allocated traffic rates
    marginal_utilities = alpha / (beta + x_star)  # derivative of the utility function
    mu_star = constraints[0].dual_value # lagrange multipliers for inequality constraints
    lambda_star = constraints[1].dual_value  # lagrange multiplier for equality constraint

    return problem.value, x_star, marginal_utilities, lambda_star
  
  # Solve for c = 10
  obj_value_c10, x_star_c10, marginal_utilities_c10, lambda_star_c10 = network_utility(10)
  # print("Optimal x: ", x_star_c10)
  # print("Marginal utilities: ", marginal_utilities_c10)
  # print("Optimal lambda: ", lambda_star_c10)

  # Solve for c = 11
  obj_value_c11, x_star_c11, marginal_utilities_c11, lambda_star_c11 = network_utility(11)

  # Solve for c = 100
  obj_value_c100, x_star_c100, marginal_utilities_c100, lambda_star_c100 = network_utility(100)
  
  # Solve for c = 101
  obj_value_c101, x_star_c101, marginal_utilities_c101, lambda_star_c101 = network_utility(101)

  # Sensitivity Analysis
  sensitivity_c10 = obj_value_c11 - obj_value_c10
  sensitivity_c100 = obj_value_c101 - obj_value_c100

  # Results
  print("Optimal objective value for c = 10: ", obj_value_c10)
  print("Optimal objective value for c = 11: ", obj_value_c11)
  print("Sensitivity at c = 10: ", sensitivity_c10)

  print("Optimal objective value for c = 100: ", obj_value_c100)
  print("Optimal objective value for c = 101: ", obj_value_c101)
  print("Sensitivity at c = 100: ", sensitivity_c100)

  return


if __name__ == "__main__":
  main()
