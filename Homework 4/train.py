# Main file to simulate the Mountain Car environment and run Q Learning
# To run this file please first install the requirements.txt file
# pip install -r requirements.txt
# After implementing your method run this file to generate the plots and
# Video of the agent in the environment

import gym
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns

from swarm_visualizer.utility.general_utils import set_plot_properties, save_fig
from swarm_visualizer.lineplot import plot_overlaid_ts

from q_learning import QLearningAgent


# Creating Mountain Car Environment
env = gym.make("MountainCar-v0")

# Number of grid points for each state dimension
N = 40

# Creating Q Learning Agent
q_learning_agent = QLearningAgent(env, N)


# Training parameters for Q Learning

alpha = 0.1  # learning rate
gamma = 0.99  # discount factor
max_epsilon = 0.3  # epsilon-greedy parameter
min_epsilon = -0.05  # epsilon-greedy parameter
num_episodes = 40000  # number of episodes
log_every = 200  # log every log_every episodes


# Training Q Learning Agent
q_learning_agent.train(
    num_episodes, min_epsilon, max_epsilon, env, alpha, gamma, log_every
)

env.close()

# Getting the Q Table, Policy and Value Function from the Q Learning Agent
action_map = {0: "L", 1: "N", 2: "R"}

q_table = q_learning_agent.get_q_table()
policy = q_learning_agent.get_policy()
value_function = q_learning_agent.get_value_function()

# Setting the plot properties before any plotting function
set_plot_properties(legend_font_size=20, xtick_label_size=20, ytick_label_size=20)

# Plotting the Q Table

# Map policy to action
fig = plt.figure(figsize=(12, 9))
cmap = sns.color_palette("deep", 3)
ax = sns.heatmap(policy, cmap=cmap)
ax.invert_yaxis()
ax.set_aspect("equal")
ax.set_xlabel("Position", fontsize=20)
ax.set_ylabel("Velocity", fontsize=20)


# Get the colorbar object from the Seaborn heatmap
colorbar = ax.collections[0].colorbar

# The list comprehension calculates the positions to place the labels to be evenly distributed across the colorbar
r = colorbar.vmax - colorbar.vmin
colorbar.set_ticks([colorbar.vmin + 0.5 * r / (3) + r * i / (3) for i in range(3)])
colorbar.set_ticklabels(list(action_map.values()))


current_loc = os.path.dirname(os.path.realpath(__file__))

save_fig(fig, os.path.join(current_loc, "mc_policy.png"))

# Plotting the Value Function

fig = plt.figure(figsize=(12, 9))
ax = sns.heatmap(value_function, cmap="viridis")
ax.invert_yaxis()
ax.set_aspect("equal")
ax.set_xlabel("Position", fontsize=20)
ax.set_ylabel("Velocity", fontsize=20)

current_loc = os.path.dirname(os.path.realpath(__file__))
save_fig(fig, os.path.join(current_loc, "mc_value_function.png"))

# Plotting Average, Min and Max Rewards per Episode

fig, ax = plt.subplots(figsize=(12, 9))

episodes = np.arange(log_every, num_episodes, log_every)

df = {
    "Average Reward": {
        "xvec": episodes,
        "ts_vector": q_learning_agent.aggr_ep_rewards["Avg"],
        "lw": 3,
        "color": "tab:green",
        "linestyle": "-",
    },
    "Max Reward": {
        "xvec": episodes,
        "ts_vector": q_learning_agent.aggr_ep_rewards["Max"],
        "lw": 3,
        "color": "tab:blue",
        "linestyle": "-.",
    },
    "Min Reward": {
        "xvec": episodes,
        "ts_vector": q_learning_agent.aggr_ep_rewards["Min"],
        "lw": 3,
        "color": "tab:orange",
        "linestyle": "--",
    },
}

plot_overlaid_ts(
    normalized_ts_dict=df,
    ylabel="Reward",
    xlabel="Episode",
    ax=ax,
)

current_loc = os.path.dirname(os.path.realpath(__file__))
save_fig(fig, os.path.join(current_loc, "mc_rewards.png"))

# Visualizing a final run of the agent

env = gym.make("MountainCar-v0", render_mode="rgb_array")

q_learning_agent.visualize(env, os.path.join(current_loc, "mc_anim.mp4"))
