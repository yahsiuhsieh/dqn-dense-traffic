import gym
import pybulletgym
import highway_env
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# configure env
env = gym.make("highway-v0")
env.config["lanes_count"] = 4
env.config["duration"] = 100
env.config["vehicles_count"] = 10
env.config["vehicles_density"] = 1.3
env.config["policy_frequency"] = 2
env.config["simulation_frequency"] = 10
env.reset() # to update configuration

class Net(nn.Module):
    def __init__(self, state_dim, action_dim):
        """
        Initialize the network

        : param state_dim: int, size of state space
        : param action_dim: int, size of action space
        """
        super(Net, self).__init__()

        hidden_nodes1 = 512
        hidden_nodes2 = 256
        self.fc1 = nn.Linear(state_dim, hidden_nodes1)
        self.fc2 = nn.Linear(hidden_nodes1, hidden_nodes2)
        self.fc3 = nn.Linear(hidden_nodes2, action_dim)

    def forward(self, state):
        """
        Define the forward pass of the actor

        : param state: ndarray, the state of the environment
        """
        x = state
        # print(x.shape)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        out = self.fc3(x)
        return out

def choose_action(net, state):
    """
    Select action using epsilon greedy method

    : param state: ndarray, the state of the environment
    : param epsilon: float, between 0 and 1
    : return: ndarray, chosen action
    """
    state = torch.FloatTensor(state).reshape(-1)  # get a 1D array
    action_value = net(state)
    action = torch.argmax(action_value).item()
    return action


def plot(avg_reward, title):
    """
    Plots for average reward over every iteration

    :param avg_reward: list, a list of average reward
    :param title: str, plot title
    """
    plt.figure()
    plt.plot(avg_reward)
    plt.title(title)
    plt.xlabel("Iterations")
    plt.ylabel("Average Reward")
    plt.savefig(title + ".png", dpi=150)
    plt.show()


if __name__ == "__main__":

    # load the model
    net = Net(25, 5) # 5 observed vehicles
    net.load_state_dict(torch.load("double_dqn_pr.pkl"))

    success = 0
    num_epochs = 100
    for epoch in tqdm(range(num_epochs)):
        state = env.reset()
        done = False
        step = 0
        while not done:
            step += 1
            if step == env.config["duration"]:
                #print("success")
                success += 1
            action = choose_action(net, state)
            state, reward, done, _ = env.step(action)
            env.render()
    print("success rate: {}%".format(success))

    env.close()
