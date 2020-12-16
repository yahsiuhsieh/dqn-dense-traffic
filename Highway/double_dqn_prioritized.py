import os
import copy
import random
import time
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

import gym
import highway_env
import pybullet
import pybulletgym.envs
import pprint

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


# Define the environment
env = gym.make("highway-v0")
env.config["lanes_count"] = 4
env.config["duration"] = 100
env.config["vehicles_count"] = 10
env.config["vehicles_density"] = 1.3
env.config["policy_frequency"] = 2
env.config["simulation_frequency"] = 10
env.reset()


class Prioritized_Replay:
    def __init__(
        self,
        buffer_size,
        init_length,
        state_dim,
        action_dim,
        est_Net,
        tar_Net,
        gamma,
    ):
        """
        A function to initialize the replay buffer.

        : param init_length: int, initial number of transitions to collect
        : param state_dim: int, size of the state space
        : param action_dim: int, size of the action space
        : param env: gym environment object
        """
        self.buffer_size = buffer_size
        self.init_length = init_length
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma

        self.priority = deque(maxlen=buffer_size)
        self._storage = []
        self._init_buffer(init_length, est_Net, tar_Net)

    def _init_buffer(self, n, est_Net, tar_Net):
        """
        Init buffer with n samples with state-transitions taken from random actions

        : param n: int, number of samples
        """
        state = env.reset()
        for _ in range(n):
            action = env.action_space.sample()
            state_next, reward, done, _ = env.step(action)
            exp = {
                "state": state,
                "action": action,
                "reward": reward,
                "state_next": state_next,
                "done": done,
            }
            self.prioritize(est_Net, tar_Net, exp, alpha=0.6)
            self._storage.append(exp)
            state = state_next

            if done:
                state = env.reset()
                done = False

    def buffer_add(self, exp):
        """
        A function to add a dictionary to the buffer

        : param exp: a dictionary consisting of state, action, reward , next state and done flag
        """
        self._storage.append(exp)
        if len(self._storage) > self.buffer_size:
            self._storage.pop(0)

    def prioritize(self, est_Net, tar_Net, exp, alpha=0.6):
        state = torch.FloatTensor(exp["state"]).to(device).reshape(-1)

        q = est_Net(state)[exp["action"]].detach().cpu().numpy()
        q_next = exp["reward"] + self.gamma * torch.max(est_Net(state).detach())
        # TD error
        p = (np.abs(q_next.cpu().numpy() - q) + (np.e ** -10)) ** alpha
        self.priority.append(p.item())

    def get_prioritized_batch(self, N):
        prob = self.priority / np.sum(self.priority)
        sample_idxes = random.choices(range(len(prob)), k=N, weights=prob)
        importance = (1 / prob) * (1 / len(self.priority))
        sampled_importance = np.array(importance)[sample_idxes]
        sampled_batch = np.array(self._storage)[sample_idxes]
        return sampled_batch.tolist(), sampled_importance

    def buffer_sample(self, N):
        """
        A function to sample N points from the buffer

        : param N: int, number of samples to obtain from the buffer
        """
        return random.sample(self._storage, N)


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

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        out = self.fc3(x)
        return out


class DQNPB(nn.Module):
    def __init__(
        self,
        env,
        state_dim,
        action_dim,
        lr=0.001,
        gamma=0.99,
        buffer_size=1000,
        batch_size=50,
        beta=1,
        beta_decay=0.995,
        beta_min=0.01,
        timestamp="",
    ):
        """
        : param env: object, a gym environment
        : param state_dim: int, size of state space
        : param action_dim: int, size of action space
        : param lr: float, learning rate
        : param gamma: float, discount factor
        : param batch_size: int, batch size for training
        """
        super(DQNPB, self).__init__()

        self.timestamp = timestamp

        self.test_env = copy.deepcopy(env)  # for evaluation purpose
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.learn_step_counter = 0

        self.target_net = Net(self.state_dim, self.action_dim).to(device)
        self.estimate_net = Net(self.state_dim, self.action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.estimate_net.parameters(), lr=lr)

        self.ReplayBuffer = Prioritized_Replay(
            buffer_size,
            100,
            self.state_dim,
            self.action_dim,
            self.estimate_net,
            self.target_net,
            gamma,
        )
        self.priority = self.ReplayBuffer.priority
        # NOTE: right here beta is equal to (1-beta) in most of website articles, notation difference
        # start from 1 and decay
        self.beta = beta
        self.beta_decay = beta_decay
        self.beta_min = beta_min

    def choose_action(self, state, epsilon=0.9):
        """
        Select action using epsilon greedy method

        : param state: ndarray, the state of the environment
        : param epsilon: float, between 0 and 1
        : return: ndarray, chosen action
        """
        state = torch.FloatTensor(state).to(device).reshape(-1)  # get a 1D array
        # print(state.shape)
        if np.random.randn() <= epsilon:
            action_value = self.estimate_net(state)
            action = torch.argmax(action_value).item()
        else:
            action = np.random.randint(0, self.action_dim)
        return action

    def train(self, num_epochs):
        """
        Train the policy for the given number of iterations

        :param num_epochs: int, number of epochs to train the policy for
        """
        loss_list = []
        avg_reward_list = []
        epoch_reward = 0

        for epoch in tqdm(range(int(num_epochs))):
            done = False
            state = env.reset()

            avg_loss = 0
            step = 0
            while not done:
                step += 1
                action = self.choose_action(state)
                state_next, reward, done, _ = env.step(action)

                # store experience to replay memory
                exp = {
                    "state": state,
                    "action": action,
                    "reward": reward,
                    "state_next": state_next,
                    "done": done,
                }
                self.ReplayBuffer.buffer_add(exp)
                state = state_next

                # importance weighting
                if self.beta > self.beta_min:
                    self.beta *= self.beta_decay

                # sample random batch from replay memory
                exp_batch, importance = self.ReplayBuffer.get_prioritized_batch(
                    self.batch_size
                )
                importance = torch.FloatTensor(importance ** (1 - self.beta)).to(device)

                # extract batch data
                state_batch = torch.FloatTensor([exp["state"] for exp in exp_batch]).to(
                    device
                )
                action_batch = torch.LongTensor(
                    [exp["action"] for exp in exp_batch]
                ).to(device)
                reward_batch = torch.FloatTensor(
                    [exp["reward"] for exp in exp_batch]
                ).to(device)
                state_next_batch = torch.FloatTensor(
                    [exp["state_next"] for exp in exp_batch]
                ).to(device)
                done_batch = torch.FloatTensor(
                    [1 - exp["done"] for exp in exp_batch]
                ).to(device)

                # reshape
                state_batch = state_batch.reshape(self.batch_size, -1)
                action_batch = action_batch.reshape(self.batch_size, -1)
                reward_batch = reward_batch.reshape(self.batch_size, -1)
                state_next_batch = state_next_batch.reshape(self.batch_size, -1)
                done_batch = done_batch.reshape(self.batch_size, -1)

                # get estimate Q value
                estimate_Q = self.estimate_net(state_batch).gather(1, action_batch)

                # get target Q value
                max_action_idx = self.estimate_net(state_next_batch).detach().argmax(1)
                target_Q = reward_batch + done_batch * self.gamma * self.target_net(
                    state_next_batch
                ).gather(1, max_action_idx.unsqueeze(1))

                # compute mse loss
                # loss = F.mse_loss(estimate_Q, target_Q)
                loss = torch.mean(
                    torch.multiply(torch.square(estimate_Q - target_Q), importance)
                )
                avg_loss += loss.item()

                # update network
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # update target network
                if self.learn_step_counter % 10 == 0:
                    # self.update_target_networks()
                    self.target_net.load_state_dict(self.estimate_net.state_dict())

                self.learn_step_counter += 1

            reward, count = self.eval()
            epoch_reward += reward

            # save
            period = 40
            if epoch % period == 0:
                # log
                avg_loss /= step
                epoch_reward /= period
                avg_reward_list.append(epoch_reward)
                loss_list.append(avg_loss)

                print(
                    "\nepoch: [{}/{}], \tavg loss: {:.4f}, \tavg reward: {:.3f}, \tsteps: {}".format(
                        epoch + 1, num_epochs, avg_loss, epoch_reward, count
                    )
                )

                epoch_reward = 0
                # create a new directory for saving
                try:
                    os.makedirs(self.timestamp)
                except OSError:
                    pass
                np.save(self.timestamp + "/double_dqn_prioritized_loss.npy", loss_list)
                np.save(
                    self.timestamp + "/double_dqn_prioritized_avg_reward.npy",
                    avg_reward_list,
                )
                torch.save(
                    self.estimate_net.state_dict(),
                    self.timestamp + "/double_dqn_prioritized.pkl",
                )

        env.close()
        return loss_list, avg_reward_list

    def eval(self):
        """
        Evaluate the policy
        """
        count = 0
        total_reward = 0
        done = False
        state = self.test_env.reset()

        while not done:
            action = self.choose_action(state, epsilon=1)
            state_next, reward, done, _ = self.test_env.step(action)
            total_reward += reward
            count += 1
            state = state_next

        return total_reward, count


if __name__ == "__main__":

    # timestamp for saving
    named_tuple = time.localtime()  # get struct_time
    time_string = time.strftime("%m%d_%H_%M", named_tuple)

    dqn_prioritized_object = DQNPB(
        env,
        state_dim=25,
        action_dim=5,
        lr=0.001,
        gamma=0.99,
        buffer_size=1000,
        batch_size=64,
        timestamp=time_string,
    )

    # Train the policy
    iterations = 4000
    avg_loss, avg_reward_list = dqn_prioritized_object.train(iterations)
    np.save(time_string + "double_dqn_prioritized_loss.npy", avg_loss)
    np.save(time_string + "double_dqn_prioritized_average_reward.npy", avg_reward_list)

    # save the dqn network
    torch.save(
        dqn_prioritized_object.estimate_net.state_dict(),
        time_string + "double_dqn_prioritized.pkl",
    )

    # plot
    # plt.figure(figsize=(10, 6))
    # plt.plot(avg_loss)
    # plt.grid()
    # plt.title("Double DQN Loss")
    # plt.xlabel("epochs")
    # plt.ylabel("loss")
    # plt.savefig("double_dqn_loss.png", dpi=150)
    # plt.show()

    # plt.figure(figsize=(10, 6))
    # plt.plot(avg_reward_list)
    # plt.grid()
    # plt.title("Double DQN Training Reward")
    # plt.xlabel("*40 epochs")
    # plt.ylabel("reward")
    # plt.savefig(time_string + "/double_dqn_prioritized_train_reward.png", dpi=150)
    # plt.show()
