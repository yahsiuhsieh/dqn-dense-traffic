# Decision Making in Dense Traffic using DQN

Deep reinforcement learning techniques apply to autonomous driving problems.

<p align="center">
  <img width="640" height="160" src="https://github.com/arthur960304/dqn-dense-traffic/blob/main/doc/highway.gif">
</p>

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Simulator

[highway-env](https://github.com/eleurent/highway-env)

### Built With

* Python 3.6.10

* PyTorch >= 1.7.0

* gym 0.17.3

* numpy >= 1.16.2

* matplotlib >= 3.1.1

## Code Organization

```
.
├── Highway                       # Scripts for highway environment
│   ├── dueling_dqn.py            # Dueling DQN
│   ├── double_dqn.py             # Double DQN
│   ├── double_dqn_cnn.py         # Double DQN with CNN architecture
│   └── double_dqn_prioritized.py # Double DQN with Prioritized Buffer
├── Intersection                  # Scripts for intersection environment
│   ├── dueling_dqn.py            # Dueling DQN
│   ├── double_dqn.py             # Double DQN
│   ├── double_dqn_cnn.py         # Double DQN with CNN architecture
│   └── double_dqn_prioritized.py # Double DQN with Prioritized Buffer
├── doc                           # Detailed info
└── README.md
```

## How to Run


## Results

Please refer to the [notebook](https://github.com/arthur960304/european_soccer/blob/main/european_soccer.ipynb) to see the result.
