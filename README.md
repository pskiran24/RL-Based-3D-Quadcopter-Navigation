# Reinforcement Learning Based 3D Quadcopter Navigation and Obstacle Avoidance

A MATLAB-based reinforcement learning project exploring autonomous 3D quadcopter navigation, waypoint reaching, and obstacle avoidance.

## Project Overview

This project investigates reinforcement learning algorithms for training a main (navigator) drone to reach a target in a three-dimensional environment containing randomly positioned obstacles.

The research compares five algorithms:

- Q-Learning
- SARSA
- Double Q-Learning
- Monte Carlo
- Deep Q-Network (DQN)

The broader research framework also considers an attacker drone and competitive pursuit-evasion. The current repository focuses on the main-drone implementation; the attacker-drone and competitive-testing components will be added separately when available.

## Environment

The research report describes a simulated 3D environment with:

- Environment dimensions: 100 m × 100 m × 50 m
- Five randomly positioned cubic obstacles per episode
- Six discrete movement actions along the positive and negative X, Y, and Z axes
- Reward-based learning for target reaching and obstacle avoidance

## Repository Structure

```text
RL-Based-3D-Quadcopter-Navigation/
├── algorithms/
│   ├── Q-Learning/
│   ├── SARSA/
│   ├── Double-Q-Learning/
│   ├── Monte-Carlo/
│   └── DQN/
├── hyperparameter-tuning/
├── results/
├── models/
├── .gitignore
└── README.md
```

## Evaluation

The project report evaluates the algorithms using average episode reward and navigation success rate. It also reports collision, timeout, and competitive testing outcomes for the broader framework.

The report identifies DQN as the best-performing main-drone algorithm in its reported training results, with an average reward of 140.90 and a success rate of 94.3%.

These figures are reported results from the study and have not been independently reproduced as part of this repository documentation.

## Technologies

- MATLAB
- Reinforcement Learning
- Q-Learning and temporal-difference methods
- Deep Q-Networks
- 3D navigation and obstacle avoidance

## Contributors

- **Kiran P. S.** — Project contributor
- **Dinesh R.** — Collaborator; attacker-drone component maintained separately

GitHub: [Phy6toAI](https://github.com/Phy6toAI)

## Scope and Reproducibility

This repository currently contains the available main-drone MATLAB scripts. Additional dependencies, saved models, and result files will be documented as they are reviewed.

The complete research framework includes components that are not yet present in this repository. Results should therefore be interpreted in the context of the accompanying research report and the code currently available.

## License

No license has been selected yet. Reuse and redistribution permissions should be confirmed with all relevant contributors before a license is added.
