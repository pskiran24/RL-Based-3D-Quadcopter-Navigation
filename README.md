# Reinforcement Learning-Based 3D Quadcopter Navigation and Obstacle Avoidance

**A MATLAB research project exploring autonomous drone navigation using reinforcement learning in a three-dimensional environment with randomly positioned obstacles.**

This project compares five reinforcement learning algorithms for training a main (navigator) drone to reach a target while avoiding collisions. The broader research framework also investigates attacker–defender interactions and competitive pursuit–evasion.

**Technologies:** MATLAB · Reinforcement Learning · Q-Learning · SARSA · Double Q-Learning · Monte Carlo · Deep Q-Network (DQN)

---

## Project Overview

Autonomous drone navigation requires agents to make sequential decisions in environments containing obstacles, spatial constraints, and changing conditions. This project investigates how reinforcement learning algorithms learn navigation policies through trial-and-error interaction with a simulated 3D environment.

The main drone learns to move toward a predefined target while avoiding obstacles and boundary violations. Five algorithms are compared under the study's experimental framework, with hyperparameter tuning used to investigate their learning performance.

### Project Highlights

- 3D quadcopter navigation and waypoint reaching
- Randomized obstacle configurations
- Comparison of five reinforcement learning algorithms
- Hyperparameter optimization and reward-based learning
- Evaluation using average reward and navigation success rate
- Research framework designed to extend to competitive multi-agent pursuit–evasion

## Project Workflow

![Reinforcement Learning Workflow](results/rl_workflow.png)


## Simulation Environment

The research report describes a three-dimensional arena with the following configuration:

| Parameter | Configuration |
|---|---|
| Environment dimensions | 100 m × 100 m × 50 m |
| Obstacles | Five randomly positioned cubic obstacles per episode |
| Obstacle edge length | 10–15 m |
| Navigator movement | 2 m per step |
| Discrete action space | ±X, ±Y, ±Z |
| Maximum episode length | 200 steps |
| Evaluation | 100 independent test episodes, as described in the report |

Random obstacle placement is intended to evaluate whether learned navigation policies can generalize beyond a single memorized route.

## Methodology

### 1. Environment and state representation

The main drone interacts with a simulated 3D environment. Its state representation includes its spatial coordinates, distance to the target, distance to the attacker where applicable, and proximity to the nearest obstacle.

The agent selects one of six discrete actions representing movement along the positive or negative X, Y, and Z axes. The environment returns a reward and transitions to the next state.

### 2. Reward formulation

The research report describes rewards and penalties designed to encourage goal-directed navigation:

- Positive reward for reaching the target
- Penalties for obstacle collisions and boundary violations
- Positive intermediate rewards for moving closer to the target
- Negative intermediate rewards for moving away from the target

The broader research framework uses different reward formulations for solo navigation, attacker training, and competitive multi-agent learning.

### 3. Reinforcement learning algorithms

| Algorithm | Learning approach |
|---|---|
| Q-Learning | Off-policy temporal-difference learning using action-value estimates |
| SARSA | On-policy temporal-difference learning |
| Double Q-Learning | Uses two action-value estimates to reduce overestimation bias |
| Monte Carlo | Estimates returns from completed episodes |
| Deep Q-Network (DQN) | Uses a neural network to approximate action values |

### 4. Hyperparameter tuning

The study investigates hyperparameters such as learning rate, discount factor, exploration rate, episode count, and DQN-specific replay and batch settings.

The selected configurations reported in the study are used to compare learning performance across algorithms.

### 5. Evaluation

The research report describes evaluation using:

- **Average reward:** Mean cumulative reward per episode.
- **Success rate:** Percentage of episodes in which the main drone reaches the target.
- **Collision and timeout outcomes:** Used to assess navigation failures and incomplete episodes.
- **Competitive outcomes:** Win, loss, and timeout counts for the broader attacker–defender framework.

## Results

The following figures are reported in the research paper for main-drone training.

| Algorithm | Average reward | Success rate |
|---|---:|---:|
| Q-Learning | 98.89 | 81.1% |
| SARSA | 130.26 | 87.3% |
| Double Q-Learning | 79.90 | 82.3% |
| Monte Carlo | 79.88 | 65.7% |
| **DQN** | **140.90** | **94.3%** |

*Source: research report, Table II. These are reported experimental results, not independently reproduced results from this repository.*

### Key observations

- **DQN** achieved the highest reported average reward and success rate among the five algorithms.
- **SARSA** achieved the second-highest reported average reward and success rate.
- **Q-Learning** achieved an 81.1% reported success rate.
- **Double Q-Learning and Monte Carlo** had lower reported average rewards than DQN and SARSA in the study's selected configurations.

These results describe the configurations reported in the study; they should not be interpreted as a universal ranking of the algorithms.

## Results and Visualizations

Selected experiment images are available in the [`results/`](results/) directory.

The available visualizations cover:

- SARSA
- Q-Learning
- Deep Q-Network (DQN)

The visualizations complement the quantitative results by providing evidence from the available experiment outputs.

## Video Demonstration

A video demonstration can help illustrate the simulated drone's navigation behavior.

**Planned video:** Main-drone navigation and obstacle avoidance.

Once the MP4 is uploaded to the repository, link it here using its actual filename. For example, if the file is uploaded as `results/main_drone_demo.mp4`, use:

[Watch the main-drone simulation](results/main_drone_demo.mp4)

GitHub's rendering behavior for repository-hosted videos can vary by format and location, so a direct link is a reliable fallback.

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
│   ├── README.md
│   └── rl_workflow.png
├── models/
├── .gitignore
└── README.md
```

The repository currently focuses on the available main-drone MATLAB scripts. Saved models and additional experiment artifacts are being reviewed separately.

## Technologies and Concepts

- MATLAB
- Reinforcement learning
- Markov Decision Processes (MDPs)
- Temporal-difference learning
- Deep Q-Networks
- Hyperparameter optimization
- 3D waypoint navigation
- Obstacle avoidance
- Multi-agent pursuit–evasion research

## Contributors

- **Kiran P. S.** — Project contributor
- **Dinesh R.** — Collaborator; attacker-drone component maintained separately

Collaborator profile: [Phy6toAI](https://github.com/Phy6toAI)

## Scope and Reproducibility

This repository currently focuses on the available main-drone MATLAB code. The attacker-drone and competitive-testing components from the broader study will be integrated when the corresponding files are available and approved for publication.

The research report documents the experimental setup and results. Complete reproduction may require the original MATLAB environment, supporting functions, saved model artifacts, and any additional files referenced by the scripts.

## Research Context

This project is based on the research work titled *Reinforcement Learning Based 3D Quadcopter Waypoint Navigation and Obstacle Avoidance*.

The findings reflect the experimental configurations described in the report. Real-world drone deployment and physical flight validation are outside the scope of the currently documented repository.

## License

No license has been selected. Redistribution and reuse permissions should be agreed upon by the relevant contributors before a license is added.
