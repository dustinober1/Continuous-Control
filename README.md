# Continuous Control with Deep Reinforcement Learning

This repository contains a Deep Deterministic Policy Gradient (DDPG) implementation that successfully solves the Unity ML-Agents Reacher environment, achieving an average score of 30.11 over 100 consecutive episodes.

## Project Details

### Environment Description

In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of the agent is to maintain its position at the target location for as many time steps as possible.

### State Space

The observation space consists of **33 variables** corresponding to:
- Position, rotation, velocity, and angular velocities of the arm
- Each observation is a continuous vector of 33 dimensions

### Action Space

Each action is a vector with **four continuous values**, corresponding to torque applicable to two joints. Every entry in the action vector is a number between -1 and 1.

### Solving Criteria

The environment is considered solved when the agent achieves an **average score of +30 over 100 consecutive episodes**.

### Environment Versions

This implementation supports both versions:
- **Version 1**: Single agent
- **Version 2**: 20 identical agents (used in this solution)

## Getting Started

### Prerequisites

- Python 3.6 or higher
- PyTorch 2.0 or higher (CUDA support recommended)
- Unity ML-Agents
- NumPy
- Matplotlib
- tqdm

### Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Download the Unity Environment:

   **Version 2 (20 agents) - Recommended:**
   - Linux: [Download](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
   - Mac OSX: [Download](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
   - Windows (64-bit): [Download](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)

3. Unzip the environment and place it in the project directory.

### File Structure

```
continuous-control/
│
├── Continuous_Control.ipynb     # Main training notebook
├── ddpg_agent.py                # DDPG agent implementation
├── README.md                    # This file
├── Report.md                    # Technical report
├── requirements.txt             # Python dependencies
├── checkpoints/                 # Saved model checkpoints
│   ├── checkpoint_actor_solved.pth
│   ├── checkpoint_critic_solved.pth
│   └── checkpoint_best_score30.11.pth
└── training_plot.png           # Training progress visualization
```

## Instructions

### Training the Agent

1. Open the Jupyter notebook:
```bash
jupyter notebook Continuous_Control.ipynb
```

2. Execute cells in order to:
   - Initialize the Unity environment
   - Load the DDPG agent
   - Train the agent (with automatic checkpointing)

3. The training includes:
   - Automatic checkpoint saving every 5 episodes
   - Resume capability from previous checkpoints
   - Early stopping when target score is reached

### Using Pre-trained Model

To load and test the solved model:

```python
from ddpg_agent import Agent
import torch

# Create agent
agent = Agent(state_size=33, action_size=4, random_seed=42)

# Load the trained weights
agent.actor_local.load_state_dict(torch.load('checkpoint_actor_solved.pth'))
agent.critic_local.load_state_dict(torch.load('checkpoint_critic_solved.pth'))

# Agent is ready for testing
```

### Resuming Training

The implementation supports resuming from checkpoints:

```python
best_checkpoint = 'checkpoints/checkpoint_best_score30.11.pth'
scores = ddpg_train_with_checkpoints(
    env, agent, brain_name,
    resume_from=best_checkpoint
)
```

## Results

- **Environment solved in 112 episodes**
- **Final average score: 30.11** (over 100 episodes)
- **Training time**: Approximately 2.5 hours on CPU

The agent successfully learned to control the robotic arm, maintaining contact with the moving target for extended periods.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Udacity Deep Reinforcement Learning Nanodegree
- Unity ML-Agents Team
- Original DDPG paper: [Lillicrap et al., 2015](https://arxiv.org/abs/1509.02971)
