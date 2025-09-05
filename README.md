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

# Continuous Control with Deep Reinforcement Learning

This repository implements a DDPG (Deep Deterministic Policy Gradient) agent for the Unity ML-Agents Reacher environment and includes tools to train, checkpoint, visualize, and demo results.

The project contains training scripts, a reusable agent implementation (`src/ddpg_agent.py`), utilities for checkpointing and plotting (`src/utils.py`), and a `generate_plot.py` helper that creates publication-ready plots and an animated GIF of training progress.

## Highlights
- Training with checkpointing and resume capability
- Animated training-curve GIF generation (`checkpoints/demos/training_progress.gif`)
- Headless-friendly plotting (matplotlib Agg backend)
- Minimal scripts to reproduce training & demos

## Quickstart

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Download and place the Unity environment binary in the project folder (use the Reacher environment from Udacity/Unity). Recommended: Version 2 (20 agents).

3. Run a quick demo that generates an animated training GIF (uses existing checkpoints if present; otherwise uses synthetic example data):

```bash
python3 scripts/generate_plot.py --checkpoints checkpoints --out checkpoints/demos --fps 8
```

Output:
- `checkpoints/demos/training_progress.gif` — animated training curve
- `training_plot.png` — static training plot

## Docker & smoke tests

You can build and run a reproducible container that generates the demo GIF and runs the lightweight smoke test included in the repo.

Build and generate demo artifacts (preferred, runs the `scripts/generate_plot.py` entrypoint):

```bash
./scripts/run.sh
```

Or run the Docker commands manually:

```bash
docker build -t continuous-control:latest .
docker run --rm -v $(pwd)/checkpoints/demos:/app/checkpoints/demos continuous-control:latest --checkpoints checkpoints --out checkpoints/demos --fps 6
```

Artifacts will be available in `checkpoints/demos/` after the container finishes.

Run the test suite inside the container (quick smoke & verification):

```bash
docker run --rm continuous-control:latest pytest -q
```

You can also run tests locally with:

```bash
pytest -q
```

## Running Training

Train from the command line (example):

```bash
python3 src/train.py --env /path/to/Reacher_Linux_x86_64/Reacher.x86_64 --episodes 300
```

Key `train.py` features:
- Saves checkpoints to `checkpoints/` every few episodes
- Can resume from a checkpoint using `--resume <path>`
- Prints progress with a tqdm bar and maintains best-score tracking

Notes: `src/train.py` expects the Unity environment file path via `--env`.

## Reproducing Results

- To reproduce a run, set a fixed seed (the agent uses `random_seed`), keep environment deterministic where possible, and pass `--resume` with a saved checkpoint to continue training.
- Example: after a successful run you can replay the best checkpoint and generate visuals:

```bash
python3 scripts/generate_plot.py --checkpoints checkpoints --out checkpoints/demos
```

Or programmatically:

```python
from src.utils import find_best_checkpoint
best = find_best_checkpoint('checkpoints')
print('Best checkpoint:', best)
```

## Demo & Visualizations

- Use `generate_plot.py` to create an animated GIF of the training curve and a static `training_plot.png` for reports.
- During training you can capture environment frames (if rendering is available) and create videos/GIFs; the utilities are written to be compatible with headless CI by using Matplotlib's Agg backend.

## Checkpoints & Model Files

- Checkpoints are saved in `checkpoints/` and include actor/critic weights, optimizer states, episode number and score history.
- Use `src/utils.find_best_checkpoint()` to locate the best checkpoint automatically.

## Project Layout

```
Continuous-Control/
├── Continuous_Control.ipynb     # Notebook with examples and walkthrough
├── scripts/                     # CLI scripts (generate_plot, run)
│   ├── generate_plot.py
│   └── run.sh
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── checkpoints/                 # Saved model checkpoints + demos
│   └── demos/                   # generated GIFs and sample media
└── src/
   ├── ddpg_agent.py            # Agent, networks, replay buffer
   ├── train.py                 # Training entry point (CLI)
   └── utils.py                 # Checkpointing, plotting, GIF helper
└── notebooks/
   └── demo.ipynb               # Demo notebook that embeds GIF and shows checkpoint save/load
```

## Dependencies

Core dependencies are declared in `requirements.txt`. Notable packages:
- torch (PyTorch)
- numpy, pandas
- matplotlib, imageio, Pillow (for plotting and GIFs)
- unityagents (for the Unity environment bridge)

## Tips for reviewers / CI

- The visualization tools work in headless CI because `src/utils.py` sets Matplotlib to use the `Agg` backend.
- If CI cannot run Unity binaries, `generate_plot.py` will still produce the animated GIF using synthetic/example scores.

## Contributing

Contributions are welcome. Suggested small improvements:
- Add a short `demo.ipynb` that loads a best checkpoint and embeds the GIF
- Add a GitHub Actions workflow to run a quick smoke test and produce demo artifacts
- Add unit tests for utility functions in `src/utils.py`

## License

This project is released under the MIT License — see `LICENSE`.

## Acknowledgments

- Udacity Deep Reinforcement Learning Nanodegree
- Unity ML-Agents
- Lillicrap et al., DDPG (2015)
