import os
import sys
import numpy as np
import torch

# Ensure project root is on sys.path so `src` modules can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.ddpg_agent import ReplayBuffer, Agent


def test_replay_add_and_len():
    buf = ReplayBuffer(action_size=2, buffer_size=100, batch_size=4, seed=0)
    assert len(buf) == 0
    for i in range(10):
        buf.add(np.zeros(3), np.zeros(2), 0.0, np.zeros(3), False)
    assert len(buf) == 10


def test_replay_sample_shape():
    buf = ReplayBuffer(action_size=2, buffer_size=100, batch_size=4, seed=0)
    for i in range(10):
        buf.add(np.ones(3)*i, np.ones(2)*i, float(i), np.ones(3)*(i+1), False)
    states, actions, rewards, next_states, dones = buf.sample()
    assert states.shape[1] == 3
    assert actions.shape[1] == 2


def test_soft_update_preserves_shapes():
    # create agent and ensure soft_update copies params with same shapes
    agent = Agent(state_size=3, action_size=2, random_seed=0)
    # run one soft update
    agent.soft_update(agent.actor_local, agent.actor_target, tau=0.1)
    for p_local, p_target in zip(agent.actor_local.parameters(), agent.actor_target.parameters()):
        assert p_local.data.shape == p_target.data.shape
