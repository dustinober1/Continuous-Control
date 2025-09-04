import os
import torch
import numpy as np
from collections import deque
from tqdm import tqdm
from datetime import datetime
from unityagents import UnityEnvironment
from ddpg_agent import Agent
from utils import save_checkpoint, load_checkpoint

def ddpg_train(env, agent, brain_name, n_episodes=300, max_t=1000,
                                print_every=10, save_every=5, checkpoint_dir='checkpoints',
                                resume_from=None):
    """
    Enhanced DDPG training with checkpoint saving and recovery.

    Args:
        save_every: Save checkpoint every N episodes
        resume_from: Path to checkpoint to resume from
    """
    env_info = env.reset(train_mode=True)[brain_name]
    num_agents = len(env_info.agents)

    scores_deque = deque(maxlen=100)
    scores_list = []
    best_score = 0.0
    start_episode = 1

    # Resume from checkpoint if specified
    if resume_from and os.path.exists(resume_from):
        print(f"📂 Resuming from checkpoint: {resume_from}")
        start_episode, scores_list, best_score = load_checkpoint(agent, resume_from)
        scores_deque.extend(scores_list[-100:])  # Restore the deque
        start_episode += 1  # Start from next episode
        print(f"   Resumed from episode {start_episode-1}, best score: {best_score:.2f}")

    # Create progress bar
    remaining_episodes = n_episodes - start_episode + 1
    episode_bar = tqdm(range(start_episode, n_episodes+1),
                       desc='Training', unit='episode',
                       total=remaining_episodes)

    # Track training metrics
    episode_times = []
    buffer_sizes = []

    for i_episode in episode_bar:
        start_time = datetime.now()
        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations
        agent.reset()
        scores = np.zeros(num_agents)

        for t in range(max_t):
            # Get actions for all agents
            actions = agent.act(states)

            # Step environment
            env_info = env.step(actions)[brain_name]
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done

            # Store experiences from all agents
            for i in range(num_agents):
                agent.step(states[i], actions[i], rewards[i], next_states[i], dones[i])

            states = next_states
            scores += rewards

            if np.any(dones):
                break

        # Record metrics
        episode_time = (datetime.now() - start_time).total_seconds()
        episode_times.append(episode_time)
        buffer_sizes.append(len(agent.memory))

        # Record scores
        avg_score = np.mean(scores)
        scores_deque.append(avg_score)
        scores_list.append(avg_score)

        # Update best score
        current_avg = np.mean(scores_deque)
        if current_avg > best_score:
            best_score = current_avg

        # Update progress bar
        episode_bar.set_postfix({
            'Score': f'{avg_score:.2f}',
            'Avg(100)': f'{current_avg:.2f}',
            'Best': f'{best_score:.2f}',
            'Time': f'{episode_time:.1f}s',
            'Buffer': f'{len(agent.memory)/1e6:.2f}M'
        })

        # Periodic printing
        if i_episode % print_every == 0:
            avg_time = np.mean(episode_times[-10:]) if len(episode_times) >= 10 else episode_time
            tqdm.write(f'Episode {i_episode} | Avg Score: {current_avg:.2f} | '
                      f'Episode Score: {avg_score:.2f} | Avg Time: {avg_time:.1f}s')

        # Save checkpoint
        if i_episode % save_every == 0:
            checkpoint_path = save_checkpoint(agent, scores_list, i_episode, best_score, checkpoint_dir)
            tqdm.write(f'💾 Checkpoint saved: {checkpoint_path}')

        # Check if solved
        if current_avg >= 30.0:
            tqdm.write(f'\n🎉 Environment solved in {i_episode-100} episodes! '
                      f'Average Score: {current_avg:.2f}')
            # Save final checkpoint
            save_checkpoint(agent, scores_list, i_episode, best_score, checkpoint_dir)
            # Save standalone model files
            torch.save(agent.actor_local.state_dict(), 'checkpoint_actor_solved.pth')
            torch.save(agent.critic_local.state_dict(), 'checkpoint_critic_solved.pth')
            episode_bar.close()
            break

    episode_bar.close()

    # Print training summary
    print("\n" + "="*60)
    print("📊 Training Summary:")
    print(f"   Total Episodes: {len(scores_list)}")
    print(f"   Best Average Score: {best_score:.2f}")
    print(f"   Final Average Score: {np.mean(scores_deque):.2f}")
    print(f"   Average Episode Time: {np.mean(episode_times):.1f}s")
    print(f"   Final Buffer Size: {len(agent.memory)/1e6:.2f}M")
    print("="*60)

    return scores_list

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Train a DDPG agent for the Reacher environment.')
    parser.add_argument('--env', type=str, required=True, help='Path to the Unity environment')
    parser.add_argument('--episodes', type=int, default=300, help='Number of episodes to train for')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    args = parser.parse_args()

    env = UnityEnvironment(file_name=args.env)
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    env_info = env.reset(train_mode=True)[brain_name]
    num_agents = len(env_info.agents)
    action_size = brain.vector_action_space_size
    states = env_info.vector_observations
    state_size = states.shape[1]

    agent = Agent(state_size=state_size, action_size=action_size, random_seed=42)

    ddpg_train(env, agent, brain_name, n_episodes=args.episodes, resume_from=args.resume)

    env.close()
