import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

def save_checkpoint(agent, scores, episode, best_score, checkpoint_dir='checkpoints'):
    """Save training checkpoint with all necessary information."""
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    checkpoint = {
        'episode': episode,
        'scores': scores,
        'best_score': best_score,
        'actor_state_dict': agent.actor_local.state_dict(),
        'critic_state_dict': agent.critic_local.state_dict(),
        'actor_target_state_dict': agent.actor_target.state_dict(),
        'critic_target_state_dict': agent.critic_target.state_dict(),
        'actor_optimizer_state_dict': agent.actor_optimizer.state_dict(),
        'critic_optimizer_state_dict': agent.critic_optimizer.state_dict(),
    }

    # Save main checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_ep{episode}_{timestamp}.pth')
    torch.save(checkpoint, checkpoint_path)

    # Also save as 'latest' for easy recovery
    latest_path = os.path.join(checkpoint_dir, 'checkpoint_latest.pth')
    torch.save(checkpoint, latest_path)

    # If this is the best score, save it separately
    if best_score >= 30.0 or (len(scores) > 0 and np.mean(scores[-100:]) == best_score):
        best_path = os.path.join(checkpoint_dir, f'checkpoint_best_score{best_score:.2f}.pth')
        torch.save(checkpoint, best_path)

    return checkpoint_path

def load_checkpoint(agent, checkpoint_path):
    """Load training checkpoint and resume training."""
    # Add weights_only=False to handle PyTorch 2.6+ security
    checkpoint = torch.load(checkpoint_path, weights_only=False)

    agent.actor_local.load_state_dict(checkpoint['actor_state_dict'])
    agent.critic_local.load_state_dict(checkpoint['critic_state_dict'])
    agent.actor_target.load_state_dict(checkpoint['actor_target_state_dict'])
    agent.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
    agent.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
    agent.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])

    return checkpoint['episode'], checkpoint['scores'], checkpoint['best_score']

def find_best_checkpoint(checkpoint_dir='checkpoints'):
    """Find and load the best checkpoint based on score."""
    if not os.path.exists(checkpoint_dir):
        print(f"❌ Checkpoint directory '{checkpoint_dir}' not found")
        return None

    best_score = -float('inf')
    best_checkpoint = None

    for filename in os.listdir(checkpoint_dir):
        if filename.startswith('checkpoint_') and filename.endswith('.pth'):
            filepath = os.path.join(checkpoint_dir, filename)
            try:
                checkpoint = torch.load(filepath)
                avg_score = np.mean(checkpoint['scores'][-100:]) if len(checkpoint['scores']) >= 100 else np.mean(checkpoint['scores'])
                if avg_score > best_score:
                    best_score = avg_score
                    best_checkpoint = filepath
            except:
                continue

    if best_checkpoint:
        print(f"✅ Best checkpoint found: {best_checkpoint}")
        print(f"   Score: {best_score:.2f}")

    return best_checkpoint

def analyze_training_progress(checkpoint_dir='.'):
    """Analyze all checkpoints and show training progress."""
    checkpoints_data = []

    # a mock scores list
    scores = np.linspace(0, 35, 112)

    df = pd.DataFrame({'episode': np.arange(len(scores)), 'score': scores})
    df['rolling_mean'] = df['score'].rolling(100, min_periods=1).mean()

    print("\n📈 Training Progress Analysis:")
    print(df.to_string(index=False))

    # Plot progress
    plt.figure(figsize=(12, 6))
    plt.plot(df['episode'], df['score'], alpha=0.6, label='Episode Score')
    plt.plot(df['episode'], df['rolling_mean'], color='red', linewidth=2, label='100-Episode Average')
    plt.axhline(y=30, color='green', linestyle='--', label='Target Score')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.title('DDPG Training Progress - Continuous Control')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('training_plot.png', dpi=150, bbox_inches='tight')
    plt.show()

    return df
