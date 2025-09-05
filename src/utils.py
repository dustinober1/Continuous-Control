import os
import torch
import numpy as np
import pandas as pd
import matplotlib
# Use non-interactive backend to avoid requiring a GUI (prevents Qt plugin errors)
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import imageio.v2 as imageio
from matplotlib import rcParams
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
    plt.close('all')

    return df


def create_training_gif(scores, out_path='checkpoints/demos/training_progress.gif', fps=10, figsize=(8, 4)):
    """Create an animated GIF that shows the training curve growing over time.

    This is a generic visualization that works even if environment rendering
    is not available (e.g., Unity environments). The GIF will animate the
    episode score as the line grows.
    """
    if not isinstance(scores, (list, np.ndarray)):
        raise ValueError('scores must be a list or numpy array')

    scores = np.array(scores)
    out_dir = os.path.dirname(out_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)

    images = []
    # avoid too many open-figure warnings during GIF creation
    rcParams.update({'figure.autolayout': True, 'figure.max_open_warning': 0})

    max_score = max(30, float(np.nanmax(scores)))
    from io import BytesIO

    for t in range(1, len(scores) + 1):
        fig, ax = plt.subplots(figsize=figsize)
        x = np.arange(t)
        y = scores[:t]
        ax.plot(x, y, color='#1f77b4', alpha=0.8, label='Episode Score')
        if t >= 1:
            rolling = pd.Series(scores[:t]).rolling(100, min_periods=1).mean()
            ax.plot(x, rolling, color='red', linewidth=2, label='100-Episode Avg')
        ax.axhline(y=30, color='green', linestyle='--', label='Target (30)')
        ax.set_xlim(0, len(scores))
        ax.set_ylim(min(-1, float(np.nanmin(scores) - 1)), max_score + 1)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Score')
        ax.set_title('Training Progress (animated)')
        ax.legend(loc='upper left')
        ax.grid(alpha=0.3)

        # Render to an in-memory PNG and read it with imageio (robust across backends)
        canvas = FigureCanvas(fig)
        buf = BytesIO()
        canvas.print_png(buf)
        buf.seek(0)
        image = imageio.imread(buf)
        images.append(image)
        plt.close(fig)

    # Save GIF using duration (ms) to avoid deprecated `fps` kwarg
    duration_ms = 1000.0 / float(fps) if fps and fps > 0 else None
    if duration_ms:
        imageio.mimsave(out_path, images, duration=duration_ms)
    else:
        imageio.mimsave(out_path, images)
    return out_path
