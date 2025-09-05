import argparse
import os
import glob
import torch
from src.utils import analyze_training_progress, create_training_gif, find_best_checkpoint


def collect_scores_from_checkpoints(checkpoint_dir='checkpoints'):
    """Collect scores lists from checkpoint files found in checkpoint_dir.

    Returns a single flattened list of scores from the latest or best checkpoint
    available. If no checkpoints are present, returns a synthetic example.
    """
    if not os.path.exists(checkpoint_dir):
        return list(range(0, 36))  # fallback synthetic scores

    # Prefer the best checkpoint if available
    best = find_best_checkpoint(checkpoint_dir)
    target = best if best else os.path.join(checkpoint_dir, 'checkpoint_latest.pth')
    if not os.path.exists(target):
        # pick any checkpoint
        files = glob.glob(os.path.join(checkpoint_dir, 'checkpoint_*.pth'))
        if not files:
            return list(range(0, 36))
        target = files[-1]

    try:
        ckpt = torch.load(target)
        scores = ckpt.get('scores', None)
        if scores is None or len(scores) == 0:
            return list(range(0, 36))
        return scores
    except Exception:
        return list(range(0, 36))


def main():
    parser = argparse.ArgumentParser(description='Generate plots and an animated GIF from training checkpoints.')
    parser.add_argument('--checkpoints', type=str, default='checkpoints', help='Path to checkpoint directory')
    parser.add_argument('--out', type=str, default='checkpoints/demos', help='Output directory for GIF and plots')
    parser.add_argument('--fps', type=int, default=8, help='Frames per second for the GIF')
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    scores = collect_scores_from_checkpoints(args.checkpoints)

    print(f"Found {len(scores)} score entries; generating animated GIF...")
    gif_path = os.path.join(args.out, 'training_progress.gif')
    create_training_gif(scores, out_path=gif_path, fps=args.fps)

    print(f"GIF created: {gif_path}")

    # Also create a static analysis plot
    analyze_training_progress()


if __name__ == '__main__':
    main()
