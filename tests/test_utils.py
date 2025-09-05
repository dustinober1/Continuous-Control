import os
import sys
import tempfile
import numpy as np

# Ensure project root is on sys.path so `src` modules can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils import create_training_gif


def test_create_training_gif_creates_file():
    scores = np.linspace(0, 35, 50)
    with tempfile.TemporaryDirectory() as td:
        out_path = os.path.join(td, 'test_progress.gif')
        path = create_training_gif(scores, out_path=out_path, fps=4)
        assert os.path.exists(path)
        assert path.endswith('.gif')
