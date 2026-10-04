import os

import pandas as pd

from src.visualizer import Visualizer


def test_plot_creates_output_file(tmp_path, monkeypatch):

    training_df = pd.DataFrame({
        "x": [1, 2, 3],
        "y1": [1, 2, 3],
        "y2": [2, 3, 4],
        "y3": [3, 4, 5],
        "y4": [4, 5, 6]
    })

    ideal_df = pd.DataFrame({
        "x": [1, 2, 3],
        "y1": [1, 2, 3],
        "y2": [2, 3, 4],
        "y3": [3, 4, 5],
        "y4": [4, 5, 6]
    })

    test_df = pd.DataFrame({
        "x": [1, 2, 3],
        "y": [1.1, 2.1, 3.1]
    })

    mapped_df = pd.DataFrame({
        "x": [1, 2],
        "y": [1.1, 2.1],
        "ideal_function": ["y1", "y1"],
        "delta": [0.1, 0.1]
    })

    matches = {
        "y1": {
            "ideal_function": "y1",
            "sse": 0.0
        }
    }

    output_path = tmp_path / "results.html"

    monkeypatch.chdir(tmp_path)

    visualizer = Visualizer()

    visualizer.plot(
        training_df,
        ideal_df,
        test_df,
        mapped_df,
        matches
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0