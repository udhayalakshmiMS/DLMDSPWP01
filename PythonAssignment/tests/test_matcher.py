import pandas as pd

from src.matcher import FunctionMatcher


def create_ideal_dataframe():

    data = {
        "x": [1, 2]
    }

    # Create all 50 required ideal functions.
    for i in range(1, 51):
        data[f"y{i}"] = [1000, 1000]

    # Known matching functions.
    data["y1"] = [1, 2]
    data["y2"] = [10, 20]
    data["y3"] = [5, 6]
    data["y4"] = [50, 60]

    return pd.DataFrame(data)


def test_calculate_sse():

    matcher = FunctionMatcher()

    training = pd.Series([1, 2, 3])
    ideal = pd.Series([1, 4, 2])

    # (1-1)^2 + (2-4)^2 + (3-2)^2
    # = 0 + 4 + 1
    # = 5

    result = matcher.calculate_sse(
        training,
        ideal
    )

    assert result == 5


def test_find_best_matches():

    matcher = FunctionMatcher()

    training_df = pd.DataFrame({
        "x": [1, 2],
        "y1": [1, 2],
        "y2": [10, 20],
        "y3": [5, 6],
        "y4": [50, 60]
    })

    ideal_df = create_ideal_dataframe()

    matches = matcher.find_best_matches(
        training_df,
        ideal_df
    )

    assert matches["y1"]["ideal_function"] == "y1"
    assert matches["y2"]["ideal_function"] == "y2"
    assert matches["y3"]["ideal_function"] == "y3"
    assert matches["y4"]["ideal_function"] == "y4"


def test_find_best_matches_tie_breaking():

    matcher = FunctionMatcher()

    training_df = pd.DataFrame({
        "x": [1, 2],
        "y1": [1, 2],
        "y2": [10, 20],
        "y3": [5, 6],
        "y4": [50, 60]
    })

    ideal_df = create_ideal_dataframe()

    # Make y1 and y2 identical.
    # Therefore both have exactly the same SSE
    # for training y1.
    ideal_df["y2"] = [1, 2]

    matches = matcher.find_best_matches(
        training_df,
        ideal_df
    )

    # y1 is encountered before y2.
    # Since matcher uses "<" rather than "<=",
    # y1 remains the selected function.
    assert matches["y1"]["ideal_function"] == "y1"