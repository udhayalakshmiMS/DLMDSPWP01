import math

import pandas as pd
import pytest

from src.mapper import TestDataMapper
from src.exceptions import MappingError


def create_training_data():
    return pd.DataFrame({
        "x": [1, 2, 3],
        "y1": [0, 0, 0],
        "y2": [10, 10, 10],
        "y3": [20, 20, 20],
        "y4": [30, 30, 30]
    })


def create_ideal_data():
    return pd.DataFrame({
        "x": [1, 2, 3],
        "y1": [0, 0, 0],
        "y2": [10, 10, 10],
        "y3": [20, 20, 20],
        "y4": [30, 30, 30]
    })


def create_matches():
    return {
        "y1": {
            "ideal_function": "y1",
            "sse": 0
        }
    }


def test_point_exactly_at_threshold_is_mapped():

    mapper = TestDataMapper()

    training_df = create_training_data()
    ideal_df = create_ideal_data()

    training_df["y1"] = [0, 1, 0]

    max_deviation = (
        training_df["y1"] - ideal_df["y1"]
    ).abs().max()

    threshold = math.sqrt(2) * max_deviation

    test_df = pd.DataFrame({
        "x": [2],
        "y": [threshold]
    })

    result = mapper.map_test_data(
        training_df,
        ideal_df,
        test_df,
        create_matches()
    )

    assert len(result) == 1
    assert result.iloc[0]["ideal_function"] == "y1"
    assert result.iloc[0]["delta"] == threshold


def test_point_just_inside_threshold_is_mapped():

    mapper = TestDataMapper()

    training_df = create_training_data()
    ideal_df = create_ideal_data()

    training_df["y1"] = [0, 1, 0]

    max_deviation = (
        training_df["y1"] - ideal_df["y1"]
    ).abs().max()

    threshold = math.sqrt(2) * max_deviation

    test_df = pd.DataFrame({
        "x": [2],
        "y": [threshold - 0.001]
    })

    result = mapper.map_test_data(
        training_df,
        ideal_df,
        test_df,
        create_matches()
    )

    assert len(result) == 1


def test_point_just_outside_threshold_is_not_mapped():

    mapper = TestDataMapper()

    training_df = create_training_data()
    ideal_df = create_ideal_data()

    training_df["y1"] = [0, 1, 0]

    max_deviation = (
        training_df["y1"] - ideal_df["y1"]
    ).abs().max()

    threshold = math.sqrt(2) * max_deviation

    test_df = pd.DataFrame({
        "x": [2],
        "y": [threshold + 0.001]
    })

    result = mapper.map_test_data(
        training_df,
        ideal_df,
        test_df,
        create_matches()
    )

    assert result.empty


def test_smaller_deviation_is_selected():

    mapper = TestDataMapper()

    training_df = pd.DataFrame({
        "x": [1, 2, 3],
        "y1": [0, 1, 0],
        "y2": [1, 2, 1],
        "y3": [20, 20, 20],
        "y4": [30, 30, 30]
    })

    ideal_df = pd.DataFrame({
        "x": [1, 2, 3],
        "y1": [0, 0, 0],
        "y2": [1, 1, 1],
        "y3": [20, 20, 20],
        "y4": [30, 30, 30]
    })

    test_df = pd.DataFrame({
        "x": [2],
        "y": [0.2]
    })

    matches = {
        "y1": {
            "ideal_function": "y1",
            "sse": 1
        },
        "y2": {
            "ideal_function": "y2",
            "sse": 1
        }
    }

    result = mapper.map_test_data(
        training_df,
        ideal_df,
        test_df,
        matches
    )

    assert len(result) == 1

    # Both functions qualify:
    # y1 -> delta = 0.2
    # y2 -> delta = 0.8
    #
    # Therefore y1 must be selected.

    assert result.iloc[0]["ideal_function"] == "y1"
    assert result.iloc[0]["delta"] == 0.2


def test_missing_x_value_raises_mapping_error():

    mapper = TestDataMapper()

    training_df = create_training_data()
    ideal_df = create_ideal_data()

    test_df = pd.DataFrame({
        "x": [999],
        "y": [10]
    })

    with pytest.raises(MappingError):

        mapper.map_test_data(
            training_df,
            ideal_df,
            test_df,
            create_matches()
        )