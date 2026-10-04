"""
mapper.py

Maps each test point to the best matching ideal function.
"""

import math

import pandas as pd

from src.exceptions import MappingError


class TestDataMapper:
    """
    Maps test data to the selected ideal functions.
    """

    def map_test_data(
        self,
        training_df: pd.DataFrame,
        ideal_df: pd.DataFrame,
        test_df: pd.DataFrame,
        matches: dict
    ) -> pd.DataFrame:
        """
        Map test points to the selected ideal functions.

        Raises:
            MappingError:
                If mapping cannot be completed.
        """

        try:

            if not matches:
                raise MappingError(
                    "No ideal-function matches were provided."
                )

            mapped_rows = []

            # Calculate threshold for each selected ideal function
            thresholds = {}

            for training_col, match in matches.items():

                ideal_col = match["ideal_function"]

                max_deviation = (
                    training_df[training_col]
                    - ideal_df[ideal_col]
                ).abs().max()

                thresholds[ideal_col] = (
                    math.sqrt(2) * max_deviation
                )

            # Process each test point
            for _, test_row in test_df.iterrows():

                x = test_row["x"]
                y = test_row["y"]

                best_function = None
                best_delta = float("inf")

                # Find ideal-function row for this x value
                ideal_rows = ideal_df[
                    ideal_df["x"] == x
                ]

                if ideal_rows.empty:
                    raise MappingError(
                        f"No ideal-function row found "
                        f"for test x-value: {x}"
                    )

                ideal_row = ideal_rows.iloc[0]

                for ideal_col, threshold in thresholds.items():

                    ideal_y = ideal_row[ideal_col]

                    delta = abs(y - ideal_y)

                    if delta <= threshold and delta < best_delta:

                        best_delta = delta
                        best_function = ideal_col

                if best_function is not None:

                    mapped_rows.append({
                        "x": x,
                        "y": y,
                        "ideal_function": best_function,
                        "delta": best_delta
                    })

            return pd.DataFrame(mapped_rows)

        except MappingError:
            raise

        except Exception as error:

            raise MappingError(
                f"Test-data mapping failed: {error}"
            ) from error