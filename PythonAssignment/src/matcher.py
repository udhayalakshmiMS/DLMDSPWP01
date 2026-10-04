"""
matcher.py

Finds the best matching ideal function for each training function
using the Least Square Method (Sum of Squared Errors).
"""

import pandas as pd


class FunctionMatcher:
    """
    Matches training functions with ideal functions.
    """

    def calculate_sse(
        self,
        training_series: pd.Series,
        ideal_series: pd.Series
    ) -> float:
        """
        Calculate Sum of Squared Errors.
        """

        return ((training_series - ideal_series) ** 2).sum()

    def find_best_matches(
        self,
        training_df: pd.DataFrame,
        ideal_df: pd.DataFrame
    ) -> dict:
        """
        Find the best ideal function for each training function.
        """

        matches = {}

        training_columns = ["y1", "y2", "y3", "y4"]
        ideal_columns = [f"y{i}" for i in range(1, 51)]

        for training_col in training_columns:

            best_sse = float("inf")
            best_function = None

            for ideal_col in ideal_columns:

                sse = self.calculate_sse(
                    training_df[training_col],
                    ideal_df[ideal_col]
                )

                if sse < best_sse:
                    best_sse = sse
                    best_function = ideal_col

            matches[training_col] = {
                "ideal_function": best_function,
                "sse": best_sse
            }

        print("\nBest Matches")
        print("-" * 30)

        for training, match in matches.items():
            print(
                f"{training} -> "
                f"{match['ideal_function']} "
                f"(SSE = {match['sse']:.4f})"
            )

        return matches