"""
visualizer.py

Creates separate Bokeh plots for each
training/ideal function pair.
"""

from bokeh.layouts import column
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.plotting import figure, output_file, save

from src.exceptions import VisualizationError


class Visualizer:
    """
    Creates visualizations for training functions,
    selected ideal functions, and test data.
    """

    def plot(
        self,
        training_df,
        ideal_df,
        test_df,
        mapped_df,
        matches
    ):
        """
        Create one figure for each training/ideal pair.

        Raises:
            VisualizationError:
                If visualization cannot be generated.
        """

        try:

            if not matches:
                raise VisualizationError(
                    "Cannot create visualization: "
                    "no function matches were provided."
                )

            required_training_columns = ["x"]
            required_test_columns = ["x", "y"]
            required_mapped_columns = [
                "x",
                "y",
                "ideal_function",
                "delta"
            ]

            # Validate training dataset
            for required_column in required_training_columns:

                if required_column not in training_df.columns:
                    raise VisualizationError(
                        f"Training dataset is missing column: "
                        f"{required_column}"
                    )

            # Validate test dataset
            for required_column in required_test_columns:

                if required_column not in test_df.columns:
                    raise VisualizationError(
                        f"Test dataset is missing column: "
                        f"{required_column}"
                    )

            # Validate mapped dataset
            for required_column in required_mapped_columns:

                if required_column not in mapped_df.columns:
                    raise VisualizationError(
                        f"Mapped dataset is missing column: "
                        f"{required_column}"
                    )

            output_file("results.html")

            figures = []

            colors = [
                "blue",
                "green",
                "orange",
                "purple"
            ]

            # All test points that were mapped to any
            # selected ideal function
            mapped_points = set(
                zip(
                    mapped_df["x"],
                    mapped_df["y"]
                )
            )

            for index, (training_col, match) in enumerate(
                matches.items()
            ):

                ideal_col = match["ideal_function"]
                sse = match["sse"]

                if training_col not in training_df.columns:
                    raise VisualizationError(
                        f"Training column not found: "
                        f"{training_col}"
                    )

                if ideal_col not in ideal_df.columns:
                    raise VisualizationError(
                        f"Ideal-function column not found: "
                        f"{ideal_col}"
                    )

                color = colors[index % len(colors)]

                # Test points mapped to this particular
                # ideal function
                mapped_for_function = mapped_df[
                    mapped_df["ideal_function"] == ideal_col
                ].copy()

                # Test points that were not mapped to ANY
                # selected ideal function
                unmapped_rows = []

                for _, row in test_df.iterrows():

                    point = (row["x"], row["y"])

                    if point not in mapped_points:
                        unmapped_rows.append(row)

                p = figure(
                    title=(
                        f"{training_col} → {ideal_col} | "
                        f"SSE = {sse:.4f} | "
                        f"Mapped = {len(mapped_for_function)} | "
                        f"Unmapped = {len(unmapped_rows)}"
                    ),
                    width=1200,
                    height=400,
                    x_axis_label="X",
                    y_axis_label="Y"
                )

                # Training function
                p.line(
                    training_df["x"],
                    training_df[training_col],
                    color=color,
                    line_width=3,
                    legend_label=f"{training_col} (Training)"
                )

                # Selected ideal function
                p.line(
                    ideal_df["x"],
                    ideal_df[ideal_col],
                    color=color,
                    line_dash="dashed",
                    line_width=2,
                    legend_label=f"{ideal_col} (Ideal)"
                )

                # Unmapped test points
                if unmapped_rows:

                    p.scatter(
                        [row["x"] for row in unmapped_rows],
                        [row["y"] for row in unmapped_rows],
                        color="gray",
                        size=7,
                        alpha=0.6,
                        legend_label="Unmapped Test Points"
                    )

                # Mapped test points
                if not mapped_for_function.empty:

                    source = ColumnDataSource(
                        mapped_for_function
                    )

                    mapped_renderer = p.scatter(
                        "x",
                        "y",
                        source=source,
                        color="black",
                        size=8,
                        legend_label="Mapped Test Points"
                    )

                    hover = HoverTool(
                        renderers=[mapped_renderer],
                        tooltips=[
                            ("X", "@x"),
                            ("Y", "@y"),
                            (
                                "Ideal Function",
                                "@ideal_function"
                            ),
                            ("Delta", "@delta")
                        ]
                    )

                    p.add_tools(hover)

                p.legend.location = "top_left"

                figures.append(p)

            # Bokeh's column() function is still available
            # because we no longer overwrite its name.
            save(column(*figures))

        except VisualizationError:
            raise

        except Exception as error:

            raise VisualizationError(
                f"Visualization generation failed: {error}"
            ) from error