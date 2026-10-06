from src.loader import DatasetLoader
from src.database import DatabaseManager
from src.matcher import FunctionMatcher
from src.mapper import TestDataMapper
from src.visualizer import Visualizer
from src.exceptions import AssignmentError


def main():

    loader = DatasetLoader()
    database = DatabaseManager()
    matcher = FunctionMatcher()
    mapper = TestDataMapper()
    visualizer = Visualizer()

    print("Loading datasets...")

    training_df = loader.load_dataset("data/train.csv")
    ideal_df = loader.load_dataset("data/ideal.csv")
    test_df = loader.load_dataset("data/test.csv")

    print("Creating database...")

    database.create_tables()

    print("Inserting datasets...")

    database.insert_training(training_df)
    database.insert_ideal(ideal_df)

    print("Finding best ideal functions...")

    matches = matcher.find_best_matches(
        training_df,
        ideal_df
    )

    print(matches)

    print("Mapping test data...")

    mapped_df = mapper.map_test_data(
        training_df,
        ideal_df,
        test_df,
        matches
    )

    database.insert_mapping(mapped_df)

    print("Generating visualization...")

    visualizer.plot(
        training_df,
        ideal_df,
        test_df,
        mapped_df,
        matches
    )

    print("Project completed successfully!")
    

if __name__ == "__main__":
    try:
        main()
    except AssignmentError as error:
        print(f"Error: {error}")
        raise SystemExit(1)
    except OSError as error:
        print(f"File system error: {error}")
        raise SystemExit(1)