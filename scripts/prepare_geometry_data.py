from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


SEED = 42
TEST_SIZE = 0.30

INPUT_PATH = Path("data/raw/cities.csv")
OUTPUT_DIR = Path("data/processed/geometry_of_truth")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_PATH)

    required_columns = {
        "statement",
        "label",
        "city",
        "country",
        "correct_country",
    }

    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Split by CITY, not by individual rows.
    # This prevents the same city appearing in both train and test sets.
    cities = df["city"].drop_duplicates().tolist()

    train_cities, test_cities = train_test_split(
        cities,
        test_size=TEST_SIZE,
        random_state=SEED,
        shuffle=True,
    )

    train_df = df[df["city"].isin(train_cities)].copy()
    test_df = df[df["city"].isin(test_cities)].copy()

    # Safety check: no city leakage between train and test.
    assert set(train_df["city"]).isdisjoint(set(test_df["city"]))

    train_df.to_csv(OUTPUT_DIR / "train.csv", index=False)
    test_df.to_csv(OUTPUT_DIR / "test.csv", index=False)

    manifest = pd.DataFrame({
        "city": cities,
        "split": [
            "train" if city in set(train_cities) else "test"
            for city in cities
        ],
    })

    manifest.to_csv(OUTPUT_DIR / "split_manifest.csv", index=False)

    print("Geometry of Truth split complete")
    print(f"Seed: {SEED}")
    print(f"Full dataset: {df.shape}")
    print(f"Train: {train_df.shape}")
    print(f"Test: {test_df.shape}")
    print("\nTrain labels:")
    print(train_df["label"].value_counts().sort_index())
    print("\nTest labels:")
    print(test_df["label"].value_counts().sort_index())


if __name__ == "__main__":
    main()
