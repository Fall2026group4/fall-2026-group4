from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_DIR = Path("data/got")
OUTPUT_DIR = DATA_DIR / "splits"
SEED = 42


def assign_splits(groups):
    groups = sorted(set(groups), key=str)

    train, remaining = train_test_split(
        groups, test_size=0.30, random_state=SEED
    )
    val, test = train_test_split(
        remaining, test_size=0.50, random_state=SEED
    )

    mapping = {group: "train" for group in train}
    mapping.update({group: "val" for group in val})
    mapping.update({group: "test" for group in test})
    return mapping


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    datasets = {
        name: pd.read_csv(DATA_DIR / f"{name}.csv")
        for name in [
            "cities", "neg_cities",
            "larger_than", "smaller_than"
        ]
    }

    # Group all city statements by city identity.
    city_groups = set(datasets["cities"]["city"])
    city_groups.update(datasets["neg_cities"]["city"])
    city_mapping = assign_splits(city_groups)

    # Group numerical statements by unordered number pair.
    number_groups = set()
    for name in ["larger_than", "smaller_than"]:
        df = datasets[name]
        number_groups.update(
            tuple(sorted((int(a), int(b))))
            for a, b in zip(df["n1"], df["n2"])
        )

    number_mapping = assign_splits(number_groups)

    for name, df in datasets.items():
        df = df.copy()
        df["row_id"] = np.arange(len(df))

        if name in ["cities", "neg_cities"]:
            df["group_id"] = df["city"].astype(str)
            df["split"] = df["city"].map(city_mapping)
        else:
            keys = [
                tuple(sorted((int(a), int(b))))
                for a, b in zip(df["n1"], df["n2"])
            ]
            df["group_id"] = [f"{a}_{b}" for a, b in keys]
            df["split"] = [number_mapping[key] for key in keys]

        assert df["split"].notna().all()

        # Verify no group crosses split boundaries.
        assert df.groupby("group_id")["split"].nunique().max() == 1

        output = OUTPUT_DIR / f"{name}_splits.csv"
        df[["row_id", "group_id", "label", "split"]].to_csv(
            output, index=False
        )

        print(f"\n{name}")
        print(pd.crosstab(df["split"], df["label"]))
        print("Saved:", output)

    # Verify related groups across datasets.
    for names in [
        ["cities", "neg_cities"],
        ["larger_than", "smaller_than"]
    ]:
        combined = pd.concat(
            [
                pd.read_csv(OUTPUT_DIR / f"{name}_splits.csv")
                for name in names
            ],
            ignore_index=True
        )
        assert combined.groupby("group_id")["split"].nunique().max() == 1

    print("\nAll grouped split checks passed.")


if __name__ == "__main__":
    main()