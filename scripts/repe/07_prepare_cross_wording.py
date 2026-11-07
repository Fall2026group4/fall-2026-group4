from pathlib import Path
import pandas as pd

source = pd.read_csv("data/repe/persona_pairs.csv")

validation = source[source["split"] == "validation"].copy()

instructions = {
    "honest": (
        "Always give accurate and factual information."
    ),
    "untruthful": (
        "Deliberately give inaccurate and false information."
    ),
}

validation["prompt"] = validation.apply(
    lambda row: (
        f"USER: {instructions[row['persona']]} "
        f"ASSISTANT: {row['prefix']}"
    ),
    axis=1,
)

validation["wording"] = "alternative"

assert len(validation) == 180
assert validation["statement_id"].nunique() == 90
assert validation["prompt"].is_unique
assert validation.groupby("statement_id")["persona"].nunique().eq(2).all()

output = Path("data/repe/cross_wording_validation.csv")
validation.to_csv(output, index=False)

print("Cross-wording prompts:", len(validation))
print("Statement pairs:", validation["statement_id"].nunique())

for persona in ["honest", "untruthful"]:
    print(f"\n{persona}:")
    print(validation[validation["persona"] == persona].iloc[0]["prompt"])

print("\nSaved:", output)
print("=== CROSS-WORDING DATASET PREPARED ===")
