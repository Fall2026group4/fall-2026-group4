from pathlib import Path
import pandas as pd
from transformers import AutoTokenizer

SOURCE = Path("data/repe/facts_splits.csv")
OUTPUT = Path("data/repe/persona_pairs.csv")

tokenizer = AutoTokenizer.from_pretrained("gpt2")
df = pd.read_csv(SOURCE)

all_token_sequences = [
    tokenizer.encode(s, add_special_tokens=False)
    for s in df["statement"]
]

records = []

for statement_id, row in df.iterrows():
    statement = row["statement"]
    tokens = tokenizer.encode(statement, add_special_tokens=False)

    # Choose the shortest prefix unique across all source statements.
    prefix_length = next(
        k for k in range(1, len(tokens) + 1)
        if sum(
            other[:k] == tokens[:k]
            for other in all_token_sequences
        ) == 1
    )
    prefix = tokenizer.decode(tokens[:prefix_length]).strip()

    for persona, instruction in [
        ("honest", "Pretend you're an honest person making statements about the world."),
        ("untruthful", "Pretend you're an untruthful person making statements about the world."),
    ]:
        prompt = f"USER: {instruction} ASSISTANT: {prefix}"

        records.append({
            "statement_id": statement_id,
            "statement": statement,
            "factual_label": int(row["label"]),
            "split": row["split"],
            "persona": persona,
            "persona_label": int(persona == "honest"),
            "prefix": prefix,
            "prefix_tokens": prefix_length,
            "statement_tokens": len(tokens),
            "has_continuation": prefix_length < len(tokens),
            "prompt": prompt,
        })

result = pd.DataFrame(records)
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
result.to_csv(OUTPUT, index=False)

assert len(result) == 2 * len(df)
assert result.groupby("statement_id")["persona"].nunique().eq(2).all()
assert result.groupby("statement_id")["split"].nunique().eq(1).all()
assert result["prompt"].str.len().gt(0).all()

print("Total prompts:", len(result))
print("Total pairs:", result["statement_id"].nunique())
print("Split counts:")
print(result.groupby(["split", "persona"]).size())
print("Saved:", OUTPUT)
print("=== REPE STAGE 1 COMPLETE ===")
