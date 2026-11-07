"""Find minimal true/false pairs in the RepE facts (a false statement that is a small edit
of a true one), for the steering readout: log P(true version) - log P(false version).

Saves data/repe/minimal_pairs.csv (all best matches) and prints the minimal ones.
Run from the repo root:  python scripts/repe_minimal_pairs.py
"""
import difflib
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
split_path = ROOT / "data" / "repe" / "facts_split.csv"
df = pd.read_csv(split_path if split_path.exists() else ROOT / "data" / "repe" / "facts_clean.csv")
if "split" not in df.columns:
    df["split"] = "unknown"
true, false = df[df["label"] == 1], df[df["label"] == 0]


def words(s):
    return s.rstrip(".").split()


rows = []
for _, f in false.iterrows():
    fw = words(f["statement"])
    best = max(((difflib.SequenceMatcher(a=words(t["statement"]), b=fw).ratio(), t)
                for _, t in true.iterrows()), key=lambda p: p[0])
    ratio, t = best
    ops = difflib.SequenceMatcher(a=words(t["statement"]), b=fw).get_opcodes()
    changed = sum(max(i2 - i1, j2 - j1) for tag, i1, i2, j1, j2 in ops if tag != "equal")
    rows.append(dict(true_statement=t["statement"], false_statement=f["statement"],
                     similarity=round(ratio, 3), words_changed=changed,
                     true_split=t["split"], false_split=f["split"]))

pairs = pd.DataFrame(rows).sort_values("similarity", ascending=False)
# keep each true statement once (its closest false edit)
pairs = pairs.drop_duplicates("true_statement")
minimal = pairs[(pairs["words_changed"] <= 3) & (pairs["similarity"] >= 0.7)]

out = ROOT / "data" / "repe" / "minimal_pairs.csv"
pairs.to_csv(out, index=False)
print(f"Saved {out.relative_to(ROOT)} ({len(pairs)} best matches)")
print(f"\nMinimal pairs (<= 3 words changed): {len(minimal)}")
print("By split (true / false):")
print(minimal.groupby(["true_split", "false_split"]).size().to_string())
print("\nExamples:")
for _, r in minimal.head(15).iterrows():
    print(f"  [{r.words_changed}]  TRUE:  {r.true_statement}\n       FALSE: {r.false_statement}")