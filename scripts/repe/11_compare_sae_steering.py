from pathlib import Path
import numpy as np
import pandas as pd
import torch
from transformer_lens import HookedTransformer

df = pd.read_csv(
    "data/repe/steering/validation_matched_tokens.csv"
)

source = torch.load(
    "data/repe/activations/gpt2_small/repe_final_token.pt",
    map_location="cpu",
    weights_only=True,
)

labels = source["persona_labels"].numpy()
train = np.array(source["splits"]) == "train"

def get_direction(x):
    x = x.numpy()
    d = (
        x[train & (labels == 1)].mean(axis=0)
        - x[train & (labels == 0)].mean(axis=0)
    )
    d = torch.tensor(d, dtype=torch.float32)
    return d / d.norm()

directions = {
    "original_gpt2": get_direction(source["activations"][6])
}

root = Path("results/repe/sae_reconstruction")

for arch in ["relu", "topk", "jumprelu"]:
    for variant in [
        "pretrained", "reconstruction_only", "truth_aware"
    ]:
        name = f"{arch}_{variant}"
        saved = torch.load(
            root / f"{name}.pt",
            map_location="cpu",
            weights_only=True,
        )
        directions[name] = get_direction(
            saved["reconstructed_activations"]
        )

generator = torch.Generator().manual_seed(42)

for i in range(20):
    d = torch.randn(768, generator=generator)
    directions[f"random_{i:02d}"] = d / d.norm()

device = "cuda" if torch.cuda.is_available() else "cpu"
model = HookedTransformer.from_pretrained(
    "gpt2-small", device=device
)
model.eval()

prepared = []

for _, row in df.iterrows():
    prompt = "Question: " + row["question"] + "\nAnswer:"
    tokens = model.to_tokens(prompt, prepend_bos=True)

    correct = model.to_tokens(
        row["correct_answer"], prepend_bos=False
    )[0]
    incorrect = model.to_tokens(
        row["incorrect_answer"], prepend_bos=False
    )[0]

    assert len(correct) == len(incorrect) == 1

    prepared.append((
        row["question"],
        tokens,
        int(correct[0]),
        int(incorrect[0]),
    ))

def evaluate(vector, alpha):
    margins = []

    for question, tokens, correct, incorrect in prepared:
        def hook_fn(activation, hook):
            modified = activation.clone()
            modified[:, -1, :] += alpha * vector
            return modified

        with torch.inference_mode():
            if vector is None or alpha == 0:
                logits = model(tokens)
            else:
                logits = model.run_with_hooks(
                    tokens,
                    fwd_hooks=[
                        ("blocks.6.hook_resid_pre", hook_fn)
                    ],
                )

        final = logits[0, -1]
        margins.append(
            (final[correct] - final[incorrect]).item()
        )

    return margins

baseline = evaluate(None, 0)
rows = []

rows.append({
    "model": "baseline",
    "alpha": 0,
    "mean_margin": np.mean(baseline),
    "delta_from_baseline": 0.0,
    "correct_preference": sum(m > 0 for m in baseline),
})

for name, direction in directions.items():
    direction = direction.to(device)

    for alpha in [-10, 10]:
        margins = evaluate(direction, alpha)

        rows.append({
            "model": name,
            "alpha": alpha,
            "mean_margin": np.mean(margins),
            "delta_from_baseline": (
                np.mean(margins) - np.mean(baseline)
            ),
            "correct_preference": sum(m > 0 for m in margins),
        })

results = pd.DataFrame(rows)

output = Path("results/repe/steering")
output.mkdir(parents=True, exist_ok=True)
results.to_csv(
    output / "sae_steering_pilot.csv",
    index=False,
)

print("\n=== ORIGINAL AND SAE STEERING RESULTS ===")
print(
    results[
        ~results["model"].str.startswith("random_")
    ].round(4).to_string(index=False)
)

print("\n=== RANDOM CONTROL SUMMARY ===")
print(
    results[
        results["model"].str.startswith("random_")
    ].groupby("alpha")["delta_from_baseline"]
    .agg(["mean", "std", "min", "max"])
    .round(4)
    .to_string()
)

print("\nSaved:", output / "sae_steering_pilot.csv")
print("=== SAE CAUSAL STEERING COMPARISON COMPLETE ===")
