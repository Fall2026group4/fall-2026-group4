import pandas as pd
import numpy as np
import torch
from transformer_lens import HookedTransformer

df = pd.read_csv(
    "data/repe/steering/validation_matched_tokens.csv"
)

saved = torch.load(
    "data/repe/activations/gpt2_small/repe_final_token.pt",
    map_location="cpu",
    weights_only=True,
)

x = saved["activations"][6].numpy()
labels = saved["persona_labels"].numpy()
train = np.array(saved["splits"]) == "train"

direction = (
    x[train & (labels == 1)].mean(axis=0)
    - x[train & (labels == 0)].mean(axis=0)
)
direction /= np.linalg.norm(direction)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = HookedTransformer.from_pretrained(
    "gpt2-small", device=device
)
model.eval()

direction = torch.tensor(
    direction, dtype=torch.float32, device=device
)

torch.manual_seed(42)
random_direction = torch.randn_like(direction)
random_direction /= random_direction.norm()

def score_question(row, vector, alpha):
    prompt = "Question: " + row["question"] + "\nAnswer:"
    tokens = model.to_tokens(prompt, prepend_bos=True)

    correct = model.to_tokens(
        row["correct_answer"], prepend_bos=False
    )[0]
    incorrect = model.to_tokens(
        row["incorrect_answer"], prepend_bos=False
    )[0]

    assert len(correct) == len(incorrect) == 1

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

    final_logits = logits[0, -1]

    return (
        final_logits[correct[0]]
        - final_logits[incorrect[0]]
    ).item()

conditions = [
    ("baseline", None, 0),
    ("original_positive", direction, 5),
    ("original_negative", direction, -5),
    ("random_positive", random_direction, 5),
    ("random_negative", random_direction, -5),
]

rows = []

for name, vector, alpha in conditions:
    margins = [
        score_question(row, vector, alpha)
        for _, row in df.iterrows()
    ]

    rows.append({
        "condition": name,
        "alpha": alpha,
        "mean_margin": np.mean(margins),
        "correct_preference": int(sum(bool(m > 0) for m in margins)),
        "individual_margins": [round(m, 4) for m in margins],
        "questions": len(margins),
    })

results = pd.DataFrame(rows)

print("\n=== REPE CAUSAL STEERING PILOT ===")
print(results.round(4).to_string(index=False))

print("\n=== STEERING PILOT COMPLETE ===")
