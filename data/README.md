# Data

This project does not vendor any dataset into the repository. `data/raw/`
and `data/processed/` are tracked as empty directories (via `.gitkeep`) so
the expected layout is visible in git, but their contents are gitignored
(see `.gitignore`) — nothing large or licensed should ever be committed
here.

## Sources

### OpenWebText (`Skylion007/openwebtext`)

Used for Phase 1/2 exploratory data analysis and initial GPT-2 Small
activation collection (`cookbooks/01_data_exploration.py`).

Fetched via the Hugging Face `datasets` library in streaming mode, so no
download/caching step is required before running the notebook:

```python
from datasets import load_dataset
dataset = load_dataset("Skylion007/openwebtext", split="train", streaming=True)
```

The EDA notebook pulls a fixed-size sample (1,000 documents) directly from
the stream rather than materializing the full ~40GB corpus to disk.

### Geometry of Truth

Used for Phase 5 concept-detection work on truth/falsehood features
(`cookbooks/03_geometry_of_truth_eda.py`, `results/geometry_of_truth_*`).

Sourced from the public Geometry of Truth dataset release
(saprmarks/geometry-of-truth on GitHub / the associated HuggingFace
dataset). See `cookbooks/03_geometry_of_truth_eda.py` for the exact
loading code and `results/geometry_of_truth_observations.md` for findings.

### Representation Engineering (RepE)

Planned for Phase 5 concept-detection experiments (contrastive
positive/negative behavioral examples). Not yet integrated — this section
will be filled in with the exact source/loading code once that phase
starts.

## Pretrained models and SAEs

Not "data" in the traditional sense, but also not vendored: GPT-2 Small
weights and the pretrained SAE (`gpt2-small-res-jb` release, block 8
`hook_resid_pre`) are downloaded on demand from HuggingFace Hub via
`transformer_lens` / `sae_lens` the first time `src/models/gpt2_sae.py`'s
`load_sae_bundle()` runs, and cached locally by those libraries (not under
this repo's `data/`).
