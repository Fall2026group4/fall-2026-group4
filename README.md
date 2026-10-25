## Team Members

- Aditi Shukla
- Dhruv Rai
- Simbanegavi Simbarashe

# SAE-Faithful

## Are Sparse Autoencoder Features Causally Faithful to Their Auto-Generated Labels?

The full project proposal (research questions, models, SAE architectures, datasets, phases, timeline, evaluation metrics, tech stack) lives in [`proposal.md`](proposal.md).

## Repository layout

- `proposal.md` — full project proposal and timeline
- `cookbooks/` — narrative Python scripts (import logic from `src/`, contain figures/tables only)
- `src/` — model loading, SAE steering, and evaluation logic (`models/`, `sae/`, `evaluation/`), with tests in `src/tests/`
- `configs/` — experiment configuration (model x SAE-architecture matrix, steering defaults)
- `results/` — figures (`figures/`, vector format) and tables (`tables/`) produced by the notebooks
- `reports/` — weekly progress report and the required formal written document (`Latex_report/`)
- `research_paper/` — final paper deliverable (LaTeX/Word)
- `presentation/` — slide decks
- `architecture/diagrams/` — system/pipeline diagrams (`.drawio` source + exported `.svg`)
- `data/` — not vendored; see `data/README.md` for sources and how to fetch them

---

## Project Overview

Sparse Autoencoders (SAEs) are increasingly used to interpret the internal representations of Large Language Models (LLMs). SAEs decompose dense neural-network activations into sparse features that can potentially be assigned human-interpretable meanings.

However, identifying and labeling an SAE feature does not necessarily mean that the feature causally controls the behavior described by its label.

SAE-Faithful investigates whether interpretable SAE features are also causally faithful. The project will compare feature activation patterns, automatically generated feature labels, concept detection performance, and causal steering behavior across different models, layers, feature frequencies, and SAE architectures.

The central benchmark is steering-based: selected SAE features will be deliberately modified, and the resulting language-model outputs will be evaluated to determine whether they move toward the concepts described by their feature labels.

---
