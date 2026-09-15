# SAE Faithfulness

## Overview

This project investigates the faithfulness and interpretability of Sparse Autoencoders (SAEs) applied to internal representations of GPT-2 Small.

Sparse Autoencoders decompose dense language-model activations into a larger set of sparse features. These features can potentially correspond to human-interpretable concepts. However, an interpretable feature does not necessarily mean that the feature faithfully represents how the language model internally uses that concept.

The goal of this project is to study SAE features through both observational analysis and causal intervention.

## Research Questions

This project focuses on the following questions:

1. Can pretrained SAE features identify interpretable concepts in GPT-2 representations?
2. Are these features consistently activated across different contexts containing the same concept?
3. Are the features selective for the identified concept compared with related or unrelated inputs?
4. Can manipulating an SAE feature causally influence GPT-2 toward the concept associated with that feature?
5. How faithfully do SAE features represent the underlying behavior of the language model?

## Project Phases

### Phase 1 — Pretrained SAE Baseline and Feature Analysis

Phase 1 establishes a baseline using GPT-2 Small and a pretrained Sparse Autoencoder.

The phase includes:

- Loading GPT-2 Small and the pretrained SAE.
- Inspecting GPT-2 tokenization and internal residual-stream activations.
- Encoding GPT-2 activations using the SAE.
- Identifying strongly activated SAE features.
- Investigating Feature 974 as a case study.
- Comparing feature activations across multiple contexts.
- Comparing Paris examples against non-Paris city controls.
- Validating feature interpretations using Neuronpedia.

### Phase 2 — Causal Intervention and Steering

Phase 2 investigates whether an interpretable SAE feature can causally influence GPT-2's behavior.

The main question is whether a feature that strongly detects a concept also causes the model to move toward that concept when the feature is manipulated.

Steering and intervention experiments will be performed at different strengths and across multiple prompts.

### Phase 3 — SAE Faithfulness Evaluation

Phase 3 will extend the experiments to systematically evaluate SAE faithfulness across multiple features, concepts, prompts, and intervention settings.

The goal will be to compare observational interpretability with causal behavior and determine when SAE features provide faithful explanations of model representations.

## Phase 1 Results

Feature 974 was identified as the strongest SAE feature at the `Paris` token position in an initial test sentence.

The feature was then evaluated across multiple Paris and non-Paris examples.

The average Feature 974 activation was:

- **Paris examples:** 58.97
- **Non-Paris city examples:** 6.91

Feature 974 therefore activated approximately **8.5 times more strongly** for the tested Paris examples.

The feature also remained strongly active across different Paris contexts, including travel, geography, work, conference, and vacation-related sentences.

Neuronpedia independently describes Feature 974 as relating to mentions or references to Paris.

These results provide preliminary evidence that Feature 974 is selective for Paris-related representations in GPT-2 Small.

The current results demonstrate representational selectivity but do not yet establish causal faithfulness.

## Current Research Question

**Does Feature 974 only detect Paris-related representations, or can it causally influence GPT-2 toward the Paris concept?**

If increasing Feature 974 makes GPT-2 more likely to generate something related to Paris, this would provide evidence that the feature does not simply detect Paris-related representations but can also causally influence the model toward the concept.

This question will be investigated in Phase 2.

## Repository Structure

    SAE_faithfulness/
    │
    ├── notebooks/
    │   ├── 01_data_exploration.ipynb
    │   └── 02_pretrained_sae_baseline.ipynb
    │
    ├── results/
    │   ├── figures/
    │   ├── logs/
    │   ├── tables/
    │   └── phase_1_observations.md
    │
    ├── scripts/
    │
    ├── requirements.txt
    ├── .gitignore
    └── README.md

## Tools and Technologies

- Python
- PyTorch
- GPT-2 Small
- TransformerLens
- SAE Lens
- Sparse Autoencoders
- Neuronpedia
- NumPy
- Pandas
- Matplotlib
- Jupyter
- VS Code
- Git and GitHub

## Status

**Phase 1: Pretrained SAE Baseline — In Progress**

Current work focuses on validating pretrained SAE features and establishing a reliable observational baseline before moving to causal steering experiments.

## Author

Aditi Shukla  
M.S. Data Science  
George Washington University
