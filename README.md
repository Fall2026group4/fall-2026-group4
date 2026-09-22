## Team Members

- Aditi Shukla
- Dhruv
- Simbanegavi Simbarashe

# SAE-Faithful

## Are Sparse Autoencoder Features Causally Faithful to Their Auto-Generated Labels?

### 7-Week Project Proposal

---

## Project Overview

Sparse Autoencoders (SAEs) are increasingly used to interpret the internal representations of Large Language Models (LLMs). SAEs decompose dense neural-network activations into sparse features that can potentially be assigned human-interpretable meanings.

However, identifying and labeling an SAE feature does not necessarily mean that the feature causally controls the behavior described by its label.

SAE-Faithful investigates whether interpretable SAE features are also causally faithful. The project will compare feature activation patterns, automatically generated feature labels, concept detection performance, and causal steering behavior across different models, layers, feature frequencies, and SAE architectures.

The central benchmark is steering-based: selected SAE features will be deliberately modified, and the resulting language-model outputs will be evaluated to determine whether they move toward the concepts described by their feature labels.

---

## Problem Statement

Sparse Autoencoders identify features inside Large Language Models, and these features can be given natural-language labels based on the text examples that activate them.

For example, a feature may be interpreted as representing concepts such as:

- Positive sentiment
- Truthfulness
- Sports
- Geographic locations
- Other semantic or behavioral concepts

However, a feature label describing a concept does not prove that the feature actually controls that behavior.

A feature may correlate with a concept without playing a causal role in the model's output.

The goal of this project is therefore to distinguish between features that are simply interpretable-looking and features whose labels accurately describe their causal effect on model behavior.

---

## Research Questions

This project will investigate the following questions:

1. Can SAE features be assigned meaningful natural-language interpretations from their highest-activating text examples?

2. Do SAE features reliably detect the concepts described by their labels?

3. Does activating or strengthening an SAE feature cause the language model to generate behavior matching the feature's label?

4. Does causal faithfulness change across different transformer layers?

5. Does feature activation frequency or sparsity affect causal faithfulness?

6. Do different SAE architectures produce different levels of interpretability and causal faithfulness?

7. How do ReLU, TopK, and JumpReLU SAEs compare?

8. Does higher auto-generated label confidence correspond to higher causal faithfulness?

9. Do findings observed in GPT-2 Small generalize to a larger language model such as Gemma-2-2B?

10. Under what conditions can an SAE-generated feature label be trusted as a causal explanation of model behavior?

---

## Models

### GPT-2 Small

GPT-2 Small will serve as the primary experimental language model.

- 124M parameters
- 12 transformer layers
- Main experimental model
- Suitable for controlled SAE experiments
- Pretrained SAEs available through SAE Lens

GPT-2 Small will be used for initial feature extraction, interpretation, concept detection, and causal steering experiments.

### Gemma-2-2B

Gemma-2-2B will serve as a secondary comparison model.

Pretrained SAEs available through Gemma Scope will be used to investigate whether findings from GPT-2 Small generalize to a different and larger language model.

---

## Sparse Autoencoder Architectures

Three SAE architectures will be investigated.

### ReLU SAE

A standard sparse autoencoder in which negative feature activations are removed.

    z = ReLU(W_enc x + b_enc)

### TopK SAE

Only the k strongest feature activations are retained.

    z = TopK(W_enc x + b_enc, k)

This forces the SAE representation to contain only a limited number of active features.

### JumpReLU SAE

A feature becomes active only when its activation exceeds a learned threshold.

    z_i = a_i if a_i > θ_i
    otherwise z_i = 0

Comparing these architectures will help determine whether the SAE activation mechanism affects interpretability and causal faithfulness.

---

## Datasets

### OpenWebText

OpenWebText will be used as the primary natural-text dataset for GPT-2 experiments.

The dataset will be passed through GPT-2 to collect transformer activations and identify text examples that strongly activate individual SAE features.

### The Pile

The Pile will provide additional text examples and will support experiments involving Gemma.

It will also provide a broader text distribution for evaluating whether feature behavior remains consistent across different sources.

### Geometry of Truth

Geometry of Truth contains true and false statements.

It will be used to investigate features associated with concepts such as:

- Truth
- Falsehood
- Factual correctness

### Representation Engineering (RepE)

RepE provides contrastive examples representing behavioral directions such as:

- Positive vs. negative sentiment
- Honest vs. dishonest
- Harmful vs. harmless

These contrastive examples will support concept-detection and faithfulness evaluation experiments.

---

# Project Phases

## Phase 1 — Environment, Dataset, and Model Setup

The first phase establishes the reproducible project environment.

Tasks include:

- Configure the development environment.
- Set up the GitHub repository.
- Configure Python and required libraries.
- Load GPT-2 Small.
- Load pretrained SAEs.
- Prepare initial datasets.
- Establish AWS EC2 and S3 connectivity when AWS execution begins.

**Deliverable:** Reproducible environment with models, SAEs, datasets, and project infrastructure available.

---

## Phase 2 — SAE Feature Extraction

Natural text will be processed through the language model and selected residual-stream activations will be collected.

These activations will be passed through pretrained SAEs.

For selected features, the project will record information such as:

- Feature ID
- Activation magnitude
- Activation frequency
- Transformer layer
- Activating token
- Activating text
- SAE architecture

**Deliverable:** Structured SAE feature-activation dataset.

---

## Phase 3 — Feature Interpretation

The highest-activating examples for selected SAE features will be collected.

These examples will be used to generate natural-language interpretations of the features.

Each selected feature will contain:

- Feature ID
- Layer
- SAE architecture
- Highest-activating examples
- Generated feature label
- Label-confidence score

Existing interpretability resources such as Neuronpedia may also be used to support feature investigation.

**Deliverable:** Labeled SAE feature dataset with feature interpretations and confidence scores.

---

## Phase 4 — Causal Steering

Selected SAE features will be deliberately manipulated to determine whether they causally influence language-model behavior.

A selected feature activation can be modified as:

    z'_j = z_j + α

where α represents the steering strength.

Multiple steering strengths will be tested.

Baseline model generations will then be compared with steered generations to determine whether increasing a feature causes the model output to move toward the concept represented by its label.

**Deliverable:** Baseline and steered model generations.

---

## Phase 5 — Concept Detection

Concept-present and concept-absent examples will be used to determine whether SAE features reliably detect their claimed concepts.

The evaluation will include:

- Precision
- Recall
- F1 Score
- AUROC

This phase separates the question:

> "Does this feature detect the concept?"

from:

> "Does this feature causally control the concept?"

**Deliverable:** Feature-level concept-detection evaluation.

---

## Phase 6 — Cross-Condition Faithfulness Analysis

Causal faithfulness will be compared across multiple experimental conditions.

Comparisons will include:

- Transformer layer
- Feature frequency
- Feature sparsity
- SAE architecture
- Label confidence
- Steering strength
- Language model

The analysis will investigate which conditions are associated with stronger or weaker causal faithfulness.

**Deliverable:** Cross-condition faithfulness evaluation table.

---

## Phase 7 — Final Analysis and Benchmark

The final phase will combine all experimental results into a single evaluation framework.

The project will compare:

    Model
        ×
    Layer
        ×
    SAE Architecture
        ×
    Feature
        ×
    Steering Magnitude

The final analysis will investigate when SAE-generated feature labels provide reliable causal explanations of language-model behavior.

**Deliverable:** Final SAE faithfulness benchmark, visualizations, report, GitHub documentation, and presentation.

---

# Evaluation Metrics

The project will use multiple metrics rather than relying on a single measurement.

### SAE Quality

- Reconstruction Loss (MSE)
- Explained Variance
- L0 Sparsity
- Feature Activation Frequency

### Concept Detection

- Precision
- Recall
- F1 Score
- AUROC

### Causal Faithfulness

- Steering Effect
- Causal Faithfulness Score (CFS)
- Steering Success Rate (SSR)

### Label vs. Causal Behavior

The relationship between auto-generated label confidence and measured causal faithfulness will also be investigated.

A proposed **Faithfulness Gap** will measure disagreement between how convincing a feature interpretation appears and how strongly the feature actually affects model behavior.

---

# 7-Week Project Plan

| Week | Phase | Main Tasks | Planned Deliverable |
|---|---|---|---|
| **Week 1** | Phase 1 — Environment + Models | Set up repository and Python environment; install dependencies; load GPT-2 Small and pretrained SAE; prepare AWS EC2/S3 structure and initial datasets. | Working project environment |
| **Week 2** | Phase 2 — SAE Feature Extraction | Inspect transformer residual-stream activations; encode activations using pretrained SAE; extract feature IDs, activation magnitudes, frequencies, and activating examples. | Initial feature-activation dataset |
| **Week 3** | Phase 3 — Feature Interpretation | Select candidate features; collect highest-activating examples; investigate feature interpretations; generate labels and confidence scores; compare with resources such as Neuronpedia. | Interpreted SAE feature dataset |
| **Week 4** | Phase 4 — Causal Steering | Implement feature intervention; establish baseline generations; test multiple steering strengths and selected features. | Baseline vs. steered generations |
| **Week 5** | Phase 5 — Concept Detection | Build concept-present and concept-absent evaluations using controlled examples, Geometry of Truth, and RepE where appropriate. Calculate Precision, Recall, F1, and AUROC. | Concept-detection evaluation |
| **Week 6** | Phase 6 — Cross-Condition Analysis | Compare layers, feature frequencies, sparsity, SAE architectures, label confidence, steering strength, and models. Calculate CFS, SSR, Faithfulness Gap, and related statistics. | Master faithfulness evaluation table |
| **Week 7** | Phase 7 — Final Analysis + Visualization | Complete statistical analysis; compare ReLU, TopK, and JumpReLU; generate figures and tables; build Tableau visualizations; finalize GitHub documentation, report, and presentation. | Final benchmark and project presentation |

---

# Planned Tableau Visualization

During the final stage of the project, experimental results will be prepared for visualization using Tableau.

The Tableau component will provide an interactive way to explore the SAE faithfulness benchmark rather than relying only on static plots and tables.

Potential dashboard views will include:

- Feature activation across concepts
- Concept-present vs. concept-absent activation
- Feature activation frequency
- SAE architecture comparison
- Layer-wise faithfulness comparison
- Steering strength vs. causal effect
- Label confidence vs. causal faithfulness
- Causal Faithfulness Score
- Steering Success Rate
- Faithfulness Gap
- GPT-2 vs. Gemma comparison

The purpose of the Tableau dashboard will be to make relationships between **feature interpretation, feature detection, and causal behavior** easier to understand and communicate.

---

# Planned Technology Stack

### Development

- Python
- VS Code
- Jupyter Notebook
- Git
- GitHub

### Machine Learning

- PyTorch
- NumPy
- Pandas
- Scikit-learn

### Language Models

- GPT-2 Small
- Gemma-2-2B
- Hugging Face Transformers
- TransformerLens

### Sparse Autoencoders

- SAE Lens
- Gemma Scope
- ReLU SAE
- TopK SAE
- JumpReLU SAE

### Analysis and Visualization

- Matplotlib
- SciPy
- Tableau

### Interpretability

- Neuronpedia
- Automatic feature-labeling methods

### Cloud Infrastructure

- Amazon EC2
- Amazon S3
- AWS CLI
- Boto3

---

# Expected Project Outcome

The project is not designed to conclude that Sparse Autoencoders are simply "good" or "bad."

Instead, the goal is to determine **under which conditions SAE-generated feature labels can be trusted as causal explanations of language-model behavior**.

The final project will provide a measurable framework for distinguishing features that appear interpretable from features whose interpretations accurately describe their causal effects on model behavior.
