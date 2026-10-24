# Geometry of Truth — Final Analysis Summary

## Project Context

This analysis evaluates whether Sparse Autoencoder (SAE) features associated with true and false factual statements are only correlated with truth labels or are also causally faithful to the underlying truth representation in GPT-2 Small.

The Geometry of Truth dataset used here contains paired factual statements about cities and countries.

---

## Stage 0 — Data Preparation

The Geometry of Truth `cities.csv` dataset contained:

- 1,496 statements
- 748 TRUE statements
- 748 FALSE statements
- 748 unique cities

Each city appears once with a true statement and once with a false statement.

To prevent city leakage between training and testing, the dataset was split by **city**, not by individual statement.

Using seed `42`:

- Training set: 1,046 statements
  - 523 TRUE
  - 523 FALSE

- Test set: 450 statements
  - 225 TRUE
  - 225 FALSE

The split was stored locally and backed up to AWS S3.

---

## Stage 1 — GPT-2 Activation Caching

Model:

- GPT-2 Small
- 12 transformer layers
- Residual-stream activations
- Last-token representation
- Hidden dimension: 768

For every statement, activations were cached for Layers 0–11.

Shapes:

- Train activations: `(1046, 12, 768)`
- Test activations: `(450, 12, 768)`

The activation files were stored in AWS S3 for reproducibility.

---

## Stage 2 — Baseline Truth Probes

A separate linear probe was trained for each GPT-2 layer.

Probe configuration:

- `StandardScaler`
- `LogisticRegression`
- Training split used for fitting
- Held-out city-level test split used for evaluation
- Seed: `42`

Performance increased substantially in deeper GPT-2 layers.

The best layer was Layer 11:

- Accuracy: `0.7778`
- AUROC: `0.8635`

Layer 10, selected for the subsequent SAE analysis, achieved:

- Accuracy: `0.7689`
- AUROC: `0.8528`

This confirmed that GPT-2 residual activations contain a strong linearly decodable truth/falsity signal.

---

## Stage 3 — SAE Encoding

The pretrained SAE catalogue available through SAE Lens was inspected rather than assuming SAE architectures from release names.

For the controlled Geometry analysis, the selected SAE was:

- Release: `gpt2-small-res-jb`
- Model: GPT-2 Small
- Hook: `blocks.10.hook_resid_pre`
- SAE class: `StandardSAE`
- Activation function: ReLU
- Input dimension: 768
- SAE dimension: 24,576
- Training corpus: OpenWebText

The layer-10 residual activations were encoded into SAE features.

Shapes:

- Train SAE features: `(1046, 24576)`
- Test SAE features: `(450, 24576)`

Average active features per statement:

- Train: approximately `48.58`
- Test: approximately `48.50`

This corresponds to roughly 0.20% of SAE features being active per example.

---

## Stage 4A — SAE Feature Detection

Candidate SAE features were selected using the training split only.

The held-out test set was not used during feature selection.

The strongest training-selected feature was:

- Feature ID: `19579`
- Training detection AUROC: approximately `0.6989`
- Training polarity: `-1`

A polarity of `-1` means that larger activation was associated with FALSE statements, while reducing the feature corresponds to the TRUE direction.

On the held-out test set, Feature 19579 achieved:

- Test AUROC: approximately `0.6891`

Therefore, the feature association generalized beyond the training cities.

However, this result establishes predictive association, not causal faithfulness.

---

## Stage 4B — SAE Reconstruction Faithfulness

To test whether the SAE preserves the underlying truth representation, SAE feature activations were decoded back into Layer-10 residual space.

A linear truth probe using the same configuration as the raw residual baseline was trained and evaluated on the SAE-reconstructed residuals.

Results:

### Raw Layer-10 Residual

- Accuracy: `0.7689`
- AUROC: `0.8528`

### SAE-Reconstructed Residual

- Accuracy: `0.7556`
- AUROC: `0.8508`

### Faithfulness Gaps

- Accuracy gap: `0.0133`
- AUROC gap: `0.0021`

The extremely small AUROC gap shows that the SAE reconstruction preserved almost all of the linearly decodable truth signal.

This supports strong **representational faithfulness of the SAE reconstruction**.

It does not by itself prove that any individual SAE feature is uniquely causal.

---

## Stage 5 — Causal Steering

### Steering Implementation

The Geometry experiment reused the steering convention already merged and tested by the team.

The intervention operates on the last-token residual representation:

`residual = residual + steering_strength × steering_direction`

For an SAE feature, the steering direction is the corresponding decoder vector:

`sae.W_dec[feature_id]`

Feature `19579` was locked before causal testing because it was the top feature selected using training data only.

Its polarity was `-1`, therefore:

- decreasing Feature 19579 represents steering toward TRUE
- increasing Feature 19579 represents steering toward FALSE

A fixed Layer-10 truth probe was used as the causal outcome measure.

---

## Stage 5A — Steering Dose Response

Feature 19579 produced a clear monotonic probe-score response.

Mean truth-probe score shifts:

| Multiplier | Feature 19579 |
|---:|---:|
| -2 | -0.3244 |
| -1 | -0.1622 |
| 0 | 0.0000 |
| +1 | +0.1622 |
| +2 | +0.3244 |

The effect was symmetric and approximately linear.

A probe-direction intervention was included as a positive control and produced much larger score changes, confirming that the fixed truth probe responded correctly to truth-directed perturbations.

---

## Stage 5B — Random SAE Feature Controls

A single random SAE feature was not sufficient to establish whether Feature 19579 was unusually causal.

Therefore, 100 random SAE features were sampled using seed `42`.

Important controls:

- Top training-selected features were excluded.
- Random features received their own polarity derived from training data only.
- Intervention magnitudes were matched to the candidate feature in residual-space L2 norm.
- Held-out test data was used only to measure the causal response.

Feature 19579 truth-directed probe shift:

- `0.162199`

Random-control results:

- Number of random controls: `100`
- Random mean shift: `0.004919`
- Random 95th percentile: `0.496438`
- Candidate percentile: `66.0%`
- Signed empirical p-value: `0.3465`
- Absolute empirical p-value: `0.6535`

Feature 19579 therefore did not produce an intervention effect that was statistically unusual relative to random SAE decoder directions.

---

## Behavioral TRUE/FALSE Prompt Check

A direct GPT-2 behavioral scoring approach was also investigated.

The planned deterministic score was:

`logit(" TRUE") - logit(" FALSE")`

However, GPT-2 Small did not reliably perform the explicit TRUE/FALSE classification task.

Initial held-out prompt performance was approximately chance:

- Test AUROC: `0.5005`

Several prompt formats were evaluated using training data only.

Best training AUROC:

- Simple statement prompt: `0.5740`

Few-shot prompting achieved:

- Training AUROC: `0.5567`

Because the explicit behavioral classifier remained weak, this output was not used as the primary causal-faithfulness measure.

The causal experiment therefore used the established Layer-10 internal truth probe instead.

---

## Main Findings

The Geometry of Truth analysis produces three distinct conclusions.

### 1. Truth Information Exists in GPT-2 Residual Representations

Linear probes achieve strong held-out discrimination between true and false statements.

Best observed raw residual AUROC:

`0.8635`

### 2. The SAE Preserves the Truth Representation

Passing Layer-10 activations through the SAE encode/decode bottleneck reduced AUROC by only:

`0.0021`

This provides strong evidence that the SAE reconstruction is representationally faithful to the truth signal.

### 3. The Best Truth-Associated SAE Feature Is Not Uniquely Causal

Feature 19579:

- generalized predictively to held-out data
- caused a clean dose-dependent truth-probe shift
- moved the probe in the expected TRUE/FALSE direction

However, its causal effect ranked only at the `66th percentile` among 100 random SAE steering controls.

Its empirical p-values were not significant.

Therefore, the analysis does **not** support the claim that Feature 19579 is a uniquely causal truth feature.

---

## Interpretation

The Geometry results demonstrate an important distinction between:

1. **Detectability**
2. **Representational faithfulness**
3. **Feature-specific causal faithfulness**

The SAE representation can preserve truth information extremely well while an individually interpretable or strongly correlated feature may still fail to demonstrate exceptional causal control.

This supports the broader project motivation:

> SAE feature labels or associations should not automatically be interpreted as evidence that individual SAE features are causally responsible for the corresponding concept.

---

## Key Challenges and Decisions

Several methodological issues were encountered during the analysis.

### SAE Architecture Availability

The pretrained SAE catalogue was inspected directly.

A same-layer, same-hook ReLU/TopK/JumpReLU comparison was not available for GPT-2 Layer 10 `hook_resid_pre`.

To avoid confounding SAE architecture with hook location, the primary controlled Geometry analysis used the compatible ReLU SAE.

### Test Leakage Prevention

Feature ranking and polarity were calculated using training data only.

The held-out test split was used only for final evaluation.

### Steering Reproducibility

The analysis reused the team's merged steering implementation and deterministic steering conventions.

### Random-Control Specificity

A single random feature initially produced a noticeable probe shift.

This motivated the stronger 100-random-feature null-distribution analysis.

### GPT-2 Behavioral Classification

GPT-2 Small was weak at explicit TRUE/FALSE next-token classification.

Rather than forcing a behavioral result, the analysis used the stronger and previously validated internal truth probe as the causal outcome.

---

## Generated Outputs

Main result tables:

- `baseline_probe_results.csv`
- `sae_encoding_summary.csv`
- `sae_top_features_train.csv`
- `sae_feature_test_results.csv`
- `sae_faithfulness_results.csv`
- `sae_steering_results.csv`
- `sae_random_control_results.csv`
- `sae_random_control_summary.csv`

Main vector figures:

- `geometry_baseline_probe_performance.svg`
- `geometry_sae_faithfulness.svg`
- `geometry_steering_dose_response.svg`
- `geometry_random_control_null.svg`

---

## Current Conclusion

For the Geometry of Truth stream:

**Detection:** Supported.

**SAE reconstruction faithfulness:** Strongly supported.

**Feature 19579 predictive generalization:** Supported.

**Feature 19579 causal directional effect:** Supported.

**Feature 19579 uniquely or exceptionally causal relative to random SAE directions:** Not supported.

The Geometry stream therefore provides evidence that SAE representations can faithfully preserve a conceptual signal without guaranteeing that a strongly associated individual SAE feature is uniquely causally faithful to that concept.