# Simbanegavi Onboarding Notes

## Project Title
SAE-Faithful: Causal Faithfulness of Sparse Autoencoder Feature Labels

## Project Goal
The project investigates whether automatically generated labels for Sparse Autoencoder (SAE) features are actually causally faithful to the behavior of large language models.

In simple terms, the project asks:

> If an SAE feature is labeled as representing a concept, does increasing or steering that feature actually cause the model to produce behavior related to that concept?

## Main Models
The project uses:

- GPT-2-small
- Gemma-2-2B

GPT-2-small is the smaller model and is useful for controlled experiments. Gemma-2-2B is larger and provides a more modern comparison.

## Sparse Autoencoder Architectures
The project compares several SAE architectures:

- ReLU SAE
- TopK SAE
- JumpReLU SAE

For a controlled architectural comparison, the same GPT-2-small layer can be used, with only the SAE architecture changed.

## Main Data Resources
The project uses different resources for different purposes:

- OpenWebText
- The Pile
- Geometry of Truth
- Representation Engineering concept datasets
- A held-out model/SAE set for generalization testing

The text corpora are used to find examples that strongly activate SAE features, while the concept datasets are used to evaluate whether those features correctly detect and represent concepts.

## Core Evaluation Questions

The project tests:

1. Whether confident SAE labels are actually causally faithful.
2. Whether SAE features perform as well as or better than supervised probes.
3. Whether faithfulness changes with layer depth or feature sparsity.
4. Whether SAE architecture affects causal faithfulness.

## Key Evaluation Metrics

Possible metrics include:

- Precision
- Recall
- AUROC
- Steering faithfulness scores
- Human validation agreement

AUROC measures how well a feature separates concept-present examples from concept-absent examples.

## My Planned Contribution

My initial focus will be on:

- Benchmarking and experiment tracking
- Evaluation of model outputs
- Data analysis
- Visualization of results
- Comparison of SAE architectures
- Layer and sparsity analysis
- Documentation
- Presentation support
- GitHub repository maintenance

## Initial Understanding of the Workflow

Text data  
→ GPT-2-small / Gemma  
→ residual stream activations  
→ Sparse Autoencoder  
→ hidden feature  
→ auto-generated feature label  
→ steering intervention  
→ evaluate whether the label is faithful

## Immediate Next Steps

- Review the existing repository
- Understand what the other group members have already completed
- Confirm individual responsibilities
- Review the available datasets and model checkpoints
- Begin contributing to benchmarking, evaluation, and visualization
