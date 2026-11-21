# Plain-Language Project Summary

## The question

Inside a language model, each word is represented at every layer by a long list of numbers called an "activation". A sparse autoencoder (SAE) rewrites an activation as a combination of a few simpler building blocks that researchers try to interpret, and it can rebuild, or "reconstruct", the activation from them.

SAEs are usually judged by reconstruction accuracy. We asked whether an SAE can reconstruct an activation accurately while still losing information or changing how the language model behaves.

A "faithful" reconstruction does more than look numerically similar: it keeps the information and effects that the language model actually uses.

## What we did

We tested faithfulness at four levels:

- **Reconstruction:** how close the rebuilt activation is to the original.
- **Information:** whether task information, such as whether a sentence is true, can still be recovered.
- **Behaviour:** whether swapping in the rebuilt activation changes the model's next-word predictions.
- **Causal effect:** whether the rebuilt activation keeps a specific effect the model really uses.

In GPT-2 Small, we trained 108 SAEs across all 12 layers, using three common designs (ReLU, TopK and JumpReLU). Our data were web text (OpenWebText), true and false statements about cities (Geometry of Truth), and instructions asking the model to act honestly or dishonestly, rewritten so both versions have equal length (originally, length alone gave the answer away).

We then tried to improve the SAEs with more data, greater width and behaviour-focused training. Finally, we checked the behavioural result on a larger model, Gemma-2-2B, with public SAEs (Gemma Scope) and different text (The Pile).

## Main findings

**1. Good reconstruction does not guarantee that useful information is kept.** Within a layer, SAEs that reconstructed better also disturbed the model's predictions less. But reconstruction accuracy did not reliably predict how much true-or-false information was lost.

**2. How you test for information matters.** A "probe" is a simple classifier that reads information, such as truth, from an activation. A probe trained on original activations and applied unchanged to rebuilt ones suggested heavy loss. Same-sized random noise caused part of that drop, so the unchanged probe is sensitive to any disturbance; but on the true/false statements at the middle and later layers, a third to a half of the drop comes from SAE errors lined up with the truth signal. Re-training the probe on the rebuilt activations showed a smaller, but real, loss. For the honesty instructions, the loss was genuine.

We also compared SAEs with PCA, a standard compression method. With equal reconstruction error, PCA often kept more task information, but used several times more dimensions. Limited to a similar number of active dimensions, no trained SAE did worse and some did better. So "SAE versus PCA" depends on what the comparison holds equal.

**3. Reconstructions keep some causal effects better than others.** The honest-or-dishonest instruction changes the model's next-token output; we first located where GPT-2 carries this effect. The rebuilt activation kept about 93 to 97 percent of the effect at an early layer, but only about 62 to 68 percent at later layers. This shows how much of the effect the reconstruction preserves, not that the model reasons through the SAE's building blocks.

**4. Better training improves some kinds of faithfulness, but not all.** Wider SAEs trained on far more data reconstructed better and damaged predictions less, and this held on data set aside for a final check. Truth information showed no confirmed improvement. Behaviour-focused training reduced the damage further, but slightly worsened reconstruction and, at one layer, kept less of the causal effect.

On Gemma-2-2B, SAE reconstruction errors also harmed next-word predictions more than random errors of the same size, at all three layers tested. This supports the broad behavioural finding in a second model, but says nothing about information or causal results in Gemma, which we did not test.

## What it means

Reconstruction quality is useful and necessary, but it is not enough by itself to show that an SAE faithfully preserves what a language model uses.

Future SAE evaluations should measure reconstruction, information preservation, behaviour and causal preservation separately.

The lesson is not that SAEs are bad. It is to judge an SAE with several tests, not one reconstruction score. A further study of Gemma SAE size and sparsity is still in progress.
