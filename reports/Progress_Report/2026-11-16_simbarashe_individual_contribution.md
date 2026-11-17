# Simbarashe Simbanegavi — Individual Contribution Explanation

My main contributions were the Method A multi-layer extension and the extension of our SAE faithfulness study to
Gemma-2-2B.

For the Method A extension, I tested whether the approach generalised across additional layers. On branch
`simba-method-a-multilayer` (results commit `7f2557b`), I trained new runs at layers 5 and 9 and reused the existing
layer-6 results. This was exploratory validation work. Only 1 of 9 cells met the success criterion, and the layer-9
pipeline disagreed with the final IMP-L pipeline. Because that discrepancy was not resolved before the technical
freeze, this extension was excluded from the final scientific scope and its results are not used as final-paper
evidence.

For the Gemma experiment, I used public Gemma Scope SAEs at three layers and tested what happened when their
reconstructed activations were placed back into Gemma-2-2B. At all three layers, SAE reconstruction error caused more
next-token prediction damage than a random error of the same size. This supported our broader finding that
reconstruction quality alone does not fully describe behavioural faithfulness.

My main takeaway is that an SAE should not be judged only by how accurately it reconstructs activations. We also need
to test whether it preserves task information and model behaviour.
