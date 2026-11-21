# TSP-VERIFY: independent verification of the TSP pilot (2026-11-14). Status: COMPLETED (VALIDATION; analysis only)

**Evidence level:** VALIDATION, post-hoc verification. No training was run and no TEST row was read. Nothing here
changes the pre-registered pilot verdict (NOT SUPPORTED, `../pilot/analysis/verdict.json`). The pilot folder was not
modified.

Full report, with interpretation, power analysis and the replication design:
`docs/2026-11-14_tsp_verification_and_replication_design.md`.

## Question

Are the pilot's numbers and statistics correct? And what do probe-noise, geometry, wording, transfer and power
checks say about why the result was inconclusive?

## Design

* Script: `scripts/geometry_of_truth/28_tsp_independent_verification.py`. Its SHA-256 is recorded in
  `verification_info.json`.
* It imports none of the pilot's evaluation code. It has its own GoT loader, TopK forward pass, sklearn probes with
  grouped C selection, grouped bootstrap (2,000 resamples of validation cities, shared by all SAEs), GPT-2 splice
  hook and statistics.
* Sources: the same 18 L8 SAEs as the pilot (R2 base, R4 MSE, and TSP ctrl / tsp / rand / pca, × 3 seeds).
* Data:
  * GoT geographic: validation 448 statements / 112 cities; group-B probes on 1,048 rows.
  * GoT numerical: validation 592 / 148 groups; group B 1,388.
  * OWT validation cache: 12,700 tokens; 12,600 spliced predictions.
* Runtime 1,966 s on the A10G (tmux session `tsp-verify`). The log is in `../verification_logs/`, gitignored.

## Outputs

| File | Contents |
|---|---|
| `per_sae.csv` | All re-computed metrics per SAE, plus checkpoint SHA-256 |
| `agreement_with_pilot.csv` | Max / mean absolute difference from the pilot's `got_independent.csv` and `ladder_layer8.csv` |
| `comparisons.csv` | Paired seed comparisons for 8 arm pairs × all metrics: the pre-registered se and t(2), plus Satterthwaite df, seed-only and statement-only p (post hoc) |
| `primary_recomputed.csv` | The frozen decision rule on the re-computed values |
| `power.csv` | Simulated pass probability by number of seeds and validation groups |
| `bootstrap.npz` | Bootstrap replicates (per SAE, per metric) |
| `verification_info.json` | Counts, original AUROCs, subspace geometry, runtime |

## Results

1. **Agreement.**
   * AUROC gaps agree within 0.0003, and the 3-net MLP within 0.0046 (MLP noise).
   * NMSE, L0, ΔCE and KL agree within 1e-5.
   * The pilot's numbers are correct.
2. **Rule reproduced.** tsp − ctrl:
   * frozen −0.039 [−0.085, +0.008], Holm p 0.140;
   * re-fit −0.020 [−0.058, +0.018], Holm p 0.151;
   * 3/3 seeds on both. NOT SUPPORTED is confirmed.
3. **Statement sampling dominates the uncertainty:** 95 % of the re-fit variance and 65 % of the frozen variance.
   * Post hoc, with Satterthwaite df: p = 0.023 (re-fit) and p = 0.0025 (frozen).
   * Not pre-registered, so not a result.
4. **The pilot's secondary CIs for error-only AUROC, MLP, sparse-z and kNN omitted statement sampling.**
   * Error-only AUROC: pilot [−0.154, −0.044], p 0.016. With statement sampling: [−0.189, −0.009], p 0.042.
5. **Construct validity.** The frozen group-B probe direction has 58 % of its norm² inside S. Chance is 1 %; the
   random subspace holds 1.0 % and PCA 0.01 %.
6. **Specificity holds for the frozen readout only.**
   * tsp − rand −0.079 (p 0.028).
   * On the re-fit logistic gap, tsp − rand and tsp − pca are −0.025 and −0.027 (p ≥ 0.12).
   * On the MLP: 2/3 seeds.
7. **MLP reader noise.** With larger reader ensembles, tsp − ctrl on the MLP gap is −0.013 (10 nets, 3/3) and
   −0.016 (5 × 256 units, 3/3). The pilot's 3-net reader read −0.010.
8. **No transfer to GoT numerical** (new template and entity type): tsp − ctrl frozen +0.014 (1/3 seeds), re-fit
   +0.004 (2/3); n.s.
   * The numerical probe direction has 1.3 % of its norm inside S, which is chance.
   * Caveat: numerical has a surface shortcut, and all SAEs reconstruct it poorly (NMSE 3.4–3.8).
9. **Wording.** A logistic probe fitted on affirmative group-B rows scores AUROC 0.16 on negated validation rows
   (inverted). There is no usable held-out wording inside cities.
10. **Cost.**
    * tsp − ctrl: OWT NMSE 0.0000, ΔCE +0.0004 [−0.0015, +0.0023], KL 0.0000.
    * ctrl − R2 base: OWT NMSE +1.6 % (CI excludes 0), KL +1.6 % (p 0.030), ΔCE +1.9 % (p 0.063).
11. **Enhancement.** tsp's re-fit gap at C = 0.01 is −0.017 ± 0.011. With strong regularisation x̂ is more linearly
    separable than x.
12. **Power.** Under the frozen rule, the probability of passing at the observed effects was 0.02 (frozen) and
    < 0.001 (re-fit). With 112 validation cities, even 15 seeds give 0.43 for the re-fit metric.

## Limitations

* M4 was checked from the pilot's raw ladder rows, not re-implemented.
* The numerical domain had been seen in earlier diagnostics, and its truth signal is partly a shortcut.
* All post-hoc p values are exploratory.
