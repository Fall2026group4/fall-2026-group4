# <Experiment name> (<YYYY-MM-DD>). Status: PLANNED / RUNNING / COMPLETED / STOPPED

Evidence level: exploratory | validation finding | confirmed test result | inconclusive.

## 1. Before implementation

* **Research question:**
* **Hypothesis:**
* **Why it's needed, and how it connects to the main SAE-faithfulness question:**
* **Design:**
  * model and layer(s) (hook);
  * SAE architectures;
  * datasets and splits;
  * seeds;
  * training budget;
  * hyperparameters.
* **Baselines and controls:**
* **Comparisons:**

  | X vs Y | What it establishes |
  |---|---|
  |  |  |

* **Metrics:**
  * Give each one a category: reconstruction, representation preservation, behavioural splicing, causal, or steering.
  * Give each one its direction: higher or lower is better.
* **Pre-registered success criteria and guards:**
* **Leakage-prevention checks:**
  * split hash and disjointness;
  * probes fitted on original activations only;
  * no test data.
* **Files to be created or modified, and why:**
* **Pre-training hashes** (scripts, split file, pre-registration):

## 2. After execution

* **Runs:** completed or failed; runtime; GPU memory; storage.
* **Integrity checks:** reproduction checks; base checkpoints and frozen model unchanged.
* **Results:** mean ± sd, per-seed values where criteria are paired, and figure paths.
* **Improvements and regressions:**
* **Criteria met or not met:** per criterion, with the numbers.
* **Limitations:**
* **What was learned, and what remains unresolved:**
* **Files created, modified and superseded:**
* **Git commit:**

## 3. If stopped early

* Current status:
* Existing artifacts:
* Unfinished work:
* Exact next steps:
