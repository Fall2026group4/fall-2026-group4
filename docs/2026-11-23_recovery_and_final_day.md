# Instance recovery and final-day execution (2026-11-23). Status: IN PROGRESS

## 1. Recovery (COMPLETED, verified)

* **What happened.** The original EC2 instance became unavailable on 2026-11-22, after the TF32 queue finished
  (15:38 UTC). Its root EBS volume (`vol-09f3ee28a9577eee3`) was attached read-only to a replacement instance and
  mounted with `ro,noload`.
* **Copy.**
  * `sae-project/` (including `venv/` and `backups/`) and the Hugging Face cache were copied to the new root EBS:
    120.7 GB, 0 rsync errors, 86,226 entries on each side; a final dry-run found no differences.
  * HF credentials were not copied.
  * Records: `../../recovery_2026-11-23/` relative to this repository (outside it).
* **Checkpoints.** 897/897 registered checkpoints match `docs/checkpoint_registry.csv` SHA-256, checked on both the
  source volume and the copy.
* **Other verification.**
  * XB and A4 `sum_x.npz`: 10/10 match the committed hashes.
  * HF cache: 62/62 large files are identical to the source.
* **Git.**
  * Repository state preserved: a full bundle (716 MB) and the old refs under `refs/recovery/old-instance/*` in the
    GitHub clone `~/fall-2026-group4`.
  * This working repository is unchanged: `sae-research` at `b0f3e84`, the local `final-week/p*` branches, and the
    accidental commit kept as `recovery/accidental-commit-05f6d68`.
* **Environment.** `../venv` works on the new instance: Python 3.14.4, torch 2.14.1+cu130, CUDA on the A10G.
* **Lost (instance storage, `/mnt/scratch`, not recoverable):**
  * TI-2's 32 checkpoints and TI-4's 18 new checkpoints (their SHA-256 values are in the ledgers);
  * the downloaded Gemma Scope 32k/65k ladder weights (re-downloadable).
* **Storage rules from now on:**
  * `/mnt/scratch` is not configured; the user deferred formatting the instance-store disk.
  * Every output goes to EBS.
  * GPU queue `logs_xm/gpu_queue.sh`: no job starts below 60 GB free; a running job is stopped below 42 GB.
  * Free space at the start: 134 GB.
* **Old volume.** Still attached and mounted read-only. Do not reboot while it is attached, because it has the same
  root UUID and label as the new root disk. Snapshot it before detaching (the snapshot could not be verified from the
  instance: no AWS CLI).

## 2. Work finished after the last GitHub push, now documented

| Item | State before recovery | Done today | Status |
|---|---|---|---|
| TI-2 TSP-MF | trained + evaluated; analysis crashed (`allow_pickle`) | analysis re-run through the frozen script with one declared deviation (`34b_…`) | **COMPLETED, VALIDATION: NO TRANSFER DETECTED** (MF-H1 +0.0020 [−0.0096, +0.0137]; MF-H4 met; cost met) |
| TI-4 TSP×KL | trained, evaluated, analysed (15:38) | reviewed against the frozen rules; all checks reproduce | **COMPLETED, VALIDATION: COMBINES** (F-H1 −0.026, F-H2 −0.012, Holm 6.5e-4; F-H3 INCONCLUSIVE; costs met) |
| TI-3 minimal pair | gate held (TF32), never trained | none (not re-run, as instructed) | **WITHDRAWN** under its pre-registered schedule rule ("withdrawn if it cannot start by 14:30 UTC"; queue log 15:38:26 `TI3_WITHDRAWN`). No result exists. |
| Ladder batch 2 (option R, L12) | XB Pile shared-span at L12 DONE for 16 SAEs (37 min); XT-G arm started, verified 11 downloads, then died with the instance | none | **PARTIAL.** Missing: (1) the XT-G truth arm for 15 ladder SAEs at L12 (needs the 9.6 GB 32k/65k re-download); (2) the XB b2 analysis. Optional (Phase 4). |

## 3. Final-day experiments (user-approved scope: XC-3, XC-2; XC-1, XF and M4 deferred)

| ID | Pre-registration | State |
|---|---|---|
| XC-3 steering parity | `results/cross_model/xc3_steering_parity/preregistration.json` (frozen 2026-11-23 ~14:40 UTC) | queued |
| XC-2 Gemma GoT causal | `results/cross_model/xc2_gemma_got_causal/preregistration.json` | smoke |
| XC-1, XF, M4 | not written | **NOT RUN**: deferred by the user on 2026-11-23 |

## 4. Git placement (open question for the user)

* The working tree is on `sae-research` at `b0f3e84`, but the final-week PR stack lives on local `final-week/p2…p16`
  branches. The previous session wrote to those through temporary worktrees, which no longer exist.
* p4–p10 (A4, E1, ladder, TI-2, TI-4, TI-3, PCA-CM-T) are not ancestors of p16.
* Today's files are therefore written to the working tree and **not committed** until the user decides where they
  belong.
