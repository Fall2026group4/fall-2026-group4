# Off-machine backup plan for local-only GPT-2 assets (2026-11-13; remediation U1). Status: BLOCKED (no destination)

## What exists only on this machine

* A SHA-256 manifest was computed on 2026-11-13 over every `.pt`, `.npy` and `.npz` file and the streamed OWT
  shards/tokens under `results/` and `data/`: **953 files, 26.2 GB**.
* Of these, **434 `.pt` files (15 GB)** are newer than the only backup, `../backups/2026-11-06_pre_migration.tar`.
  That tar is on the **same disk**, so it is not an independent copy.
* Per-folder sizes are in `issue_register.md` §E.
* Git holds every result CSV, so losing the disk would cost:
  * the checkpoints (R2: about 5.5 GPU h to retrain; R4, IMP-L, fine-tunes);
  * the activation caches;
  * every evaluation that needs a checkpoint.

## Why nothing was copied

* No authorised remote destination is configured: there are no AWS credentials or CLI, no rclone or gsutil, and no
  mounted external volume.
* The second NVMe device (`nvme1n1`, 233 GB) is unmounted. On this instance type it is most likely ephemeral instance
  storage, which would not survive a stop. It was not touched.
* Uploading private research assets to an arbitrary service is out of scope without the user's choice and credentials.

## Command to run once a destination is chosen (copy only; nothing is moved or deleted)

From the repository root:

```bash
# 1. manifest (about 3 min)
find results data -type f \( -name '*.pt' -o -name '*.npy' -o -name '*.npz' \
     -o -path '*stream_shards*' -o -path '*stream_tokens*' \) -print0 | sort -z \
  | xargs -0 sha256sum > ../backups/2026-11-13_gpt2_assets.sha256

# 2a. to another machine or external disk over SSH
rsync -a --files-from=<(awk '{print $2}' ../backups/2026-11-13_gpt2_assets.sha256) \
      ./ USER@HOST:/path/sae_backup_2026-11-13/
#  or 2b. to S3 (after `aws configure` with the team's bucket)
# aws s3 cp . s3://BUCKET/sae_backup_2026-11-13/ --recursive --exclude '*' --include 'results/*.pt' ...

# 3. verify ON THE DESTINATION before anything is ever deleted locally
cd /path/sae_backup_2026-11-13 && sha256sum -c 2026-11-13_gpt2_assets.sha256 | grep -v ': OK$'   # must print nothing
```

* Storage needed: about **27 GB**, or about 16 GB if only the 434 post-tar checkpoints and the RepE/GoT caches are
  copied.
* The backup counts as complete only when step 3 prints no failures.
