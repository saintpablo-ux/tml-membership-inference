# TML Task 1 - Membership Inference Attack

This repository contains the code needed to reproduce our best leaderboard submission.

## Required files

Place the following assignment-provided files in this folder:

- pub.pt
- priv.pt
- model.pt

The file reference_lira_loss_cache.npz is included and contains the cached reference-model loss scores used by our final attack.

## Method

The final attack uses:

- negative cross-entropy loss as the base membership signal
- 32 reference models trained on random 50% subsets of pub.pt + priv.pt
- classwise Gaussian LiRA scoring
- RMIA-style per-sample normalization
- true RMIA-style same-class population comparison
- final blend: 87.5% LiRA/RMIA score + 12.5% population RMIA score with gamma = 1.05

## Reproduce submission

Run:
```bash
python make_submission.py
```

This creates:
```text
submission.csv
```

To submit using the provided template:
```bash
python task_template.py
```

## Optional: rebuild the reference cache

Rebuilding is **not required** because `reference_lira_loss_cache.npz` is already included.

The file `reference_lira_loss_diagnostic.py` is included for optional cache rebuilding. This script was used to train the 32 reference models and create `reference_lira_loss_cache.npz`.

To recreate the cache from scratch, delete the existing cache file and run:

```bash
python reference_lira_loss_diagnostic.py
