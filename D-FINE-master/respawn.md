# Respawn Notes

Last updated: 2026-04-13

## Goal

This file is the quick-resume note for the current TrustKD / TrustLSD-style D-FINE experiments.

- Proposal document: `/home/com_2/suan/TOD/TrustKD-DETR.md`
- Current completed run config: `/home/com_2/suan/TOD/D-FINE-master/D-FINE-master/configs/dfine/custom/dfine_hgnetv2_s_aitod.yml`
- Current completed run log: `/home/com_2/suan/TOD/D-FINE-master/D-FINE-master/output/D-Fine_S_filter_trust_weight/log.txt`
- Current baseline log to use: `/home/com_2/suan/TOD/D-FINE-master/D-FINE-master/output/dfine_s_baseline_filtering/log.txt`

The purpose of the current stage is not to implement the full proposal yet. The current stage is to verify whether the minimal learned trust-weight extension helps under the filtered tiny/small protocol.

## Research Framing

The proposal in `TrustKD-DETR.md` is broader than the code currently trained.

Important framing:

- D-FINE already has localization quality estimation and GO-LSD-style localization self-distillation.
- Because of that, the paper claim cannot just be "we add reliability weighting."
- The defensible novelty is a tiny-object-aware trust mechanism and possibly stronger teacher construction for tiny objects.

## Current Protocol

The current experiments are no longer using the old unrestricted sample protocol.

Active protocol:

- Sample AI-TOD setup is still intentional for fast iteration.
- Training GT is filtered to `tiny` and `small`.
- Validation GT is filtered to `tiny` and `small`.
- Images with no remaining retained GT are excluded from train and val.
- Prediction boxes are **not** filtered by predicted area.

Equivalent retained GT size range:

- `[8^2, 32^2)`, meaning `tiny` + `small`

Primary metrics under this protocol:

- `mAP@50:95`
- `AP50`
- `AP75`
- `AP_tiny`
- `AP_s`

Do not use these old metrics as decision targets for the current protocol:

- `AP_very_tiny`
- `AP_m`
- `AP_l`

Also important:

- Older unrestricted logs such as `output/dfine_s_baseline/log.txt`, `output/D-Fine_S_smaple/log.txt`, and `output/D-Fine_S_trust_weight/log.txt` are now historical context only.
- They are not numerically comparable to the current filtered protocol.

## Current Implementation Scope

The currently completed run is the minimal trust-weight stage, not the full TrustKD proposal.

What is implemented now:

- `DFINETransformer` contains a train-time trust head.
- Trust is supervised only on matched queries.
- Trust target is continuous:
  `trust_alpha * IoU + (1 - trust_alpha) * center_term`
- Localization KD for matched queries can use predicted trust scores.
- Final-layer teacher is still unchanged.

What is **not** implemented in the completed run:

- no scale-aware trust weighting
- no matched-only KD reweight config from the trust subfolder
- no teacher mixture across decoder layers
- no hard trust mask
- no confidence-only or entropy-only heuristic baseline
- no transfer study

## Active Config Snapshot

Current completed config:

- File: `/home/com_2/suan/TOD/D-FINE-master/D-FINE-master/configs/dfine/custom/dfine_hgnetv2_s_aitod.yml`
- Output dir: `./output/D-Fine_S_filter_trust_weight`

Key settings in that config:

- `DFINETransformer.trust_enabled: True`
- `DFINETransformer.trust_topk: 4`
- `DFINETransformer.trust_hidden_dim: 64`
- `DFINECriterion.weight_dict.loss_trust: 1.0`
- `DFINECriterion.losses: ['vfl', 'boxes', 'local', 'trust']`
- `DFINECriterion.trust_alpha: 0.5`
- `DFINECriterion.trust_eps: 1.0e-6`
- `train_dataloader.dataset.allowed_area_labels: ['tiny', 'small']`
- `val_dataloader.dataset.allowed_area_labels: ['tiny', 'small']`
- `train_dataloader.dataset.exclude_images_without_valid_annotations: True`
- `val_dataloader.dataset.exclude_images_without_valid_annotations: True`

Related code locations:

- trust head: `/home/com_2/suan/TOD/D-FINE-master/D-FINE-master/src/zoo/dfine/dfine_decoder.py`
- trust loss / KD weighting: `/home/com_2/suan/TOD/D-FINE-master/D-FINE-master/src/zoo/dfine/dfine_criterion.py`
- filtered dataset protocol: `/home/com_2/suan/TOD/D-FINE-master/D-FINE-master/src/data/dataset/area_filter.py`

## Current Baseline

Use this baseline for the current protocol:

- `/home/com_2/suan/TOD/D-FINE-master/D-FINE-master/output/dfine_s_baseline_filtering/log.txt`

File state:

- 300 log lines
- last epoch: 299

Best metrics:

- `mAP@50:95`: epoch 288, `0.214030`
- `AP50`: epoch 296, `0.469210`
- `AP75`: epoch 287, `0.154222`
- `AP_tiny`: epoch 289, `0.175111`
- `AP_s`: epoch 289, `0.281033`

Last epoch metrics:

- `mAP@50:95`: `0.207484`
- `AP50`: `0.466139`
- `AP75`: `0.145365`
- `AP_tiny`: `0.168804`
- `AP_s`: `0.276072`

## Current Completed Experiment

Current completed experiment log:

- `/home/com_2/suan/TOD/D-FINE-master/D-FINE-master/output/D-Fine_S_filter_trust_weight/log.txt`

This is the filtered-protocol minimal trust-weight run that corresponds to the current active config.

File state:

- 300 log lines
- last epoch: 299

Best metrics:

- `mAP@50:95`: epoch 293, `0.214885`
- `AP50`: epoch 255, `0.491472`
- `AP75`: epoch 293, `0.168967`
- `AP_tiny`: epoch 277, `0.175616`
- `AP_s`: epoch 259, `0.284889`

Last epoch metrics:

- `mAP@50:95`: `0.212186`
- `AP50`: `0.472602`
- `AP75`: `0.165400`
- `AP_tiny`: `0.174099`
- `AP_s`: `0.278920`

Training signal note:

- `train_loss_trust` decreased from `0.044039` at epoch 0 to `0.009869` at epoch 299

Best-to-best delta vs current baseline:

- `mAP@50:95`: `+0.000855`
- `AP50`: `+0.022262`
- `AP75`: `+0.014745`
- `AP_tiny`: `+0.000505`
- `AP_s`: `+0.003856`

Current interpretation:

- Under the filtered `tiny+small` protocol, the minimal trust-weight run is slightly better than the current baseline on every tracked metric.
- This is the main current result.
- Older notes about `AP_very_tiny` collapse belong to the pre-filter protocol and should not be used to summarize the current phase.

## Next Experiment To Run

The prepared next run is the scale-aware matched-KD version:

- Config: `/home/com_2/suan/TOD/D-FINE-master/D-FINE-master/configs/dfine/custom/dfine_hgnetv2_s_aitod_trust_scaleaware_matchedkd.yml`
- Output dir: `./output/D-Fine_S_filter_trust_scaleaware_matchedkd`

This config adds:

- `trust_scale_weight_very_tiny: 2.5`
- `trust_scale_weight_tiny: 1.75`
- `trust_scale_weight_small: 1.25`
- `trust_unmatched_kd_weight: 0.0`

Meaning:

- keep the same base config
- bias trust / KD more toward smaller objects
- make localization KD matched-query-focused

Status:

- config is prepared
- this run has not been trained yet in the current note state
- it should be launched as a fresh run, not as a resume from `D-Fine_S_filter_trust_weight`

## Resume Checklist

If a new chat needs to resume quickly, the key facts are:

- use `output/dfine_s_baseline_filtering/log.txt` as the current baseline
- use `output/D-Fine_S_filter_trust_weight/log.txt` as the current completed experiment
- current protocol is GT-only filtering to `tiny + small`
- current completed result is mildly positive across all tracked filtered metrics
- active completed config is `configs/dfine/custom/dfine_hgnetv2_s_aitod.yml`
- next config to run is `configs/dfine/custom/dfine_hgnetv2_s_aitod_trust_scaleaware_matchedkd.yml`
- do not summarize the current stage using old unrestricted-protocol `AP_very_tiny` results
