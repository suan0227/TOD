# Respawn Notes

Last updated: 2026-04-05

## Goal

This file records the discussion and experiment status so the next experiment can resume cleanly.

The project direction is based on:

- Proposal: `/home/com_2/suan/TOD/TrustKD-DETR.md`
- Active config: `/home/com_2/suan/TOD/D-FINE-master/D-FINE-master/configs/dfine/custom/dfine_hgnetv2_s_aitod.yml`

The main research target is a tiny-object-aware trust mechanism for localization self-distillation on D-FINE.

## Scope Agreed In Discussion

- We are intentionally using the sample AI-TOD setup for faster iteration.
- We do not need to switch to the full dataset at this stage.
- Baseline reference is:
  `/home/com_2/suan/TOD/D-FINE-master/D-FINE-master/output/D-Fine_S_smaple/log.txt`
- We proceed step by step.
- Code modifications are allowed only after explicit permission. Permission was later granted for the first TrustLSD-style experiment.

## Important Conceptual Conclusion

After reviewing the proposal and the existing D-FINE implementation, the key conclusion was:

- D-FINE already contains localization quality estimation and GO-LSD style localization self-distillation.
- Therefore, the paper claim should not be "introducing reliability weighting in general."
- The real novelty should be a tiny-object-aware trust formulation and stronger teacher selection for tiny objects.

Relevant implementation facts that motivated this:

- D-FINE already has an LQE module in `src/zoo/dfine/dfine_decoder.py`
- D-FINE already uses final-layer teacher supervision for auxiliary localization distillation
- Tiny-object metrics are already supported through the custom evaluator bins

## Baseline Decision

There was an early discussion snapshot where a different output folder was inspected first, but that was not accepted as the true baseline.

The user explicitly confirmed that the correct baseline is:

- `output/D-Fine_S_smaple/log.txt`

This should be treated as the frozen comparison target for future experiments on the sample dataset.

Later, a cleaner explicit baseline rerun was also prepared and used for direct experiment comparison:

- `output/dfine_s_baseline/log.txt`

Going forward, the most practical baseline reference for new comparisons should be:

- `output/dfine_s_baseline/log.txt`

## Current On-Disk Baseline Snapshot

Historical baseline file from the earlier phase:

- `/home/com_2/suan/TOD/D-FINE-master/D-FINE-master/output/D-Fine_S_smaple/log.txt`

Current file state:

- 315 log lines
- Last recorded epoch: 293

Current best metrics extracted from the present on-disk log:

- Best overall AP:
  epoch 245, AP 0.215123, AP_very_tiny 0.051789, AP_tiny 0.172121, AP_s 0.278094, AP_m 0.404668
- Best AP_very_tiny:
  epoch 285, AP_very_tiny 0.069293
- Best AP_tiny:
  epoch 244, AP_tiny 0.172839
- Best AP_s:
  epoch 284, AP_s 0.278764
- Best AP_m:
  epoch 190, AP_m 0.411856
- Last epoch snapshot:
  epoch 293, AP 0.204538, AP_very_tiny 0.066670, AP_tiny 0.170721, AP_s 0.270328, AP_m 0.389050

For future comparison, the primary metrics should be:

- AP_very_tiny
- AP_tiny
- AP_s

Secondary metrics:

- AP
- AP_m

Note:

- Earlier discussion in this session referenced stronger peak numbers from an older state of the same baseline log.
- The current on-disk file should be treated as the authoritative reference unless the older log is restored externally.

## Active Comparison Baseline Snapshot

For the completed trust-weight experiment comparison, the baseline used was:

- `/home/com_2/suan/TOD/D-FINE-master/D-FINE-master/output/dfine_s_baseline/log.txt`

Current file state:

- 300 log lines
- Last recorded epoch: 299

Best metrics from `dfine_s_baseline/log.txt`:

- Best overall AP:
  epoch 293, AP 0.213011
- Best AP_very_tiny:
  epoch 240, AP_very_tiny 0.093225
- Best AP_tiny:
  epoch 289, AP_tiny 0.176913
- Best AP_s:
  epoch 293, AP_s 0.278647
- Best AP_m:
  epoch 299, AP_m 0.402397

## First Trust Experiment Chosen

We selected the smallest clean experiment first, not the full proposal.

What this first experiment does:

- Keeps the current D-FINE final-layer teacher unchanged
- Adds a learned trust score for matched queries during training
- Supervises that trust score with a continuous localization-quality target
- Uses the learned trust score to replace the matched-query weighting inside localization KD

What this first experiment does not do yet:

- No reliability-aware teacher mixture
- No tiny-object bucketed scale weighting
- No hard trust mask
- No RT-DETR transfer study
- No teacher reconstruction across multiple decoder layers

This means the current implementation corresponds to the earliest TrustLSD stage:

- learned trust
- continuous trust target
- trust-weighted localization KD
- final-layer teacher remains unchanged

## Reason For The Implementation Design

One important codebase-specific decision was necessary:

- The trust head must live inside the model, not inside the criterion

Reason:

- The optimizer in this repository is built from `model.named_parameters()`
- If the trust head were added only inside `DFINECriterion`, it would not be optimized

Therefore the trust head was added inside `DFINETransformer`.

## Files Modified

The following files were modified for the first trust experiment:

- `/home/com_2/suan/TOD/D-FINE-master/D-FINE-master/configs/dfine/custom/dfine_hgnetv2_s_aitod.yml`
- `/home/com_2/suan/TOD/D-FINE-master/D-FINE-master/src/zoo/dfine/dfine_decoder.py`
- `/home/com_2/suan/TOD/D-FINE-master/D-FINE-master/src/zoo/dfine/dfine_criterion.py`

## Config Changes

In `configs/dfine/custom/dfine_hgnetv2_s_aitod.yml`:

- Enabled trust in `DFINETransformer`
- Added:
  `trust_enabled: True`
- Added:
  `trust_topk: 4`
- Added:
  `trust_hidden_dim: 64`
- Extended `DFINECriterion.weight_dict` with:
  `loss_trust: 1.0`
- Extended `DFINECriterion.losses` with:
  `trust`
- Added:
  `trust_alpha: 0.5`
- Added:
  `trust_eps: 1.0e-6`

Current output directory for the new experiment remains:

- `./output/dfine_hgnetv2_s_aitod`

This is separate from the frozen sample baseline folder.

## Model-Side Code Changes

In `src/zoo/dfine/dfine_decoder.py`:

- Added new constructor arguments to `DFINETransformer`:
  `trust_enabled`, `trust_topk`, `trust_hidden_dim`
- Added a train-time `trust_head`
- Initialized the last layer of `trust_head` to zero
- Added trust score prediction from final decoder outputs
- Passed `trust_scores` into the main output dict
- Passed `trust_scores` into auxiliary decoder outputs

The current trust feature vector uses:

- localization entropy from final-layer corner distributions
- top-1 probability mass
- top-2 mass
- top-k mass
- classification confidence
- cross-layer box drift between final and previous decoder layers
- log box area

This trust head is training-only and does not modify inference outputs.

## Criterion-Side Code Changes

In `src/zoo/dfine/dfine_criterion.py`:

- Added `trust_alpha` and `trust_eps` to the criterion constructor
- Added a new `loss_trust`
- `loss_trust` is defined only when `trust_scores` exist
- `loss_trust` is computed only for matched queries
- The continuous trust target is:
  `trust_alpha * IoU + (1 - trust_alpha) * center_term`
- `center_term` is:
  `exp(-center_distance / (sqrt(gt_area) + eps))`
- The trust regression loss uses `SmoothL1`

The local distillation weighting was also changed:

- For matched queries, the KD weight now uses predicted trust score when available
- If trust scores are missing, it falls back to the previous matched-query IoU behavior
- Unmatched/background positions still use the original teacher-confidence style weighting

To avoid duplicating `loss_trust` on every auxiliary branch:

- `trust` loss is skipped for aux decoder outputs
- `trust` loss is skipped for pre outputs
- `trust` loss is skipped for encoder auxiliary outputs
- `trust` loss is skipped for denoising outputs

## Additional Solver Change

After the first trust implementation, the checkpoint-saving logic was extended so that training can keep a dedicated best checkpoint for very tiny objects.

In `src/solver/det_solver.py`:

- Added tracking for best `AP_very_tiny`
- Added checkpoint outputs:
  `best_very_tiny.pth`
- Added stage-specific outputs:
  `best_very_tiny_stg1.pth`
- Added stage-specific outputs:
  `best_very_tiny_stg2.pth`
- Added log fields:
  `best_test_AP_very_tiny`
- Added log fields:
  `best_test_AP_very_tiny_epoch`

## Validation Performed

What was checked:

- Python syntax check passed for:
  `src/zoo/dfine/dfine_decoder.py`
- Python syntax check passed for:
  `src/zoo/dfine/dfine_criterion.py`

What was not fully checked in this shell:

- Full model instantiation with the project runtime environment
- End-to-end training launch
- A forward pass with real tensors

Reason:

- The shell environment available during this session did not have `torch` available in the invoked Python
- It also lacked `yaml` in the invoked Python for lightweight config parsing

So the code was edited and syntax-checked, but not executed end-to-end inside this shell session.

## First Trust Experiment Results

The first completed trust experiment was compared against the explicit baseline rerun.

Compared files:

- Baseline:
  `/home/com_2/suan/TOD/D-FINE-master/D-FINE-master/output/dfine_s_baseline/log.txt`
- Trust experiment:
  `/home/com_2/suan/TOD/D-FINE-master/D-FINE-master/output/D-Fine_S_trust_weight/log.txt`

Trust experiment file state:

- 301 log lines
- Last recorded epoch: 299

Best metrics from `D-Fine_S_trust_weight/log.txt`:

- Best overall AP:
  epoch 284, AP 0.218697
- Best AP_very_tiny:
  epoch 248, AP_very_tiny 0.071421
- Best AP_tiny:
  epoch 284, AP_tiny 0.170536
- Best AP_s:
  epoch 249, AP_s 0.294985
- Best AP_m:
  epoch 285, AP_m 0.385402

Best-to-best comparison versus `dfine_s_baseline/log.txt`:

- AP:
  0.213011 -> 0.218697, delta `+0.005685`
- AP_very_tiny:
  0.093225 -> 0.071421, delta `-0.021804`
- AP_tiny:
  0.176913 -> 0.170536, delta `-0.006377`
- AP_s:
  0.278647 -> 0.294985, delta `+0.016338`
- AP_m:
  0.402397 -> 0.385402, delta `-0.016995`

Important same-epoch pattern:

- Around epochs 240 to 299, the trust model consistently achieved higher AP and often higher AP_s than baseline
- But AP_very_tiny remained consistently worse than baseline in the same region
- This means the result is not just a best-epoch selection artifact

Observed training behavior:

- `train_loss_trust` decreased smoothly during training
- So the trust objective was learned
- However, the learned weighting did not improve the very tiny regime

## Interpretation Of The First Result

The first trust-weight version did not solve the main research target.

What happened:

- General AP improved
- AP_s improved
- AP_very_tiny became much worse
- AP_tiny also became worse

Most likely interpretation:

- The learned trust signal favored easier or more stable queries
- This helped medium-small general localization quality
- But it suppressed or underweighted the hardest tiny-object matches, especially very tiny objects

In other words:

- the current trust formulation is learnable
- but it is not yet tiny-object-aware enough

## Recommended Next Experiment

The next experiment should not jump directly to the full reliability-aware teacher mixture.

The highest-priority next change should be:

- keep the current trust head
- add explicit scale-aware weighting for `very_tiny`, `tiny`, and `small`
- apply localization KD only to matched queries, or strongly reduce non-matched/background influence

This is the most direct response to the observed failure mode.

Reason:

- the current trust model improves overall AP but hurts very tiny objects
- therefore the next experiment should bias the trust/KD mechanism toward tiny-object preservation
- adding teacher mixture first would make diagnosis harder

## Diagnostics Needed Next

Before or alongside the next experiment, add analysis for:

- mean trust score for `very_tiny`, `tiny`, `small`
- correlation between trust score and IoU
- how many very tiny matched queries receive low trust
- whether trust behaves like a classification-confidence proxy

These diagnostics are now important because the first experiment showed that the trust objective is active but misaligned with the desired subset.

## Recommended Evaluation Rule For The Next Run

When the trust-enabled experiment is trained, compare it against the frozen sample baseline using:

- AP_very_tiny as the main decision metric
- AP_tiny as the second metric
- AP_s as the third metric

Suggested interpretation:

- If AP_very_tiny improves but AP collapses, inspect stability and overfitting
- If AP_tiny and AP_s improve consistently, the trust signal is probably useful
- If learned trust does not beat the baseline clearly, the next ablation should compare it directly against simpler heuristic trust weights

After the completed first experiment, this interpretation can now be specialized:

- The current learned trust did not beat baseline on AP_very_tiny
- Therefore the next ablation should prioritize tiny-object-aware weighting rather than broader complexity
- Confidence-only and entropy-only heuristic comparisons are still important, but the most urgent fix is scale awareness

## What Has Not Been Done Yet

The following planned items from the proposal are still pending:

- entropy-only trust weighting baseline
- confidence-only trust weighting baseline
- LQE-style heuristic trust weighting baseline
- hard trust mask vs soft trust weighting comparison
- binary trust target vs continuous trust target comparison
- tiny-object scale bucket weighting
- reliability-aware teacher mixture across decoder layers
- transfer study to RT-DETR
- reliability analysis plots
- calibration and correlation analysis

## Suggested Next Step

The next step should be:

- keep the current trust branch as the starting point
- add tiny-object-aware scale weighting
- restrict KD more explicitly to matched queries if needed
- rerun on the sample dataset
- compare against `output/dfine_s_baseline/log.txt`

After that, the next ablation priority should be:

- confidence heuristic vs learned trust
- entropy heuristic vs learned trust

## Quick Resume Summary

If resuming from scratch in the next session, remember this:

- Sample dataset is intentional
- Use `output/dfine_s_baseline/log.txt` as the active comparison baseline
- `output/D-Fine_S_smaple/log.txt` is older historical baseline context
- The current trust experiment folder is `output/D-Fine_S_trust_weight`
- The first trust experiment is complete
- The current experiment is a minimal trust extension, not the full TrustLSD method
- Final-layer teacher is still unchanged
- Trust head lives in `DFINETransformer`
- Trust supervision and trust-weighted local KD live in `DFINECriterion`
- First result summary:
  AP improved, AP_s improved, AP_very_tiny worsened strongly
- Most likely next step:
  add scale-aware weighting and make the method more tiny-object-aware before trying teacher mixture
