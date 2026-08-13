# ADR-0011 — Phase 1 training protocol on real data: dev-split validation, class weighting, and full-session coverage

- **Status:** Accepted
- **Date:** 2026-08-13
- **Phase:** 1 (Centralized Multimodal Baseline)
- **Related objectives:** H4 (prototype), H5 (evaluation)

## Context

The first run of the Phase 1 baseline against the real corpus (see
[ADR-0010](ADR-0010-real-corpus-ingestion.md)) completed end-to-end but learned
nothing: `F1 = 0.0000`, `ROC-AUC = 0.5432`, and `accuracy = 0.6667` — exactly the
majority-class rate. The model predicted "not depressed" for every session.

Every phase that follows is measured *against* this baseline: the DP cost curve
(H1), the capability-aware aggregation gain (H2), and the final comparison table
(H5) are all differences from it. A baseline with no signal makes those
differences meaningless, so the training protocol is fixed here before Phase 3+
numbers are generated.

Three causes were identified, addressed below. A fourth issue — the model still
not fitting the *training* set — is recorded as unresolved.

## Decisions

### 1. Validate on the official dev split, not a random slice of train

The Phase 1 script split the 107-session train set 75/25, validating on ~27
random sessions and never touching the official `dev` split. That is both
noisier and not comparable to published DAIC-WOZ/AVEC2017 results.

Real-data runs now train on all of `train` (107) and validate on the official
`dev` split (34 after the 440 exclusion). `build_train_val_loaders` gained an
optional `val_dataset`; when supplied, `val_fraction` is ignored. Mock runs are
unchanged and still use the random split.

### 2. Weight the positive class in the BCE loss

DAIC-WOZ train is 30/107 positive (28%). Unweighted `BCEWithLogitsLoss` on 107
samples minimizes by predicting the majority class, which is exactly what
happened. `DepressionObjective` now accepts `pos_weight`, and
`positive_class_weight()` **measures** it from the training loader as
`n_negative / n_positive` (2.567 here) rather than taking it from config — so it
stays correct across folds, splits, and federated client shards, which will have
different balances. The config flag `train.class_weighting` only chooses whether
to apply it.

### 3. Cover the whole interview, not its first 50 seconds

The loader keeps every `frame_stride`-th frame and stops at `max_frames`, so
`stride x max_frames` bounds how much of a session is seen. The inherited
setting (`max_frames: 1000`, `frame_stride: 5`) covered 5,000 COVAREP frames —
at 100 Hz that is the **first ~50 seconds** of a ~15-minute interview, i.e.
~90% of every session was silently discarded, including essentially all of the
clinically informative later dialogue.

Sampling is now coarser but complete:

| Modality | Rate | Frames/session | Setting | Coverage |
|---|---|---|---|---|
| audio (COVAREP) | 100 Hz | ~90,000 | `stride 30 x max 3000` | full session @ ~3.3 Hz |
| video (CLNF AUs) | 30 Hz | ~27,000 | `stride 10 x max 3000` | full session @ 3 Hz |

Sequence length stays ~3,000 either way, so cost is comparable while coverage
goes from ~6% to ~100%.

### 4. Early stopping and explicit model selection

`epochs` is now an upper bound (60) with `early_stopping_patience` (10) on a
configurable `selection_metric`. ROC-AUC is the default: it is
threshold-independent, which matters on a 34-session imbalanced validation set
where a single flipped prediction moves F1 sharply.

### 5. Device is configurable, defaulting to `auto`

`train.device` accepts `auto | cpu | cuda`; `resolve_device()` picks CUDA when
available. The trainer previously hardcoded CPU.

## Result after these changes

```
Device=cuda  pos_weight=2.567  epochs_run=11/60
Best  (epoch 1, by roc_auc) — F1=0.0000  ROC-AUC=0.6838  acc=0.6765
Final (epoch 11)            — F1=0.4889  ROC-AUC=0.6008  acc=0.3235
```

ROC-AUC improved from 0.543 to 0.684, and the model no longer sits permanently
on the majority class. But this is **not yet an acceptable baseline**.

## Unresolved: the model does not fit its training data

`train_loss` is flat across all 11 epochs (0.995 → 1.018, no downward trend).
The model is not learning the training set at all, let alone generalizing. Its
validation behaviour is a collapse in the other direction — by epoch 9 it
predicts *every* session positive (`recall = 1.0`, `accuracy = 0.3235` = the
positive rate), so it is oscillating between degenerate constant predictors
rather than finding a boundary.

Selecting on ROC-AUC picks epoch 1, whose F1 is 0 — the selection metric and the
reported headline metric disagree, which is itself a signal that no epoch is
actually good.

Plausible causes, to be separated by experiment rather than guessed:

- **Too few samples for the model class.** 107 training sessions against a
  bi-GRU over 3,000 timesteps is a very high-capacity model on very little data.
- **Gradient signal lost over long sequences.** A GRU followed by mean-pooling
  over 3,000 steps averages away localized cues; the effective gradient path is
  long even for a gated unit.
- **Optimizer settings inherited from mock data.** `lr = 1e-3` was tuned on 32
  synthetic sessions with strong planted signal.

**Next step:** an overfit sanity check — train on ~20 sessions with
regularization off and confirm the model can drive training loss toward zero. If
it cannot, the defect is structural (architecture/gradient flow) and no amount
of hyperparameter tuning will help; if it can, the problem is capacity/data and
the response is a smaller encoder plus stronger regularization.

## Consequences

- Phase 1's Definition of Done in `docs/implementation-plan.md` should **not**
  be marked complete on real data until the training loss actually decreases and
  dev F1 is non-degenerate.
- Reported Chapter 4 numbers must state validation on the official dev split
  (n=34) so they are comparable to the AVEC2017 literature.
- `configs/baseline.yaml` now carries training-protocol knobs
  (`class_weighting`, `selection_metric`, `early_stopping_patience`, `device`)
  that later phases inherit; DP-SGD runs in Phase 3 must set them consistently
  or the privacy-utility curve will not be comparable to this baseline.
