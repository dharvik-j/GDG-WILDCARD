# Audit — WaferGuard / Fab-7 Wafer Excursion Detector

## Summary

The reported F1 score of at least 0.90 is not sufficient evidence for release. The notebook was repaired to address leakage, tensor broadcasting, and loss of the PS-L station invariant.

**Data limitation:** the exact organiser-provided `secom_telemetry_modified.csv` could not be retrieved. The bundled CSV is a clearly disclosed fallback derived from UCI SECOM with an artificial PS-L overlay. The measurements below are real notebook outputs on that fallback, not official Fab-7 results. See `DATASET_PROVENANCE.md`.

## The three silent bugs and the repairs

### 1. Data leakage from the evaluation window / future observations

**Failure mode:** Any preprocessing fitted before the split (for example, imputing missing values or scaling with the entire CSV), oversampling before splitting, or randomly mixing chronological observations can let information from later wafers influence the model and its preparation. Tuning a decision threshold on the test set creates another evaluation leak.

**Repair:** The corrected notebook orders rows by a fully parseable timestamp or sequence column when one exists; otherwise it preserves CSV row order and warns that row order must be confirmed as acquisition order. CSV values are initially read as strings; numeric feature selection and then the contiguous 70% train / 15% validation / 15% test windows are handled so that feature typing and learned preprocessing are based on the training window, not held-out values. All learned preprocessing statistics come from the training window only. Class balancing uses a weighted sampler over training examples only. Validation is used for early stopping and threshold selection; the final chronological test window is never resampled and never used to select a threshold. Identifier and time columns are excluded from model inputs.

**Production caveat:** Causality cannot be proven from a CSV alone if neither the timestamp nor its row order is guaranteed to reflect acquisition time. Verify this data contract before production use.

### 2. Corrupted loss gradients caused by silent tensor broadcasting

**Failure mode:** A reconstruction loss reduced per feature or kept as shape `[batch, 1]` can be combined with a label/logit tensor of shape `[batch]`. PyTorch broadcasting may expand the operands into an unintended matrix (for example, `[batch, 1]` against `[batch]` becoming `[batch, batch]`) rather than comparing one prediction with its matching wafer label. This produces incorrect gradients without necessarily raising an exception.

**Repair:** The fixed model explicitly reshapes logits and labels to `[batch]`, asserts the reconstruction tensor exactly matches the input tensor `[batch, features]`, and reduces reconstruction MSE across the feature dimension to `[batch]`. `BCEWithLogitsLoss(reduction="none")` also returns `[batch]`. The two per-wafer losses are shape-checked before being combined and averaged. Training also checks loss and gradient finiteness and clips gradients.

### 3. Invariant blindness at legacy Probe Station PS-L

**Failure mode:** Treating `sensor_0 == 0.0` as a physical measurement makes the detector learn a false continuous value. Treating `sensor_1` values `100.0`, `200.0`, and `300.0` on legacy wafers as continuous measurements (or scaling/imputing over them) destroys the firmware-code semantics. A model can then become blind to the tool/station invariant and to the categorical fault signal.

**Repair:** The preprocessor derives an explicit `ps_l_legacy_probe_station` indicator from the exact `sensor_0 == 0.0` sentinel, masks that sentinel out of the continuous `sensor_0` statistics, and uses a neutral standardized placeholder only in the inapplicable continuous channel. For legacy wafers, `sensor_1` firmware codes are represented by separate categorical indicators for 100, 200, and 300, plus an unknown/missing category. These codes never enter a continuous scaler or the continuous-channel imputer. The continuous `sensor_1` scaler is fit only on non-legacy training rows. The code-category columns and PS-L indicator preserve the physical/data contract explicitly.

## Evaluation protocol

- Split: contiguous chronological 70% train / 15% validation / 15% test.
- Class balance: training-only inverse-frequency weighted sampling; no validation or test rows are sampled or duplicated.
- Threshold: selected on the naturally distributed validation split by F1 and frozen before evaluating the test split.
- Final test: all rows in the final test window, at observed natural defect prevalence. Precision, recall, F1, accuracy, balanced accuracy, ROC-AUC, average precision, and the confusion matrix are reported.
- Reproducibility: fixed random seed; training history and exact metrics are written by the notebook.

## Measured evaluation metrics — fallback proxy

The numbers below were measured by executing `wafer_guard_fixed.ipynb` against the bundled derived fallback. Overall defect prevalence is 6.6369%; the chronological holdout has 3.8136% prevalence because the source labels vary over time. The test window was not resampled. These are not official Fab-7 results because the exact organiser-provided file was unavailable. Re-run the notebook if the official file becomes available; it writes `metrics.json` and refreshes this table.

<!-- MEASURED_METRICS_START -->
| Metric | Measured result |
|---|---:|
| Natural test defect prevalence | 3.814% |
| Test sample count | 236 |
| Decision threshold (selected on validation) | 0.0132479 |
| Precision | 0.0474 |
| Recall | 1.0000 |
| F1 score | 0.0905 |
| Accuracy | 0.2331 |
| Balanced accuracy | 0.6013 |
| ROC-AUC | 0.6427 |
| Average precision (PR-AUC summary) | 0.0711 |
| Confusion matrix [ [TN, FP], [FN, TP] ] | `[[46, 181], [0, 9]]` |
<!-- MEASURED_METRICS_END -->

## Verification evidence

`tests/test_smoke.py` passed on 19 October 2026 using a separate disposable fixture. It checked end-to-end notebook execution, PS-L sentinel and firmware-category handling, tensor shapes, finite loss and gradients, exclusion of a held-out-only feature, and untouched test-row metric bookkeeping. These checks validate implementation mechanics only; smoke-test metrics are not predictive-performance evidence.

## Release checks still required

1. Replace the fallback with the exact organiser CSV when available and rerun the notebook.
2. Confirm the timestamp or row order is true acquisition order.
3. The fallback overall prevalence is 6.6369%, but the chronological test window is 3.8136%. Never rebalance the test set to force a target prevalence.
4. The fallback F1 is low; this model is not production-ready based on these results. Review false negatives, alert volume, and validation design before any operational use.
