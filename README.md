# GDG-WILDCARD

WaferGuard: Causality-Safe Wafer Excursion Detector
WaferGuard is a repaired PyTorch notebook for the Fab-7 wafer-defect assessment. It trains a supervised autoencoder, preserves the special semantics of legacy Probe Station PS-L telemetry, selects a classification threshold on validation data, and reports metrics on an untouched chronological test window.
Evaluation status: The real assessment CSV was not available in the build workspace when this repository package was prepared. The notebook and audit deliberately do not invent a real-data F1 score. Run the notebook on secom_telemetry_modified.csv to produce the official measured metrics.

Repository contents
- wafer_guard_fixed.ipynb — full preprocessing, model, training, threshold selection, evaluation, and audit-metric update pipeline.
- AUDIT.md — three silent defects, corresponding repairs, evaluation protocol, measured-metrics block, and release caveats.
- requirements.txt — Python libraries required for notebook execution and the smoke test.
- tests/test_smoke.py — fast end-to-end mechanics test using a synthetic fixture. It is not a substitute for evaluation on the assessment data.
- .gitignore — excludes environments, notebook caches, generated model files, and the assessment CSV so private or restricted data is not accidentally committed.
Prerequisites
- Python 3.10 or newer (Python 3.11 recommended).
- The authorised secom_telemetry_modified.csv assessment dataset.
- Jupyter Notebook/Lab, or another environment that can execute a Python 3 notebook.
- CPU is supported; a compatible CUDA-enabled PyTorch installation may be used when available.


#Installation

Create and activate a virtual environment, then install dependencies:

python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
Install or launch Jupyter if your environment does not already include it:
python -m pip install jupyterlab
jupyter lab
If PyTorch installation differs for your operating system or accelerator, use the official PyTorch installation selector and keep the resulting version compatible with requirements.txt.
Dataset and configuration
Place the authorised file named secom_telemetry_modified.csv in the repository root beside the notebook. The notebook also checks common hosted-notebook locations. To use a different location, update DATA_PATH in the configuration cell before execution.
The loader identifies a target column from common names and supports 0/1, conventional SECOM -1/+1 labels (+1 means defective), and common explicit pass/fail strings. Ambiguous labels fail explicitly rather than being guessed. A parseable timestamp or sequence column is used to order rows; if none exists, CSV row order is retained and a warning is displayed. Before production use, confirm that the selected order truly reflects acquisition time.
The model configuration lives in the first code cell of the notebook. It includes the random seed, chronological split fractions, batch size, learning rate, epoch limit, early-stopping patience, and supervised-loss weight.
Run the full evaluation
1. Confirm the assessment CSV is present and its use is authorised.
2. Open wafer_guard_fixed.ipynb in Jupyter.
3. Run every cell from top to bottom without shuffling or resampling validation/test data.
4. Review the reported temporal order, row counts, and class prevalence before interpreting the metrics.
5. Inspect the generated artifacts:
   - metrics.json — exact test metrics, class prevalence, selected threshold, split sizes, and confusion matrix.
   - training_history.csv — training and validation loss history.
   - AUDIT.md — automatically updated measured-metrics table, if the notebook runs from the repository directory.
Threshold selection uses the natural validation distribution and maximises validation F1. The threshold is frozen before the test window is scored. Test metrics are calculated once on all chronological test rows without resampling.
Architecture and implementation decisions
1. Ordering and split: the notebook identifies a parseable sequence/time column when available, otherwise keeps CSV order with a prominent warning. It creates contiguous 70% train, 15% validation, and 15% test windows before fitting preprocessing.
2. Training-only preprocessing: continuous-feature medians, means, and scales are learned on training rows only. IDs, target labels, and time/order columns are excluded from model inputs.
3. PS-L data contract: exact sensor_0 == 0.0 is treated as the legacy probe sentinel, not a true continuous reading. For legacy wafers, sensor_1 codes 100.0, 200.0, and 300.0 are represented as separate categorical indicator features (plus an unknown/missing indicator). These codes are not passed through the continuous scaler or overwritten with fabricated continuous measurements. A PS-L indicator preserves the station invariant explicitly.
4. Custom supervised autoencoder: a PyTorch encoder feeds both a reconstruction decoder and a binary defect-classification head. Per-row reconstruction MSE and binary cross-entropy are explicitly shaped as [batch] before combination. Assertions reject accidental [batch, 1]/[batch] broadcasting. Non-finite losses/gradients are checked and gradients are clipped.
5. Training-only balancing: inverse-frequency weighted sampling is applied only to the training dataset. Validation and test retain their observed prevalence.
6. Evaluation: the validation window chooses the F1 threshold; the final test window is not used for threshold tuning or preprocessing. Precision, recall, F1, accuracy, balanced accuracy, ROC-AUC, average precision, and a confusion matrix are written where mathematically defined.
Test and verification instructions
Run the fast end-to-end smoke test:
python tests/test_smoke.py
The smoke test creates a disposable synthetic fixture with approximately 6.67% positive labels and PS-L sentinel/code examples, executes a shortened copy of the notebook, and asserts:
- the notebook completes end to end;
- legacy firmware values are represented categorically and not scaled as continuous values;
- preprocessed tensors are finite and shapes are aligned;
- the loss is scalar and gradients are finite;
- test-row count and reported synthetic prevalence agree, with no test resampling.
Verified in the build environment: the smoke test passed all of the above checks on 19 October 2026. This is evidence of implementation mechanics only, not evidence of predictive performance on the assessment dataset. The synthetic metrics must never be reported as Fab-7 model metrics.
For actual evaluation evidence, execute the full notebook using the authorised assessment CSV and retain metrics.json, training_history.csv, and the updated AUDIT.md. Review them before publishing.
External libraries and assets
- PyTorch: neural-network modules, loss operations, optimisers, training tensors, and weighted sampling.
- pandas and NumPy: CSV handling and numerical preprocessing.
- scikit-learn: evaluation metrics and precision-recall curve utilities.
- Jupyter/IPython, nbformat, and nbclient: notebook execution and the repeatable smoke test.
- Dataset: secom_telemetry_modified.csv is assessment-provided data; it is not redistributed by this repository package.
The feature transformation and supervised autoencoder/loss implementation are custom to the task. The pipeline does not replace the required core logic with a prebuilt anomaly-detection package.
Limitations and release gates
- Temporal ordering is a data contract. A CSV without a trustworthy timestamp/sequence cannot prove causality from the file alone. Verify acquisition order with the data owner.
- A chronological holdout can have a different defect prevalence from the overall dataset. Report the observed final test prevalence; do not rebalance test rows to force it to 6.6%.
- A chronological window missing either class cannot support all requested classification metrics. The notebook fails rather than silently shuffling the split.
- Threshold selection optimises validation F1; production operation may require a different threshold based on false-negative/false-positive costs and the line's alert capacity.
- Metrics alone do not constitute manufacturing or process-safety approval. False-negative counts and alert behaviour must be reviewed with the process owner before production use.
- This package does not claim a real-data score until the exact assessment CSV has been evaluated.
