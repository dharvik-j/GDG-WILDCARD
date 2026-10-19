# Dataset provenance

## What this file is

`secom_telemetry_modified.csv` in this package is an **explicit fallback proxy**, not the original task-organiser file. The supporting folder did not expose the required CSV, so the proxy was assembled to let the notebook and PS-L preprocessing run end to end without inventing model scores or pretending the exact assessment file was available.

## Source

- Dataset: SECOM semiconductor manufacturing data
- Authors: Michael McCann and Adrian Johnston
- Repository: UCI Machine Learning Repository
- DOI: `10.24432/C54305`
- Licence reported by UCI: Creative Commons Attribution 4.0 International (CC BY 4.0)
- Source data: 1,567 rows, 590 sensor measurements, labels, and timestamps

UCI label convention is `-1` for pass and `+1` for fail. In the CSV, these are encoded as `0` and `1`, respectively.

## PS-L overlay used for this fallback

The following deterministic transformation was applied after sorting rows by timestamp:

1. Select 313 rows (approximately 20% of 1,567) without using the defect labels, using NumPy random generator seed `7042026`.
2. Set `sensor_0` to exactly `0.0` for those rows to represent the task's PS-L sentinel.
3. Replace `sensor_1` on those rows with one of the category codes `100.0`, `200.0`, or `300.0`, sampled using the same seeded generator.
4. Leave other telemetry fields, labels, timestamps, and source missing values unchanged.

This creates the required data shape and exercises the sentinel/category handling. It does **not** establish that these 313 rows actually used a legacy probe station, and the simulated PS-L assignments are not supplied or verified by the organisers.

## Prevalence

The source has 104 defective rows out of 1,567, or 6.6369% overall. Because the notebook uses a chronological split, the final 15% window has a different observed rate (3.8136%) in this source data. Test rows are not rebalanced; the notebook reports both overall and test-window prevalence.

## Use in the challenge

Use this proxy to inspect code execution and preprocessing behavior only. If the task-specific `secom_telemetry_modified.csv` becomes available, replace the fallback with that exact file, run the notebook again, and update the audit using its measured metrics. Do not describe the fallback's metrics as measured on the official Fab-7 dataset.
