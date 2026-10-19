#!/usr/bin/env python3
"""Fast, reproducible end-to-end smoke test using a synthetic-only fixture.

This is a mechanics test, not model-performance evidence. It must never be used
as the reported Fab-7 evaluation result.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "wafer_guard_fixed.ipynb"


def make_fixture(path: Path, n: int = 600) -> None:
    rng = np.random.default_rng(7)
    sequence = np.arange(n)
    # 1 positive per 15 wafers = 6.67% prevalence. Distributed throughout time
    # so each chronological split contains both classes.
    defect = (sequence % 15 == 0).astype(int)
    sensor_0 = rng.normal(10.0, 1.5, n)
    legacy = (sequence % 5 == 0)
    sensor_0[legacy] = 0.0  # PS-L sentinel; never a physical measurement
    sensor_1 = rng.normal(20.0, 3.0, n)
    firmware = np.array([100.0, 200.0, 300.0])[(sequence // 5) % 3]
    sensor_1[legacy] = firmware[legacy]
    sensor_2 = rng.normal(0.0, 1.0, n) + defect * 1.2
    # Include missing telemetry and an extreme value confined to the held-out window.
    sensor_2[sequence % 47 == 0] = np.nan
    sensor_2[540:] = 100_000.0
    future_only_sensor = np.full(n, np.nan)
    future_only_sensor[420:] = 50_000.0
    frame = pd.DataFrame({
        "sequence": sequence,
        "sensor_0": sensor_0,
        "sensor_1": sensor_1,
        "sensor_2": sensor_2,
        "future_only_sensor": future_only_sensor,
        "label": defect,
    })
    frame.to_csv(path, index=False)


def main() -> None:
    if not NOTEBOOK.exists():
        raise FileNotFoundError(NOTEBOOK)
    source_nb = nbformat.read(NOTEBOOK, as_version=4)

    # Keep the official notebook untouched and shorten only this test copy.
    source_nb.cells[1].source = source_nb.cells[1].source.replace(
        "MAX_EPOCHS = 60", "MAX_EPOCHS = 2"
    ).replace("PATIENCE = 10", "PATIENCE = 2")

    checks = r'''
# Assertions specific to the synthetic fixture; never treat these scores as the real assessment.
assert "ps_l_legacy_probe_station" in preprocessor.feature_names_out_
assert "ps_l_firmware_100" in preprocessor.feature_names_out_
assert "ps_l_firmware_200" in preprocessor.feature_names_out_
assert "ps_l_firmware_300" in preprocessor.feature_names_out_
assert X_train.ndim == X_val.ndim == X_test.ndim == 2
assert np.isfinite(X_train).all() and np.isfinite(X_val).all() and np.isfinite(X_test).all()
assert "future_only_sensor" not in feature_cols
assert preprocessor.generic_stats_["sensor_2"]["mean"] < 10.0

# Confirm firmware codes are categories, not values passed through a continuous scaler.
legacy_test = X_test_raw["sensor_0"].eq(0.0).to_numpy()
raw_codes_test = pd.to_numeric(X_test_raw["sensor_1"], errors="coerce").to_numpy()
name_to_idx = {name: i for i, name in enumerate(preprocessor.feature_names_out_)}
if legacy_test.any():
    assert np.all(X_test[legacy_test, name_to_idx["sensor_1_continuous_scaled"]] == 0.0)
    for code in (100, 200, 300):
        col = name_to_idx[f"ps_l_firmware_{code}"]
        expected = legacy_test & (raw_codes_test == code)
        assert np.array_equal(X_test[:, col].astype(bool), expected)

# Explicit shape and gradient smoke check (including a short final batch shape).
probe_x = torch.as_tensor(X_train[:7], dtype=torch.float32, device=DEVICE)
probe_y = torch.as_tensor(y_train[:7], dtype=torch.float32, device=DEVICE)
probe_recon, probe_logits = model(probe_x)
assert probe_recon.shape == probe_x.shape
assert probe_logits.shape == probe_y.shape == (7,)
probe_loss, _, _ = calculate_batch_loss(probe_recon, probe_logits, probe_x, probe_y)
assert probe_loss.ndim == 0 and torch.isfinite(probe_loss)
model.zero_grad(set_to_none=True)
probe_loss.backward()
assert all(p.grad is None or torch.isfinite(p.grad).all().item() for p in model.parameters())

# The smoke-test test rows remain untouched, and metrics reflect their actual prevalence.
with open("metrics.json", encoding="utf-8") as f:
    smoke_metrics = json.load(f)
assert smoke_metrics["test_rows_resampled"] is False
assert smoke_metrics["test_natural_prevalence_untouched"] is True
assert smoke_metrics["n_test"] == len(y_test)
assert abs(smoke_metrics["test_defect_prevalence"] - float(y_test.mean())) < 1e-12
print("SMOKE_TEST_ASSERTIONS_PASSED")
'''
    source_nb.cells.append(nbformat.v4.new_code_cell(checks))

    with tempfile.TemporaryDirectory(prefix="waferguard-smoke-") as tmp:
        workdir = Path(tmp)
        make_fixture(workdir / "secom_telemetry_modified.csv")
        (workdir / "AUDIT.md").write_text(
            "# Synthetic smoke-test audit\n\n"
            "<!-- MEASURED_METRICS_START -->\nPending\n<!-- MEASURED_METRICS_END -->\n",
            encoding="utf-8",
        )
        client = NotebookClient(source_nb, timeout=600, kernel_name="python3", resources={"metadata": {"path": str(workdir)}})
        executed = client.execute()
        outputs = []
        for cell in executed.cells:
            for out in cell.get("outputs", []):
                if out.output_type == "stream":
                    outputs.append(out.text)
                elif out.output_type in {"execute_result", "display_data"}:
                    data = out.get("data", {})
                    if "text/plain" in data:
                        outputs.append(str(data["text/plain"]))
        combined = "\n".join(outputs)
        if "SMOKE_TEST_ASSERTIONS_PASSED" not in combined:
            raise AssertionError("Smoke-test assertions did not complete successfully.\n" + combined[-5000:])
        print("PASS: end-to-end notebook execution on synthetic fixture")
        print("PASS: PS-L sentinel and firmware-code assertions")
        print("PASS: training-only feature selection and scaling")
        print("PASS: tensor shapes, finite loss, and gradients")
        print("PASS: untouched test rows and natural-prevalence metric bookkeeping")
        print("NOTE: synthetic smoke-test metrics are not valid Fab-7 evaluation metrics.")


if __name__ == "__main__":
    main()
