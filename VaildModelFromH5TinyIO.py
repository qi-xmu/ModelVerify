#!/usr/bin/env python3
"""Compatibility launcher for models with different output formats.

This script does not modify existing project files. It monkey patches
`base.model.InertialNetwork` at runtime, then delegates execution to
`VaildModelFromH5atr.py`.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

import VaildModelFromH5atr as atr_runner
from base.model import InertialNetwork


def _flatten_tensor(t: torch.Tensor) -> NDArray:
    return t.detach().cpu().numpy().astype(np.float64).reshape(-1)


def _extract_meas_and_cov(output: Any) -> tuple[NDArray, NDArray]:
    """Normalize model outputs into `(meas, meas_cov)`.

    Supported output forms:
    - tuple/list: `(meas, meas_cov, ...)` or `(meas,)`
    - tensor: either `[... , 6+]` where first 3 dims are meas and next 3 dims are
      covariance/log-cov, or `[... , 3]` meas only.

    For meas-only outputs, covariance is filled with zeros to keep downstream
    evaluation logic unchanged.
    """

    meas_t: torch.Tensor
    cov_t: torch.Tensor

    if isinstance(output, (tuple, list)):
        if len(output) == 0:
            raise RuntimeError("Model returned an empty output sequence")

        if not torch.is_tensor(output[0]):
            raise TypeError(f"Unsupported output[0] type: {type(output[0])}")

        meas_t = output[0]
        if len(output) >= 2 and torch.is_tensor(output[1]):
            cov_t = output[1]
        else:
            cov_t = torch.zeros_like(meas_t)

    elif torch.is_tensor(output):
        if output.shape[-1] >= 6:
            meas_t = output[..., :3]
            cov_t = output[..., 3:6]
        else:
            meas_t = output
            cov_t = torch.zeros_like(meas_t)
    else:
        raise TypeError(f"Unsupported model output type: {type(output)}")

    meas = _flatten_tensor(meas_t)
    meas_cov = _flatten_tensor(cov_t)

    if meas_cov.shape != meas.shape:
        meas_cov = np.zeros_like(meas, dtype=np.float64)

    return meas, meas_cov


def _patched_predict(self: InertialNetwork, block: NDArray) -> tuple[NDArray, NDArray]:
    if self.input_shape:
        assert block.shape == self.input_shape, (
            f"Input shape mismatch: {block.shape} != {self.input_shape}"
        )

    inputs = torch.from_numpy(np.ascontiguousarray(block)).to(
        dtype=torch.float32,
        device=self.device,
    )

    with torch.inference_mode():
        output = self.model(inputs)

    return _extract_meas_and_cov(output)


def _patched_predict_with_time(
    self: InertialNetwork,
    block: NDArray,
) -> tuple[tuple[NDArray, NDArray], float]:
    start_time = time.perf_counter()

    if self.input_shape:
        assert block.shape == self.input_shape, (
            f"Input shape mismatch: {block.shape} != {self.input_shape}"
        )

    inputs = torch.from_numpy(np.ascontiguousarray(block)).to(
        dtype=torch.float32,
        device=self.device,
    )

    with torch.inference_mode():
        output = self.model(inputs)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    meas, meas_cov = _extract_meas_and_cov(output)
    end_time = time.perf_counter()

    return (meas, meas_cov), end_time - start_time


def _apply_runtime_patch() -> None:
    InertialNetwork.predict = _patched_predict  # type: ignore[method-assign]
    InertialNetwork.predict_with_time = _patched_predict_with_time  # type: ignore[method-assign]


def main() -> None:
    _apply_runtime_patch()
    atr_runner.main()


if __name__ == "__main__":
    main()
