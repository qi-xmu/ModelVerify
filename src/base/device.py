import os

import torch


class Device:
    device_name: str
    device: torch.device

    def __init__(self):
        if torch.cuda.is_available():
            self.device_name = "cuda"
        elif torch.mps.is_available():
            self.device_name = "mps"
        else:
            self.device_name = "cpu"
        self.device = torch.device(self.device_name)


CpuDevice = torch.device("cpu")


def resolve_device(device_name: str = "auto") -> torch.device:
    normalized = device_name.lower().strip()
    if normalized == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.mps.is_available():
            return torch.device("mps")
        return CpuDevice

    if normalized == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available, please use --device cpu")
        return torch.device("cuda")

    if normalized == "mps":
        if not torch.mps.is_available():
            raise RuntimeError("MPS is not available, please use --device cpu")
        return torch.device("mps")

    if normalized == "cpu":
        return CpuDevice

    raise ValueError(f"Unsupported device: {device_name}")


def set_default_device(device_name: str = "auto") -> torch.device:
    global DefaultDevice
    DefaultDevice = resolve_device(device_name)
    return DefaultDevice


DefaultDevice = resolve_device(os.getenv("MODELVERIFY_DEVICE", "auto"))
