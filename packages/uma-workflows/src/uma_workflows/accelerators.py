from __future__ import annotations


def synchronize(device: str) -> None:
    """Wait for asynchronous work on the selected accelerator to finish."""
    if device in {"mps", "cuda"} or device.startswith("cuda:"):
        import torch

        if device == "mps":
            torch.mps.synchronize()
        else:
            torch.cuda.synchronize(device)
