"""v4 with TSFlow's lag specification (not our simplified version)."""
import torch
import numpy as np
from .model_v4 import S4DMeanFlowNetV4 as _BaseNet
from gluonts.time_feature import get_lags_for_frequency

# TSFlow-style lag indices using GluonTS's get_lags_for_frequency
_TSFLOW_LAGS = {}

def get_tsflow_lags(freq):
    """Get the actual lag indices TSFlow uses."""
    if freq not in _TSFLOW_LAGS:
        _TSFLOW_LAGS[freq] = get_lags_for_frequency(freq)
    return _TSFLOW_LAGS[freq]


def extract_lags_tsflow(past_target, ctx_len, freq="B"):
    """
    Extract lag features matching TSFlow's lag specification.
    past_target: (B, past_len) — full history
    Returns: (B, 1 + n_lags, ctx_len)

    IMPORTANT: requires past_target to have length >= ctx_len + max(lags).
    For exchange (B), max lag is 780, so need at least 810 steps.
    """
    lag_offsets = get_tsflow_lags(freq)
    B = past_target.shape[0]
    context = past_target[:, -ctx_len:]  # (B, ctx_len)

    channels = [context.unsqueeze(1)]  # (B, 1, ctx_len)
    for offset in lag_offsets:
        start = past_target.shape[1] - ctx_len - offset
        end = past_target.shape[1] - offset
        if start >= 0:
            lag = past_target[:, start:end]
        else:
            lag = torch.zeros(B, ctx_len, device=past_target.device)
            if end > 0:
                available = past_target[:, :end]
                lag[:, -available.shape[1]:] = available
        channels.append(lag.unsqueeze(1))

    return torch.cat(channels, dim=1)  # (B, 1 + n_lags, ctx_len)
