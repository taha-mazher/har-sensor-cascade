"""The paper's 7 features computed from raw accelerometer data, following the UCI-HAPT recipe.

Recipe (UCI README / features_info): 50 Hz; median filter (3 samples) + 3rd-order Butterworth low-pass 20 Hz for noise;
3rd-order Butterworth low-pass 0.3 Hz splits gravity from body acceleration. Filters are causal
(lfilter), which is what UCI used (r >= 0.9999 vs their features) and what streaming hardware can do; 2.56 s windows (128 samples)
with 50% overlap. Per window:
  tBodyAcc-STD-1..3   std of body acceleration per axis
  tBodyAcc-SMA-1      signal magnitude area = mean(|bx| + |by| + |bz|)
  tGravityAcc-Mean-1..3  mean of gravity per axis
Units: g. Datasets in m/s^2 are divided by 9.80665; other rates are resampled to 50 Hz first.
"""
import numpy as np
from scipy.signal import butter, lfilter, medfilt, resample_poly

FS, WIN, STEP = 50, 128, 64
G = 9.80665
NAMES = ['tBodyAcc-STD-1', 'tBodyAcc-STD-2', 'tBodyAcc-STD-3', 'tBodyAcc-SMA-1',
         'tGravityAcc-Mean-1', 'tGravityAcc-Mean-2', 'tGravityAcc-Mean-3']
_NOISE = butter(3, 20, fs=FS)
_GRAV = butter(3, 0.3, fs=FS)


def split_gravity(acc):
    """acc: (N, 3) in g at 50 Hz, one continuous recording. Returns body, gravity."""
    a = lfilter(*_NOISE, medfilt(acc, (3, 1)), axis=0)
    g = lfilter(*_GRAV, a, axis=0)
    return a - g, g


def window_features(body, grav, starts):
    """Features for windows starting at the given sample indices."""
    out = np.empty((len(starts), 7))
    for i, s in enumerate(starts):
        b, g = body[s:s + WIN], grav[s:s + WIN]
        out[i, :3] = b.std(0, ddof=1)
        out[i, 3] = np.abs(b).sum(1).mean()
        out[i, 4:] = g.mean(0)
    return out


def segment_starts(first, last):
    """Window starts inside a labelled segment [first, last] (inclusive sample indices)."""
    return list(range(first, last - WIN + 2, STEP))


def to_50hz(acc, fs):
    """Resample (N, 3) from fs to 50 Hz with a polyphase filter."""
    from math import gcd
    k = gcd(int(fs), FS)
    return acc if fs == FS else resample_poly(acc, FS // k, int(fs) // k, axis=0)


if __name__ == '__main__':
    # self-check: a still device tilted on x gives zero body motion and gravity on x
    # (causal filters need a few seconds to settle, so test a window well after the start)
    acc = np.tile([1.0, 0.0, 0.0], (3000, 1))
    b, g = split_gravity(acc)
    f = window_features(b, g, [2800])[0]
    assert np.allclose(f[:4], 0, atol=1e-6) and np.allclose(f[4:], [1, 0, 0], atol=1e-3), f
    print('self-check passed')
