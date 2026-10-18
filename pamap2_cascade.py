"""Cascade on PAMAP2: stage 1 on the chest accelerometer, stage 2 switches on a second sensor.

Same continuous-recording simulation and policies as gyro_policies.py (sticky / oneshot), leave-one-subject-out.
Second-sensor candidates (each powered only when the policy says so):
  chest gyro   : std x, y, z of the chest IMU gyroscope
  ankle accel  : the 7 paper features computed on the ankle IMU accelerometer
  wrist accel  : the 7 paper features computed on the hand (wrist) IMU accelerometer
Columns (0-based): activity 1; IMU blocks of 17 start at 3 (hand), 20 (chest), 37 (ankle):
  +1..+3 acc16 (m/s^2), +7..+9 gyro (rad/s).
"""
import os

import numpy as np
import pandas as pd
from scipy.signal import butter, lfilter, medfilt

from extract_all import PAMAP2
from features import FS, G, STEP, WIN, split_gravity, to_50hz, window_features
from gyro_policies import TAUS, evaluate

MAP = {4: 0, 12: 0, 13: 0, 2: 1, 3: 2, 1: 3}
HAND, CHEST, ANKLE = 3, 20, 37


def imu(d, base, part):
    cols = slice(base + 1, base + 4) if part == 'acc' else slice(base + 7, base + 10)
    x = pd.DataFrame(d[:, cols]).interpolate(limit_direction='both').to_numpy()
    return to_50hz(x / G if part == 'acc' else x, 100)


def build():
    noise = butter(3, 20, fs=FS)
    out = {k: [] for k in ('chest gyro', 'ankle accel', 'wrist accel')}
    for sid in range(101, 109):  # 109 has none of the scored activities
        d = np.loadtxt(os.path.join(PAMAP2, f'subject{sid}.dat'))
        chest = split_gravity(imu(d, CHEST, 'acc'))
        lab = d[::2, 1].astype(int)[:len(chest[0])]
        starts = list(range(0, len(lab) - WIN + 1, STEP))
        y = np.array([MAP.get(lab[s], -1) if (lab[s:s + WIN] == lab[s]).all() else -1 for s in starts])
        X = window_features(*chest, starts)
        gyro = lfilter(*noise, medfilt(imu(d, CHEST, 'gyro'), (3, 1)), axis=0)
        blocks = {
            'chest gyro': np.array([gyro[s:s + WIN].std(0) for s in starts]),
            'ankle accel': window_features(*split_gravity(imu(d, ANKLE, 'acc')), starts),
            'wrist accel': window_features(*split_gravity(imu(d, HAND, 'acc')), starts),
        }
        for k, Gk in blocks.items():
            out[k].append(dict(X=X, G=Gk, y=y, subject=sid))
        print(f'subject {sid}: {len(starts)} windows, {(y >= 0).sum()} scored', flush=True)
    return out


def main():
    data = build()
    n = len(next(iter(data.values())))
    folds = [(np.array([j for j in range(n) if j != i]), np.array([i])) for i in range(n)]
    for source, exps in data.items():
        print(f'\n== stage 2 sensor: {source}  (leave-one-subject-out, 8 folds)')
        print(f"{'tau':>5} {'policy':8s} | {'acc':>6} {'sserr':>6} {'sensor on':>9}")
        for tau in TAUS:
            res = [evaluate(exps, a, b, tau) for a, b in folds]
            for k in (('stage1',) if tau == 0 else ('oneshot', 'sticky')):
                c = {m: np.mean([r[k][m] for r in res]) for m in ('acc', 'sserr', 'gyro_on')}
                print(f"{tau:>5} {k:8s} | {c['acc']:6.2f} {c['sserr']:5.1f}% {c['gyro_on']:8.1f}%", flush=True)


if __name__ == '__main__':
    main()
