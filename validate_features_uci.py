"""Check features.py against UCI-HAPT's own precomputed features.

UCI's features are min-max scaled to [-1, 1], so the same feature must match up to a straight line:
we report window counts and the per-feature correlation over windows matched one-to-one.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import NAMES, segment_starts, split_gravity, window_features  # noqa: E402

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'exact_paper_implementation', 'data', 'UCI HAR Dataset')
names = [l.split()[-1] for l in open(os.path.join(D, 'features.txt')) if l.strip()]
cols = [names.index(n) for n in NAMES]


def build(subjects):
    labels = np.loadtxt(os.path.join(D, 'RawData', 'labels.txt'), dtype=int)  # exp user act start end (1-based)
    feats, acts, cache = [], [], {}
    for exp, user, act, start, end in labels:
        if user not in subjects:
            continue
        if exp not in cache:
            acc = np.loadtxt(os.path.join(D, 'RawData', f'acc_exp{exp:02d}_user{user:02d}.txt'))
            cache = {exp: split_gravity(acc)}
        body, grav = cache[exp]
        st = segment_starts(start - 1, end - 1)
        feats.append(window_features(body, grav, st))
        acts += [act] * len(st)
    return np.vstack(feats), np.array(acts)


for part in ('Train', 'Test'):
    sub = set(np.loadtxt(os.path.join(D, part, f'subject_id_{part.lower()}.txt'), dtype=int))
    ref = np.loadtxt(os.path.join(D, part, f'X_{part.lower()}.txt'))[:, cols]
    yref = np.loadtxt(os.path.join(D, part, f'y_{part.lower()}.txt'), dtype=int)
    mine, y = build(sub)
    print(f'{part}: my windows {len(mine)}, UCI windows {len(ref)}')
    if len(mine) == len(ref):
        print('  labels identical:', bool((y == yref).all()))
        for j, n in enumerate(NAMES):
            print(f'  {n:20s} r = {np.corrcoef(mine[:, j], ref[:, j])[0, 1]:.4f}')
