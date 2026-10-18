"""Extract the 7 features from raw data for UCI-HAPT, PAMAP2 and WISDM with one pipeline.

Writes data/<name>_features.npz with X (n, 7), y (paper's 5 classes), subject (n,).
Classes: 0 Walking (incl. stairs), 1 Sitting, 2 Standing, 3 Laying, 4 Transitions.
Activities outside these (jogging, cycling, ironing, ...) are left out.
"""
import collections
import os

import numpy as np
import pandas as pd

from features import FS, G, STEP, WIN, segment_starts, split_gravity, to_50hz, window_features

HERE = os.path.dirname(os.path.abspath(__file__))
UCI = os.path.join(HERE, '..', 'exact_paper_implementation', 'data', 'UCI HAR Dataset')
PAMAP2 = r'C:\Users\taham\datasets\PAMAP2\PAMAP2_Dataset\Protocol'
WISDM = r'C:\Users\taham\datasets\WISDM\WISDM_ar_v1.1\WISDM_ar_v1.1_raw.txt'
SETTLE = 5 * FS  # WISDM segments are separate recordings: skip 5 s while the causal filters settle


def runs(labels):
    """(label, first, last) for each run of equal labels."""
    edges = np.flatnonzero(np.diff(labels)) + 1
    for a, b in zip(np.r_[0, edges], np.r_[edges, len(labels)]):
        yield labels[a], a, b - 1


def uci():
    m = {1: 0, 2: 0, 3: 0, 4: 1, 5: 2, 6: 3, **{k: 4 for k in range(7, 13)}}
    X, y, s, cache = [], [], [], {}
    for exp, user, act, start, end in np.loadtxt(os.path.join(UCI, 'RawData', 'labels.txt'), dtype=int):
        if exp not in cache:
            cache = {exp: split_gravity(np.loadtxt(os.path.join(UCI, 'RawData', f'acc_exp{exp:02d}_user{user:02d}.txt')))}
        st = segment_starts(start - 1, end - 1)
        X.append(window_features(*cache[exp], st)); y += [m[act]] * len(st); s += [user] * len(st)
    return np.vstack(X), np.array(y), np.array(s)


def pamap2():
    m = {4: 0, 12: 0, 13: 0, 2: 1, 3: 2, 1: 3}  # walking, stairs up/down, sitting, standing, lying
    X, y, s = [], [], []
    for sid in range(101, 110):
        d = np.loadtxt(os.path.join(PAMAP2, f'subject{sid}.dat'))
        acc = pd.DataFrame(d[:, 21:24]).interpolate(limit_direction='both').to_numpy() / G  # chest acc16
        acc50 = to_50hz(acc, 100)
        lab = d[::2, 1].astype(int)[:len(acc50)]
        body, grav = split_gravity(acc50)
        for act, a, b in runs(lab):
            if act in m:
                st = segment_starts(a, b)
                if st:
                    X.append(window_features(body, grav, st)); y += [m[act]] * len(st); s += [sid] * len(st)
    return np.vstack(X), np.array(y), np.array(s)


def wisdm():
    m = {'Walking': 0, 'Upstairs': 0, 'Downstairs': 0, 'Sitting': 1, 'Standing': 2}
    rows = []
    for line in open(WISDM):
        for rec in line.strip().split(';'):
            p = rec.strip().strip(',').split(',')
            if len(p) >= 6:
                try:
                    rows.append((int(p[0]), p[1], int(p[2]), float(p[3]), float(p[4]), float(p[5])))
                except ValueError:
                    pass
    df = pd.DataFrame(rows, columns=['user', 'act', 't', 'x', 'y', 'z'])
    # a new segment starts when user or activity changes or the clock jumps by > 1 s either way.
    # Small repeats/backsteps (~4% of rows) are phone clock jitter, not breaks in the recording.
    dt = df.t.diff()
    new = (df.user != df.user.shift()) | (df.act != df.act.shift()) | (dt.abs() > 1e9)
    X, y, s = [], [], []
    for _, seg in df.groupby(new.cumsum()):
        act = seg.act.iloc[0]
        if act not in m:
            continue
        acc50 = to_50hz(seg[['x', 'y', 'z']].to_numpy() / G, 20)
        if len(acc50) < SETTLE + WIN:
            continue
        body, grav = split_gravity(acc50)
        st = segment_starts(SETTLE, len(acc50) - 1)
        X.append(window_features(body, grav, st)); y += [m[act]] * len(st); s += [seg.user.iloc[0]] * len(st)
    return np.vstack(X), np.array(y), np.array(s)


if __name__ == '__main__':
    os.makedirs(os.path.join(HERE, 'data'), exist_ok=True)
    for name, fn in [('uci', uci), ('pamap2', pamap2), ('wisdm', wisdm)]:
        X, y, s = fn()
        assert np.isfinite(X).all(), name
        np.savez(os.path.join(HERE, 'data', f'{name}_features.npz'), X=X, y=y, subject=s)
        c = collections.Counter(y.tolist())
        print(f'{name}: {len(y)} windows, {len(set(s))} subjects, per class',
              {k: c.get(i, 0) for i, k in enumerate(['Walk', 'Sit', 'Stand', 'Lay', 'Trans'])}, flush=True)
