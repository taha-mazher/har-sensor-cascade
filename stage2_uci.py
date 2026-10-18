"""Stage 2 on UCI-HAPT: a small Sitting-vs-Standing classifier that only runs when stage 1 is unsure.

1. Pool of extra per-window features from the same filtered signals (accelerometer), plus gyroscope
   features as a separate option (the phone has a gyro; using it means powering a second sensor).
2. Forward feature selection on TRAINING subjects' Sitting/Standing windows only (grouped CV).
3. Stage 2 = logistic regression on the 7 base features + k extras (7 + k MACs in hardware).
4. Gate: send a window to stage 2 if stage 1 says Sitting or Standing with top-2 logit margin < tau.
Evaluated on the official subject split and 5-fold by subject.
"""
import json
import os

import numpy as np
import torch
from scipy.signal import butter, lfilter, medfilt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from baseline import train
from features import FS, WIN, segment_starts, split_gravity, window_features

HERE = os.path.dirname(os.path.abspath(__file__))
UCI = os.path.join(HERE, '..', 'exact_paper_implementation', 'data', 'UCI HAR Dataset')
MAP = {1: 0, 2: 0, 3: 0, 4: 1, 5: 2, 6: 3, **{k: 4 for k in range(7, 13)}}
SIT, STAND = 1, 2
TAUS = [0, 0.5, 1, 2, 3, 5, np.inf]


def pool_names():
    ax = 'xyz'
    n = [f'b_{s}_{a}' for s in ('mean', 'max', 'min', 'energy', 'jerkstd') for a in ax]
    n += [f'g_{s}_{a}' for s in ('std', 'min', 'max', 'cos') for a in ax]
    n += ['bmag_mean', 'bmag_std', 'gmag_mean', 'corr_xy', 'corr_xz', 'corr_yz']
    n += [f'gyro_{s}_{a}' for s in ('mean', 'std') for a in ax]
    return n


def pool_features(body, grav, gyro, starts):
    rows = []
    for s in starts:
        b, g, w = body[s:s + WIN], grav[s:s + WIN], gyro[s:s + WIN]
        bm, gm = np.linalg.norm(b, axis=1), np.linalg.norm(g, axis=1)
        jerk = np.diff(b, axis=0) * FS
        c = np.corrcoef(b.T)
        rows.append(np.r_[b.mean(0), b.max(0), b.min(0), (b ** 2).mean(0), jerk.std(0),
                          g.std(0), g.min(0), g.max(0), g.mean(0) / gm.mean(),
                          bm.mean(), bm.std(), gm.mean(), c[0, 1], c[0, 2], c[1, 2],
                          w.mean(0), w.std(0)])
    return np.nan_to_num(np.array(rows).reshape(-1, len(pool_names())))  # short segments give 0 rows


def build():
    noise = butter(3, 20, fs=FS)
    X7, XP, y, s, cache = [], [], [], [], {}
    for exp, user, act, start, end in np.loadtxt(os.path.join(UCI, 'RawData', 'labels.txt'), dtype=int):
        if exp not in cache:
            acc = np.loadtxt(os.path.join(UCI, 'RawData', f'acc_exp{exp:02d}_user{user:02d}.txt'))
            gyr = np.loadtxt(os.path.join(UCI, 'RawData', f'gyro_exp{exp:02d}_user{user:02d}.txt'))
            cache = {exp: (*split_gravity(acc), lfilter(*noise, medfilt(gyr, (3, 1)), axis=0))}
        body, grav, gyro = cache[exp]
        st = segment_starts(start - 1, end - 1)
        X7.append(window_features(body, grav, st)); XP.append(pool_features(body, grav, gyro, st))
        y += [MAP[act]] * len(st); s += [user] * len(st)
    return np.vstack(X7), np.vstack(XP), np.array(y, dtype=np.int64), np.array(s)


def lr():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))


def forward_select(X7, XP, y, g, allowed, kmax=6):
    """Greedy selection of extra features for Sit-vs-Stand, grouped CV on training subjects only."""
    m = np.isin(y, [SIT, STAND])
    X7, XP, y, g = X7[m], XP[m], y[m], g[m]
    chosen, curve = [], []
    for _ in range(kmax):
        best = None
        for j in allowed:
            if j in chosen:
                continue
            X = np.c_[X7, XP[:, chosen + [j]]]
            sc = []
            for tr, te in GroupKFold(5).split(X, y, g):
                sc.append(balanced_accuracy_score(y[te], lr().fit(X[tr], y[tr]).predict(X[te])))
            if best is None or np.mean(sc) > best[1]:
                best = (j, np.mean(sc))
        chosen.append(best[0]); curve.append(round(100 * best[1], 2))
    return chosen, curve


def cascade(X7, XP, y, g, tr, te, extra, seed=0):
    """Train stage 1 and stage 2 on training subjects; evaluate the gate sweep on test subjects."""
    a, b = next(GroupShuffleSplit(1, test_size=0.15, random_state=seed).split(tr, groups=g[tr]))
    a, b = tr[a], tr[b]
    sc = StandardScaler().fit(X7[a])
    m1 = train(sc.transform(X7[a]), y[a], sc.transform(X7[b]), y[b], seed)
    with torch.no_grad():
        logits = m1(torch.tensor(sc.transform(X7[te]), dtype=torch.float32)).numpy()
    p1 = logits.argmax(1)
    top2 = np.sort(logits, 1)[:, -2:]
    margin = top2[:, 1] - top2[:, 0]

    ss = tr[np.isin(y[tr], [SIT, STAND])]
    X2 = lambda idx: np.c_[X7[idx], XP[idx][:, extra]]
    m2 = lr().fit(X2(ss), y[ss])
    p2 = m2.predict(X2(te))

    rows = []
    for tau in TAUS:
        route = np.isin(p1, [SIT, STAND]) & (margin < tau)
        p = np.where(route, p2, p1)
        yt = y[te]
        sserr = np.isin(yt, [SIT, STAND])
        rows.append(dict(tau=tau, routed_pct=round(100 * route.mean(), 1), acc=round(100 * (p == yt).mean(), 2),
                         bal_acc=round(100 * balanced_accuracy_score(yt, p), 2),
                         sit_stand_err=round(100 * (np.isin(p[sserr], [SIT, STAND]) & (p[sserr] != yt[sserr])).mean(), 1)))
    return rows


def main():
    X7, XP, y, g = build()
    names = pool_names()
    acc_only = [i for i, n in enumerate(names) if not n.startswith('gyro')]
    test_subj = set(np.loadtxt(os.path.join(UCI, 'Test', 'subject_id_test.txt'), dtype=int))
    te = np.isin(g, list(test_subj)); idx = np.arange(len(y))
    tr = idx[~te]
    out = {}
    for label, allowed in [('accelerometer only', acc_only), ('accelerometer + gyro', list(range(len(names))))]:
        chosen, curve = forward_select(X7[tr], XP[tr], y[tr], g[tr], allowed)
        print(f'\n== stage 2 extras: {label}')
        print('  selected (training subjects only):', [names[j] for j in chosen])
        print('  Sit/Stand balanced acc (train-subject CV) vs k extras:', curve)
        res = {}
        for k in (0, 2, 4, 6):
            extra = chosen[:k]
            off = cascade(X7, XP, y, g, tr, idx[te], extra)
            folds = [cascade(X7, XP, y, g, a, b, extra) for a, b in GroupKFold(5).split(X7, y, g)]
            cv = [dict(tau=r['tau'], acc=round(float(np.mean([f[i]['acc'] for f in folds])), 2),
                       routed_pct=round(float(np.mean([f[i]['routed_pct'] for f in folds])), 1),
                       sit_stand_err=round(float(np.mean([f[i]['sit_stand_err'] for f in folds])), 1))
                  for i, r in enumerate(off)]
            res[k] = dict(official=off, cv5=cv)
            print(f'  k={k}: ' + ' | '.join(f"tau={r['tau']}: acc {r['acc']} sserr {r['sit_stand_err']}% routed {r['routed_pct']}%"
                                           for r in off if r['tau'] in (0, 1, 3, np.inf)))
            print(f'        5-fold: ' + ' | '.join(f"tau={r['tau']}: acc {r['acc']} sserr {r['sit_stand_err']}% routed {r['routed_pct']}%"
                                                for r in cv if r['tau'] in (0, 1, 3, np.inf)), flush=True)
        out[label] = dict(selected=[names[j] for j in chosen], curve=curve, results=res)
    with open(os.path.join(HERE, 'data', 'stage2_uci_results.json'), 'w') as f:
        json.dump(out, f, indent=1, default=str)


if __name__ == '__main__':
    main()
