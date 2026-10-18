"""Does the cascade survive the gyro being switched on only after stage 1 decides?

Realistic timing: stage 1 finishes at the end of window [s, s+128); the gyro then records the next fresh
window [s+128, s+256) (start-up ~50 ms, ignored). Stage 2 = current 7 accelerometer features + the
DELAYED gyro features; the label is still the current window's. Windows whose delayed gyro window runs
past the end of the recording keep the stage-1 answer. Compared against the ideal same-window gyro.
"""
import os

import numpy as np
import torch
from scipy.signal import butter, lfilter, medfilt
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

from baseline import train
from features import FS, WIN, segment_starts, split_gravity, window_features
from stage2_uci import MAP, SIT, STAND, UCI, lr

GYRO = ['gyro_std_x', 'gyro_std_z']  # selected on training subjects in stage2_uci.py
TAUS = [0, 0.5, 1, 2, 3, np.inf]


def gyro_std(gyro, starts):
    """std of x and z gyro per window; NaN if the window runs past the recording."""
    out = np.full((len(starts), 2), np.nan)
    for i, s in enumerate(starts):
        if s + WIN <= len(gyro):
            out[i] = gyro[s:s + WIN][:, [0, 2]].std(0)
    return out


def build():
    noise = butter(3, 20, fs=FS)
    X7, Gnow, Gnext, y, s, cache = [], [], [], [], [], {}
    for exp, user, act, start, end in np.loadtxt(os.path.join(UCI, 'RawData', 'labels.txt'), dtype=int):
        if exp not in cache:
            acc = np.loadtxt(os.path.join(UCI, 'RawData', f'acc_exp{exp:02d}_user{user:02d}.txt'))
            gyr = np.loadtxt(os.path.join(UCI, 'RawData', f'gyro_exp{exp:02d}_user{user:02d}.txt'))
            cache = {exp: (*split_gravity(acc), lfilter(*noise, medfilt(gyr, (3, 1)), axis=0))}
        body, grav, gyro = cache[exp]
        st = segment_starts(start - 1, end - 1)
        X7.append(window_features(body, grav, st))
        Gnow.append(gyro_std(gyro, st)); Gnext.append(gyro_std(gyro, [x + WIN for x in st]))
        y += [MAP[act]] * len(st); s += [user] * len(st)
    return (np.vstack(X7), np.vstack(Gnow), np.vstack(Gnext), np.array(y, dtype=np.int64), np.array(s))


def cascade(X7, G, y, g, tr, te, seed=0):
    a, b = next(GroupShuffleSplit(1, test_size=0.15, random_state=seed).split(tr, groups=g[tr]))
    a, b = tr[a], tr[b]
    sc = StandardScaler().fit(X7[a])
    m1 = train(sc.transform(X7[a]), y[a], sc.transform(X7[b]), y[b], seed)
    with torch.no_grad():
        logits = m1(torch.tensor(sc.transform(X7[te]), dtype=torch.float32)).numpy()
    p1 = logits.argmax(1)
    top2 = np.sort(logits, 1)[:, -2:]
    margin = top2[:, 1] - top2[:, 0]

    ok = ~np.isnan(G).any(1)
    ss = tr[np.isin(y[tr], [SIT, STAND]) & ok[tr]]
    m2 = lr().fit(np.c_[X7[ss], G[ss]], y[ss])
    p2 = np.full(len(te), -1)
    tok = te[ok[te]]
    p2[ok[te]] = m2.predict(np.c_[X7[tok], G[tok]])

    yt, rows = y[te], []
    for tau in TAUS:
        route = np.isin(p1, [SIT, STAND]) & (margin < tau) & ok[te]
        p = np.where(route, p2, p1)
        m = np.isin(yt, [SIT, STAND])
        rows.append(dict(tau=tau, routed=100 * route.mean(), acc=100 * (p == yt).mean(),
                         bal=100 * balanced_accuracy_score(yt, p),
                         sserr=100 * (np.isin(p[m], [SIT, STAND]) & (p[m] != yt[m])).mean()))
    return rows


def main():
    X7, Gnow, Gnext, y, g = build()
    print(f'windows without a complete next gyro window: {np.isnan(Gnext).any(1).sum()} of {len(y)}')
    test_subj = set(np.loadtxt(os.path.join(UCI, 'Test', 'subject_id_test.txt'), dtype=int))
    te = np.isin(g, list(test_subj)); idx = np.arange(len(y))
    for label, G in [('ideal: gyro on the same window', Gnow), ('realistic: gyro on the NEXT window', Gnext)]:
        off = cascade(X7, G, y, g, idx[~te], idx[te])
        folds = [cascade(X7, G, y, g, a, b) for a, b in GroupKFold(5).split(X7, y, g)]
        print(f'\n== {label}')
        print(f"{'tau':>5} | {'official acc':>12} {'sserr':>6} {'routed':>7} | {'5-fold acc':>10} {'sserr':>6} {'routed':>7}")
        for i, r in enumerate(off):
            cv = {k: np.mean([f[i][k] for f in folds]) for k in ('acc', 'sserr', 'routed')}
            print(f"{r['tau']:>5} | {r['acc']:12.2f} {r['sserr']:5.1f}% {r['routed']:6.1f}% | {cv['acc']:10.2f} {cv['sserr']:5.1f}% {cv['routed']:6.1f}%")


if __name__ == '__main__':
    main()
