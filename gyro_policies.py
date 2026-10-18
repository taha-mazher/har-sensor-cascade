"""Gyro power policies for the cascade, simulated in time order on continuous UCI-HAPT recordings.

Stage 1 runs on every window of a continuous 50%-overlap grid (hop = 64 samples = 1.28 s); only windows that
lie wholly inside one labelled segment are scored. Window w covers hops w and w+1; stage 1 decides at the end
of hop w+1, so a gyro switched on then first covers a whole window at w+2.

Policies:
  stage1       : accelerometer only
  oneshot      : unsure Sit/Stand at w -> gyro on for one window (w+2); its features answer window w
  sticky       : unsure Sit/Stand -> gyro on, and it stays on while stage 1 keeps predicting Sit/Stand.
                 Any Sit/Stand window fully covered by the gyro uses stage 2 on its own gyro features;
                 the triggering window keeps the stage-1 answer.
  +smooth      : causal majority vote over the last 3 decisions (ties keep the current one)
Reported: accuracy, Sit/Stand error, and % of time the gyro is powered (hops on / all hops).
"""
import os

import numpy as np
import torch
from scipy.signal import butter, lfilter, medfilt
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

from baseline import train
from features import FS, STEP, WIN, split_gravity, window_features
from stage2_uci import MAP, SIT, STAND, UCI, lr

TAUS = [0, 0.5, 1, 2, np.inf]  # tau=0 is a sanity check: every policy must equal stage 1, gyro 0%


def build():
    """Per experiment: X7, gyro std (x, z), label (-1 = not scored), subject, on a continuous window grid."""
    noise = butter(3, 20, fs=FS)
    labels = np.loadtxt(os.path.join(UCI, 'RawData', 'labels.txt'), dtype=int)
    exps = []
    for exp in np.unique(labels[:, 0]):
        rows = labels[labels[:, 0] == exp]
        user = rows[0, 1]
        acc = np.loadtxt(os.path.join(UCI, 'RawData', f'acc_exp{exp:02d}_user{user:02d}.txt'))
        gyr = lfilter(*noise, medfilt(np.loadtxt(os.path.join(UCI, 'RawData', f'gyro_exp{exp:02d}_user{user:02d}.txt')), (3, 1)), axis=0)
        body, grav = split_gravity(acc)
        lab = np.zeros(len(acc), int)
        for _, _, act, a, b in rows:
            lab[a - 1:b] = act
        starts = list(range(0, len(acc) - WIN + 1, STEP))
        y = np.array([MAP[lab[s]] if lab[s] and (lab[s:s + WIN] == lab[s]).all() else -1 for s in starts])
        G = np.array([gyr[s:s + WIN][:, [0, 2]].std(0) for s in starts])
        exps.append(dict(X=window_features(body, grav, starts), G=G, y=y, subject=user))
    return exps


def majority3(p):
    out = p.copy()
    for i in range(2, len(p)):
        v, c = np.unique(p[i - 2:i + 1], return_counts=True)
        if c.max() >= 2:
            out[i] = v[c.argmax()]
    return out


def simulate(e, m1, sc, m2, tau):
    X, G = e['X'], e['G']
    with torch.no_grad():
        logits = m1(torch.tensor(sc.transform(X), dtype=torch.float32)).numpy()
    p1 = logits.argmax(1)
    t2 = np.sort(logits, 1)[:, -2:]
    unsure = np.isin(p1, [SIT, STAND]) & (t2[:, 1] - t2[:, 0] < tau)
    p2_own = m2.predict(np.c_[X, G])
    n = len(p1); hops = n + 1

    # oneshot: window w answered with gyro from window w+2
    one, on1 = p1.copy(), np.zeros(hops + 2, bool)
    for w in np.flatnonzero(unsure):
        if w + 2 < n:
            one[w] = m2.predict(np.c_[X[w:w + 1], G[w + 2:w + 3]])[0]
            on1[w + 2:w + 4] = True

    # sticky: on after an unsure Sit/Stand decision, off after any non-Sit/Stand decision
    st, on2, gyro_on_from = p1.copy(), np.zeros(hops + 2, bool), None
    for w in range(n):
        covered = gyro_on_from is not None and gyro_on_from <= w
        if covered and p1[w] in (SIT, STAND):
            st[w] = p2_own[w]
        if p1[w] not in (SIT, STAND):
            if gyro_on_from is not None:
                on2[gyro_on_from:w + 2] = True
            gyro_on_from = None
        elif unsure[w] and gyro_on_from is None:
            gyro_on_from = w + 2
    if gyro_on_from is not None:
        on2[gyro_on_from:hops] = True

    preds = dict(stage1=p1, oneshot=one, sticky=st,
                 **{'stage1+smooth': majority3(p1), 'sticky+smooth': majority3(st)})
    on = dict(stage1=0, oneshot=on1[:hops].sum(), sticky=on2[:hops].sum(),
              **{'stage1+smooth': 0, 'sticky+smooth': on2[:hops].sum()})
    return preds, on, hops


class NetStage2:
    """Small ReLU6 MLP for Sit vs Stand, with the predict() interface simulate() expects."""

    def __init__(self, hidden):
        self.hidden = hidden

    def fit(self, Z, y, g, seed=0):
        a, b = next(GroupShuffleSplit(1, test_size=0.15, random_state=seed).split(Z, groups=g))
        self.sc = StandardScaler().fit(Z[a])
        self.m = train(self.sc.transform(Z[a]), y[a], self.sc.transform(Z[b]), y[b], seed, self.hidden)
        return self

    def predict(self, Z):
        with torch.no_grad():
            out = self.m(torch.tensor(self.sc.transform(Z), dtype=torch.float32)).numpy()
        return np.where(out[:, SIT] >= out[:, STAND], SIT, STAND)


def fit_models(exps, train_idx, seed=0, h1=6, h2=None):
    """Stage 1 (7-h1-5 ReLU6 MLP) and stage 2 (logistic regression, or a ReLU6 MLP with h2 hidden)."""
    tr = [exps[i] for i in train_idx]
    X = np.vstack([e['X'] for e in tr]); y = np.concatenate([e['y'] for e in tr])
    G = np.vstack([e['G'] for e in tr]); g = np.concatenate([[e['subject']] * len(e['y']) for e in tr])
    keep = y >= 0
    X, y, G, g = X[keep], y[keep].astype(np.int64), G[keep], g[keep]
    a, b = next(GroupShuffleSplit(1, test_size=0.15, random_state=seed).split(X, groups=g))
    sc = StandardScaler().fit(X[a])
    m1 = train(sc.transform(X[a]), y[a], sc.transform(X[b]), y[b], seed, h1)
    ss = np.isin(y, [SIT, STAND])
    Z = np.c_[X[ss], G[ss]]
    m2 = lr().fit(Z, y[ss]) if h2 is None else NetStage2(h2).fit(Z, y[ss], g[ss], seed)
    return m1, sc, m2


def evaluate(exps, train_idx, test_idx, tau, seed=0, models=None):
    m1, sc, m2 = models if models is not None else fit_models(exps, train_idx, seed)
    tot = {}
    for i in test_idx:
        preds, on, hops = simulate(exps[i], m1, sc, m2, tau)
        yt = exps[i]['y']; s = yt >= 0
        for k, p in preds.items():
            t = tot.setdefault(k, np.zeros(5))
            ssm = s & np.isin(yt, [SIT, STAND])
            t += [(p[s] == yt[s]).sum(), s.sum(), (np.isin(p[ssm], [SIT, STAND]) & (p[ssm] != yt[ssm])).sum(), ssm.sum(), 0]
            t[4] += on[k]
        tot['_hops'] = tot.get('_hops', 0) + hops
    return {k: dict(acc=100 * v[0] / v[1], sserr=100 * v[2] / v[3], gyro_on=100 * v[4] / tot['_hops'])
            for k, v in tot.items() if k != '_hops'}


def main():
    exps = build()
    subj = np.array([e['subject'] for e in exps])
    test_subj = set(np.loadtxt(os.path.join(UCI, 'Test', 'subject_id_test.txt'), dtype=int))
    off_tr, off_te = np.flatnonzero(~np.isin(subj, list(test_subj))), np.flatnonzero(np.isin(subj, list(test_subj)))
    folds = list(GroupKFold(5).split(subj, subj, subj))
    print(f"scored windows: {sum((e['y'] >= 0).sum() for e in exps)}  (continuous grid; previous runs used 10929)")
    for tau in TAUS:
        off = evaluate(exps, off_tr, off_te, tau)
        cv = [evaluate(exps, a, b, tau) for a, b in folds]
        print(f'\n== tau = {tau}')
        print(f"{'policy':15s} | {'official acc':>12} {'sserr':>6} {'gyro on':>8} | {'5-fold acc':>10} {'sserr':>6} {'gyro on':>8}")
        for k in off:
            c = {m: np.mean([f[k][m] for f in cv]) for m in ('acc', 'sserr', 'gyro_on')}
            o = off[k]
            print(f"{k:15s} | {o['acc']:12.2f} {o['sserr']:5.1f}% {o['gyro_on']:7.1f}% | {c['acc']:10.2f} {c['sserr']:5.1f}% {c['gyro_on']:7.1f}%", flush=True)


if __name__ == '__main__':
    main()
