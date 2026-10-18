"""General early-exit gate: ANY window where stage 1 is unsure (top-2 logit margin < tau) goes to a
multi-class stage 2 that sees stage-1 features + the second sensor (UCI: gyro std x,z; PAMAP2: wrist accel 7).

Policies (same timing model as gyro_policies.py: a sensor switched on after window w first covers w+2):
  oneshot : unsure at w -> sensor on for window w+2, whose features answer window w
  sticky  : unsure -> sensor on; it stays on until stage 1 makes a confident decision.
            Covered unsure windows use stage 2 on their own features; the triggering window keeps stage 1.
Stage 1 = 7-6-5 ReLU6 MLP; stage 2 = n2-h2-5 ReLU6 MLP trained on all labelled training windows.
"""
import os

import numpy as np
import torch
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

from baseline import train
from gyro_policies import UCI, build as build_uci
from pamap2_cascade import build as build_pamap2

TAUS = [0, 0.5, 1, 2, np.inf]
SIT, STAND = 1, 2


def net(X, y, g, hidden, seed=0):
    a, b = next(GroupShuffleSplit(1, test_size=0.15, random_state=seed).split(X, groups=g))
    sc = StandardScaler().fit(X[a])
    m = train(sc.transform(X[a]), y[a], sc.transform(X[b]), y[b], seed, hidden)

    def logits(Z):
        with torch.no_grad():
            return m(torch.tensor(sc.transform(Z), dtype=torch.float32)).numpy()
    return logits


def fit(exps, tr_idx, h2):
    tr = [exps[i] for i in tr_idx]
    X = np.vstack([e['X'] for e in tr]); G = np.vstack([e['G'] for e in tr])
    y = np.concatenate([e['y'] for e in tr]); g = np.concatenate([[e['subject']] * len(e['y']) for e in tr])
    k = y >= 0
    X, G, y, g = X[k], G[k], y[k].astype(np.int64), g[k]
    return net(X, y, g, 6), net(np.c_[X, G], y, g, h2)


def simulate(e, s1, s2, tau):
    X, G = e['X'], e['G']
    l1 = s1(X); p1 = l1.argmax(1)
    t = np.sort(l1, 1)[:, -2:]
    unsure = (t[:, 1] - t[:, 0]) < tau
    p2_own = s2(np.c_[X, G]).argmax(1)
    n = len(p1); hops = n + 1

    one, on1 = p1.copy(), np.zeros(hops + 2, bool)
    idx = np.flatnonzero(unsure); idx = idx[idx + 2 < n]
    if len(idx):
        one[idx] = s2(np.c_[X[idx], G[idx + 2]]).argmax(1)
        for w in idx:
            on1[w + 2:w + 4] = True

    st, on2, on_from = p1.copy(), np.zeros(hops + 2, bool), None
    for w in range(n):
        if on_from is not None and on_from <= w and unsure[w]:
            st[w] = p2_own[w]
        if not unsure[w]:
            if on_from is not None:
                on2[on_from:w + 2] = True
            on_from = None
        elif on_from is None:
            on_from = w + 2
    if on_from is not None:
        on2[on_from:hops] = True
    return dict(stage1=p1, oneshot=one, sticky=st), dict(stage1=0, oneshot=on1[:hops].sum(), sticky=on2[:hops].sum()), hops


def score(exps, te_idx, models, tau, cm_for=None):
    tot, hops_all, cm = {}, 0, np.zeros((5, 5), int)
    for i in te_idx:
        preds, on, hops = simulate(exps[i], *models, tau)
        y = exps[i]['y']; s = y >= 0; m = s & np.isin(y, [SIT, STAND])
        for k, p in preds.items():
            t = tot.setdefault(k, np.zeros(5))
            t += [(p[s] == y[s]).sum(), s.sum(), (np.isin(p[m], [SIT, STAND]) & (p[m] != y[m])).sum(), m.sum(), on[k]]
        if cm_for:
            np.add.at(cm, (y[s], preds[cm_for][s]), 1)
        hops_all += hops
    out = {k: dict(acc=100 * v[0] / v[1], sserr=100 * v[2] / v[3], on=100 * v[4] / hops_all) for k, v in tot.items()}
    return out, cm


def report(name, exps, splits, n2):
    print(f'\n######## {name}')
    print(f"{'stage2':>8} {'MACs':>9} {'tau':>5} {'policy':8s} | {'acc':>6} {'sserr':>6} {'sensor on':>9}")
    for h2 in (8, 16):
        fits = [(te, fit(exps, tr, h2)) for tr, te in splits]
        for tau in TAUS:
            res = [score(exps, te, m, tau)[0] for te, m in fits]
            for k in (('stage1',) if tau == 0 else ('oneshot', 'sticky')):
                c = {q: np.mean([r[k][q] for r in res]) for q in ('acc', 'sserr', 'on')}
                print(f"{f'{n2}-{h2}-5':>8} {f'72+{n2 * h2 + h2 * 5}':>9} {tau:>5} {k:8s} | {c['acc']:6.2f} {c['sserr']:5.1f}% {c['on']:8.1f}%", flush=True)
    # where do the errors go? stage 1 vs the general sticky cascade (h2 = 16, tau = 1)
    cms = [score(exps, te, m, 1, cm_for=k)[1] for k in ('stage1', 'sticky') for te, m in [fits[0]]] if len(splits) == 1 else None
    if cms is None:
        cm1 = sum(score(exps, te, m, 1, cm_for='stage1')[1] for te, m in fits)
        cm2 = sum(score(exps, te, m, 1, cm_for='sticky')[1] for te, m in fits)
    else:
        cm1, cm2 = cms
    present = [c for c in range(5) if cm1[c].sum()]
    print('  confusion (rows = true, cols = predicted; classes Walk Sit Stand Lay Trans), stage 1 then sticky tau=1, h2=16:')
    for a, b in zip(cm1[present][:, present], cm2[present][:, present]):
        print('   ', ' '.join(f'{v:5d}' for v in a), '   |   ', ' '.join(f'{v:5d}' for v in b))


def main():
    uci = build_uci()
    subj = np.array([e['subject'] for e in uci])
    test_subj = list(set(np.loadtxt(os.path.join(UCI, 'Test', 'subject_id_test.txt'), dtype=int)))
    report('UCI official split', uci, [(np.flatnonzero(~np.isin(subj, test_subj)), np.flatnonzero(np.isin(subj, test_subj)))], 9)
    report('UCI 5-fold by subject', uci, list(GroupKFold(5).split(subj, subj, subj)), 9)
    pam = build_pamap2()['wrist accel']
    n = len(pam)
    report('PAMAP2 leave-one-subject-out', pam, [(np.array([j for j in range(n) if j != i]), np.array([i])) for i in range(n)], 14)


if __name__ == '__main__':
    main()
