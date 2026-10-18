"""Nested gate-threshold selection over 5 seeds: tau is chosen on training subjects only, never on the test set.

For each seed and each outer fold:
  1. Inner GroupKFold(3) over the training subjects: train the cascade on 2/3, simulate the hardware state machine
     (cascade_hw.policy) on the held-out third for every tau in TAUS, pool the inner accuracies.
  2. Rule fixed in advance: the smallest tau whose inner accuracy is within 0.5 points of the inner accuracy at
     tau = inf (sensor on whenever the mask allows). Small tau = less sensor time.
  3. Retrain on all training subjects, apply the chosen tau to the test subjects.
Also reported on the same seeds and folds: stage 1 alone, the frozen tau (1 UCI, 2 PAMAP2, chosen earlier by
looking at test results), and an always-on n2-16-5 network that sees the second sensor on every window.
Architectures stay frozen (UCI stage 2 9-8-5 Sit/Stand, PAMAP2 14-16-5): this run tests the threshold choice.
Usage: python nested_tau.py uci_official | uci_5fold | pamap2
"""
import json
import os
import sys

import numpy as np
import torch
from sklearn.model_selection import GroupKFold

from cascade_hw import ALL, SS, pamap2_models, policy
from general_gate import net
from gyro_policies import UCI, build as build_uci, fit_models

HERE = os.path.dirname(os.path.abspath(__file__))
TAUS = [0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, np.inf]
SEEDS = range(5)
TOL = 0.5
torch.set_num_threads(3)


def cfg_for(ds, tau):
    if ds == 'pamap2':
        return dict(M=ALL, M2=ALL, tau=tau, tau_keep=tau, keep_all=False)
    return dict(M=SS, M2=SS, tau=tau, tau_keep=0.0, keep_all=True)


def T(m, sc, Z):
    with torch.no_grad():
        return m(torch.tensor(sc.transform(Z), dtype=torch.float32)).numpy()


def models(ds, exps, idx, seed):
    with torch.enable_grad():
        if ds == 'pamap2':
            m1, sc1, m2, sc2 = pamap2_models(exps, idx, 16, seed)
        else:
            m1, sc1, s2 = fit_models(exps, idx, seed, h2=8)
            m2, sc2 = s2.m, s2.sc
    return lambda X: T(m1, sc1, X), lambda Z: T(m2, sc2, Z)


def streams(exps, idx, f1, f2):
    return [(f1(exps[i]['X']), f2(np.c_[exps[i]['X'], exps[i]['G']]), exps[i]['y']) for i in idx]


def always_on(exps, tr, te, seed):
    """n2-16-5 network on [X, G] for every window (the second sensor always powered)."""
    X = np.vstack([exps[i]['X'] for i in tr]); G = np.vstack([exps[i]['G'] for i in tr])
    y = np.concatenate([exps[i]['y'] for i in tr]); g = np.concatenate([[exps[i]['subject']] * len(exps[i]['y']) for i in tr])
    k = y >= 0
    with torch.enable_grad():
        f = net(np.c_[X, G][k], y[k].astype(np.int64), g[k], 16, seed)
    c = np.zeros(2)
    for i in te:
        e = exps[i]; s = e['y'] >= 0
        c += [(f(np.c_[e['X'], e['G']])[s].argmax(1) == e['y'][s]).sum(), s.sum()]
    return c


def main(proto):
    ds = 'pamap2' if proto == 'pamap2' else 'uci'
    if ds == 'pamap2':
        from pamap2_cascade import build as build_pamap2
        exps = build_pamap2()['wrist accel']
    else:
        exps = build_uci()
    subj = np.array([e['subject'] for e in exps])
    if proto == 'uci_official':
        test_subj = list(set(np.loadtxt(os.path.join(UCI, 'Test', 'subject_id_test.txt'), dtype=int)))
        outer = [(np.flatnonzero(~np.isin(subj, test_subj)), np.flatnonzero(np.isin(subj, test_subj)))]
    elif proto == 'uci_5fold':
        outer = list(GroupKFold(5).split(subj, subj, subj))
    else:
        outer = [(np.flatnonzero(subj != s), np.flatnonzero(subj == s)) for s in np.unique(subj)]
    frozen = 2.0 if ds == 'pamap2' else 1.0

    runs = []
    for seed in SEEDS:
        tot = {k: np.zeros(3) for k in ('stage1', 'nested', 'frozen')}
        ao, chosen = np.zeros(2), []
        for tr, te in outer:
            # 1. inner selection on training subjects only
            inner = {t: np.zeros(3) for t in TAUS}
            for a, b in GroupKFold(3).split(tr, groups=subj[tr]):
                st = streams(exps, tr[b], *models(ds, exps, tr[a], seed))
                for t in TAUS:
                    for l1, l2, y in st:
                        cls, en, _ = policy(l1, l2, cfg_for(ds, t)); s = y >= 0
                        inner[t] += [(cls[s] == y[s]).sum(), s.sum(), en.sum()]
            acc = {t: 100 * v[0] / v[1] for t, v in inner.items()}
            tau = min(t for t in TAUS if acc[t] >= acc[np.inf] - TOL)
            chosen.append(tau)
            # 2. retrain on all training subjects, evaluate on test subjects
            st = streams(exps, te, *models(ds, exps, tr, seed))
            for k, t in (('stage1', 0.0), ('nested', tau), ('frozen', frozen)):
                for l1, l2, y in st:
                    cls, en, _ = policy(l1, l2, cfg_for(ds, t)); s = y >= 0
                    tot[k] += [(cls[s] == y[s]).sum(), s.sum(), en.sum()]
            ao += always_on(exps, tr, te, seed)
        nwin = sum(len(exps[i]['y']) for _, te in outer for i in te)
        r = {k: dict(acc=100 * v[0] / v[1], sensor=100 * v[2] / nwin) for k, v in tot.items()}
        r['always_on'] = dict(acc=100 * ao[0] / ao[1], sensor=100.0)
        r['chosen_tau'] = [float(t) for t in chosen]
        runs.append(r)
        print(f"{proto} seed {seed}: " + '  '.join(f"{k} {r[k]['acc']:.2f}% ({r[k]['sensor']:.1f}%)" for k in ('stage1', 'nested', 'frozen', 'always_on'))
              + f"  tau {sorted(set(r['chosen_tau']))}", flush=True)

    print(f'\n== {proto}: mean +- sd over {len(runs)} seeds (accuracy pooled over test windows; sensor = % of windows on)')
    for k in ('stage1', 'nested', 'frozen', 'always_on'):
        a = np.array([r[k]['acc'] for r in runs]); s = np.array([r[k]['sensor'] for r in runs])
        print(f'  {k:10s} acc {a.mean():6.2f} +- {a.std(ddof=1):.2f}   sensor on {s.mean():5.1f} +- {s.std(ddof=1):.1f}%')
    taus = [t for r in runs for t in r['chosen_tau']]
    print('  chosen tau counts:', {str(t): taus.count(t) for t in sorted(set(taus))})
    with open(os.path.join(HERE, 'data', f'nested_tau_{proto}.json'), 'w') as f:
        json.dump(runs, f, indent=1)


if __name__ == '__main__':
    main(sys.argv[1])
