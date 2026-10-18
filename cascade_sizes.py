"""Cascade with a bigger stage 1 and/or a small-network stage 2, on UCI-HAPT and PAMAP2.

Compared with the always-on networks from uci_upper_bound.py / pamap2_upper_bound.py
(UCI 9-16-5: 95.13 official / 95.48 5-fold; PAMAP2 14-16-5: 89.37, 14-64-5: 90.96).
MACs per window: stage 1 = 7*h1 + h1*5 always; stage 2 = n2 (linear) or n2*h2 + h2*5, only when routed.
"""
import os

import numpy as np
from sklearn.model_selection import GroupKFold

from gyro_policies import UCI, build as build_uci, evaluate, fit_models
from pamap2_cascade import build as build_pamap2

CONFIGS = [(6, None), (16, None), (6, 8), (16, 8)]  # (stage-1 hidden, stage-2 hidden or None = linear)
TAUS = [0, 0.5, 1, np.inf]


def macs(h1, h2, n2):
    return 7 * h1 + h1 * 5, (n2 if h2 is None else n2 * h2 + h2 * 5)


def report(name, exps, splits, n2):
    print(f'\n######## {name}')
    print(f"{'stage1':>7} {'stage2':>7} {'MACs s1+s2':>11} {'tau':>5} {'policy':8s} | {'acc':>6} {'sserr':>6} {'sensor on':>9}")
    for h1, h2 in CONFIGS:
        fits = [(te, fit_models(exps, tr, h1=h1, h2=h2)) for tr, te in splits]
        m1c, m2c = macs(h1, h2, n2)
        for tau in TAUS:
            res = [evaluate(exps, None, te, tau, models=m) for te, m in fits]
            for k in (('stage1',) if tau == 0 else ('oneshot', 'sticky')):
                c = {m: np.mean([r[k][m] for r in res]) for m in ('acc', 'sserr', 'gyro_on')}
                s2 = 'linear' if h2 is None else f'{n2}-{h2}'
                print(f"{'7-' + str(h1) + '-5':>7} {s2:>7} {f'{m1c}+{m2c}':>11} {tau:>5} {k:8s} | "
                      f"{c['acc']:6.2f} {c['sserr']:5.1f}% {c['gyro_on']:8.1f}%", flush=True)


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
