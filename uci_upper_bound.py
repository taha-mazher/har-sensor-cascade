"""Upper bound for the UCI cascade: waist accelerometer 7 features + gyro std (x, z) always on, one 9-6-5 MLP.

Same continuous grid, scored windows, official split and 5 subject folds as gyro_policies.py, so it compares
directly with the sticky cascade (gyro on ~14% of the time).
"""
import os

import numpy as np
import torch
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

from baseline import train
from gyro_policies import build
from stage2_uci import SIT, STAND, UCI


def run(exps, tr_idx, te_idx, cols, hidden=6):
    tr = [exps[i] for i in tr_idx]
    X = np.vstack([np.c_[e['X'], e['G']] for e in tr])[:, cols]
    y = np.concatenate([e['y'] for e in tr]); g = np.concatenate([[e['subject']] * len(e['y']) for e in tr])
    k = y >= 0
    X, y, g = X[k], y[k].astype(np.int64), g[k]
    a, b = next(GroupShuffleSplit(1, test_size=0.15, random_state=0).split(X, groups=g))
    sc = StandardScaler().fit(X[a])
    m = train(sc.transform(X[a]), y[a], sc.transform(X[b]), y[b], 0, hidden)
    c = np.zeros(4)
    for i in te_idx:
        e = exps[i]; s = e['y'] >= 0
        with torch.no_grad():
            p = m(torch.tensor(sc.transform(np.c_[e['X'], e['G']][s][:, cols]), dtype=torch.float32)).argmax(1).numpy()
        yt = e['y'][s]; mm = np.isin(yt, [SIT, STAND])
        c += [(p == yt).sum(), len(yt), (np.isin(p[mm], [SIT, STAND]) & (p[mm] != yt[mm])).sum(), mm.sum()]
    return 100 * c[0] / c[1], 100 * c[2] / c[3]


def main():
    exps = build()
    subj = np.array([e['subject'] for e in exps])
    test_subj = list(set(np.loadtxt(os.path.join(UCI, 'Test', 'subject_id_test.txt'), dtype=int)))
    off = (np.flatnonzero(~np.isin(subj, test_subj)), np.flatnonzero(np.isin(subj, test_subj)))
    folds = list(GroupKFold(5).split(subj, subj, subj))
    for label, cols, h in [('accel only, 7-6-5 (stage 1 reference)', list(range(7)), 6),
                           ('accel + gyro always on, 9-6-5', list(range(9)), 6),
                           ('accel + gyro always on, 9-16-5', list(range(9)), 16),
                           ('accel + gyro always on, 9-64-5', list(range(9)), 64)]:
        oa, os_ = run(exps, *off, cols, h)
        cv = [run(exps, a, b, cols, h) for a, b in folds]
        print(f'{label:46s} official {oa:6.2f} (sserr {os_:4.1f}%) | 5-fold {np.mean([c[0] for c in cv]):6.2f} '
              f'(sserr {np.mean([c[1] for c in cv]):4.1f}%) | gyro on {0 if len(cols) == 7 else 100}%', flush=True)


if __name__ == '__main__':
    main()
