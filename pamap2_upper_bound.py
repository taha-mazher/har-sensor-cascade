"""Upper bound for the PAMAP2 cascade: chest + wrist features both always on, one 14-6-5 ReLU6 MLP.

Same continuous grid, scored windows and leave-one-subject-out folds as pamap2_cascade.py, so the numbers
compare directly with the cascade (which powers the wrist sensor only 22-34% of the time).
Also reports chest-only (7-6-5) from the same runs as the stage-1 reference.
"""
import numpy as np
import torch
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

from baseline import train
from pamap2_cascade import build

SIT, STAND = 1, 2


def loso(exps, cols, hidden=6):
    accs, ss_err, ss_n, corr, n = [], 0, 0, 0, 0
    for i in range(len(exps)):
        tr = [e for j, e in enumerate(exps) if j != i]
        X = np.vstack([np.c_[e['X'], e['G']] for e in tr])[:, cols]
        y = np.concatenate([e['y'] for e in tr]); g = np.concatenate([[e['subject']] * len(e['y']) for e in tr])
        k = y >= 0
        X, y, g = X[k], y[k].astype(np.int64), g[k]
        a, b = next(GroupShuffleSplit(1, test_size=0.15, random_state=0).split(X, groups=g))
        sc = StandardScaler().fit(X[a])
        m = train(sc.transform(X[a]), y[a], sc.transform(X[b]), y[b], 0, hidden)
        e = exps[i]; s = e['y'] >= 0
        with torch.no_grad():
            p = m(torch.tensor(sc.transform(np.c_[e['X'], e['G']][s][:, cols]), dtype=torch.float32)).argmax(1).numpy()
        yt = e['y'][s]
        accs.append(100 * (p == yt).mean())
        mm = np.isin(yt, [SIT, STAND])
        ss_err += (np.isin(p[mm], [SIT, STAND]) & (p[mm] != yt[mm])).sum(); ss_n += mm.sum()
    return np.mean(accs), 100 * ss_err / ss_n


def main():
    exps = build()['wrist accel']
    for label, cols, h in [('chest only, 7-6-5 (stage 1 reference)', list(range(7)), 6),
                           ('chest + wrist always on, 14-6-5', list(range(14)), 6),
                           ('chest + wrist always on, 14-16-5', list(range(14)), 16),
                           ('chest + wrist always on, 14-64-5', list(range(14)), 64)]:
        acc, sserr = loso(exps, cols, h)
        print(f'{label:48s} acc {acc:6.2f}  sit/stand err {sserr:5.1f}%  wrist sensor on {0 if len(cols) == 7 else 100}%', flush=True)


if __name__ == '__main__':
    main()
