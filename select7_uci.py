"""Would a different 7 of the 561 UCI features make a better stage 1?

Forward-select 7 features with LDA (fast) using grouped CV on the official TRAINING subjects only,
from three pools of increasing hardware cost, then test each set with the paper's 7-6-5 ReLU6 MLP
on the official test subjects. The paper's 7 features are the reference.
"""
import os

import numpy as np
import torch
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

from baseline import train
from features import NAMES as PAPER7

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, '..', 'exact_paper_implementation', 'data', 'UCI HAR Dataset')
MAP = {1: 0, 2: 0, 3: 0, 4: 1, 5: 2, 6: 3, **{k: 4 for k in range(7, 13)}}

ld = lambda p: np.loadtxt(os.path.join(D, p))
names = [l.split()[-1] for l in open(os.path.join(D, 'features.txt')) if l.strip()]
Xtr, Xte = ld('Train/X_train.txt'), ld('Test/X_test.txt')
ytr = np.array([MAP[int(v)] for v in ld('Train/y_train.txt')]); yte = np.array([MAP[int(v)] for v in ld('Test/y_test.txt')])
gtr = ld('Train/subject_id_train.txt').astype(int)

acc = [i for i, n in enumerate(names) if 'Gyro' not in n]
POOLS = {
    'accel, time-domain only': [i for i in acc if names[i].startswith(('t', 'angle'))],
    'accel incl. FFT features': acc,
    'all 561 (gyro included)': list(range(len(names))),
}


def forward7(pool):
    chosen = []
    for _ in range(7):
        best = None
        for j in pool:
            if j in chosen:
                continue
            cols = chosen + [j]
            s = np.mean([balanced_accuracy_score(ytr[b], LinearDiscriminantAnalysis().fit(Xtr[a][:, cols], ytr[a]).predict(Xtr[b][:, cols]))
                         for a, b in GroupKFold(5).split(Xtr, ytr, gtr)])
            if best is None or s > best[1]:
                best = (j, s)
        chosen.append(best[0])
    return chosen


def mlp_test(cols, seeds=(0, 1, 2)):
    accs, ss = [], []
    for seed in seeds:
        a, b = next(GroupShuffleSplit(1, test_size=0.15, random_state=seed).split(Xtr, groups=gtr))
        sc = StandardScaler().fit(Xtr[a][:, cols])
        m = train(sc.transform(Xtr[a][:, cols]), ytr[a].astype(np.int64), sc.transform(Xtr[b][:, cols]), ytr[b].astype(np.int64), seed)
        with torch.no_grad():
            p = m(torch.tensor(sc.transform(Xte[:, cols]), dtype=torch.float32)).argmax(1).numpy()
        accs.append(100 * (p == yte).mean())
        m2 = np.isin(yte, [1, 2])
        ss.append(100 * (np.isin(p[m2], [1, 2]) & (p[m2] != yte[m2])).mean())
    return np.mean(accs), np.std(accs), np.mean(ss)


paper = [names.index(n) for n in PAPER7]
mu, sd, ss = mlp_test(paper)
print(f"{'paper 7 features':28s} test acc {mu:5.2f} +/- {sd:.2f}   sit/stand err {ss:4.1f}%", flush=True)
for label, pool in POOLS.items():
    cols = forward7(pool)
    mu, sd, ss = mlp_test(cols)
    print(f"{label:28s} test acc {mu:5.2f} +/- {sd:.2f}   sit/stand err {ss:4.1f}%   {[names[c] for c in cols]}", flush=True)
