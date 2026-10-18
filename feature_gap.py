"""Is the 94.8% vs 97% gap (TIM 2025, 19 features) about the number of features?

Rebuilds split A exactly (same stratified splits, seed 42) with all 561 features, then trains a
small ReLU6-style MLP on 7, 19 and more features chosen by mutual information on the training set only.
"""
import os

import numpy as np
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
P = os.path.join(HERE, '..', 'exact_paper_implementation')
D = os.path.join(P, 'data', 'UCI HAR Dataset')
MAP = {1: 0, 2: 0, 3: 0, 4: 1, 5: 2, 6: 3, **{k: 4 for k in range(7, 13)}}

ld = lambda f: np.loadtxt(os.path.join(D, f))
X = np.vstack([ld('Train/X_train.txt'), ld('Test/X_test.txt')])
y = np.array([MAP[int(v)] for v in np.concatenate([ld('Train/y_train.txt'), ld('Test/y_test.txt')])])
names = [l.split()[-1] for l in open(os.path.join(D, 'features.txt')) if l.strip()]
SEVEN = [names.index(n) for n in ['tBodyAcc-STD-1', 'tBodyAcc-STD-2', 'tBodyAcc-STD-3', 'tBodyAcc-SMA-1',
                                  'tGravityAcc-Mean-1', 'tGravityAcc-Mean-2', 'tGravityAcc-Mean-3']]

# split A: 55/30/15, exactly as data_preprocessing.py
Xtv, Xte, ytv, yte = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
Xtr, Xva, ytr, yva = train_test_split(Xtv, ytv, test_size=0.30 / 0.85, random_state=42, stratify=ytv)

saved = np.load(os.path.join(P, 'relu6_implementation', 'data', 'processed', 'dataset_a', 'X_test.npy'))
sc7 = StandardScaler().fit(Xtr[:, SEVEN])
assert np.allclose(sc7.transform(Xte[:, SEVEN]), saved), 'split rebuild does not match the saved split'

mi = mutual_info_classif(Xtr, ytr, random_state=0)
order = np.argsort(-mi)


def run(cols, hidden):
    s = StandardScaler().fit(Xtr[:, cols])
    accs = []
    for seed in range(3):
        m = MLPClassifier(hidden, activation='relu', max_iter=2000, early_stopping=True, random_state=seed)
        m.fit(s.transform(Xtr[:, cols]), ytr)
        accs.append((m.predict(s.transform(Xte[:, cols])) == yte).mean())
    return 100 * np.mean(accs), 100 * np.std(accs)


print('split A rebuilt and matches the saved split')
for label, cols, hidden in [('paper 7 features, 6 hidden', SEVEN, (6,)),
                            ('paper 7 features, 32 hidden', SEVEN, (32,)),
                            ('top-19 MI features, 6 hidden', order[:19], (6,)),
                            ('top-19 MI features, 32 hidden', order[:19], (32,)),
                            ('7 paper + top-12 MI (19), 6 hidden', list(SEVEN) + [i for i in order if i not in SEVEN][:12], (6,)),
                            ('all 561 features, 32 hidden', np.arange(561), (32,))]:
    mu, sd = run(list(cols), hidden)
    print(f'{label:38s} {mu:5.2f} +/- {sd:.2f}', flush=True)
