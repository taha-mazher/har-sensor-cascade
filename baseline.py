"""Stage-1 baseline: the paper's 7-6-5 ReLU6 MLP on three datasets, evaluated across subjects.

Protocols (no subject ever appears in both training and test):
  UCI-HAPT : official split (21 training / 9 test subjects), plus 5-fold grouped CV
  PAMAP2   : leave-one-subject-out (8 folds)
  WISDM    : 6-fold grouped CV over 36 users
Also, for UCI only, the AEJ paper's random window split, to show how much subject leakage inflates accuracy.
Early stopping uses a validation set of held-out *training* subjects; the scaler is fit on training data only.
Output layer is always 5 wide so the hardware is the same for every dataset.
"""
import json
import os

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, LeaveOneGroupOut, train_test_split
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
CLASSES = ['Walking', 'Sitting', 'Standing', 'Laying', 'Transitions']
torch.set_num_threads(6)


class MLP(torch.nn.Module):
    def __init__(self, n_in=7, hidden=6):
        super().__init__()
        self.l1, self.l2 = torch.nn.Linear(n_in, hidden), torch.nn.Linear(hidden, 5)

    def forward(self, x):
        return self.l2(torch.clamp(self.l1(x), 0, 6))  # ReLU6


def train(Xtr, ytr, Xva, yva, seed, hidden=6):
    torch.manual_seed(seed)
    m, best, bad, state = MLP(Xtr.shape[1], hidden), np.inf, 0, None
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    Xt, yt = torch.tensor(Xtr, dtype=torch.float32), torch.tensor(ytr)
    Xv, yv = torch.tensor(Xva, dtype=torch.float32), torch.tensor(yva)
    lossf = torch.nn.CrossEntropyLoss()
    for _ in range(400):
        m.train()
        for idx in torch.randperm(len(Xt)).split(64):
            opt.zero_grad(); lossf(m(Xt[idx]), yt[idx]).backward(); opt.step()
        m.eval()
        with torch.no_grad():
            v = lossf(m(Xv), yv).item()
        if v < best - 1e-4:
            best, bad, state = v, 0, {k: t.clone() for k, t in m.state_dict().items()}
        elif (bad := bad + 1) >= 30:
            break
    m.load_state_dict(state)
    return m


def fit_predict(X, y, g, tr, te, seed=0):
    # early-stopping validation = 15% of the training subjects (or windows, if no groups)
    if g is None:
        a, b = train_test_split(tr, test_size=0.3 / 0.85, random_state=seed, stratify=y[tr])
    else:
        a, b = next(GroupShuffleSplit(1, test_size=0.15, random_state=seed).split(tr, groups=g[tr]))
        a, b = tr[a], tr[b]
    sc = StandardScaler().fit(X[a])
    m = train(sc.transform(X[a]), y[a], sc.transform(X[b]), y[b], seed)
    with torch.no_grad():
        return m(torch.tensor(sc.transform(X[te]), dtype=torch.float32)).argmax(1).numpy()


def score(y, p):
    present = sorted(set(y.tolist()))
    cm = confusion_matrix(y, p, labels=range(5))
    recall = {CLASSES[c]: round(100 * cm[c, c] / cm[c].sum(), 1) for c in present}
    return dict(acc=round(100 * (p == y).mean(), 2), bal_acc=round(100 * balanced_accuracy_score(y, p), 2),
                recall=recall, sit_as_stand=int(cm[1, 2]), stand_as_sit=int(cm[2, 1]),
                sit_stand_err=round(100 * (cm[1, 2] + cm[2, 1]) / max(1, cm[1].sum() + cm[2].sum()), 1),
                confusion=cm.tolist())


def run(name, splits, X, y, g):
    """Pool predictions over folds, so every window is tested exactly once (per protocol)."""
    p, yy, fold_acc = [], [], []
    for tr, te in splits:
        pr = fit_predict(X, y, g, tr, te)
        p.append(pr); yy.append(y[te]); fold_acc.append(100 * (pr == y[te]).mean())
    r = score(np.concatenate(yy), np.concatenate(p))
    r['fold_acc_mean_sd'] = [round(float(np.mean(fold_acc)), 2), round(float(np.std(fold_acc)), 2)]
    print(f"{name:34s} acc {r['acc']:6.2f}  bal {r['bal_acc']:6.2f}  folds {r['fold_acc_mean_sd']}  "
          f"sit/stand err {r['sit_stand_err']:5.1f}%  recall {r['recall']}", flush=True)
    return r


def main():
    out = {}
    for ds in ('uci', 'pamap2', 'wisdm'):
        d = np.load(os.path.join(HERE, 'data', f'{ds}_features.npz'))
        X, y, g = d['X'], d['y'].astype(np.int64), d['subject']
        idx = np.arange(len(y))
        if ds == 'uci':
            D = os.path.join(HERE, '..', 'exact_paper_implementation', 'data', 'UCI HAR Dataset')
            test_subj = set(np.loadtxt(os.path.join(D, 'Test', 'subject_id_test.txt'), dtype=int))
            te = np.isin(g, list(test_subj))
            out['uci_official'] = run('UCI official subject split', [(idx[~te], idx[te])], X, y, g)
            out['uci_group5'] = run('UCI 5-fold by subject', GroupKFold(5).split(X, y, g), X, y, g)
            tv, te2 = train_test_split(idx, test_size=0.15, random_state=42, stratify=y)
            out['uci_random_window'] = run('UCI random windows (AEJ protocol)', [(tv, te2)], X, y, None)
        elif ds == 'pamap2':
            out['pamap2_loso'] = run('PAMAP2 leave-one-subject-out', LeaveOneGroupOut().split(X, y, g), X, y, g)
        else:
            out['wisdm_group6'] = run('WISDM 6-fold by user', GroupKFold(6).split(X, y, g), X, y, g)
    with open(os.path.join(HERE, 'data', 'baseline_results.json'), 'w') as f:
        json.dump(out, f, indent=1)


if __name__ == '__main__':
    main()
