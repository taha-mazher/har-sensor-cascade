"""Better gates for the general cascade: which windows should switch the second sensor on?

Gate scores (higher = more likely to need stage 2):
  margin  : -(top-1 minus top-2 stage-1 logit)        (current gate)
  entropy : entropy of stage-1 softmax
  learned : logistic regression predicting "stage 1 is wrong" from the 7 features + 5 stage-1 logits
            (12 MACs). Trained on OUT-OF-FOLD stage-1 predictions (4 subject folds inside the training set),
            so it learns stage 1's mistakes on unseen people.
For a fair comparison every gate routes the same budget: thresholds are the training-set quantiles that send
the top q% of windows to stage 2. Stage 2 = general n2-16-5 ReLU6 MLP. Policies as in general_gate.py.
"""
import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from general_gate import net
from gyro_policies import UCI, build as build_uci
from pamap2_cascade import build as build_pamap2

BUDGETS = [10, 20, 30, 40, 50]
SIT, STAND = 1, 2


def entropy(l):
    p = np.exp(l - l.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
    return -(p * np.log(p + 1e-12)).sum(1)


def margin(l):
    t = np.sort(l, 1)[:, -2:]
    return -(t[:, 1] - t[:, 0])


def stack(exps, idx):
    tr = [exps[i] for i in idx]
    X = np.vstack([e['X'] for e in tr]); G = np.vstack([e['G'] for e in tr])
    y = np.concatenate([e['y'] for e in tr]); g = np.concatenate([[e['subject']] * len(e['y']) for e in tr])
    k = y >= 0
    return X[k], G[k], y[k].astype(np.int64), g[k]


def fit(exps, tr_idx):
    X, G, y, g = stack(exps, tr_idx)
    s1, s2 = net(X, y, g, 6), net(np.c_[X, G], y, g, 16)
    # out-of-fold stage-1 logits on the training subjects, for the learned gate and for the quantile thresholds
    oof = np.zeros((len(y), 5))
    for a, b in GroupKFold(4).split(X, y, g):
        oof[b] = net(X[a], y[a], g[a], 6)(X[b])
    wrong = (oof.argmax(1) != y).astype(int)
    gate = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(np.c_[X, oof], wrong)
    scores = dict(margin=margin, entropy=entropy, learned=lambda l, X: gate.decision_function(np.c_[X, l]))
    train_scores = dict(margin=margin(oof), entropy=entropy(oof), learned=gate.decision_function(np.c_[X, oof]))
    return s1, s2, scores, train_scores


def simulate(e, s1, s2, route):
    X, G = e['X'], e['G']
    p1 = s1(X).argmax(1); p2_own = s2(np.c_[X, G]).argmax(1)
    n = len(p1); hops = n + 1
    one, on1 = p1.copy(), np.zeros(hops + 2, bool)
    idx = np.flatnonzero(route); idx = idx[idx + 2 < n]
    if len(idx):
        one[idx] = s2(np.c_[X[idx], G[idx + 2]]).argmax(1)
        for w in idx:
            on1[w + 2:w + 4] = True
    st, on2, on_from = p1.copy(), np.zeros(hops + 2, bool), None
    for w in range(n):
        if on_from is not None and on_from <= w and route[w]:
            st[w] = p2_own[w]
        if not route[w]:
            if on_from is not None:
                on2[on_from:w + 2] = True
            on_from = None
        elif on_from is None:
            on_from = w + 2
    if on_from is not None:
        on2[on_from:hops] = True
    return dict(oneshot=one, sticky=st), dict(oneshot=on1[:hops].sum(), sticky=on2[:hops].sum()), hops


def report(name, exps, splits):
    fits = [(te, fit(exps, tr)) for tr, te in splits]
    print(f'\n######## {name}')
    print(f"{'gate':8s} {'budget':>6} {'policy':8s} | {'acc':>6} {'sserr':>6} {'sensor on':>9}")
    base = []
    for te, (s1, s2, scores, trs) in fits:
        for i in te:
            y = exps[i]['y']; s = y >= 0
            base.append(((s1(exps[i]['X']).argmax(1)[s] == y[s]).sum(), s.sum()))
    print(f"{'none':8s} {0:>6} {'stage1':8s} | {100 * sum(a for a, _ in base) / sum(b for _, b in base):6.2f}")
    for gname in ('margin', 'entropy', 'learned'):
        for q in BUDGETS:
            tot, hops_all = {}, 0
            for te, (s1, s2, scores, trs) in fits:
                thr = np.percentile(trs[gname], 100 - q)
                for i in te:
                    e = exps[i]; l = s1(e['X'])
                    sc = scores[gname](l, e['X']) if gname == 'learned' else scores[gname](l)
                    preds, on, hops = simulate(e, s1, s2, sc > thr)
                    y = e['y']; s = y >= 0; m = s & np.isin(y, [SIT, STAND])
                    for k, p in preds.items():
                        t = tot.setdefault(k, np.zeros(5))
                        t += [(p[s] == y[s]).sum(), s.sum(), (np.isin(p[m], [SIT, STAND]) & (p[m] != y[m])).sum(), m.sum(), on[k]]
                    hops_all += hops
            for k, v in tot.items():
                print(f"{gname:8s} {q:>5}% {k:8s} | {100 * v[0] / v[1]:6.2f} {100 * v[2] / v[3]:5.1f}% {100 * v[4] / hops_all:8.1f}%", flush=True)


def main():
    uci = build_uci()
    subj = np.array([e['subject'] for e in uci])
    test_subj = list(set(np.loadtxt(os.path.join(UCI, 'Test', 'subject_id_test.txt'), dtype=int)))
    report('UCI official split', uci, [(np.flatnonzero(~np.isin(subj, test_subj)), np.flatnonzero(np.isin(subj, test_subj)))])
    report('UCI 5-fold by subject', uci, list(GroupKFold(5).split(subj, subj, subj)))
    pam = build_pamap2()['wrist accel']
    n = len(pam)
    report('PAMAP2 leave-one-subject-out', pam, [(np.array([j for j in range(n) if j != i]), np.array([i])) for i in range(n)])


if __name__ == '__main__':
    main()
