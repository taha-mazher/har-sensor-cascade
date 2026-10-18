"""WISDM (phone accelerometer only): does the frozen gate transfer to a third dataset?

No second sensor exists, so stage 2 cannot run. Reported per gate setting, 6-fold grouped CV by user:
  stage-1 accuracy; % windows flagged; error rate of flagged vs unflagged windows; share of all stage-1
  errors the gate catches; and % of windows the sensor would be on under the hardware sticky state machine.
Windows are taken in file order per user (single-activity segments back to back), so sensor-on time is an
estimate: real recordings would have transitions between the segments.
"""
import os

import numpy as np
from sklearn.model_selection import GroupKFold

from cascade_hw import ALL, SS, policy
from general_gate import net

HERE = os.path.dirname(os.path.abspath(__file__))
GATES = {'UCI gate (Sit/Stand, tau 1, keep while Sit/Stand)': dict(M=SS, M2=SS, tau=1.0, tau_keep=0.0, keep_all=True),
         'PAMAP2 gate (all classes, tau 2)': dict(M=ALL, M2=ALL, tau=2.0, tau_keep=2.0, keep_all=False)}


def main():
    d = np.load(os.path.join(HERE, 'data', 'wisdm_features.npz'))
    X, y, g = d['X'], d['y'].astype(np.int64), d['subject']
    logits = np.zeros((len(y), 5))
    for tr, te in GroupKFold(6).split(X, y, g):
        logits[te] = net(X[tr], y[tr], g[tr], 6)(X[te])
    p1 = logits.argmax(1)
    wrong = p1 != y
    s = np.sort(logits, 1)
    margin = s[:, -1] - s[:, -2]
    cm = np.zeros((3, 3), int); np.add.at(cm, (y, p1), 1)
    print(f'stage 1 accuracy {100 * (~wrong).mean():.2f}%  ({wrong.sum()} errors of {len(y)})')
    print('confusion (Walk, Sit, Stand):', cm.tolist())
    for name, cfg in GATES.items():
        M = np.array(cfg['M'], bool)
        flag = M[p1] & (margin < cfg['tau'])
        on = 0
        for u in np.unique(g):  # one stream per user, state machine reset per user as in the testbench
            k = g == u
            on += policy(logits[k], logits[k], cfg)[1].sum()
        print(f'\n{name}')
        print(f'  flagged {100 * flag.mean():5.1f}% of windows; sensor on {100 * on / len(y):5.1f}%')
        print(f'  error rate: flagged {100 * wrong[flag].mean():5.1f}%  unflagged {100 * wrong[~flag].mean():5.2f}%')
        print(f'  stage-1 errors caught by the gate: {flag[wrong].sum()} of {wrong.sum()} ({100 * flag[wrong].mean():.1f}%)')


if __name__ == '__main__':
    main()
