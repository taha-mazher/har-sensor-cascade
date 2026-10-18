"""Step 1: integer-only (bit-exact) reference for the 7-6-5 ReLU6 MLP.

Trains the paper's ReLU6 model on splits A/B/C, quantises it, and measures accuracy with
integer arithmetic only. Exports weights and test vectors as .mem files for the Verilog check.

Number formats (all signed):
  x   : Q4.12 int16 (standardised features, |x| < 8)
  W1  : int16, scale 2^F1;  b1 : scale 2^(12+F1)
  h   : (x.W1 + b1) >> F1  -> Q4.12, then ReLU6 = clip(h, 0, 6<<12)
  W2  : int16, scale 2^F2;  b2 : scale 2^(12+F2)
  out : argmax(h.W2 + b2), first index wins ties (matches a '>' compare chain)
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.join(HERE, '..', 'exact_paper_implementation')
sys.path.insert(0, PROJ)
from relu6_implementation.mlp_relu6 import MLP765_ReLU6  # noqa: E402

XF = 12                      # input / hidden fractional bits
RELU6_MAX = 6 << XF
SPLITS = {'A': 'dataset_a', 'B': 'dataset_B', 'C': 'dataset_C'}


def frac_bits(w):
    """Largest F such that round(w * 2^F) fits int16."""
    return int(15 - np.ceil(np.log2(np.abs(w).max() + 1e-12)))


def quantise(m):
    F1, F2 = frac_bits(m.W1), frac_bits(m.W2)
    q = lambda a, s: np.round(a * 2.0 ** s).astype(np.int64)
    return dict(F1=F1, F2=F2, W1=q(m.W1, F1), b1=q(m.b1.ravel(), XF + F1),
                W2=q(m.W2, F2), b2=q(m.b2.ravel(), XF + F2))


def int_forward(xq, Q):
    """Integer forward pass. Returns classes plus the widest accumulator seen (for RTL sizing)."""
    acc1 = xq @ Q['W1'] + Q['b1']
    h = np.clip(acc1 >> Q['F1'], 0, RELU6_MAX)
    acc2 = h @ Q['W2'] + Q['b2']
    return acc2.argmax(1), max(np.abs(acc1).max(), np.abs(acc2).max())


def hexs(v, bits):
    return format(int(v) & ((1 << bits) - 1), f'0{bits // 4}x')


def main():
    out = os.path.join(HERE, 'out')
    os.makedirs(out, exist_ok=True)
    report = {}
    for name, folder in SPLITS.items():
        d = os.path.join(PROJ, 'relu6_implementation', 'data', 'processed', folder)
        L = lambda s: np.load(os.path.join(d, s + '.npy'))
        Xtr, ytr, Xva, yva, Xte, yte = (L(s) for s in ('X_train', 'y_train', 'X_val', 'y_val', 'X_test', 'y_test'))
        yte = yte.astype(int)

        np.random.seed(42)
        m = MLP765_ReLU6(learning_rate=0.2, epochs=6000, batch_size=64, momentum=0.95, weight_decay=0.0001,
                         use_lr_schedule=True, use_momentum=True, early_stopping_patience=1000)
        m.train(Xtr, ytr, Xva, yva, verbose=False)
        float_acc = (m.predict(Xte) == yte).mean()

        Q = quantise(m)
        xq = np.clip(np.round(Xte * 2 ** XF), -32768, 32767).astype(np.int64)
        pred, accmax = int_forward(xq, Q)
        fixed_acc = (pred == yte).mean()
        cm = np.zeros((5, 5), int)
        np.add.at(cm, (yte, pred), 1)

        np.savez(os.path.join(out, f'float_weights_{name}.npz'), W1=m.W1, b1=m.b1, W2=m.W2, b2=m.b2)
        with open(os.path.join(out, f'weights_{name}.mem'), 'w') as f:  # W1 row-major, b1, W2, b2
            for v in Q['W1'].ravel(): f.write(hexs(v, 16) + '\n')
            for v in Q['b1']: f.write(hexs(v, 48) + '\n')
            for v in Q['W2'].ravel(): f.write(hexs(v, 16) + '\n')
            for v in Q['b2']: f.write(hexs(v, 48) + '\n')
        with open(os.path.join(out, f'test_{name}.mem'), 'w') as f:  # 7 inputs, python-fixed class, true label
            for x, p, y in zip(xq, pred, yte):
                f.write(' '.join(hexs(v, 16) for v in x) + f' {p:x} {y:x}\n')

        report[name] = dict(n_test=len(yte), float_acc=round(100 * float_acc, 2), fixed_acc=round(100 * fixed_acc, 2),
                            F1=Q['F1'], F2=Q['F2'], acc_bits_needed=int(accmax).bit_length() + 1,
                            sitting_as_standing=int(cm[1, 2]), standing_as_sitting=int(cm[2, 1]),
                            confusion=cm.tolist())
        print(name, {k: v for k, v in report[name].items() if k != 'confusion'}, flush=True)

    with open(os.path.join(out, 'report.json'), 'w') as f:
        json.dump(report, f, indent=1)


def selfcheck():
    """Tiny sanity check of the integer path against float on random data."""
    rng = np.random.default_rng(0)
    class M: pass
    m = M(); m.W1, m.b1, m.W2, m.b2 = rng.normal(size=(7, 6)), rng.normal(size=(1, 6)), rng.normal(size=(6, 5)), rng.normal(size=(1, 5))
    X = rng.normal(size=(2000, 7))
    ref = (np.clip(X @ m.W1 + m.b1, 0, 6) @ m.W2 + m.b2).argmax(1)
    pred, _ = int_forward(np.round(X * 2 ** XF).astype(np.int64), quantise(m))
    assert (pred == ref).mean() > 0.99, (pred == ref).mean()


if __name__ == '__main__':
    selfcheck()
    main()
