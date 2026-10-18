"""Frozen cascade for hardware: integer (bit-exact) model, ROM files and test vectors for rtl/cascade.v.

One design covers both datasets, set by parameters:
  stage 1 : 7-6-5 ReLU6 MLP on the always-on accelerometer (x = 7 features)
  gate    : margin = top1 - top2 stage-1 logit; class mask M on the stage-1 class
            trigger = M[c1] and margin < TAU         (switch the second sensor on)
            in_set  = M[c1] and margin < TAU_KEEP    (keep it on; KEEP_ALL means TAU_KEEP = inf)
  sensor  : on after a trigger, off after the first window not in_set. A window is covered once the sensor
            has been on for two whole windows (it was switched on at the end of window w, so it first covers w+2).
  stage 2 : N2-H2-5 ReLU6 MLP on [x, g], used for covered in_set windows; argmax over class mask M2.
  UCI    : M = M2 = {Sit, Stand}, KEEP_ALL, tau 1, stage 2 9-8-5 (gyroscope std x, z)   = gyro_policies sticky
  PAMAP2 : M = M2 = all, TAU_KEEP = TAU = 2, stage 2 14-16-5 (wrist accelerometer)        = general_gate sticky
Stage 2's scaler is folded into its first layer so both stages share the stage-1-scaled x.
Number formats as fixed_point_reference.py: x, g, h Q4.12; hidden weights 2^F, biases 2^(12+F), int32 sums.
"""
import os

import numpy as np
import torch
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

from baseline import train
from fixed_point_reference import XF, frac_bits, hexs
from gyro_policies import UCI, build as build_uci, fit_models
from pamap2_cascade import build as build_pamap2

HERE = os.path.dirname(os.path.abspath(__file__))
RELU6_MAX = 6 << XF
ALL, SS = [1] * 5, [0, 1, 1, 0, 0]


def wb(m):
    return m.l1.weight.detach().numpy().T.astype(float), m.l1.bias.detach().numpy().astype(float), \
        m.l2.weight.detach().numpy().T.astype(float), m.l2.bias.detach().numpy().astype(float)


def safe_bits(W, b, amax):
    """frac_bits, lowered until no partial sum can leave int32 for ANY input with |a| <= amax."""
    F = frac_bits(W)
    while 2.0 ** F * (amax * np.abs(W).sum(0) + np.abs(b) * 2 ** XF).max() + 2 ** 15 >= 2 ** 31:
        F -= 1
    return F


def quantise(W1, b1, W2, b2):
    F1, F2 = safe_bits(W1, b1, 2 ** 15), safe_bits(W2, b2, RELU6_MAX)
    q = lambda a, s: np.round(a * 2.0 ** s).astype(np.int64)
    return dict(F1=F1, F2=F2, W1=q(W1, F1), b1=q(b1, XF + F1), W2=q(W2, F2), b2=q(b2, XF + F2))


def make_hw(m1, sc1, m2, sc2):
    """Integer networks. Stage 2 sees x scaled by sc1 (7 cols) and g scaled by sc2's last columns."""
    W1, b1, W2, b2 = wb(m2)
    r = sc1.scale_ / sc2.scale_[:7]
    b1 = b1 + ((sc1.mean_ - sc2.mean_[:7]) / sc2.scale_[:7]) @ W1[:7]
    W1 = W1.copy(); W1[:7] *= r[:, None]
    return quantise(*wb(m1)), quantise(W1, b1, W2, b2)


def qin(Z):
    return np.clip(np.round(Z * 2 ** XF), -32768, 32767).astype(np.int64)


def forward(xq, Q):
    """Logits plus the largest |partial sum| in MAC order (bias first), for 32-bit accumulator sizing."""
    p1 = np.cumsum(xq[:, :, None] * Q['W1'][None], 1) + Q['b1']
    h = np.clip(p1[:, -1] >> Q['F1'], 0, RELU6_MAX)
    p2 = np.cumsum(h[:, :, None] * Q['W2'][None], 1) + Q['b2']
    peak = max(np.abs(Q['b1']).max(), np.abs(Q['b2']).max(), np.abs(p1).max(), np.abs(p2).max())
    return p2[:, -1], peak


def policy(l1, l2, cfg):
    """Sensor state machine, identical to rtl/cascade.v. l1, l2 = stage-1 / stage-2 logits per window."""
    s = np.sort(l1, 1)
    c1, margin = l1.argmax(1), s[:, -1] - s[:, -2]
    M, M2 = np.array(cfg['M'], bool), np.array(cfg['M2'], bool)
    in_set = M[c1] & (cfg['keep_all'] | (margin < cfg['tau_keep']))
    trigger = M[c1] & (margin < cfg['tau'])
    c2 = np.where(M2, l2, -np.inf).argmax(1)
    cls, en, used = c1.copy(), np.zeros(len(c1), int), np.zeros(len(c1), int)
    on, cnt = 0, 0
    for w in range(len(c1)):
        if on and cnt == 2 and in_set[w]:
            cls[w], used[w] = c2[w], 1
        if on:
            on, cnt = (1, min(cnt + 1, 2)) if in_set[w] else (0, 0)
        elif trigger[w]:
            on, cnt = 1, 1
        en[w] = on
    return cls, en, used


def run(name, exps, te, m1, sc1, m2, sc2, cfg):
    Q1, Q2 = make_hw(m1, sc1, m2, sc2)
    tau_int = lambda t: int(round(t * 2 ** (XF + Q1['F2'])))
    icfg = dict(cfg, tau=tau_int(cfg['tau']), tau_keep=tau_int(cfg['tau_keep']))
    T = lambda m, sc, Z: m(torch.tensor(sc.transform(Z), dtype=torch.float32)).detach().numpy()
    res, peak, lines = dict(float=[0, 0, 0], int=[0, 0, 0]), 0, []
    hops = 0
    for i in te:
        e = exps[i]; X, G, y = e['X'], e['G'], e['y']
        lf = (T(m1, sc1, X), T(m2, sc2, np.c_[X, G]))
        xq = qin(np.c_[sc1.transform(X), (G - sc2.mean_[7:]) / sc2.scale_[7:]])
        (a, pa), (b, pb) = forward(xq[:, :7], Q1), forward(xq, Q2)
        peak = max(peak, pa, pb)
        s = y >= 0
        for k, (l1, l2, c) in dict(float=(*lf, cfg), int=(a, b, icfg)).items():
            cls, en, used = policy(l1, l2, c)
            res[k][0] += (cls[s] == y[s]).sum(); res[k][1] += s.sum(); res[k][2] += en.sum()
        hops += len(y)
        for w in range(len(y)):  # new-recording flag, N2 inputs, class, sensor_en, used_stage2, label
            lines.append(f"{int(w == 0)} {' '.join(hexs(v, 16) for v in xq[w])} {cls[w]:x} {en[w]:x} {used[w]:x} {y[w] & 15:x}")
    assert peak < 2 ** 31, f'partial sums overflow 32 bits: {peak}'

    d = os.path.join(HERE, 'rtl', 'mem', f'cascade_{name}')
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, 'w.mem'), 'w') as f:  # neuron-major, in MAC order: one pointer walks it
        for Q in (Q1, Q2):
            for W in (Q['W1'], Q['W2']):
                f.writelines(hexs(v, 16) + '\n' for v in W.T.ravel())
    with open(os.path.join(d, 'b.mem'), 'w') as f:
        f.writelines(hexs(v, 32) + '\n' for Q in (Q1, Q2) for v in np.r_[Q['b1'], Q['b2']])
    with open(os.path.join(HERE, 'out', f'cascade_{name}.mem'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    params = dict(N2=Q2['W1'].shape[0], H2=Q2['W1'].shape[1], FA=Q1['F1'], FB=Q2['F1'], TAU=icfg['tau'],
                  TAU_KEEP=icfg['tau_keep'] if not cfg['keep_all'] else 0, KEEP_ALL=int(cfg['keep_all']),
                  MASK=int(''.join(map(str, cfg['M'][::-1])), 2), MASK2=int(''.join(map(str, cfg['M2'][::-1])), 2))
    with open(os.path.join(d, 'params.txt'), 'w') as f:
        f.write(' '.join(f'{k}={v}' for k, v in params.items()) + '\n')
    acc = {k: (100 * v[0] / v[1], 100 * v[2] / hops) for k, v in res.items()}
    print(f"{name:10s} float {acc['float'][0]:6.2f}% (sensor {acc['float'][1]:4.1f}%)  int {acc['int'][0]:6.2f}% "
          f"(sensor {acc['int'][1]:4.1f}%)  peak {int(peak).bit_length() + 1} bits  {params}", flush=True)
    return res


def pamap2_models(exps, tr, h2, seed=0):
    """Same training as general_gate.fit, but keeping the networks and scalers."""
    tr = [exps[i] for i in tr]
    X = np.vstack([e['X'] for e in tr]); G = np.vstack([e['G'] for e in tr])
    y = np.concatenate([e['y'] for e in tr]); g = np.concatenate([[e['subject']] * len(e['y']) for e in tr])
    k = y >= 0
    X, G, y, g = X[k], G[k], y[k].astype(np.int64), g[k]
    out = []
    for Z, h in ((X, 6), (np.c_[X, G], h2)):
        a, b = next(GroupShuffleSplit(1, test_size=0.15, random_state=seed).split(Z, groups=g))
        sc = StandardScaler().fit(Z[a])
        out += [train(sc.transform(Z[a]), y[a], sc.transform(Z[b]), y[b], seed, h), sc]
    return out


def main():
    torch.set_grad_enabled(False)
    uci = build_uci()
    subj = np.array([e['subject'] for e in uci])
    test_subj = list(set(np.loadtxt(os.path.join(UCI, 'Test', 'subject_id_test.txt'), dtype=int)))
    tr, te = np.flatnonzero(~np.isin(subj, test_subj)), np.flatnonzero(np.isin(subj, test_subj))
    with torch.enable_grad():
        m1, sc1, m2 = fit_models(uci, tr, h2=8)
    run('uci', uci, te, m1, sc1, m2.m, m2.sc, dict(M=SS, M2=SS, tau=1.0, tau_keep=0.0, keep_all=True))

    pam = build_pamap2()['wrist accel']
    tot = np.zeros(3)
    for i in range(len(pam)):
        with torch.enable_grad():
            m = pamap2_models(pam, [j for j in range(len(pam)) if j != i], 16)
        tot += run(f'pamap2_{pam[i]["subject"]}', pam, [i], *m, dict(M=ALL, M2=ALL, tau=2.0, tau_keep=2.0, keep_all=False))['int']
    print(f'PAMAP2 pooled int accuracy {100 * tot[0] / tot[1]:.2f}%')


if __name__ == '__main__':
    main()
