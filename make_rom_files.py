"""Split out/weights_*.mem into per-array ROM files for $readmemh, and check 32-bit accumulators.

Writes rtl/mem/<split>/{w1,b1,w2,b2}.mem. Also replays every test sample in integers and asserts that
every partial sum (not just the final one) fits a signed 32-bit accumulator.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPORT = json.load(open(os.path.join(HERE, 'out', 'report.json')))
s = lambda v, b: v - (1 << b) if v >> (b - 1) else v


def main():
    for name, r in REPORT.items():
        assert (r['F1'], r['F2']) == (12, 13), 'RTL parameters assume F1=12, F2=13'
        vals = [l.strip() for l in open(os.path.join(HERE, 'out', f'weights_{name}.mem'))]
        w1 = [s(int(v, 16), 16) for v in vals[:42]]
        b1 = [s(int(v, 16), 48) for v in vals[42:48]]
        w2 = [s(int(v, 16), 16) for v in vals[48:78]]
        b2 = [s(int(v, 16), 48) for v in vals[78:83]]
        assert len(vals) == 83

        peak = 0
        for line in open(os.path.join(HERE, 'out', f'test_{name}.mem')):
            t = line.split()
            x, py = [s(int(v, 16), 16) for v in t[:7]], int(t[7], 16)
            h = []
            for n in range(6):
                acc = b1[n]; peak = max(peak, abs(acc))
                for k in range(7):
                    acc += x[k] * w1[k * 6 + n]; peak = max(peak, abs(acc))
                h.append(min(max(acc >> 12, 0), 6 << 12))
            best, cls = None, 0
            for o in range(5):
                acc = b2[o]; peak = max(peak, abs(acc))
                for k in range(6):
                    acc += h[k] * w2[k * 5 + o]; peak = max(peak, abs(acc))
                if best is None or acc > best:
                    best, cls = acc, o
            assert cls == py, 'replay disagrees with step 1'
        assert peak < 2 ** 31, f'partial sums overflow 32 bits: {peak}'

        d = os.path.join(HERE, 'rtl', 'mem', name)
        os.makedirs(d, exist_ok=True)
        for fn, arr, bits in [('w1', w1, 16), ('b1', b1, 32), ('w2', w2, 16), ('b2', b2, 32)]:
            with open(os.path.join(d, fn + '.mem'), 'w') as f:
                f.writelines(format(v & ((1 << bits) - 1), f'0{bits // 4}x') + '\n' for v in arr)
        print(f'{name}: ROM files written, peak |partial sum| = {peak} ({peak.bit_length() + 1} bits signed)')


if __name__ == '__main__':
    main()
