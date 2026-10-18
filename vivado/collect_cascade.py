"""Table of routed results for every built cascade model (out_cascade_*)."""
import glob
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))


def grab(path, pat):
    m = re.search(pat, open(path).read())
    return m.group(1) if m else '?'


print(f"{'model':20s} {'LUTs':>5} {'LUTRAM':>6} {'FFs':>5} {'slices':>6} {'DSP':>3} {'WNS ns':>7} {'power W':>7} {'dyn W':>6} {'sim':>5}")
for d in sorted(glob.glob(os.path.join(HERE, 'out_cascade_*'))):
    name = os.path.basename(d)[4:]
    u, t, p = (os.path.join(d, f) for f in ('utilization_routed.rpt', 'timing_routed.rpt', 'power_saif.rpt'))
    if not os.path.exists(p):
        print(f'{name:20s} incomplete'); continue
    log = os.path.join(HERE, f'xsim_{name}.log')
    sim = 'PASS' if 'RESULT: PASS' in open(log).read() else 'FAIL'
    print(f"{name:20s} {grab(u, r'\| Slice LUTs\s*\|\s*(\d+)'):>5} {grab(u, r'\|\s+LUT as Memory\s*\|\s*(\d+)'):>6} "
          f"{grab(u, r'\| Slice Registers\s*\|\s*(\d+)'):>5} {grab(u, r'\| Slice\s+\|\s*(\d+)'):>6} {grab(u, r'\| DSPs\s*\|\s*(\d+)'):>3} "
          f"{grab(t, r'WNS\(ns\).*\n.*\n\s*(-?[\d.]+)'):>7} {grab(p, r'Total On-Chip Power \(W\)\s*\|\s*([\d.]+)'):>7} "
          f"{grab(p, r'\| Dynamic \(W\)\s*\|\s*([\d.]+)'):>6} {sim:>5}")
