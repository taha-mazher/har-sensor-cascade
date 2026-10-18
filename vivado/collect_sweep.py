"""Parallelism sweep table for stage 1 (split A): out_A_ser, out_A_np, out_A_par.

Energy per window: dynamic = P_dyn x simulated period (latency + 1 cycle; SAIF power is averaged over that period),
total = P_total x latency (the usual 'energy per inference' figure, dominated by static power on Artix-7).
"""
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
T_CLK = 20e-9


def grab(path, pat):
    m = re.search(pat, open(path).read())
    return m.group(1) if m else None


print(f"{'variant':6s} {'LUTs':>5} {'FFs':>5} {'slices':>6} {'DSP':>4} {'WNS ns':>7} {'P mW':>5} {'Pdyn mW':>7} "
      f"{'cycles':>6} {'lat ns':>7} {'E_tot nJ':>8} {'E_dyn nJ':>8} {'sim':>5}")
for tag in ('A_ser', 'A_np', 'A_par'):
    d = os.path.join(HERE, f'out_{tag}')
    u, t, p = (os.path.join(d, f) for f in ('utilization_routed.rpt', 'timing_routed.rpt', 'power_saif_mw.rpt'))
    log = os.path.join(HERE, f'xsim_{tag}.log')
    if not all(map(os.path.exists, (u, t, p, log))):
        print(f'{tag:6s} incomplete'); continue
    lg = open(log).read()
    cyc = int(grab(log, r'latency_cycles=(\d+)'))
    P, Pd = (1e-3 * float(grab(p, r'%s \(mW\)\s*\|\s*([\d.]+)' % k)) for k in ('Total On-Chip Power', r'\| Dynamic'))
    print(f"{tag:6s} {grab(u, r'\| Slice LUTs\s*\|\s*(\d+)'):>5} {grab(u, r'\| Slice Registers\s*\|\s*(\d+)'):>5} "
          f"{grab(u, r'\| Slice\s+\|\s*(\d+)'):>6} {grab(u, r'\| DSPs\s*\|\s*(\d+)'):>4} "
          f"{grab(t, r'WNS\(ns\).*\n.*\n\s*(-?[\d.]+)'):>7} {1e3 * P:5.1f} {1e3 * Pd:7.2f} {cyc:>6} {cyc * 20:>7} "
          f"{P * cyc * T_CLK * 1e9:8.1f} {Pd * (cyc + 1) * T_CLK * 1e9:8.2f} {'PASS' if 'RESULT: PASS' in lg else 'FAIL':>5}")
