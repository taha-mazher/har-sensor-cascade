"""Figures for the cascade paper (AEJ). Writes PNGs to hw_fix/paper/.

fig_arch.png    : system architecture (also the graphical abstract)
fig_trace.png   : hardware sensor-control trace on one UCI test recording (from the bit-exact test vectors)
fig_tradeoff.png: accuracy vs second-sensor on-time, 5 seeds, nested tau (data/nested_tau_*.json)
"""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'paper')
os.makedirs(OUT, exist_ok=True)
C = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100']   # validated categorical palette (dataviz)
INK, MUTED = '#222222', '#666666'
NL = chr(10)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK,
                     'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.spines.top': False, 'axes.spines.right': False})


def box(ax, x, y, w, h, lines, fc='white', fs=6.0, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.08', fc=fc, ec=INK, lw=1))
    ax.text(x + w / 2, y + h / 2, NL.join(lines), ha='center', va='center', fontsize=fs, color=INK,
            weight='bold' if bold else 'normal')


def arrow(ax, x0, y0, x1, y1, text=None):
    ax.annotate('', (x1, y1), (x0, y0), arrowprops=dict(arrowstyle='-|>', color=INK, lw=1))
    if text:
        ax.text((x0 + x1) / 2, (y0 + y1) / 2 + 0.12, text, ha='center', va='bottom', fontsize=6.3, color=MUTED)


def note(ax, x, y, lines, color=MUTED, ha='left'):
    ax.text(x, y, NL.join(lines), fontsize=6.3, color=color, ha=ha, va='center')


def fig_arch():
    fig, ax = plt.subplots(figsize=(7.2, 3.1))
    ax.set_xlim(0, 15.4); ax.set_ylim(0, 6.2); ax.axis('off')
    box(ax, 0.1, 3.9, 2.1, 1.0, ['Accelerometer', '(always on)'], fc='#e8f0fb')
    box(ax, 0.1, 0.6, 2.1, 1.0, ['Second sensor', '(gyro or wrist)'], fc='#fdeee7')
    box(ax, 2.7, 3.9, 2.0, 1.0, ['7 features', 'per window'])
    box(ax, 2.7, 0.6, 2.0, 1.0, ['2 or 7 extra', 'features'])
    ax.add_patch(FancyBboxPatch((5.2, 0.25), 7.75, 5.55, boxstyle='round,pad=0.02,rounding_size=0.1',
                                fc='#f6f6f4', ec=MUTED, lw=0.8, ls='--'))
    ax.text(9.07, 5.55, 'FPGA: one shared 16 x 16 multiply-accumulate unit', ha='center', fontsize=6.8, color=MUTED)
    box(ax, 5.5, 3.9, 2.2, 1.0, ['Stage 1 MLP', '7-6-5, ReLU6'], fc='#e8f0fb', bold=True)
    box(ax, 8.6, 3.9, 2.2, 1.0, ['Gate: margin < τ', 'and class mask'])
    box(ax, 8.6, 2.05, 2.2, 1.0, ['Sensor control', '(sticky FSM)'])
    box(ax, 5.5, 0.6, 2.2, 1.0, ['Stage 2 MLP', '9-8-5 or 14-16-5'], fc='#fdeee7', bold=True)
    box(ax, 11.4, 2.05, 1.3, 1.0, ['Output', 'select'])
    box(ax, 13.3, 2.05, 2.0, 1.0, ['Activity class', '(5 classes)'], fc='#e9f7f1')
    arrow(ax, 2.2, 4.4, 2.7, 4.4); arrow(ax, 4.7, 4.4, 5.5, 4.4)
    arrow(ax, 2.2, 1.1, 2.7, 1.1); arrow(ax, 4.7, 1.1, 5.5, 1.1)
    arrow(ax, 4.7, 4.05, 5.5, 1.5)
    arrow(ax, 7.7, 4.4, 8.6, 4.4, 'logits')
    arrow(ax, 9.7, 3.9, 9.7, 3.05)
    note(ax, 9.8, 3.45, ['unsure'])
    arrow(ax, 8.6, 2.3, 7.7, 1.45)
    note(ax, 6.6, 2.3, ['covered:', 'run stage 2'], ha='center')
    ax.annotate('', (1.15, 1.6), (8.6, 2.75), arrowprops=dict(arrowstyle='-|>', color=C[1], lw=1.2, ls='--',
                                                              connectionstyle='arc3,rad=0.22'))
    note(ax, 5.75, 3.25, ['sensor_en (power gating)'], color=C[1])
    arrow(ax, 10.8, 4.4, 12.05, 3.05)
    note(ax, 11.9, 4.55, ['confident:', 'stage-1 class'])
    arrow(ax, 7.7, 1.1, 12.05, 2.05)
    note(ax, 10.4, 1.25, ['stage-2 class'])
    arrow(ax, 12.7, 2.55, 13.3, 2.55)
    fig.savefig(os.path.join(OUT, 'fig_arch.png'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def fig_arch_column():
    """Same system as fig_arch, laid out to fit one journal column."""
    fig, ax = plt.subplots(figsize=(3.4, 4.3))
    ax.set_xlim(0, 7.9); ax.set_ylim(0.3, 9.6); ax.axis('off')
    fs = 6.3
    ax.add_patch(FancyBboxPatch((0.05, 2.0), 7.1, 4.55, boxstyle='round,pad=0.02,rounding_size=0.1',
                                fc='#f6f6f4', ec=MUTED, lw=0.8, ls='--'))
    ax.text(7.0, 2.2, 'FPGA (one shared MAC unit)', ha='right', fontsize=5.8, color=MUTED)
    box(ax, 0.2, 8.5, 2.6, 0.95, ['Accelerometer', '(always on)'], fc='#e8f0fb', fs=fs)
    box(ax, 4.4, 8.5, 2.6, 0.95, ['Second sensor', '(gyro or wrist)'], fc='#fdeee7', fs=fs)
    box(ax, 0.2, 7.0, 2.6, 0.95, ['7 features'], fs=fs)
    box(ax, 4.4, 7.0, 2.6, 0.95, ['2 or 7 extra', 'features'], fs=fs)
    box(ax, 0.2, 5.0, 2.6, 0.95, ['Stage 1 MLP', '7-6-5, ReLU6'], fc='#e8f0fb', fs=fs, bold=True)
    box(ax, 4.4, 5.0, 2.6, 0.95, ['Stage 2 MLP', '9-8-5 or 14-16-5'], fc='#fdeee7', fs=fs, bold=True)
    box(ax, 0.2, 3.3, 2.6, 0.95, ['Gate: margin < τ', 'and class mask'], fs=fs)
    box(ax, 4.4, 3.3, 2.6, 0.95, ['Sensor control', '(sticky FSM)'], fs=fs)
    box(ax, 2.3, 1.15, 2.6, 0.8, ['Output select'], fs=fs)
    box(ax, 2.3, 0.35, 2.6, 0.6, ['Activity class'], fc='#e9f7f1', fs=fs)
    arrow(ax, 1.5, 8.5, 1.5, 7.95); arrow(ax, 5.7, 8.5, 5.7, 7.95)
    arrow(ax, 1.5, 7.0, 1.5, 5.95); arrow(ax, 5.7, 7.0, 5.7, 5.95)
    arrow(ax, 2.8, 7.2, 4.4, 5.8)
    arrow(ax, 1.5, 5.0, 1.5, 4.25)
    note(ax, 1.6, 4.62, ['logits'])
    arrow(ax, 2.8, 3.78, 4.4, 3.78)
    note(ax, 3.6, 4.0, ['unsure'], ha='center')
    arrow(ax, 5.7, 4.25, 5.7, 5.0)
    note(ax, 5.8, 4.62, ['run'])
    ax.annotate('', (7.05, 8.95), (7.05, 3.75), arrowprops=dict(arrowstyle='-|>', color=C[1], lw=1.1, ls='--',
                                                              connectionstyle='arc3,rad=0.3'))
    ax.text(7.5, 6.4, 'sensor_en', fontsize=5.8, color=C[1], rotation=90, va='center')
    arrow(ax, 1.5, 3.3, 3.0, 1.95)
    note(ax, 0.3, 2.6, ['confident'])
    arrow(ax, 4.4, 5.1, 4.2, 1.95)
    note(ax, 4.5, 2.6, ['stage-2 class'])
    arrow(ax, 3.6, 1.15, 3.6, 0.95)
    fig.savefig(os.path.join(OUT, 'fig_arch_col.png'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def fig_trace(wide=False):
    rows = [l.split() for l in open(os.path.join(HERE, 'out', 'cascade_uci.mem'))]
    # recordings start where the flag is 1; take the first test recording with a long sensor-on stretch
    starts = [i for i, r in enumerate(rows) if r[0] == '1'] + [len(rows)]
    seg = None
    for a, b in zip(starts, starts[1:]):
        r = rows[a:b]
        if sum(x[-3] == '1' for x in r) > 15:    # sensor_en column
            seg = r; break
    n2 = len(seg[0]) - 5
    cls = [int(x[1 + n2], 16) for x in seg]; en = [int(x[2 + n2], 16) for x in seg]
    used = [int(x[3 + n2], 16) for x in seg]; lab = [int(x[4 + n2], 16) for x in seg]
    t = [1.28 * i for i in range(len(seg))]
    names = ['Walk', 'Sit', 'Stand', 'Lay', 'Trans']
    fig, ax = plt.subplots(2, 1, figsize=(7.0, 2.5) if wide else (3.4, 2.7), sharex=True, gridspec_kw=dict(height_ratios=[3, 1], hspace=0.12))
    ax[0].step(t, [l if l != 15 else None for l in lab], where='post', color=MUTED, lw=3, alpha=0.45, label='True activity')
    ax[0].step(t, cls, where='post', color=C[0], lw=1.1, label='Cascade output (hardware)')
    s2 = [(ti, c) for ti, c, u in zip(t, cls, used) if u]
    ax[0].plot([a for a, _ in s2], [b for _, b in s2], 'o', ms=3.2, color=C[1], mec='white', mew=0.5, label='Answered by stage 2')
    ax[0].set_yticks(range(5)); ax[0].set_yticklabels(names); ax[0].set_ylim(-0.5, 4.5)
    ax[0].legend(loc='lower left', ncol=3 if wide else 2, frameon=False, fontsize=6.5 if wide else 5.8, bbox_to_anchor=(-0.02, 1.0), handlelength=1.5, columnspacing=0.8)
    ax[0].grid(axis='y', color='#e6e6e6', lw=0.6)
    ax[1].fill_between(t, en, step='post', color=C[1], alpha=0.85, lw=0)
    ax[1].set_yticks([0, 1]); ax[1].set_yticklabels(['off', 'on']); ax[1].set_ylabel('Gyro', fontsize=7)
    ax[1].set_xlabel('Time (s)')
    fig.savefig(os.path.join(OUT, 'fig_trace_wide.png' if wide else 'fig_trace.png'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return len(seg), sum(en), sum(used)


def fig_tradeoff(wide=False):
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.4)) if wide else plt.subplots(3, 1, figsize=(3.4, 5.6))
    for ax, (proto, title) in zip(axs, [('uci_official', 'UCI-HAPT, official split'), ('uci_5fold', 'UCI-HAPT, 5-fold by subject'),
                                         ('pamap2', 'PAMAP2, leave-one-subject-out')]):
        runs = json.load(open(os.path.join(HERE, 'data', f'nested_tau_{proto}.json')))
        for k, col, mk, lab in [('stage1', C[0], 's', 'Stage 1 only'), ('nested', C[1], 'o', 'Cascade (nested τ)'),
                                ('always_on', C[2], '^', 'Always-on network')]:
            xs = [r[k]['sensor'] for r in runs]; ys = [r[k]['acc'] for r in runs]
            ax.scatter(xs, ys, s=22, marker=mk, color=col, edgecolor='white', linewidth=0.6, label=lab, zorder=3)
        ax.set_title(title, fontsize=7.5, color=INK)
        ax.set_xlim(-5, 105)
        if wide: ax.set_xlabel('Second sensor on (% of windows)')
        if not wide or ax is axs[0]: ax.set_ylabel('Accuracy (%)')
        ax.grid(color='#eeeeee', lw=0.6)
    if not wide: axs[-1].set_xlabel('Second sensor on (% of windows)')
    axs[0].legend(frameon=False, fontsize=6.5, loc='lower right')
    fig.tight_layout(w_pad=1.2) if wide else fig.tight_layout(h_pad=1.6)
    fig.savefig(os.path.join(OUT, 'fig_tradeoff_wide.png' if wide else 'fig_tradeoff.png'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def fig_graphical_abstract():
    """Sized for the AEJ graphical-abstract box (3.13 x 2.54 in): few elements, large text, headline numbers."""
    fig, ax = plt.subplots(figsize=(3.13, 2.54))
    ax.set_xlim(0, 10); ax.set_ylim(0, 8.1); ax.axis('off')
    fs = 6.2
    box(ax, 0.1, 6.3, 2.8, 1.3, ['Accelerometer', '(always on)'], fc='#e8f0fb', fs=fs)
    box(ax, 3.6, 6.3, 2.8, 1.3, ['Stage 1 MLP', '7 features'], fc='#e8f0fb', fs=fs, bold=True)
    box(ax, 7.1, 6.3, 2.8, 1.3, ['Activity', 'class'], fc='#e9f7f1', fs=fs)
    box(ax, 0.1, 3.5, 2.8, 1.3, ['Gyroscope', 'when unsure'], fc='#fdeee7', fs=fs)
    box(ax, 3.6, 3.5, 2.8, 1.3, ['Stage 2 MLP', '7 + 2 features'], fc='#fdeee7', fs=fs, bold=True)
    arrow(ax, 2.9, 6.95, 3.6, 6.95); arrow(ax, 6.4, 6.95, 7.1, 6.95)
    note(ax, 6.75, 7.85, ['confident'], ha='center')
    ax.annotate('', (1.5, 4.8), (4.2, 6.3), arrowprops=dict(arrowstyle='-|>', color=C[1], lw=1.3, ls='--'))
    note(ax, 0.9, 5.55, ['unsure'], color=C[1])
    arrow(ax, 2.9, 4.15, 3.6, 4.15)
    ax.annotate('', (8.5, 6.3), (6.4, 4.15), arrowprops=dict(arrowstyle='-|>', color=INK, lw=1,
                                                            connectionstyle='angle,angleA=0,angleB=90,rad=0'))
    ax.add_patch(FancyBboxPatch((0.1, 0.15), 9.8, 2.65, boxstyle='round,pad=0.02,rounding_size=0.1', fc='#f6f6f4', ec=MUTED, lw=0.8))
    ax.text(5.0, 2.3, 'Artix-7 FPGA: 358 LUTs, 1 DSP, 63 mW', ha='center', fontsize=6, color=INK)
    ax.text(2.6, 1.0, NL.join(['94.6% accuracy', '(always-on 94.7%)']), ha='center', fontsize=6, color=INK, weight='bold')
    ax.text(7.4, 1.0, NL.join(['gyro on 11% of time', '71% less sensor current']), ha='center', fontsize=6, color=C[1], weight='bold')
    fig.savefig(os.path.join(OUT, 'fig_ga.png'), dpi=400, bbox_inches='tight', facecolor='white')
    plt.close(fig)

if __name__ == '__main__':
    fig_arch()
    fig_arch_column()
    print('trace windows, sensor-on, stage-2:', fig_trace())
    fig_tradeoff()
    fig_tradeoff(wide=True)
    fig_trace(wide=True)
    fig_graphical_abstract()
    print(sorted(os.listdir(OUT)))
