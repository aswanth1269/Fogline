import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from pathlib import Path
FIG = Path(__file__).parent / 'figs'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'figure.dpi': 220, 'savefig.bbox': 'tight'})
INK = '#1c3040'; EDGE = '#dbe9f8'; FOG = '#fde3d6'; STORE = '#d8f1e6'; GREY = '#eeeeee'

def box(ax, x, y, w, h, text, fc, fs=7.6, bold=False, ec='#5b6b78'):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.12', fc=fc, ec=ec, lw=1))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fs, color=INK, weight='bold' if bold else 'normal', wrap=True)

def arrow(ax, a, b, label=None, style='-|>', color='#44515c', ls='-', off=(0, 0.12), fs=7.5, rad=0):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=11, color=color, lw=1.2, linestyle=ls,
                                 connectionstyle=f'arc3,rad={rad}'))
    if label:
        ax.text((a[0] + b[0]) / 2 + off[0], (a[1] + b[1]) / 2 + off[1], label, ha='center', fontsize=fs, color='#33424e', style='italic')

def region(ax, x, y, w, h, title, fc):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.2', fc=fc, ec='none', alpha=.45))
    ax.text(x + .15, y + h - .3, title, fontsize=9.5, weight='bold', color=INK, va='top')

# ---------- 1. System architecture ----------
fig, ax = plt.subplots(figsize=(11, 7.4)); ax.set_xlim(0, 22); ax.set_ylim(0, 15); ax.axis('off')
region(ax, .2, 8.4, 21.6, 6.4, 'EDGE LAYER  (lightweight, always-on)', EDGE)
region(ax, .2, 3.4, 21.6, 4.7, 'FOG LAYER  (invoked only on escalation)', FOG)
region(ax, .2, .1, 21.6, 3.0, 'TEMPORAL GRAPH INTELLIGENCE LAYER (multi-vector prediction)', STORE)

box(ax, .6, 11.2, 3.4, 1.6, 'Traffic observation\nx ∈ ℝ⁶ (simulated,\nmanual or CSV)', 'white')
box(ax, 4.5, 11.2, 3.2, 1.6, 'Schema & range\nvalidation (finite,\n≥0, SYN in [0,1])', 'white')
box(ax, 8.3, 12.6, 3.4, 1.0, 'Logistic Regression', 'white', fs=7.2)
box(ax, 8.3, 11.4, 3.4, 1.0, 'Decision Tree (depth 6)', 'white', fs=7.2)
box(ax, 8.3, 10.2, 3.4, 1.0, 'Random Forest (140 trees)', 'white', fs=7.2)
box(ax, 12.6, 11.2, 3.2, 1.6, 'Risk engine\nR(x) ∈ [0,100]\n+ feature indicators', 'white')
box(ax, 16.6, 10.9, 4.8, 2.2, 'Escalation gate\nmax conf < 0.72  OR\nmodels disagree  OR\nR(x) ≥ 70 ?', '#fff7d6', bold=False)
box(ax, 16.6, 8.75, 4.8, 1.3, 'Local decision:\nAllow / monitor or Local alert', 'white')
arrow(ax, (4.0, 12.0), (4.5, 12.0)); arrow(ax, (7.7, 12.0), (8.3, 13.1)); arrow(ax, (7.7, 12.0), (8.3, 11.9)); arrow(ax, (7.7, 12.0), (8.3, 10.7))
for yy in (13.1, 11.9, 10.7): arrow(ax, (11.7, yy), (12.6, 12.0))
ax.text(11.85, 13.3, 'P(attack)', fontsize=7, style='italic', color='#33424e')
arrow(ax, (15.8, 12.0), (16.6, 12.0))
arrow(ax, (19.0, 10.9), (19.0, 10.05), 'No', off=(.45, -.15))

box(ax, .6, 4.1, 4.2, 2.6, 'Resource forecaster\nreal CPU / RAM (psutil)\nĉ(t+1) = c(t) + mean of\nlast 3 differences', 'white')
box(ax, 5.6, 4.1, 4.4, 2.6, 'Candidate set\nif CPU>78% or RAM>80%:\n{DT, LR}\nelse: {RF, DT, LR}', 'white')
box(ax, 10.8, 4.1, 5.2, 2.6, 'Utility-based model selection\nU(m) = 0.5·conf + 0.45·F1\n + risk bonus − cost(m)·λ', 'white')
box(ax, 16.8, 4.1, 4.6, 2.6, 'Fog decision\nBlock + alert  or  Monitor\n(recommendation only)', 'white')
arrow(ax, (21.2, 10.9), (21.2, 6.7), 'Yes', off=(.45, .9))
arrow(ax, (4.8, 5.4), (5.6, 5.4)); arrow(ax, (10.0, 5.4), (10.8, 5.4)); arrow(ax, (16.0, 5.4), (16.8, 5.4))

box(ax, 16.2, .35, 5.2, 1.9, 'Event store + dynamic graph\nnetwork, identity, endpoint,\nalert events as timestamped edges', 'white')
box(ax, 8.4, .35, 6.8, 1.9, 'TG-AMV model\ntime-encoded neighbourhood attention\n+ adaptive vector weights + GRU\n→ P(next stage), next-victim ranking', 'white')
box(ax, .6, .35, 6.8, 1.9, 'SOC output\ncurrent and next stage, likely next\nvictim, risk, confidence,\nearly-warning lead time', 'white')
arrow(ax, (16.2, 1.5), (15.2, 1.5)); arrow(ax, (8.4, 1.5), (7.4, 1.5))
arrow(ax, (19.1, 4.1), (19.1, 2.25), 'logged', ls='--', off=(.6, -.1))
arrow(ax, (16.6, 9.2), (16.45, 2.25), None, ls='--')
plt.savefig(FIG / 'arch.png'); plt.close()

# ---------- 2. Methodology pipeline ----------
fig, ax = plt.subplots(figsize=(11, 4.3)); ax.set_xlim(0, 22); ax.set_ylim(0, 8.6); ax.axis('off')
steps = [('1. Data\ngeneration', 'Seeded synthetic\ngenerators\n(Mulberry32,\nBox–Muller)'),
         ('2. Validation\n& EDA', 'schema, ranges,\nduplicates,\ndistributions,\ncorrelation'),
         ('3. Splitting', 'binary: stratified\n75/25 + 5-fold CV\nsequence: chrono-\nlogical by episode'),
         ('4. Feature\nconstruction', 'raw 6 features\n(binary)\nW×6 + Δ window\n(sequence, 21-d)'),
         ('5. Model\ntraining', 'LR, DT, RF\n(+ NB, k-NN, GB\nbaselines)'),
         ('6. Evaluation', 'Acc, P, R, F1,\nROC-AUC, macro-F1,\nreserved eval set,\nablation'),
         ('7. Deployment', 'atomic joblib\npersist, Flask\nAPI, edge–fog\nrouting')]
w = 2.75
for i, (t, d) in enumerate(steps):
    x = .3 + i * 3.1
    box(ax, x, 5.0, w, 1.6, t, EDGE if i < 4 else FOG, fs=8.8, bold=True)
    box(ax, x, 1.0, w, 3.4, d, 'white', fs=7.8)
    if i < len(steps) - 1: arrow(ax, (x + w, 5.8), (x + 3.1, 5.8))
    arrow(ax, (x + w / 2, 5.0), (x + w / 2, 4.4), style='-')
ax.text(11, 7.5, 'End-to-end data science pipeline followed in Fogline', ha='center', fontsize=11, weight='bold', color=INK)
arrow(ax, (20.9, 1.0), (1.6, 1.0), 'retraining on uploaded CSV re-enters at step 2 (invalid data keeps previous model)', ls='--', rad=-.12, off=(0, -1.0), fs=7.8)
plt.savefig(FIG / 'pipeline.png'); plt.close()

# ---------- 3. Windowing and chronological split ----------
fig, axes = plt.subplots(2, 1, figsize=(11, 5.6), gridspec_kw={'height_ratios': [1, 1.1]})
ax = axes[0]; ax.set_xlim(0, 22); ax.set_ylim(0, 4); ax.axis('off')
stages = ['Recon', 'Recon', 'Recon', 'Cred', 'Cred', 'Lateral', 'Lateral', 'Lateral', 'Exfil', 'Exfil']
cols = {'Recon': '#dbe9f8', 'Cred': '#fde3d6', 'Lateral': '#fff0c2', 'Exfil': '#d8f1e6'}
for i, s in enumerate(stages):
    x = .5 + i * 2.1
    ax.add_patch(Rectangle((x, 1.4), 1.8, 1.0, fc=cols[s], ec='#5b6b78'))
    ax.text(x + .9, 1.9, f'o{i+1}\n{s}', ha='center', va='center', fontsize=7.8)
ax.add_patch(Rectangle((.4 + 2 * 2.1, 1.25), 3 * 2.1 - .1, 1.3, fc='none', ec='#2a78d6', lw=2))
ax.add_patch(Rectangle((.4 + 5 * 2.1, 1.25), 2.0, 1.3, fc='none', ec='#eb6834', lw=2, ls='--'))
ax.text(.4 + 3.5 * 2.1, 2.8, 'input window: 3 × 6 features + (o5 − o3) = 21 values', ha='center', fontsize=8, color='#2a78d6')
ax.text(.4 + 5.5 * 2.1, .85, 'target y = stage of o6', ha='center', fontsize=8, color='#eb6834')
ax.text(.5, 3.6, 'Sliding window within one episode (example: synthetic APT path)', fontsize=9.5, weight='bold', color=INK)
ax.text(.5, .3, 'Windows never cross episodes and are dropped when any gap between consecutive observations exceeds 30 minutes.', fontsize=8, color='#33424e')

ax = axes[1]; ax.set_xlim(0, 22); ax.set_ylim(0, 4.6); ax.axis('off')
ax.text(.5, 4.3, 'Episode-isolated chronological split (no temporal leakage)', fontsize=9.5, weight='bold', color=INK)
import numpy as np
rng = np.random.default_rng(3)
starts = np.sort(rng.uniform(.5, 19, 18)); cut = starts[int(len(starts) * .75)]
for i, s in enumerate(starts):
    L = rng.uniform(.8, 1.6); yy = 3.5 - (i % 6) * .45
    fc = '#2a78d6' if s + L < cut else ('#eb6834' if s >= cut else '#b8b8b8')
    ax.add_patch(Rectangle((s, yy), L, .3, fc=fc, ec='white'))
ax.axvline(cut, ymin=.08, ymax=.88, color='#222', ls='--', lw=1.2)
ax.text(cut + .1, .45, 'cut-off = start of the (0.75·N)-th episode', fontsize=8)
ax.text(.5, .45, 'time →', fontsize=8)
for x, c, t in ((14.5, '#2a78d6', 'training episodes (end before cut-off)'), (14.5, '#eb6834', 'holdout episodes (start after cut-off)'), (14.5, '#b8b8b8', 'straddling episodes purged')):
    pass
lx = 20.2
for j, (c, t) in enumerate((('#2a78d6', 'train'), ('#eb6834', 'holdout'), ('#b8b8b8', 'purged'))):
    ax.add_patch(Rectangle((lx, 3.4 - j * .5), .4, .3, fc=c)); ax.text(lx + .55, 3.47 - j * .5, t, fontsize=8)
plt.tight_layout(); plt.savefig(FIG / 'windowing.png'); plt.close()
print('ok')
