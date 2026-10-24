"""Report experiments for Fogline. All numbers in the report come from this script."""
import json, time, pickle, sys, warnings, tempfile
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
                             roc_curve, precision_recall_curve, average_precision_score, confusion_matrix,
                             classification_report)
from sklearn.inspection import permutation_importance

warnings.filterwarnings('ignore')
ROOT = Path(__file__).parent; sys.path.insert(0, str(ROOT))
from backend.config import FEATURES
from backend.utils.sample_generator import generate_dataset, generate_live_sample
from backend.services import next_move as nm
from backend.services.risk_engine import calculate_risk
from backend.services.fog_orchestrator import FogOrchestrator

FIG = ROOT / 'figs'; FIG.mkdir(exist_ok=True)
R = {}
C = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7', '#e34948']
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.grid': True, 'grid.color': '#e6e6e6',
                     'grid.linewidth': .6, 'axes.edgecolor': '#888', 'figure.dpi': 200,
                     'savefig.bbox': 'tight', 'lines.linewidth': 2})
def save(name): plt.savefig(FIG / f'{name}.png'); plt.close()

# ---------------- A. Binary IDS -----------------
df = pd.read_csv(ROOT / 'data/synthetic_binary_balanced.csv')
X, y = df[FEATURES].astype(float), df['label'].astype(int)
R['binary_rows'] = len(df); R['binary_class_counts'] = y.value_counts().to_dict()
desc = df.groupby('label')[FEATURES].agg(['mean', 'std']).round(3)
R['binary_desc'] = {f: {str(l): [float(desc.loc[l, (f, 'mean')]), float(desc.loc[l, (f, 'std')])] for l in (0, 1)} for f in FEATURES}
R['missing_values'] = int(df[FEATURES].isna().sum().sum())
R['duplicates'] = int(df.duplicated().sum())
R['skew'] = df[FEATURES].skew().round(3).to_dict()

# EDA: distributions
fig, axes = plt.subplots(2, 3, figsize=(11, 6))
for ax, f in zip(axes.ravel(), FEATURES):
    for lab, col, nm_ in ((0, C[0], 'Normal'), (1, C[1], 'Attack')):
        ax.hist(df.loc[df.label == lab, f], bins=40, alpha=.55, color=col, label=nm_, density=True)
    ax.set_title(f, fontsize=10); ax.set_yticks([])
axes[0, 0].legend(frameon=False)
plt.suptitle('Feature distributions by class (synthetic_binary_balanced.csv, n = 6,000)', fontsize=11)
plt.tight_layout(); save('eda_distributions')

corr = df[FEATURES + ['label']].corr()
R['corr_with_label'] = corr['label'].drop('label').round(3).to_dict()
plt.figure(figsize=(6.4, 5.2))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r', vmin=-1, vmax=1, square=True,
            cbar_kws={'shrink': .75}, annot_kws={'size': 8}, linewidths=.5, linecolor='white')
plt.title('Pearson correlation matrix'); save('eda_correlation')

plt.figure(figsize=(9, 3.6))
m = df.melt(id_vars='label', value_vars=FEATURES)
m['z'] = m.groupby('variable')['value'].transform(lambda s: (s - s.mean()) / s.std())
m['Class'] = m.label.map({0: 'Normal', 1: 'Attack'})
sns.boxplot(data=m, x='variable', y='z', hue='Class', palette=[C[0], C[1]], fliersize=1.2, linewidth=.8)
plt.ylabel('z-score'); plt.xlabel(''); plt.title('Standardised feature spread by class'); plt.legend(frameon=False)
save('eda_boxplot')

def models():
    return {
        'Logistic Regression': Pipeline([('s', StandardScaler()), ('m', LogisticRegression(max_iter=500, random_state=42))]),
        'Decision Tree': DecisionTreeClassifier(max_depth=6, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=140, max_depth=9, random_state=42),
        'Gaussian NB (baseline)': GaussianNB(),
        'k-NN, k=5 (baseline)': Pipeline([('s', StandardScaler()), ('m', KNeighborsClassifier(5))]),
        'Gradient Boosting (baseline)': GradientBoostingClassifier(random_state=42),
    }

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.25, random_state=42, stratify=y)
hold, fitted = {}, {}
for name, mdl in models().items():
    t = time.perf_counter(); mdl.fit(Xtr, ytr); tt = time.perf_counter() - t
    p = mdl.predict(Xte); pr = mdl.predict_proba(Xte)[:, 1]
    one = Xte.iloc[[0]]
    for _ in range(50): mdl.predict_proba(one)
    lat = []
    for _ in range(1000):
        t = time.perf_counter(); mdl.predict_proba(one); lat.append(time.perf_counter() - t)
    hold[name] = dict(accuracy=accuracy_score(yte, p), precision=precision_score(yte, p), recall=recall_score(yte, p),
                      f1=f1_score(yte, p), roc_auc=roc_auc_score(yte, pr), fit_s=tt,
                      latency_ms=float(np.median(lat) * 1000), size_kb=len(pickle.dumps(mdl)) / 1024,
                      cm=confusion_matrix(yte, p).tolist(),
                      fpr_rate=float(confusion_matrix(yte, p)[0, 1] / (yte == 0).sum()))
    fitted[name] = (mdl, pr)
R['holdout'] = hold; R['test_rows'] = len(yte)

skf = StratifiedKFold(5, shuffle=True, random_state=42)
cvr = {}
for name, mdl in models().items():
    s = cross_validate(mdl, X, y, cv=skf, scoring=['accuracy', 'precision', 'recall', 'f1', 'roc_auc'])
    cvr[name] = {k[5:]: [float(s[k].mean()), float(s[k].std())] for k in s if k.startswith('test_')}
R['cv5'] = cvr

core = ['Logistic Regression', 'Decision Tree', 'Random Forest']
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
for i, n in enumerate(core):
    pr = fitted[n][1]; fpr, tpr, _ = roc_curve(yte, pr)
    axes[0].plot(fpr, tpr, color=C[i], label=f"{n} (AUC = {roc_auc_score(yte, pr):.3f})")
    p_, r_, _ = precision_recall_curve(yte, pr)
    axes[1].plot(r_, p_, color=C[i], label=f"{n} (AP = {average_precision_score(yte, pr):.3f})")
axes[0].plot([0, 1], [0, 1], '--', color='#999', lw=1)
axes[0].set(xlabel='False positive rate', ylabel='True positive rate', title='ROC curves (holdout, n = 1,500)')
axes[1].set(xlabel='Recall', ylabel='Precision', title='Precision-recall curves')
for a in axes: a.legend(frameon=False, fontsize=8.5, loc='lower right' if a is axes[0] else 'lower left')
# zoomed inset
ins = axes[0].inset_axes([.42, .3, .5, .4])
for i, n in enumerate(core):
    fpr, tpr, _ = roc_curve(yte, fitted[n][1]); ins.plot(fpr, tpr, color=C[i], lw=1.5)
ins.set_xlim(0, .1); ins.set_ylim(.9, 1.0); ins.tick_params(labelsize=7); ins.set_title('zoom', fontsize=7)
plt.tight_layout(); save('roc_pr')

fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
for ax, n in zip(axes, core):
    cm = np.array(hold[n]['cm'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax, annot_kws={'size': 11},
                xticklabels=['Normal', 'Attack'], yticklabels=['Normal', 'Attack'], linewidths=1, linecolor='white')
    ax.set_title(n, fontsize=10); ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
plt.tight_layout(); save('confusion_binary')

rf = fitted['Random Forest'][0]
pi = permutation_importance(rf, Xte, yte, n_repeats=15, random_state=42, scoring='f1')
imp = pd.DataFrame({'impurity': rf.feature_importances_, 'perm': pi.importances_mean, 'perm_sd': pi.importances_std}, index=FEATURES)
R['importance'] = imp.round(4).to_dict()
lr = fitted['Logistic Regression'][0]
R['lr_coef'] = dict(zip(FEATURES, np.round(lr.named_steps['m'].coef_[0], 3).tolist()))
imp = imp.sort_values('impurity')
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
axes[0].barh(imp.index, imp.impurity, color=C[0], height=.6); axes[0].set_title('Random Forest: mean decrease in impurity')
axes[1].barh(imp.index, imp.perm, xerr=imp.perm_sd, color=C[2], height=.6); axes[1].set_title('Permutation importance (drop in F1)')
plt.tight_layout(); save('feature_importance')

# cross-generator: deployed model trained on built-in generator, tested on dataset file
builtin = generate_dataset()
cross = {}
for name in core:
    mdl = models()[name]; mdl.fit(builtin[FEATURES], builtin.label)
    p = mdl.predict(X); cross[name] = dict(accuracy=accuracy_score(y, p), f1=f1_score(y, p), recall=recall_score(y, p),
                                           precision=precision_score(y, p))
R['cross_generator'] = cross

# Learning curve for RF/LR/DT
sizes = [40, 80, 160, 320, 640, 1280, 2500, 4500]
lc = {n: [] for n in core}
for s in sizes:
    for n in core:
        sc = []
        for seed in range(5):
            Xs, _, ys, _ = train_test_split(Xtr, ytr, train_size=min(s, len(Xtr) - 2), random_state=seed, stratify=ytr)
            mdl = models()[n]; mdl.fit(Xs, ys); sc.append(f1_score(yte, mdl.predict(Xte)))
        lc[n].append((float(np.mean(sc)), float(np.std(sc))))
R['learning_curve'] = {'sizes': sizes, **lc}
plt.figure(figsize=(6.8, 3.8))
for i, n in enumerate(core):
    mu = np.array([a for a, _ in lc[n]]); sd = np.array([b for _, b in lc[n]])
    plt.plot(sizes, mu, marker='o', ms=4, color=C[i], label=n); plt.fill_between(sizes, mu - sd, mu + sd, color=C[i], alpha=.15)
plt.xscale('log'); plt.xlabel('Training rows (log scale)'); plt.ylabel('Holdout F1'); plt.title('Learning curves (mean ± sd over 5 subsamples)')
plt.legend(frameon=False); save('learning_curve')

# ---------------- B. Edge-fog gate simulation -----------------
dep = {'tiny_ids': fitted['Logistic Regression'][0], 'decision_tree': fitted['Decision Tree'][0], 'random_forest': fitted['Random Forest'][0]}
pretty = {'tiny_ids': 'Logistic Regression', 'decision_tree': 'Decision Tree', 'random_forest': 'Random Forest'}
metrics = {k: {'f1': hold[pretty[k]]['f1']} for k in dep}
orch = FogOrchestrator()
rows = []
for kind in ('normal', 'uncertain', 'attack'):
    for i in range(500):
        s = generate_live_sample(kind, seed={'normal': 10_000, 'uncertain': 20_000, 'attack': 30_000}[kind] + i)
        xx = pd.DataFrame([s])[FEATURES]
        res = []
        for k, mdl in dep.items():
            ap = float(mdl.predict_proba(xx)[0][1]); res.append(dict(key=k, model=pretty[k], prediction='Attack' if ap >= .5 else 'Normal',
                                                                       attack_probability=ap, confidence=max(ap, 1 - ap)))
        risk, level = calculate_risk(s, res)
        for state, resources in (('normal', {'cpu_forecast': 35, 'ram_forecast': 55}), ('constrained', {'cpu_forecast': 90, 'ram_forecast': 85})):
            d = orch.decide(res, resources, risk, metrics)
            rows.append(dict(kind=kind, state=state, risk=risk, level=level, escalated=d['escalated'], model=d['selected_model'],
                             action=d['action'], disagree=len({r['prediction'] for r in res}) > 1,
                             lowconf=max(r['confidence'] for r in res) < .72))
sim = pd.DataFrame(rows)
g = sim[sim.state == 'normal'].groupby('kind')
R['gate'] = {k: dict(escalation_rate=float(v.escalated.mean()), disagree=float(v.disagree.mean()), lowconf=float(v.lowconf.mean()),
                     mean_risk=float(v.risk.mean()), sd_risk=float(v.risk.std()),
                     levels=v.level.value_counts().to_dict(), actions=v.action.value_counts().to_dict()) for k, v in g}
R['gate_model_choice'] = {st: sim[(sim.state == st) & sim.escalated].model.value_counts().to_dict() for st in ('normal', 'constrained')}
R['gate_offload_overall'] = float(sim[sim.state == 'normal'].escalated.mean())
fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.7))
for i, k in enumerate(('normal', 'uncertain', 'attack')):
    axes[0].hist(sim[(sim.kind == k) & (sim.state == 'normal')].risk, bins=30, color=C[i], alpha=.65, label=k.capitalize())
for t, lab in ((45, 'Medium'), (75, 'High')):
    axes[0].axvline(t, ls='--', color='#555', lw=1); axes[0].text(t + 1, axes[0].get_ylim()[1] * .9, lab, fontsize=8)
axes[0].set(xlabel='Risk score (0–100)', ylabel='Samples', title='Risk score by scenario (500 each)'); axes[0].legend(frameon=False)
tab = sim[sim.escalated].groupby(['state', 'model']).size().unstack(fill_value=0)
tab = tab.reindex(index=['normal', 'constrained'], columns=['random_forest', 'decision_tree', 'tiny_ids'], fill_value=0)
bottom = np.zeros(len(tab))
for i, col in enumerate(tab.columns):
    axes[1].bar(['Resources normal', 'Resources constrained'], tab[col].values, bottom=bottom, color=C[[0, 1, 2][i]], label=pretty[col], width=.5,
                edgecolor='white', linewidth=2)
    bottom += tab[col].values
axes[1].set(ylabel='Escalated samples', title='Fog model chosen by utility function', ylim=(0, 780)); axes[1].legend(frameon=False, fontsize=8.5, ncol=3, loc='upper center')
plt.tight_layout(); save('gate_simulation')

# ---------------- C. Next-move -----------------
def load_seq(p): return nm.validate_frame(pd.read_csv(ROOT / p))
var, base, ev = load_seq('data/synthetic_sequence_variable.csv'), load_seq('data/synthetic_sequence_baseline.csv'), load_seq('data/synthetic_sequence_evaluation.csv')
R['seq_rows'] = {'variable': len(var), 'baseline': len(base), 'evaluation': len(ev)}
R['seq_stage_counts'] = {k: d.attack_stage.value_counts().to_dict() for k, d in (('variable', var), ('evaluation', ev))}
R['eval_starts_after_train'] = str(ev.timestamp.min()) + ' > ' + str(var.timestamp.max())

with tempfile.TemporaryDirectory() as tmp:
    pv = nm.NextMovePredictor.__new__(nm.NextMovePredictor); pv.path = Path(tmp) / 'nm.joblib'; pv.bundle = None
    st = pv.train(pd.read_csv(ROOT / 'data/synthetic_sequence_variable.csv'), source='synthetic')
    R['nm_internal_holdout'] = {k: v for k, v in st['metrics'].items() if k != 'per_class'}
    model = pv.bundle['model']
xe, ye, _, _ = nm.make_windows(ev)
pe = model.predict(xe); proba = model.predict_proba(xe)
labels = list(model.classes_)
R['nm_eval'] = dict(windows=len(ye), accuracy=accuracy_score(ye, pe), macro_f1=f1_score(ye, pe, average='macro'),
                    weighted_f1=f1_score(ye, pe, average='weighted'),
                    majority=accuracy_score(ye, [pd.Series(nm.make_windows(var)[1]).value_counts().idxmax()] * len(ye)),
                    per_class=classification_report(ye, pe, output_dict=True, zero_division=0))
order = ['benign', 'reconnaissance', 'credential_access', 'lateral_movement', 'exfiltration', 'denial_of_service']
short = ['Benign', 'Recon', 'Cred.\naccess', 'Lateral\nmove', 'Exfil.', 'DoS']
cm = confusion_matrix(ye, pe, labels=order); cmn = cm / cm.sum(1, keepdims=True)
plt.figure(figsize=(6.6, 5.2))
sns.heatmap(cmn, annot=True, fmt='.2f', cmap='Blues', xticklabels=short, yticklabels=short, vmin=0, vmax=1, linewidths=1, linecolor='white',
            cbar_kws={'shrink': .75}, annot_kws={'size': 9})
plt.xlabel('Predicted next stage'); plt.ylabel('Actual next stage')
plt.title(f'Next-move confusion matrix, reserved evaluation set (n = {len(ye)} windows)', fontsize=10); save('nm_confusion')
R['nm_cm'] = cm.tolist()

# abstention curve
top = proba.max(1); correct = (pe == ye)
ths = np.round(np.arange(0.2, 0.96, .05), 2); cov, acc = [], []
for t in ths:
    k = top >= t; cov.append(float(k.mean())); acc.append(float(correct[k].mean()) if k.any() else np.nan)
R['abstain'] = dict(thresholds=ths.tolist(), coverage=cov, accuracy=acc,
                    at045=dict(coverage=float((top >= .45).mean()), accuracy=float(correct[top >= .45].mean()),
                               acc_abstained=float(correct[top < .45].mean()) if (top < .45).any() else None))
fig, ax = plt.subplots(figsize=(6.8, 3.8))
ax.plot(ths, acc, marker='o', ms=4, color=C[0], label='Accuracy on answered windows')
ax.plot(ths, cov, marker='s', ms=4, color=C[1], label='Coverage (share answered)')
ax.axvline(.45, ls='--', color='#555', lw=1); ax.text(.46, .45, 'deployed threshold 0.45', fontsize=8)
ax.set(xlabel='Minimum top-class score', ylabel='Proportion', ylim=(0, 1.02), title='Abstention trade-off on the reserved evaluation set')
ax.legend(frameon=False, loc='lower left'); save('nm_abstention')

# window ablation + model comparison (train on variable chronological-train side, test on eval file)
spans = var.groupby('episode_id')['timestamp'].agg(['min', 'max']).sort_values('min')
def windows_w(frame, W):
    x_, y_ = [], []
    for _, gdf in frame.groupby('episode_id', sort=False):
        rws = gdf.sort_values('timestamp').to_dict('records')
        for i in range(W - 1, len(rws) - 1):
            blk = rws[i - W + 1:i + 2]
            gaps = [(b['timestamp'] - a['timestamp']).total_seconds() for a, b in zip(blk, blk[1:])]
            if any(gp <= 0 or gp > 1800 for gp in gaps): continue
            v = np.array([[r[f] for f in FEATURES] for r in blk[:-1]], float)
            x_.append(np.concatenate([v.ravel(), v[-1] - v[0]])); y_.append(blk[-1]['attack_stage'])
    return np.array(x_), np.array(y_)
abl = {}
for W in (1, 2, 3, 4, 5):
    xtr_, ytr_ = windows_w(var, W); xte_, yte_ = windows_w(ev, W)
    mdl = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_leaf=3, class_weight='balanced_subsample', random_state=42)
    mdl.fit(xtr_, ytr_); p_ = mdl.predict(xte_)
    abl[W] = dict(train=len(ytr_), test=len(yte_), accuracy=accuracy_score(yte_, p_), macro_f1=f1_score(yte_, p_, average='macro'))
R['window_ablation'] = abl
xtr3, ytr3 = windows_w(var, 3); xte3, yte3 = windows_w(ev, 3)
comp = {}
cands = {
    'Majority class': None,
    'Logistic Regression': Pipeline([('s', StandardScaler()), ('m', LogisticRegression(max_iter=2000, class_weight='balanced'))]),
    'Decision Tree': DecisionTreeClassifier(max_depth=10, min_samples_leaf=3, class_weight='balanced', random_state=42),
    'Random Forest (Fogline)': RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_leaf=3, class_weight='balanced_subsample', random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
}
for n, mdl in cands.items():
    if mdl is None:
        maj = pd.Series(ytr3).value_counts().idxmax(); p_ = np.array([maj] * len(yte3))
    else:
        mdl.fit(xtr3, ytr3); p_ = mdl.predict(xte3)
    comp[n] = dict(accuracy=accuracy_score(yte3, p_), macro_f1=f1_score(yte3, p_, average='macro', zero_division=0))
# Markov oracle: uses TRUE current stage (not available at inference) as a reference
trans = {}
for _, gdf in var.groupby('episode_id'):
    s_ = gdf.sort_values('timestamp').attack_stage.tolist()
    for a, b in zip(s_, s_[1:]): trans.setdefault(a, {}).setdefault(b, 0); trans[a][b] += 1
mk_t, mk_p = [], []
for _, gdf in ev.groupby('episode_id'):
    s_ = gdf.sort_values('timestamp').attack_stage.tolist()
    for i in range(2, len(s_) - 1):
        mk_t.append(s_[i + 1]); mk_p.append(max(trans[s_[i]], key=trans[s_[i]].get))
comp['Markov chain (oracle current stage)'] = dict(accuracy=accuracy_score(mk_t, mk_p), macro_f1=f1_score(mk_t, mk_p, average='macro', zero_division=0))
R['nm_model_comparison'] = comp
R['transition_matrix'] = {a: {b: c / sum(d.values()) for b, c in d.items()} for a, d in trans.items()}

# transition heatmap
T = pd.DataFrame(R['transition_matrix']).T.reindex(index=order, columns=order).fillna(0)
plt.figure(figsize=(6.6, 5.2))
sns.heatmap(T, annot=True, fmt='.2f', cmap='Blues', xticklabels=short, yticklabels=short, vmin=0, vmax=1, linewidths=1, linecolor='white',
            cbar_kws={'shrink': .75}, annot_kws={'size': 9})
plt.xlabel('Next stage'); plt.ylabel('Current stage'); plt.title('Empirical stage-transition probabilities (training file)', fontsize=10)
save('nm_transitions')

fig, axes = plt.subplots(1, 2, figsize=(11, 3.9))
Ws = list(abl); axes[0].plot(Ws, [abl[w]['accuracy'] for w in Ws], marker='o', color=C[0], label='Accuracy')
axes[0].plot(Ws, [abl[w]['macro_f1'] for w in Ws], marker='s', color=C[1], label='Macro F1')
axes[0].set(xlabel='Window length W (observations)', ylabel='Score', title='Window-length ablation (reserved eval set)', xticks=Ws)
axes[0].legend(frameon=False)
names = list(comp); vals = [comp[n]['macro_f1'] for n in names]
cols = ['#b8b8b8' if 'Majority' in n or 'oracle' in n else (C[0] if 'Fogline' in n else '#7fa9de') for n in names]
axes[1].barh(names[::-1], vals[::-1], color=cols[::-1], height=.6)
for i, v in enumerate(vals[::-1]): axes[1].text(v + .01, i, f'{v:.3f}', va='center', fontsize=8.5)
axes[1].set(xlabel='Macro F1', title='Next-move model comparison (W = 3)', xlim=(0, 1.05))
plt.tight_layout(); save('nm_ablation_comparison')

# per-class bar
pc = R['nm_eval']['per_class']
fig, ax = plt.subplots(figsize=(8, 3.6)); xs = np.arange(len(order)); w = .27
for i, met in enumerate(('precision', 'recall', 'f1-score')):
    ax.bar(xs + (i - 1) * w, [pc[o][met] for o in order], w, color=C[i], label=met.capitalize(), edgecolor='white', linewidth=1)
ax.set_xticks(xs, [s.replace('\n', ' ') for s in short]); ax.set_ylim(0, 1.22); ax.set_yticks([0, .2, .4, .6, .8, 1.0]); ax.legend(frameon=False, ncol=3, loc='upper center')
ax.set_title('Per-stage precision, recall and F1 (reserved evaluation set)'); save('nm_per_class')

# Sequence EDA: stage counts
fig, ax = plt.subplots(figsize=(8, 3.3))
cnt = pd.DataFrame({'Training (variable)': var.attack_stage.value_counts(), 'Evaluation (reserved)': ev.attack_stage.value_counts()}).reindex(order)
cnt.plot.bar(ax=ax, color=[C[0], C[1]], width=.7, edgecolor='white', linewidth=1.5)
ax.set_xticklabels([s.replace('\n', ' ') for s in short], rotation=0); ax.set_ylabel('Observations'); ax.legend(frameon=False)
ax.set_title('Stage distribution in sequence datasets'); save('seq_stage_counts')

json.dump(R, open(ROOT / 'results.json', 'w'), indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o))
print(json.dumps(R, indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o))[:20000])
