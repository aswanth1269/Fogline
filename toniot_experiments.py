"""Fogline models evaluated on the real ToN-IoT network dataset (Train_Test_Network subset)."""
import json, time, pickle, warnings, hashlib
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve,
                             confusion_matrix, classification_report)
warnings.filterwarnings('ignore')
ROOT = Path(__file__).parent; FIG = ROOT / 'figs'
SRC = ROOT / 'data' / 'TON_IoT_Train_Test_Network.csv'
if not SRC.exists():
    SRC = Path('/home/claude/probe/SecurityArchitectureSmartCityIoT/data/TON_IoT_Train_Test_Network.csv')
C = ['#2a78d6', '#eb6834', '#1baf7a']
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.color': '#e6e6e6', 'figure.dpi': 200, 'savefig.bbox': 'tight', 'lines.linewidth': 2})
def save(n): plt.savefig(FIG / f'{n}.png'); plt.close()
R = {}

raw = pd.read_csv(SRC, encoding='utf-8-sig', low_memory=False)
R['sha256'] = hashlib.sha256(SRC.read_bytes()).hexdigest()
R['raw_rows'] = len(raw); R['raw_cols'] = raw.shape[1]
R['type_counts_raw'] = raw['type'].value_counts().to_dict()

# ---- preprocessing ----
drop_id = ['src_ip', 'dst_ip', 'src_port', 'dst_port']              # identity columns leak the attacker
drop_text = ['dns_query', 'ssl_subject', 'ssl_issuer', 'http_uri', 'http_user_agent', 'weird_name', 'weird_addl',
             'http_orig_mime_types', 'http_resp_mime_types', 'ssl_cipher', 'ssl_version', 'http_method', 'http_version']
df = raw.drop(columns=drop_id + drop_text)
for c in ['dns_AA', 'dns_RD', 'dns_RA', 'dns_rejected', 'ssl_resumed', 'ssl_established', 'weird_notice']:
    df[c] = df[c].map({'T': 1, 'F': 0}).fillna(-1)
num = ['duration', 'src_bytes', 'dst_bytes', 'missed_bytes', 'src_pkts', 'src_ip_bytes', 'dst_pkts', 'dst_ip_bytes',
       'dns_qclass', 'dns_qtype', 'dns_rcode', 'http_trans_depth', 'http_request_body_len', 'http_response_body_len', 'http_status_code']
for c in num: df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
df['http_trans_depth'] = df['http_trans_depth'].clip(lower=0)
R['missing_after'] = int(df.isna().sum().sum())
before = len(df); df = df.drop_duplicates().reset_index(drop=True)
R['duplicates_removed'] = before - len(df); R['clean_rows'] = len(df)
R['type_counts_clean'] = df['type'].value_counts().to_dict()
R['label_counts_clean'] = df['label'].value_counts().to_dict()

# Fogline six-feature mapping (proxies)
pk = df.src_pkts + df.dst_pkts
fog = pd.DataFrame({
    'packet_rate': pk / df.duration.clip(lower=1e-3),
    'failed_logins': df.conn_state.isin(['REJ', 'RSTO', 'RSTOS0', 'RSTR', 'RSTRH']).astype(int),
    'unique_ports': df.service.ne('-').astype(int),
    'avg_packet_size': (df.src_ip_bytes + df.dst_ip_bytes) / pk.clip(lower=1),
    'connection_duration': df.duration,
    'syn_ratio': df.conn_state.isin(['S0', 'SH', 'REJ']).astype(float),
})
# NOTE: per-flow data has no logins or per-source port counts; documented as proxies in the report.
full = pd.get_dummies(df.drop(columns=['label', 'type']), columns=['proto', 'service', 'conn_state'], dtype=float)
R['n_full_features'] = full.shape[1]
y = df['label'].astype(int).values; ytype = df['type'].values

def models():
    return {'Logistic Regression': Pipeline([('s', StandardScaler()), ('m', LogisticRegression(max_iter=2000, random_state=42))]),
            'Decision Tree': DecisionTreeClassifier(max_depth=6, random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=140, max_depth=9, random_state=42, n_jobs=-1)}

idx_tr, idx_te = train_test_split(np.arange(len(df)), test_size=.25, random_state=42, stratify=ytype)
R['train_rows'] = len(idx_tr); R['test_rows'] = len(idx_te)

def evaluate(X, tag):
    out, fitted = {}, {}
    Xtr, Xte = X.iloc[idx_tr], X.iloc[idx_te]
    for n, m in models().items():
        t = time.perf_counter(); m.fit(Xtr, y[idx_tr]); ft = time.perf_counter() - t
        p = m.predict(Xte); pr = m.predict_proba(Xte)[:, 1]; cm = confusion_matrix(y[idx_te], p)
        out[n] = dict(accuracy=accuracy_score(y[idx_te], p), precision=precision_score(y[idx_te], p), recall=recall_score(y[idx_te], p),
                      f1=f1_score(y[idx_te], p), roc_auc=roc_auc_score(y[idx_te], pr), fpr=cm[0, 1] / cm[0].sum(), cm=cm.tolist(), fit_s=ft)
        fitted[n] = (m, pr, p)
    return out, fitted

R['bin_fog6'], fit6 = evaluate(fog, 'fog6')
R['bin_full'], fitF = evaluate(full, 'full')

# 5-fold CV on full features
skf = StratifiedKFold(5, shuffle=True, random_state=42)
R['cv_full'] = {}
for n, m in models().items():
    s = cross_validate(m, full, y, cv=skf, scoring=['accuracy', 'f1'])
    R['cv_full'][n] = {k[5:]: [float(s[k].mean()), float(s[k].std())] for k in s if k.startswith('test_')}

# per attack type detection rate (RF, full) -- recall of attack per type
rf_pred = fitF['Random Forest'][2]; te_types = ytype[idx_te]
R['detect_by_type'] = {t: float((rf_pred[te_types == t] == (0 if t == 'normal' else 1)).mean()) for t in sorted(set(te_types))}

# multiclass RF on 10 types
rfm = RandomForestClassifier(n_estimators=140, max_depth=None, random_state=42, n_jobs=-1)
rfm.fit(full.iloc[idx_tr], ytype[idx_tr]); pm = rfm.predict(full.iloc[idx_te])
R['multi'] = dict(accuracy=accuracy_score(ytype[idx_te], pm), macro_f1=f1_score(ytype[idx_te], pm, average='macro'),
                  report=classification_report(ytype[idx_te], pm, output_dict=True, zero_division=0))
labs = sorted(set(ytype))
cmm = confusion_matrix(ytype[idx_te], pm, labels=labs); cmn = cmm / cmm.sum(1, keepdims=True)
plt.figure(figsize=(7.4, 6))
sns.heatmap(cmn, annot=True, fmt='.2f', cmap='Blues', xticklabels=labs, yticklabels=labs, vmin=0, vmax=1, linewidths=1, linecolor='white',
            annot_kws={'size': 8}, cbar_kws={'shrink': .7})
plt.xlabel('Predicted type'); plt.ylabel('Actual type'); plt.title('ToN-IoT attack type confusion matrix (Random Forest)', fontsize=10)
save('ton_multiclass_cm')

# feature importance (full RF, top 12)
imp = pd.Series(fitF['Random Forest'][0].feature_importances_, index=full.columns).sort_values(ascending=False).head(12)[::-1]
R['top_features'] = imp[::-1].round(4).to_dict()
plt.figure(figsize=(7, 4)); plt.barh(imp.index, imp.values, color=C[0], height=.6); plt.title('Top 12 features, Random Forest (ToN-IoT)')
plt.xlabel('Importance'); save('ton_importance')

# Class distribution plot
tc = df['type'].value_counts()
plt.figure(figsize=(8, 3.4)); plt.bar(tc.index, tc.values, color=[C[0] if t == 'normal' else C[1] for t in tc.index], width=.65)
plt.ylabel('Rows after cleaning'); plt.title('ToN-IoT traffic types after removing duplicates'); plt.xticks(rotation=20); save('ton_classes')

# ROC: fog6 vs full for RF and DT
plt.figure(figsize=(6, 4.2))
for i, (n, fitd, lab) in enumerate([('Random Forest', fitF, 'RF, all features'), ('Decision Tree', fitF, 'DT, all features'),
                                     ('Random Forest', fit6, 'RF, Fogline 6 features')]):
    fpr, tpr, _ = roc_curve(y[idx_te], fitd[n][1]); plt.plot(fpr, tpr, color=C[i], label=f'{lab} (AUC = {roc_auc_score(y[idx_te], fitd[n][1]):.3f})')
plt.plot([0, 1], [0, 1], '--', color='#999', lw=1); plt.xlabel('False positive rate'); plt.ylabel('True positive rate')
plt.title('ROC curves on ToN-IoT test set'); plt.legend(frameon=False, fontsize=8.5, loc='lower right'); save('ton_roc')

# Edge-fog gate on real data (confidence + disagreement rule, full features)
probs = np.vstack([fitF[n][1] for n in ['Logistic Regression', 'Decision Tree', 'Random Forest']])
preds = (probs >= .5).astype(int); conf = np.maximum(probs, 1 - probs).max(0)
disagree = preds.min(0) != preds.max(0); lowc = conf < .72
esc = disagree | lowc
best = np.where(np.maximum(probs, 1 - probs).argmax(0)[None, :] == np.arange(3)[:, None], preds, 0).sum(0)
yte = y[idx_te]
edge_acc = float((best[~esc] == yte[~esc]).mean()); rf_on_esc = float((preds[2][esc] == yte[esc]).mean()); edge_on_esc = float((best[esc] == yte[esc]).mean())
R['gate'] = dict(escalation_rate=float(esc.mean()), disagree=float(disagree.mean()), lowconf=float(lowc.mean()), kept=int((~esc).sum()), sent=int(esc.sum()),
                 edge_acc_kept=edge_acc, best_edge_acc_on_sent=edge_on_esc, rf_acc_on_sent=rf_on_esc,
                 overall_with_gate=float(((best == yte) & ~esc).sum() + (preds[2][esc] == yte[esc]).sum()) / len(yte))
json.dump(R, open(ROOT / 'toniot_results.json', 'w'), indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o))
print(json.dumps({k: v for k, v in R.items() if k not in ('multi',)}, indent=1, default=str)[:6000])
print('multi', R['multi']['accuracy'], R['multi']['macro_f1'])
for k, v in R['multi']['report'].items():
    if isinstance(v, dict): print(k, {a: round(b, 3) for a, b in v.items()})
