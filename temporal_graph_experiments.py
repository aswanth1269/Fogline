"""Temporal graph intelligence experiments: next attack stage and next victim prediction.

Compares: majority, Random Forest (Fogline window), XGBoost, LSTM, static graph model and the
adaptive multi-vector temporal graph model (TG-AMV). Chronological split by incident start time.
"""
import json, bisect, time, warnings
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from xgboost import XGBClassifier
import sys; sys.path.insert(0, str(Path(__file__).parent))
from backend.utils.graph_generator import generate, STAGES, VECTORS, FEATURES

warnings.filterwarnings('ignore'); torch.manual_seed(0); np.random.seed(0); torch.set_num_threads(2)
ROOT = Path(__file__).parent; FIG = ROOT / 'figs'
C = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7']
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.color': '#e6e6e6', 'figure.dpi': 200, 'savefig.bbox': 'tight', 'lines.linewidth': 2})
def save(n): plt.savefig(FIG / f'{n}.png'); plt.close()
R = {}

df, meta = generate()
hosts = meta['hosts']; HID = {h: i for i, h in enumerate(hosts)}
SID = {s: i for i, s in enumerate(STAGES)}; VID = {v: i for i, v in enumerate(VECTORS)}
F = df[FEATURES].to_numpy(float)
mu, sd = F.mean(0), F.std(0) + 1e-9; FZ = (F - mu) / sd
TS = df.ts.to_numpy(float); VEC = df.vector.map(VID).to_numpy(); STG = df.stage.map(SID).to_numpy()
INC = df.incident.to_numpy(); SRC = df.src.to_numpy(); DST = df.dst.to_numpy()
R['events'] = len(df); R['incident_events'] = int((INC >= 0).sum()); R['incidents'] = int(INC.max() + 1)
R['vector_counts'] = df.vector.value_counts().to_dict(); R['stage_counts'] = df.stage.value_counts().to_dict()
R['nodes'] = int(len(set(SRC) | set(DST)))

# node -> sorted event index list (temporal adjacency)
node_ev = {}
for i, (s, d) in enumerate(zip(SRC, DST)):
    node_ev.setdefault(s, []).append(i)
    if d != s: node_ev.setdefault(d, []).append(i)
node_ts = {n: TS[idx] for n, idx in node_ev.items()}

def ev_vec(i, t_now):
    oh = np.zeros(4); oh[VEC[i]] = 1
    dt = np.log1p(max(t_now - TS[i], 0) / 60)
    return np.concatenate([FZ[i], oh, [dt]])   # 13 dims

# ---- prediction points ----
inc_events = {k: np.array(sorted(v)) for k, v in df[INC >= 0].groupby('incident').groups.items()}
starts = {k: TS[v[0]] for k, v in inc_events.items()}
order = sorted(starts, key=starts.get); cut = int(len(order) * .7)
train_inc, test_inc = set(order[:cut]), set(order[cut:])
split_time = starts[order[cut]]
R['train_incidents'] = len(train_inc); R['test_incidents'] = len(test_inc)
R['split_time'] = str(pd.to_datetime(split_time, unit='s', utc=True))
# train incidents must end before test incidents start (purge overlap)
train_inc = {k for k in train_inc if TS[inc_events[k][-1]] < split_time}
R['train_incidents_after_purge'] = len(train_inc)

W, K, L = 3, 24, 6
points = []
for k, idx in inc_events.items():
    if k not in train_inc and k not in test_inc: continue
    for j in range(W - 1, len(idx) - 1):
        i = idx[j]; nxt = idx[j + 1]
        victim = DST[i] if DST[i] in HID else None
        points.append(dict(inc=k, i=i, j=j, nxt=nxt, y=STG[nxt], t=TS[i], victim=DST[i], split='train' if k in train_inc else 'test'))
P = pd.DataFrame(points); R['points'] = P.split.value_counts().to_dict()

def window_feats(p):
    idx = inc_events[p['inc']][p['j'] - W + 1:p['j'] + 1]
    rows = [ev_vec(i, p['t']) for i in idx]
    return np.concatenate(rows + [FZ[idx[-1]] - FZ[idx[0]]])
def seq_feats(p):
    idx = inc_events[p['inc']][max(0, p['j'] - L + 1):p['j'] + 1]
    s = np.zeros((L, 13)); rows = [ev_vec(i, p['t']) for i in idx]; s[L - len(rows):] = rows
    return s
def static_node_feats():
    hist = df[df.ts < split_time]
    out = {}
    for n in hosts:
        e = hist[(hist.src == n) | (hist.dst == n)]
        out[n] = np.array([np.log1p(len(e)), e.src.nunique(), e.dst.nunique(), (e.vector == 'identity').mean() if len(e) else 0,
                           (e.vector == 'alert').mean() if len(e) else 0, int(n in meta['servers'])])
    return out
SN = static_node_feats()
def static_feats(p):
    v = SN.get(p['victim'], np.zeros(6))
    return np.concatenate([window_feats(p), v])
def temporal_nbhd(p):
    """last K events touching the current victim or attacker-side source up to time t (multi-vector)."""
    t = p['t']; nodes = {p['victim'], SRC[p['i']]}
    cand = []
    for n in nodes:
        if n not in node_ev: continue
        pos = bisect.bisect_right(node_ts[n], t)
        cand += node_ev[n][max(0, pos - K):pos]
    cand = sorted(set(cand), key=lambda i: TS[i])[-K:]
    X = np.zeros((K, 14)); M = np.zeros(K, bool); V = np.zeros(K, int)
    for r, i in enumerate(cand[::-1]):
        X[r, :13] = ev_vec(i, t); X[r, 13] = float(INC[i] == p['inc']); M[r] = True; V[r] = VEC[i]
    return X, M, V

t0 = time.time()
Xw = np.stack([window_feats(p) for p in points]); Xs = np.stack([seq_feats(p) for p in points]); Xst = np.stack([static_feats(p) for p in points])
NB = [temporal_nbhd(p) for p in points]; Xn = np.stack([a for a, _, _ in NB]); Mn = np.stack([b for _, b, _ in NB]); Vn = np.stack([c for _, _, c in NB])
y = P.y.to_numpy(); tr = (P.split == 'train').to_numpy(); te = ~tr
R['feature_build_s'] = time.time() - t0
classes = sorted(set(y)); cls_names = [STAGES[c] for c in classes]
ymap = {c: i for i, c in enumerate(classes)}; yi = np.array([ymap[c] for c in y])

res = {}
def rec(name, pred, prob=None):
    res[name] = dict(accuracy=accuracy_score(yi[te], pred), macro_f1=f1_score(yi[te], pred, average='macro'))
    return pred
maj = np.bincount(yi[tr]).argmax(); rec('Majority class', np.full(te.sum(), maj))
rf = RandomForestClassifier(n_estimators=200, max_depth=12, min_samples_leaf=2, class_weight='balanced_subsample', random_state=42, n_jobs=2).fit(Xw[tr], yi[tr])
rec('Random Forest (window)', rf.predict(Xw[te])); prob_rf = rf.predict_proba(Xw[te])
xgb = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=.08, subsample=.9, colsample_bytree=.8, random_state=42, n_jobs=2, verbosity=0).fit(Xw[tr], yi[tr])
rec('XGBoost (window)', xgb.predict(Xw[te]))
sgr = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=.08, subsample=.9, colsample_bytree=.8, random_state=42, n_jobs=2, verbosity=0).fit(Xst[tr], yi[tr])
rec('Static graph + XGBoost', sgr.predict(Xst[te]))

nC = len(classes)
class LSTMNet(nn.Module):
    def __init__(s): super().__init__(); s.l = nn.LSTM(13, 64, batch_first=True); s.o = nn.Sequential(nn.Dropout(.2), nn.Linear(64, nC))
    def forward(s, x): return s.o(s.l(x)[0][:, -1])
class TGAMV(nn.Module):
    """Adaptive multi-vector temporal graph model: time-encoded temporal neighbourhood, attention pooling
    per vector, learned adaptive vector weights, fused with a GRU over the incident's own recent events."""
    def __init__(s, d=64):
        super().__init__()
        s.freq = nn.Parameter(torch.linspace(0.1, 3, 8), requires_grad=True)
        s.enc = nn.Sequential(nn.Linear(14 + 16, d), nn.ReLU(), nn.Linear(d, d))
        s.att = nn.Linear(d, 1); s.gate = nn.Linear(d * 4 + 64, 4)
        s.gru = nn.GRU(13, 64, batch_first=True)
        s.out = nn.Sequential(nn.Linear(d + 64, 64), nn.ReLU(), nn.Dropout(.2), nn.Linear(64, nC))
    def forward(s, xn, mn, vn, xs, return_w=False):
        dt = xn[..., 12:13]; te_ = torch.cat([torch.cos(dt * s.freq), torch.sin(dt * s.freq)], -1)
        h = s.enc(torch.cat([xn, te_], -1)); a = s.att(h).squeeze(-1)
        summ = []
        for v in range(4):
            m = mn & (vn == v); av = a.masked_fill(~m, -1e9); w = torch.softmax(av, 1) * m.any(1, keepdim=True)
            summ.append((w.unsqueeze(-1) * h).sum(1))
        g = s.gru(xs)[1][0]
        vw = torch.softmax(s.gate(torch.cat(summ + [g], -1)), -1)          # adaptive vector weights
        fused = sum(vw[:, v:v + 1] * summ[v] for v in range(4))
        o = s.out(torch.cat([fused, g], -1))
        return (o, vw) if return_w else o

def train_torch(model, inputs, epochs=40, lr=2e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    yt = torch.tensor(yi[tr]); n = len(yt)
    wts = torch.tensor(len(yi[tr]) / (nC * np.bincount(yi[tr], minlength=nC)), dtype=torch.float32)
    lossf = nn.CrossEntropyLoss(weight=wts)
    tins = [torch.tensor(a[tr]) for a in inputs]
    for ep in range(epochs):
        model.train(); perm = torch.randperm(n)
        for b in range(0, n, 128):
            ix = perm[b:b + 128]; opt.zero_grad(); loss = lossf(model(*[t[ix] for t in tins]), yt[ix]); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad(): return model(*[torch.tensor(a[te]) for a in inputs])

f32 = lambda a: a.astype(np.float32)
lstm_out = train_torch(LSTMNet(), [f32(Xs)]); rec('LSTM (sequence)', lstm_out.argmax(1).numpy())
tg = TGAMV(); tg_out = train_torch(tg, [f32(Xn), Mn, Vn.astype(np.int64), f32(Xs)], epochs=40)
pred_tg = rec('Temporal graph (TG-AMV, ours)', tg_out.argmax(1).numpy()); prob_tg = torch.softmax(tg_out, 1).numpy()
with torch.no_grad():
    _, vw = tg(torch.tensor(f32(Xn[te])), torch.tensor(Mn[te]), torch.tensor(Vn[te].astype(np.int64)), torch.tensor(f32(Xs[te])), return_w=True)
vw = vw.numpy()
# ablation: no adaptive weights (uniform), no temporal neighbourhood (zero mask)
with torch.no_grad():
    Mz = torch.zeros_like(torch.tensor(Mn[te]))
    o_nonb = tg(torch.tensor(f32(Xn[te])), Mz, torch.tensor(Vn[te].astype(np.int64)), torch.tensor(f32(Xs[te])))
R['ablation_no_graph_context_at_test'] = dict(accuracy=accuracy_score(yi[te], o_nonb.argmax(1).numpy()), macro_f1=f1_score(yi[te], o_nonb.argmax(1).numpy(), average='macro'))
R['stage_results'] = res
cur_stage = STG[P.i.to_numpy()][te]
R['vector_weights_by_current_stage'] = {STAGES[s]: dict(zip(VECTORS, vw[cur_stage == s].mean(0).round(3).tolist())) for s in sorted(set(cur_stage))}
pc = f1_score(yi[te], pred_tg, average=None); R['tg_per_class_f1'] = dict(zip(cls_names, pc.round(3).tolist()))
R['rf_per_class_f1'] = dict(zip(cls_names, f1_score(yi[te], rf.predict(Xw[te]), average=None).round(3).tolist()))

# ---- early warning lead time: first test point where P(exfiltration or impact) > .5 before it happens ----
crit = [ymap[SID[s]] for s in ('exfiltration', 'impact') if SID[s] in ymap]
Pte = P[te].reset_index(drop=True)
def lead(prob):
    leads, warned, total = [], 0, 0
    for k, g in Pte.groupby('inc'):
        idx = inc_events[k]; first = next((TS[i] for i in idx if STG[i] in (SID['exfiltration'], SID['impact'])), None)
        if first is None: continue
        total += 1
        g = g[g.t < first]
        hit = g[prob[g.index][:, crit].sum(1) > .5]
        if len(hit): warned += 1; leads.append((first - hit.t.iloc[0]) / 60)
    return dict(incidents=total, warned=warned, rate=warned / max(total, 1), median_lead_min=float(np.median(leads)) if leads else 0)
R['early_warning'] = {'Random Forest (window)': lead(prob_rf), 'LSTM (sequence)': lead(torch.softmax(lstm_out, 1).numpy()), 'Temporal graph (TG-AMV, ours)': lead(prob_tg)}

# ---- next victim prediction (temporal link prediction) ----
def moves():
    out = []
    for k, idx in inc_events.items():
        if k not in train_inc and k not in test_inc: continue
        seen = []
        for a, b in zip(idx, idx[1:]):
            if DST[a] in HID and DST[a] not in seen: seen.append(DST[a])
            if DST[b] in HID and DST[b] not in seen and seen:
                out.append(dict(inc=k, t=TS[a], cur=seen[-1], target=DST[b], compromised=list(seen), lateral=STG[b] == SID['lateral_movement'], split='train' if k in train_inc else 'test'))
    return out
MV = moves(); R['victim_moves'] = pd.Series([m['split'] for m in MV]).value_counts().to_dict()
hist_static = df[df.ts < split_time]
def adjacency(sub, decay_t=None):
    A = {}
    for s, d, v, t in zip(sub.src, sub.dst, sub.vector, sub.ts):
        w = 1.0 if decay_t is None else np.exp(-(decay_t - t) / (5 * 86400))
        A[(s, d, v)] = A.get((s, d, v), 0) + w
    return A
st_net = hist_static[hist_static.vector == 'network'].groupby(['src', 'dst']).size().to_dict()
st_user = hist_static[hist_static.vector == 'identity'].groupby('src')['dst'].apply(set).to_dict()
pop = hist_static[hist_static.incident >= 0].dst.value_counts().to_dict()
def cand_feats(m, temporal):
    t, cur = m['t'], m['cur']; cands = [h for h in hosts if h not in m['compromised']]
    if temporal:
        lo = t - 7 * 86400; sub = df[(df.ts < t) & (df.ts >= lo)]
        ide = sub[sub.vector == 'identity']; net = sub[sub.vector == 'network']
        dec = lambda s: np.exp(-(t - s.ts.to_numpy()) / (3 * 86400))
        uw = {}; [uw.__setitem__((u, h), uw.get((u, h), 0) + w) for u, h, w in zip(ide.src, ide.dst, dec(ide))]
        nw = {}; [nw.__setitem__((a, b), nw.get((a, b), 0) + w) for a, b, w in zip(net.src, net.dst, dec(net))]
        users_cur = {u: w for (u, h), w in uw.items() if h == cur}
        alerts = sub[sub.vector == 'alert'].dst.value_counts().to_dict()
        rows = [[sum(min(w, uw.get((u, c), 0)) for u, w in users_cur.items()), nw.get((cur, c), 0) + nw.get((c, cur), 0),
                 alerts.get(c, 0), int(c in meta['servers']), np.log1p(len(sub[(sub.dst == c)]))] for c in cands]
    else:
        uc = {u for u, hs in st_user.items() if cur in hs}
        rows = [[sum(1 for u in uc if c in st_user.get(u, set())), st_net.get((cur, c), 0) + st_net.get((c, cur), 0), 0,
                 int(c in meta['servers']), np.log1p(pop.get(c, 0))] for c in cands]
    return cands, np.array(rows, float)
def rank_eval(temporal):
    Xtr, ytr = [], []
    for m in MV:
        if m['split'] != 'train': continue
        c, X = cand_feats(m, temporal); Xtr.append(X); ytr += [int(h == m['target']) for h in c]
    sc = StandardScaler().fit(np.vstack(Xtr)); clf = LogisticRegression(max_iter=2000, class_weight='balanced').fit(sc.transform(np.vstack(Xtr)), ytr)
    ranks, lat = [], []
    for m in MV:
        if m['split'] != 'test': continue
        c, X = cand_feats(m, temporal); s = clf.decision_function(sc.transform(X))
        ranks.append(1 + int((s > s[c.index(m['target'])]).sum()) if m['target'] in c else len(c)); lat.append(m['lateral'])
    r = np.array(ranks); lat = np.array(lat)
    met = lambda r: dict(hits1=float((r == 1).mean()), hits3=float((r <= 3).mean()), mrr=float((1 / r).mean()), n=len(r))
    return dict(**met(r), lateral=met(r[lat])), clf.coef_[0].round(3).tolist()
tests = [m for m in MV if m['split'] == 'test']
LAT = np.array([m['lateral'] for m in tests])
met = lambda r: dict(hits1=float((r == 1).mean()), hits3=float((r <= 3).mean()), mrr=float((1 / r).mean()), n=len(r))
rr = np.array([1 + np.random.default_rng(i).integers(len([h for h in hosts if h not in m['compromised']])) for i, m in enumerate(tests)])
R['victim_results'] = {'Random guess': dict(**met(rr), lateral=met(rr[LAT]))}
pr = []
for m in tests:
    c = [h for h in hosts if h not in m['compromised']]; s = np.array([pop.get(h, 0) for h in c]); pr.append(1 + int((s > s[c.index(m['target'])]).sum()))
pr = np.array(pr); R['victim_results']['Most attacked host (popularity)'] = dict(**met(pr), lateral=met(pr[LAT]))
R['victim_results']['Static graph'], _ = rank_eval(False)
R['victim_results']['Temporal graph (ours)'], R['victim_temporal_coef'] = rank_eval(True)

# ---- figures ----
names = list(res); f1s = [res[n]['macro_f1'] for n in names]; accs = [res[n]['accuracy'] for n in names]
fig, ax = plt.subplots(figsize=(8.5, 3.8)); yy = np.arange(len(names))
ax.barh(yy + .18, f1s[::-1] if False else f1s, .36, color=C[0], label='Macro F1'); ax.barh(yy - .18, accs, .36, color=C[1], label='Accuracy')
ax.set_yticks(yy, names); ax.invert_yaxis(); ax.set_xlim(0, 1.08); ax.legend(frameon=False, loc='lower right')
for i, (a, b) in enumerate(zip(f1s, accs)): ax.text(a + .01, i + .18, f'{a:.3f}', va='center', fontsize=8); ax.text(b + .01, i - .18, f'{b:.3f}', va='center', fontsize=8)
ax.set_title('Next attack stage prediction on the test incidents'); save('tg_stage_compare'); plt.rcParams['axes.grid'] = False
cm = confusion_matrix(yi[te], pred_tg); cmn = cm / cm.sum(1, keepdims=True)
short = {'benign': 'Benign', 'reconnaissance': 'Recon', 'credential_access': 'Cred.', 'execution': 'Exec.', 'lateral_movement': 'Lateral', 'exfiltration': 'Exfil.', 'impact': 'Impact'}
plt.figure(figsize=(6.4, 5)); sns.heatmap(cmn, annot=True, fmt='.2f', cmap='Blues', xticklabels=[short[c] for c in cls_names], yticklabels=[short[c] for c in cls_names],
    vmin=0, vmax=1, linewidths=1, linecolor='white', annot_kws={'size': 9}, cbar_kws={'shrink': .75})
plt.xlabel('Predicted next stage'); plt.ylabel('Actual next stage'); plt.title('Temporal graph model, confusion matrix (test)', fontsize=10); save('tg_confusion')
VW = pd.DataFrame(R['vector_weights_by_current_stage']).T.reindex([s for s in STAGES if s in R['vector_weights_by_current_stage']])
plt.figure(figsize=(6.4, 3.8)); sns.heatmap(VW, annot=True, fmt='.2f', cmap='Blues', vmin=0, vmax=VW.values.max(), linewidths=1, linecolor='white',
    yticklabels=[short[s] for s in VW.index], cbar_kws={'shrink': .75}); plt.xlabel('Evidence vector'); plt.ylabel('Current stage')
plt.title('Adaptive vector weights learned by the model', fontsize=10); save('tg_vector_weights')
VR = {k: v['lateral'] for k, v in R['victim_results'].items()}; nm_ = list(VR)
plt.rcParams['axes.grid'] = True
fig, ax = plt.subplots(figsize=(7.5, 3.4)); x = np.arange(len(nm_))
for i, k in enumerate(['hits1', 'hits3', 'mrr']):
    ax.bar(x + (i - 1) * .26, [VR[n][k] for n in nm_], .26, color=C[i], label={'hits1': 'Hits@1', 'hits3': 'Hits@3', 'mrr': 'MRR'}[k], edgecolor='white')
ax.set_xticks(x, ['Random guess', 'Most attacked\nhost', 'Static graph', 'Temporal graph\n(ours)'], fontsize=8.5); ax.set_ylim(0, .5); ax.legend(frameon=False, ncol=3); ax.set_title('Next victim prediction during lateral movement (test)'); save('tg_victim'); plt.rcParams['axes.grid'] = False
# temporal graph snapshot illustration
import networkx as nx
k0 = max(sorted(test_inc), key=lambda k: len({DST[i] for i in inc_events[k] if DST[i] in HID}))
idx = inc_events[k0]; ih = {DST[i] for i in idx if DST[i] in HID}; t_end = TS[idx[-1]]; t_beg = TS[idx[0]] - 3600 * 24
bg = df[(df.ts >= t_beg) & (df.ts <= t_end) & (df.incident < 0) & (df.dst.isin(ih) | df.src.isin(ih))]
bg = bg[bg.vector.isin(['identity', 'network'])].groupby(['src', 'dst']).head(1).head(30)
sub = pd.concat([df.loc[idx], bg])
G = nx.MultiDiGraph()
for s_, d_, v_, inc_ in zip(sub.src, sub.dst, sub.vector, sub.incident): G.add_edge(s_, d_, v=v_, a=inc_ == k0)
pos = nx.spring_layout(nx.Graph(G), seed=3, k=.9)
plt.figure(figsize=(8, 5.6)); ncol = ['#e34948' if n.startswith('X') else ('#eda100' if n.startswith('U') else ('#4a3aa7' if n in meta['servers'] else '#2a78d6')) for n in G.nodes]
vc = {'network': '#2a78d6', 'identity': '#eda100', 'endpoint': '#1baf7a', 'alert': '#e34948'}
for (s_, d_, dd) in G.edges(data=True):
    if s_ == d_: continue
    plt.annotate('', xy=pos[d_], xytext=pos[s_], arrowprops=dict(arrowstyle='-|>', color=vc[dd['v']], lw=2.2 if dd['a'] else .6, alpha=.9 if dd['a'] else .35, shrinkA=8, shrinkB=8))
nx.draw_networkx_nodes(nx.Graph(G), pos, node_color=ncol, node_size=260, edgecolors='white')
nx.draw_networkx_labels(nx.Graph(G), pos, font_size=6.5, font_color='white')
from matplotlib.lines import Line2D
plt.legend(handles=[Line2D([0], [0], color=c, lw=2, label=v) for v, c in vc.items()] + [Line2D([0], [0], marker='o', color='w', markerfacecolor=c, ms=8, label=l) for l, c in
    (('host', '#2a78d6'), ('server', '#4a3aa7'), ('user', '#eda100'), ('attacker IP', '#e34948'))], frameon=False, fontsize=7.5, loc='upper left', bbox_to_anchor=(1, 1))
plt.axis('off'); plt.title(f'Temporal security graph around test incident {k0} (bold edges = attack events)', fontsize=10); save('tg_graph_example')

json.dump(R, open(ROOT / 'temporal_graph_results.json', 'w'), indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o))
print(json.dumps(R, indent=1, default=str))
