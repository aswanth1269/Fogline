"""Three-observation next-event baseline with episode-isolated temporal evaluation."""
import hashlib
import os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from backend.config import FEATURES, MODEL_DIR
from backend.utils.sequence_generator import sequence_fixture, STAGES

WINDOW = 3
MAX_GAP_SECONDS = 1800
MIN_SCORE = .45  # Display abstention rule, not a calibrated confidence guarantee.
RECOMMENDATIONS = {
    'benign': 'Continue monitoring and check whether the suspicious activity has stopped.',
    'reconnaissance': 'Review connection logs for repeated probes and unusual destination-port breadth.',
    'credential_access': 'Correlate with authentication logs, failed sign-ins and account protection alerts.',
    'lateral_movement': 'Review internal host-to-host connections and endpoint authentication evidence.',
    'exfiltration': 'Check outbound byte volumes, destinations and data-loss-prevention alerts.',
    'denial_of_service': 'Review service availability, request volume and connection saturation.',
}

def vector(samples):
    values = np.array([[sample[f] for f in FEATURES] for sample in samples], dtype=float)
    # Historical features and observed changes only; no stage labels or future rows.
    return np.concatenate([values.ravel(), values[-1] - values[0]])

def validate_frame(frame):
    required = FEATURES + ['episode_id', 'timestamp', 'attack_stage']
    missing = [key for key in required if key not in frame]
    if missing:
        raise ValueError('Sequence CSV needs: ' + ', '.join(missing) + '. The binary IDS CSV cannot train next moves.')
    if not 48 <= len(frame) <= 50000:
        raise ValueError('Sequence CSV must contain 48–50,000 rows.')
    df = frame.copy()
    if df[required].isna().any().any():
        raise ValueError('Sequence fields cannot be missing.')
    df['episode_id'] = df['episode_id'].astype(str).str.strip()
    if (df['episode_id'].str.len() == 0).any() or (df['episode_id'].str.len() > 128).any():
        raise ValueError('Each episode_id must contain 1–128 characters.')
    if not set(df.attack_stage).issubset(STAGES):
        raise ValueError('attack_stage must be one of: ' + ', '.join(STAGES))
    try:
        df[FEATURES] = df[FEATURES].apply(pd.to_numeric, errors='raise')
        # ISO 8601 UTC/offset timestamps required; never silently interpret local time.
        if not df.timestamp.astype(str).str.contains(r'(?:Z|[+-]\d{2}:\d{2})$', regex=True).all():
            raise ValueError()
        df['timestamp'] = pd.to_datetime(df.timestamp, utc=True, format='ISO8601', errors='raise')
    except (TypeError, ValueError):
        raise ValueError('Use numeric features and ISO 8601 timestamps with Z or a UTC offset.')
    values = df[FEATURES].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any() or (values > 1e9).any() or (df.syn_ratio > 1).any():
        raise ValueError('Features must be finite, between 0 and 1e9; SYN ratio must be 0–1.')
    if df.duplicated(['episode_id', 'timestamp']).any():
        raise ValueError('Each episode needs unique observation timestamps; tied times are ambiguous.')
    return df.sort_values(['timestamp', 'episode_id'], kind='stable')

def make_windows(frame):
    x, y, ids, times = [], [], [], []
    for episode, group in frame.groupby('episode_id', sort=False):
        rows = group.sort_values('timestamp').to_dict('records')
        for index in range(WINDOW-1, len(rows)-1):
            block = rows[index-WINDOW+1:index+2]
            gaps = [(b['timestamp']-a['timestamp']).total_seconds() for a,b in zip(block,block[1:])]
            if any(gap <= 0 or gap > MAX_GAP_SECONDS for gap in gaps):
                continue
            x.append(vector(block[:-1]))
            y.append(block[-1]['attack_stage'])
            ids.append(str(episode))
            times.append(block[-1]['timestamp'])
    return np.asarray(x), np.asarray(y), ids, times

class NextMovePredictor:
    def __init__(self, model_dir=None):
        self.path = Path(model_dir or MODEL_DIR) / 'next_move.joblib'
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.bundle = None
        if self.path.exists():
            self.bundle = joblib.load(self.path)
        else:
            self.train(sequence_fixture(), source='synthetic')

    def status(self):
        return {key:value for key,value in self.bundle.items() if key not in {'model','feature_min','feature_max'}}

    def train(self, frame, source='uploaded'):
        df = validate_frame(frame)
        spans = df.groupby('episode_id')['timestamp'].agg(['min','max']).sort_values('min')
        if len(spans) < 8:
            raise ValueError('Provide at least eight separate episodes with chronological observations.')
        cut = max(1, int(len(spans)*.75))
        cutoff = spans.iloc[cut]['min']
        train_ids = set(spans[spans['max'] < cutoff].index)
        test_ids = set(spans[spans['min'] >= cutoff].index)
        # Episodes spanning the boundary are purged, never shared across splits.
        train = df[df.episode_id.isin(train_ids)]
        test = df[df.episode_id.isin(test_ids)]
        x_train,y_train,_,train_times = make_windows(train)
        x_test,y_test,_,test_times = make_windows(test)
        if len(x_train) < 24 or len(x_test) < 8 or len(train_ids) < 4 or len(test_ids) < 2:
            raise ValueError('Need 24 training and 8 holdout windows, with 4 earlier and 2 later nonoverlapping episodes. Each window needs 4 observations within 30-minute gaps.')
        if len(set(y_train)) < 2:
            raise ValueError('Earlier episodes must contain at least two different next-event behaviors.')
        model = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_leaf=3,
                                       class_weight='balanced_subsample', random_state=42, n_jobs=1)
        model.fit(x_train, y_train)
        predicted = model.predict(x_test)
        majority = pd.Series(y_train).value_counts().idxmax()
        labels = sorted(set(y_train) | set(y_test))
        provenance = 'synthetic' if source == 'synthetic' or ('data_source' in df and df.data_source.astype(str).str.lower().eq('synthetic').any()) else 'uploaded_unverified'
        bundle = dict(model=model, version=(self.bundle or {}).get('version',0)+1,
                      source=provenance, window=WINDOW, max_gap_seconds=MAX_GAP_SECONDS,
                      method='Random forest over three ordered observations and their changes',
                      target='Behavior at the next recorded observation in the same episode; continuation is allowed. No time-to-attack estimate.',
                      scores_calibrated=False,
                      feature_min=train[FEATURES].min().to_dict(), feature_max=train[FEATURES].max().to_dict(),
                      dataset_sha256=hashlib.sha256(df.to_csv(index=False).encode()).hexdigest(),
                      metrics=dict(accuracy=float(accuracy_score(y_test,predicted)),
                                   macro_f1=float(f1_score(y_test,predicted,labels=labels,average='macro',zero_division=0)),
                                   majority_accuracy=float(accuracy_score(y_test,[majority]*len(y_test))),
                                   train_windows=len(y_train),test_windows=len(y_test),
                                   train_episodes=len(train_ids),test_episodes=len(test_ids),
                                   purged_episodes=len(spans)-len(train_ids)-len(test_ids),
                                   train_end=max(train_times).isoformat(),test_start=min(test_times).isoformat(),
                                   unseen_holdout_classes=sorted(set(y_test)-set(y_train)),
                                   per_class=classification_report(y_test,predicted,labels=labels,output_dict=True,zero_division=0)),
                      classes=list(model.classes_))
        temporary = self.path.with_suffix('.pending')
        joblib.dump(bundle, temporary)
        os.replace(temporary, self.path)
        self.bundle = bundle
        return self.status()

    def predict(self, events):
        base = dict(model_version=self.bundle['version'], training_source=self.bundle['source'],
                    target=self.bundle['target'], scores_calibrated=False, history_count=min(len(events),WINDOW),
                    required_history=WINDOW, candidates=[])
        if len(events) < WINDOW:
            return dict(base,status='insufficient_history',message=f'Add {WINDOW-len(events)} more related observation(s) to this investigation.')
        history = events[-WINDOW:]
        samples = [event['sample'] for event in history]
        for sample in samples:
            if any(sample[f] < self.bundle['feature_min'][f] or sample[f] > self.bundle['feature_max'][f] for f in FEATURES):
                return dict(base,status='out_of_range',message='Observed features fall outside the training range. Collect representative labeled sequences before relying on a forecast.')
        scores = self.bundle['model'].predict_proba([vector(samples)])[0]
        ranked = sorted(zip(self.bundle['model'].classes_,scores),key=lambda pair:pair[1],reverse=True)
        candidates = [dict(stage=stage,label=STAGES[stage],score=round(float(score),4),recommendation=RECOMMENDATIONS[stage]) for stage,score in ranked[:3]]
        uncertain = float(ranked[0][1]) < MIN_SCORE
        return dict(base,status='uncertain' if uncertain else 'forecast', candidates=candidates,
                    message='No clear next move; gather additional observations.' if uncertain else 'Ranked next-observation behaviors, not a confirmed attacker plan.',
                    history=[dict(event_id=e.get('id'),timestamp=e['timestamp']) for e in history],
                    limitation='Six aggregate traffic features cannot establish host compromise, lateral movement or exfiltration. Correlate with endpoint, identity and flow evidence.')
