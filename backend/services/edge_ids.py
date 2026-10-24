import joblib
import pandas as pd
import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from backend.config import MODEL_DIR, FEATURES
from backend.utils.sample_generator import generate_dataset


class EdgeIDSManager:
    def __init__(self):
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        self.models = {}
        self.metrics = {}
        bundle = MODEL_DIR / 'bundle.joblib'
        if bundle.exists():
            saved = joblib.load(bundle)
            self.models, self.metrics = saved['models'], saved['metrics']
            self.version = saved.get('version', 1)
            self.source = saved.get('source', 'Synthetic demonstration data')
        else:
            self.version = 0
            self.source = 'Synthetic demonstration data'
            self.train(generate_dataset(), persist=True)

    def _build_models(self):
        return {
            'tiny_ids': Pipeline([
                ('scale', StandardScaler()),
                ('model', LogisticRegression(max_iter=500, random_state=42))
            ]),
            'random_forest': RandomForestClassifier(
                n_estimators=140, max_depth=9, random_state=42
            ),
            'decision_tree': DecisionTreeClassifier(
                max_depth=6, random_state=42
            )
        }

    def train(self, dataframe, persist=True):
        required = FEATURES + ['label']
        missing = [c for c in required if c not in dataframe.columns]
        if missing:
            raise ValueError('Dataset is missing columns: ' + ', '.join(missing))

        df = dataframe[required].copy()
        if len(df) > 50000:
            raise ValueError('Use at most 50,000 rows per training upload.')
        try:
            df = df.apply(pd.to_numeric, errors='raise')
        except (ValueError, TypeError):
            raise ValueError('All dataset values must be numeric.')
        if not np.isfinite(df.to_numpy(dtype=float)).all():
            raise ValueError('Dataset values must be complete and finite.')
        if not set(df['label'].unique()).issubset({0, 1}):
            raise ValueError('Labels must be exactly 0 (normal) or 1 (attack).')
        if (df[FEATURES] < 0).any().any() or (df['syn_ratio'] > 1).any():
            raise ValueError('Features must be nonnegative; SYN ratio must be between 0 and 1.')
        if df['label'].value_counts().min() < 4:
            raise ValueError('Provide at least four rows of each class.')
        if len(df) < 40:
            raise ValueError('Dataset needs at least 40 complete rows.')
        if df['label'].nunique() < 2:
            raise ValueError('The label column must contain both normal (0) and attack (1) rows.')

        X = df[FEATURES].astype(float)
        y = df['label'].astype(int)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )

        models = self._build_models()
        metrics = {}
        pretty = {
            'tiny_ids': 'Tiny IDS (Logistic Regression)',
            'random_forest': 'Random Forest IDS',
            'decision_tree': 'Decision Tree IDS'
        }

        for key, model in models.items():
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            metrics[key] = {
                'name': pretty[key],
                'accuracy': round(float(accuracy_score(y_test, pred)), 4),
                'precision': round(float(precision_score(y_test, pred, zero_division=0)), 4),
                'recall': round(float(recall_score(y_test, pred, zero_division=0)), 4),
                'f1': round(float(f1_score(y_test, pred, zero_division=0)), 4),
                'test_rows': int(len(y_test))
            }


        version = self.version + 1
        if persist:
            temporary = MODEL_DIR / 'bundle.pending'
            joblib.dump({'models': models, 'metrics': metrics, 'version': version, 'source': self.source}, temporary)
            os.replace(temporary, MODEL_DIR / 'bundle.joblib')
        self.models, self.metrics, self.version = models, metrics, version
        return self.metrics

    def predict_all(self, sample):
        x = pd.DataFrame([{f: float(sample[f]) for f in FEATURES}])
        pretty = {
            'tiny_ids': 'Tiny IDS (Logistic Regression)',
            'random_forest': 'Random Forest IDS',
            'decision_tree': 'Decision Tree IDS'
        }
        out = []
        for key, model in self.models.items():
            attack_probability = float(model.predict_proba(x)[0][1])
            confidence = max(attack_probability, 1.0 - attack_probability)
            out.append({
                'key': key,
                'model': pretty[key],
                'prediction': 'Attack' if attack_probability >= 0.5 else 'Normal',
                'attack_probability': round(attack_probability, 4),
                'confidence': round(confidence, 4),
                'validation_f1': self.metrics.get(key, {}).get('f1', 0)
            })
        return out
