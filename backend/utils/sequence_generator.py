"""Synthetic sequence fixtures for exercising the forecasting pipeline, not threat evidence."""
from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd
from backend.config import FEATURES

STAGES = {
    'benign': 'Normal activity',
    'reconnaissance': 'Reconnaissance / scanning',
    'credential_access': 'Credential-access attempt',
    'lateral_movement': 'Lateral-movement attempt',
    'exfiltration': 'Possible data exfiltration',
    'denial_of_service': 'Denial-of-service activity',
}
# Illustrative feature distributions, deliberately overlapping. They are not
# measurements or definitive signatures of these intrusion behaviors.
CENTERS = {
    'benign': [125, 1, 4, 760, 32, .20],
    'reconnaissance': [230, 1, 35, 450, 8, .45],
    'credential_access': [210, 12, 7, 560, 14, .35],
    'lateral_movement': [310, 5, 22, 680, 26, .40],
    'exfiltration': [430, 1, 6, 1250, 75, .17],
    'denial_of_service': [780, 2, 12, 230, 3, .90],
}
DEMO_PATH = ['reconnaissance'] * 3 + ['credential_access'] * 2 + ['lateral_movement'] * 3 + ['exfiltration'] * 2

def stage_sample(stage, rng):
    scales = [65, 2.5, 5, 160, 9, .10]
    values = np.maximum(0, rng.normal(CENTERS[stage], scales))
    values[5] = np.clip(values[5], 0, 1)
    values[1:3] = np.round(values[1:3])
    return {key: round(float(value), 4) for key, value in zip(FEATURES, values)}

def sequence_fixture(episodes=120, seed=42017):
    rng = np.random.default_rng(seed)
    paths = [
        ['benign'] * 12,
        ['reconnaissance'] * 3 + ['credential_access'] * 2 + ['lateral_movement'] * 3 + ['exfiltration'] * 3,
        ['reconnaissance'] * 3 + ['denial_of_service'] * 6,
        ['credential_access'] * 4 + ['benign'] * 5,
        ['reconnaissance'] * 4 + ['benign'] * 5,
        ['credential_access'] * 3 + ['lateral_movement'] * 3 + ['exfiltration'] * 4,
    ]
    rows = []
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    for episode in range(episodes):
        path = paths[episode % len(paths)]
        for step, stage in enumerate(path):
            rows.append(dict(episode_id=f'synthetic-{episode:04}',
                             timestamp=(start + timedelta(hours=episode, seconds=step*30)).isoformat(),
                             attack_stage=stage, data_source='synthetic', **stage_sample(stage, rng)))
    return pd.DataFrame(rows)
