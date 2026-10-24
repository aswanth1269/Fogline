import numpy as np
import pandas as pd

FEATURES = [
    'packet_rate',
    'failed_logins',
    'unique_ports',
    'avg_packet_size',
    'connection_duration',
    'syn_ratio',
]


def generate_dataset(n=1200, seed=42):
    r = np.random.default_rng(seed)
    m = n // 2

    benign = pd.DataFrame({
        'packet_rate': np.clip(r.normal(120, 35, m), 10, 250),
        'failed_logins': r.poisson(0.8, m),
        'unique_ports': np.clip(r.normal(4, 2, m), 1, 12),
        'avg_packet_size': np.clip(r.normal(720, 160, m), 100, 1500),
        'connection_duration': np.clip(r.normal(28, 14, m), 1, 90),
        'syn_ratio': np.clip(r.normal(0.22, 0.08, m), 0.02, 0.6),
    })
    benign['label'] = 0

    attack = pd.DataFrame({
        'packet_rate': np.clip(r.normal(420, 120, m), 150, 900),
        'failed_logins': r.poisson(8, m),
        'unique_ports': np.clip(r.normal(28, 11, m), 5, 80),
        'avg_packet_size': np.clip(r.normal(460, 220, m), 60, 1450),
        'connection_duration': np.clip(r.normal(9, 8, m), 0.2, 45),
        'syn_ratio': np.clip(r.normal(0.68, 0.16, m), 0.15, 0.98),
    })
    attack['label'] = 1

    return (
        pd.concat([benign, attack], ignore_index=True)
        .sample(frac=1, random_state=seed)
        .reset_index(drop=True)
    )


def generate_live_sample(kind='attack', seed=None):
    r = np.random.default_rng(seed)

    if kind == 'normal':
        return {
            'packet_rate': float(np.clip(r.normal(125, 25), 20, 220)),
            'failed_logins': int(r.poisson(1)),
            'unique_ports': int(np.clip(r.normal(4, 1.8), 1, 10)),
            'avg_packet_size': float(np.clip(r.normal(760, 130), 180, 1450)),
            'connection_duration': float(np.clip(r.normal(32, 11), 2, 80)),
            'syn_ratio': float(np.clip(r.normal(0.2, 0.06), 0.03, 0.45)),
        }

    if kind == 'uncertain':
        return {
            'packet_rate': float(np.clip(r.normal(245, 35), 150, 360)),
            'failed_logins': int(r.poisson(4)),
            'unique_ports': int(np.clip(r.normal(13, 3), 5, 25)),
            'avg_packet_size': float(np.clip(r.normal(590, 120), 180, 1200)),
            'connection_duration': float(np.clip(r.normal(18, 7), 2, 50)),
            'syn_ratio': float(np.clip(r.normal(0.45, 0.08), 0.2, 0.75)),
        }

    return {
        'packet_rate': float(np.clip(r.normal(520, 75), 280, 850)),
        'failed_logins': int(r.poisson(10)),
        'unique_ports': int(np.clip(r.normal(35, 8), 12, 75)),
        'avg_packet_size': float(np.clip(r.normal(420, 130), 80, 1100)),
        'connection_duration': float(np.clip(r.normal(7, 4), 0.5, 30)),
        'syn_ratio': float(np.clip(r.normal(0.77, 0.08), 0.4, 0.98)),
    }
