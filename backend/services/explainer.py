FEATURE_LABELS = {
    'failed_logins': 'Failed login burst',
    'unique_ports': 'Unique ports / scan behavior',
    'packet_rate': 'Packet-rate anomaly',
    'syn_ratio': 'SYN ratio',
    'connection_duration': 'Short connection duration',
    'avg_packet_size': 'Packet-size deviation'
}


def explain(s):
    raw = {
        'failed_logins': min(float(s['failed_logins']) / 12, 1) * 100,
        'unique_ports': min(float(s['unique_ports']) / 40, 1) * 100,
        'packet_rate': min(float(s['packet_rate']) / 600, 1) * 100,
        'syn_ratio': min(float(s['syn_ratio']) / .8, 1) * 100,
        'connection_duration': max(0, min((30 - float(s['connection_duration'])) / 30, 1)) * 100,
        'avg_packet_size': min(abs(float(s['avg_packet_size']) - 700) / 600, 1) * 100,
    }
    ranked = sorted(raw.items(), key=lambda x: x[1], reverse=True)
    return [
        {'feature': FEATURE_LABELS[k], 'importance': round(v, 1)}
        for k, v in ranked
    ]
