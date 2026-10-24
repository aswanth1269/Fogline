"""Multi-vector temporal security graph generator.

Produces a timestamped event stream over a small enterprise network. Nodes are users, hosts,
servers, processes and external IPs. Edges are typed events from four vectors (network, identity,
endpoint/process, alert). Attack incidents follow ATT&CK-style stages and move laterally along the
real login/communication structure of the network, so the next victim depends on graph history.
This is synthetic data for the temporal graph experiments; it is not captured traffic.
"""
import numpy as np
import pandas as pd

STAGES = ['benign', 'reconnaissance', 'credential_access', 'execution', 'lateral_movement', 'exfiltration', 'impact']
VECTORS = ['network', 'identity', 'endpoint', 'alert']
STAGE_VECTORS = {   # probability of each vector for events of a stage
    'reconnaissance': [.80, .05, .05, .10],
    'credential_access': [.25, .60, .05, .10],
    'execution': [.10, .10, .65, .15],
    'lateral_movement': [.35, .40, .15, .10],
    'exfiltration': [.70, .05, .15, .10],
    'impact': [.45, .05, .35, .15],
}
# feature centres: packet_rate, failed_logins, unique_ports, avg_packet_size, duration, syn_ratio, bytes_out(kB), priv_proc
CENTRES = {
    'benign': [120, .3, 3, 760, 30, .2, 40, .02],
    'reconnaissance': [260, .5, 30, 420, 6, .55, 8, .02],
    'credential_access': [200, 8, 5, 560, 12, .35, 15, .05],
    'execution': [130, 1, 4, 700, 25, .2, 30, .7],
    'lateral_movement': [240, 3, 12, 650, 20, .35, 60, .45],
    'exfiltration': [380, .5, 5, 1200, 70, .15, 900, .2],
    'impact': [650, 1, 10, 300, 5, .8, 20, .6],
}
SCALE = np.array([70, 2.2, 5, 170, 10, .11, 120, .2])
PATHS = [
    ['reconnaissance', 'credential_access', 'execution', 'lateral_movement', 'exfiltration'],
    ['reconnaissance', 'credential_access', 'lateral_movement', 'lateral_movement', 'impact'],
    ['credential_access', 'execution', 'lateral_movement', 'exfiltration'],
    ['reconnaissance', 'impact'],
    ['reconnaissance', 'credential_access', 'execution', 'impact'],
    ['credential_access', 'lateral_movement', 'execution', 'exfiltration'],
]
FEATURES = ['packet_rate', 'failed_logins', 'unique_ports', 'avg_packet_size', 'connection_duration', 'syn_ratio', 'bytes_out', 'privileged']


def generate(n_hosts=40, n_users=30, n_incidents=480, days=60, seed=7):
    rng = np.random.default_rng(seed)
    hosts = [f'H{i:02d}' for i in range(n_hosts)]
    servers = hosts[:4]                                   # DC, DB, FILE, WEB
    users = [f'U{i:02d}' for i in range(n_users)]
    # each user regularly logs into 2-4 hosts -> identity graph; hosts talk to a few peers -> network graph
    t0 = pd.Timestamp('2025-01-01', tz='UTC').value // 10**9
    horizon = days * 86400
    epoch_len = 10 * 86400  # who-logs-in-where and who-talks-to-whom drift every 10 days
    maps = []
    for _ in range(days // 10 + 1):
        uh = {u: list(rng.choice(hosts[4:], size=rng.integers(2, 5), replace=False)) + [str(rng.choice(servers))] for u in users}
        pr = {h: list(rng.choice(hosts, size=4, replace=False)) for h in hosts}
        maps.append((uh, pr))
    def mapping(t):
        return maps[int((t - t0) // epoch_len)]
    rows = []

    def feat(stage):
        v = np.maximum(0, rng.normal(CENTRES[stage], SCALE))
        v[5] = min(v[5], 1); v[7] = min(v[7], 1); v[1] = round(v[1]); v[2] = round(v[2])
        return v

    # benign background
    n_bg = 26000
    for _ in range(n_bg):
        t = t0 + rng.uniform(0, horizon)
        user_hosts, peers = mapping(t)
        u = users[rng.integers(n_users)]; vec = rng.choice(4, p=[.5, .3, .18, .02])
        if vec == 1:
            src, dst = u, str(rng.choice(user_hosts[u]))
        else:
            src = str(rng.choice(user_hosts[u])); dst = str(rng.choice(peers[src] + servers))
        rows.append((t, src, dst, VECTORS[vec], 'benign', -1, *feat('benign')))

    # incidents, spread over time
    starts = np.sort(rng.uniform(0, horizon - 86400, n_incidents)) + t0
    for k, ts in enumerate(starts):
        path = PATHS[rng.integers(len(PATHS))]
        attacker = f'X{rng.integers(200):03d}'
        victim = str(rng.choice(hosts[4:])); compromised = {victim}
        t = ts
        for stage in path:
            user_hosts, peers = mapping(t)
            if stage == 'lateral_movement':
                # move along real structure: hosts that share a user with victim, or network peers
                shared = [h for u in users if victim in user_hosts[u] for h in user_hosts[u]]
                cands = [h for h in shared + peers[victim] if h not in compromised] or [h for h in hosts if h not in compromised]
                victim = str(rng.choice(cands)); compromised.add(victim)
            if stage == 'exfiltration' and rng.random() < .5:
                victim = str(rng.choice(servers[1:3])); compromised.add(victim)  # data is pulled from DB/FILE
            n_ev = int(rng.integers(2, 6))
            for _ in range(n_ev):
                t += rng.exponential(240)
                vec = rng.choice(4, p=STAGE_VECTORS[stage])
                if stage == 'reconnaissance': src, dst = attacker, victim
                elif vec == 1: src, dst = str(rng.choice(users)), victim
                else: src, dst = (attacker if stage in ('exfiltration', 'impact') and rng.random() < .5 else victim), victim
                rows.append((t, src, dst, VECTORS[vec], stage, k, *feat(stage)))
            t += rng.exponential(900)
    df = pd.DataFrame(rows, columns=['ts', 'src', 'dst', 'vector', 'stage', 'incident'] + FEATURES)
    df = df.sort_values('ts', kind='stable').reset_index(drop=True)
    df['timestamp'] = pd.to_datetime(df.ts, unit='s', utc=True)
    return df, dict(hosts=hosts, servers=servers, users=users)
