import json
import sqlite3
from contextlib import contextmanager
from backend.config import DATA_DIR


class EventStore:
    def __init__(self, path=None):
        DATA_DIR.mkdir(exist_ok=True)
        self.path = str(path or DATA_DIR / 'events.sqlite3')
        with self.connect() as db:
            db.execute('CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY, payload TEXT NOT NULL)')

    @contextmanager
    def connect(self):
        connection = sqlite3.connect(self.path, timeout=15)
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def add(self, event):
        with self.connect() as db:
            cursor = db.execute('INSERT INTO events(payload) VALUES (?)', (json.dumps(event),))
            event['id'] = cursor.lastrowid
        return event

    def list(self, page=1, size=10, risk='all'):
        clause = '' if risk == 'all' else " WHERE json_extract(payload, '$.risk_level') = ?"
        args = [] if risk == 'all' else [risk]
        with self.connect() as db:
            count = db.execute('SELECT COUNT(*) FROM events' + clause, args).fetchone()[0]
            page = min(page, max(1, (count + size - 1) // size))
            rows = db.execute('SELECT id,payload FROM events' + clause + ' ORDER BY id DESC LIMIT ? OFFSET ?', args + [size, (page-1)*size]).fetchall()
        return {'items': [dict(json.loads(payload), id=key) for key, payload in rows], 'total': count, 'page': page, 'page_size': size}

    def summary(self):
        with self.connect() as db:
            row = db.execute("SELECT COUNT(*), COALESCE(SUM(json_extract(payload,'$.decision.escalated')),0), COALESCE(SUM(json_extract(payload,'$.risk_level')='High'),0), COALESCE(AVG(json_extract(payload,'$.latency_ms')),0) FROM events").fetchone()
        return dict(zip(['total', 'escalated', 'high_risk', 'avg_latency'], row))

    def investigation_history(self, investigation_id, source, limit=3):
        with self.connect() as db:
            rows = db.execute(
                "SELECT id,payload FROM events WHERE json_extract(payload,'$.investigation_id')=? AND json_extract(payload,'$.sequence_source')=? ORDER BY id DESC LIMIT ?",
                (investigation_id, source, limit),
            ).fetchall()
        return [dict(json.loads(payload), id=key) for key,payload in reversed(rows)]

    def investigation_count(self, investigation_id, source):
        with self.connect() as db:
            return db.execute(
                "SELECT COUNT(*) FROM events WHERE json_extract(payload,'$.investigation_id')=? AND json_extract(payload,'$.sequence_source')=?",
                (investigation_id, source),
            ).fetchone()[0]
