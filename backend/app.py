from datetime import datetime, timezone
from threading import RLock
import math
import time
import re
import numpy as np
from flask import Flask, jsonify, request, send_from_directory
from werkzeug.exceptions import HTTPException
import pandas as pd
from backend.config import FRONTEND_DIR, DATA_DIR, FEATURES
from backend.storage import EventStore
from backend.services.next_move import NextMovePredictor, WINDOW, MAX_GAP_SECONDS
from backend.utils.sequence_generator import sequence_fixture, stage_sample, DEMO_PATH
from backend.services.edge_ids import EdgeIDSManager
from backend.services.resource_forecaster import ResourceForecaster
from backend.services.risk_engine import calculate_risk
from backend.services.explainer import explain
from backend.services.fog_orchestrator import FogOrchestrator
from backend.utils.sample_generator import generate_live_sample

app = Flask(__name__, static_folder=str(FRONTEND_DIR), static_url_path='')
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024
edge, resources, fog = EdgeIDSManager(), ResourceForecaster(), FogOrchestrator()
store, lock = EventStore(), RLock()
next_move = NextMovePredictor()

@app.before_request
def protect_local_api():
    if request.method == 'POST':
        origin = request.headers.get('Origin')
        if origin and origin != request.host_url.rstrip('/'):
            return jsonify(error='Cross-origin requests are not allowed.'), 403

@app.errorhandler(HTTPException)
def http_error(exc):
    return jsonify(error=exc.description), exc.code

@app.errorhandler(ValueError)
def invalid(exc):
    return jsonify(error=str(exc)), 400

def payload():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        raise ValueError('Send a JSON object.')
    return data

def investigation(data):
    value = data.get('investigation_id')
    if value is None:
        return None
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9_.-]{1,64}', value):
        raise ValueError('Investigation ID must use 1–64 letters, digits, dots, dashes or underscores.')
    return value


def analyse(sample, scenario, investigation_id=None, sequence_source=None):
    start = time.perf_counter()
    with lock:
        results = edge.predict_all(sample)
        telemetry = resources.snapshot()
        risk, level = calculate_risk(sample, results)
        decision = fog.decide(results, telemetry, risk, edge.metrics)
        event = dict(timestamp=datetime.now(timezone.utc).isoformat(), scenario=scenario,
                     sample=sample, edge_results=results, resources=telemetry, risk_score=risk,
                     risk_level=level, decision=decision, explainability=explain(sample),
                     model_version=edge.version, latency_ms=round((time.perf_counter()-start)*1000, 2),
                     traffic_source='Synthetic scenario' if scenario != 'custom' else 'Manual feature input',
                     enforcement='Recommendation only; no network traffic is blocked')
        event['investigation_id'] = investigation_id
        event['sequence_source'] = sequence_source
        if not investigation_id or sequence_source not in {'manual', 'sequence_demo'}:
            event['next_move'] = {'status': 'unrelated_samples', 'candidates': [],
                                  'message': 'Single randomized scenarios are not an intrusion sequence. Use the sequence demo or related custom observations in one investigation.'}
        else:
            previous = store.investigation_history(investigation_id, sequence_source, WINDOW-1)
            history = [event]
            for old in reversed(previous):
                gap = (datetime.fromisoformat(history[0]['timestamp']) - datetime.fromisoformat(old['timestamp'])).total_seconds()
                if gap <= 0 or gap > MAX_GAP_SECONDS:
                    break
                history.insert(0, old)
            event['next_move'] = next_move.predict(history)
            event['next_move']['investigation_id'] = investigation_id
            event['next_move']['input_source'] = sequence_source
        event['latency_ms'] = round((time.perf_counter()-start)*1000,2)
        return store.add(event)

@app.get('/')
def index():
    return send_from_directory(FRONTEND_DIR, 'index.html')

@app.get('/api/status')
def status():
    with lock:
        return jsonify(status='online', model_metrics=edge.metrics, model_version=edge.version,
                       training_source=edge.source, summary=store.summary(), resources=resources.snapshot(),
                       confidence_threshold=0.72, next_move_model=next_move.status())

@app.post('/api/simulate')
def simulate():
    data = payload()
    scenario = data.get('scenario')
    if scenario not in {'normal', 'uncertain', 'attack'}:
        raise ValueError('Choose normal, uncertain, or attack.')
    return jsonify(analyse(generate_live_sample(scenario), scenario, investigation(data), 'random_scenario')), 201

@app.post('/api/analyze')
@app.post('/api/analyse')
def custom():
    data = payload()
    sample = {}
    for field in FEATURES:
        try:
            if isinstance(data.get(field), bool):
                raise ValueError()
            value = float(data[field])
        except (KeyError, ValueError, TypeError):
            raise ValueError(f'{field}: enter a number.')
        if not math.isfinite(value) or value < 0 or value > (1 if field == 'syn_ratio' else 1e9):
            raise ValueError(f'{field}: value is outside the allowed range.')
        sample[field] = value
    return jsonify(analyse(sample, 'custom', investigation(data), 'manual')), 201

@app.get('/api/events')
def events():
    try:
        page = max(1, int(request.args.get('page', 1)))
        size = min(50, max(1, int(request.args.get('page_size', 10))))
    except ValueError:
        raise ValueError('Page and page_size must be integers.')
    risk = request.args.get('risk', 'all')
    if risk not in {'all', 'High', 'Medium', 'Low'}:
        raise ValueError('Invalid risk filter.')
    return jsonify(store.list(page, size, risk))

@app.get('/api/sample-dataset')
def sample_dataset():
    return send_from_directory(DATA_DIR, 'sample_traffic.csv', as_attachment=True)

@app.post('/api/train')
def train():
    upload = request.files.get('file')
    if not upload or not upload.filename.lower().endswith('.csv'):
        raise ValueError('Choose a CSV file.')
    try:
        frame = pd.read_csv(upload, nrows=50001)
    except Exception:
        raise ValueError('Unable to read CSV. Use a UTF-8 CSV with a header row.')
    with lock:
        previous = edge.source
        edge.source = 'Uploaded CSV (random stratified holdout)'
        try:
            metrics = edge.train(frame)
        except Exception:
            edge.source = previous
            raise
        return jsonify(ok=True, rows_used=len(frame), metrics=metrics, model_version=edge.version)


@app.post('/api/prediction/demo-step')
def demo_step():
    identity = investigation(payload())
    if not identity:
        raise ValueError('An investigation_id is required for the sequence demo.')
    with lock:
        step = store.investigation_count(identity, 'sequence_demo')
        if step >= len(DEMO_PATH):
            return jsonify(error='This demonstration is complete. Start a new investigation to replay it.'), 409
        # The planned future and synthetic stage label are never passed to predictor.
        event = analyse(stage_sample(DEMO_PATH[step], np.random.default_rng(98301+step)),
                        'sequence demo', identity, 'sequence_demo')
        return jsonify(event), 201


@app.get('/api/prediction/example-csv')
def sequence_example():
    return app.response_class(sequence_fixture().to_csv(index=False), mimetype='text/csv',
                              headers={'Content-Disposition': 'attachment; filename=synthetic_intrusion_sequences.csv'})


@app.post('/api/prediction/train')
def train_prediction():
    upload = request.files.get('file')
    if not upload or not upload.filename.lower().endswith('.csv'):
        raise ValueError('Choose a labeled sequence CSV file.')
    try:
        frame = pd.read_csv(upload, nrows=50001)
    except Exception:
        raise ValueError('Unable to read CSV. Use a UTF-8 CSV with a header row.')
    with lock:
        return jsonify(ok=True, next_move_model=next_move.train(frame))
