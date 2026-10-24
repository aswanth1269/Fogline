import io
import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch
import numpy as np
from backend import app as api_module
from backend.storage import EventStore
from backend.services.next_move import NextMovePredictor, make_windows, validate_frame
from backend.utils.sequence_generator import sequence_fixture, stage_sample
from backend.utils.sample_generator import generate_dataset

class SequenceTrainingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.predictor = NextMovePredictor(cls.temp.name)
        cls.frame = sequence_fixture(24)

    def test_temporal_holdout_and_reload(self):
        result = self.predictor.train(self.frame)
        self.assertLess(result['metrics']['train_end'], result['metrics']['test_start'])
        self.assertEqual(result['source'], 'synthetic')
        self.assertEqual(result['metrics']['train_episodes'],18)
        restored = NextMovePredictor(self.temp.name)
        self.assertEqual(restored.status(), self.predictor.status())

    def test_windows_only_target_next_row_and_never_cross_episodes(self):
        df = validate_frame(self.frame)
        x,y,ids,times = make_windows(df)
        episode = df[df.episode_id == ids[0]].sort_values('timestamp').to_dict('records')
        self.assertEqual(y[0],episode[3]['attack_stage'])
        self.assertEqual(times[0],episode[3]['timestamp'])
        changed = df.copy()
        changed.loc[(changed.episode_id == ids[0]) & (changed.timestamp >= times[0]), 'packet_rate'] = 9999
        x2,_,_,_ = make_windows(changed)
        np.testing.assert_array_equal(x[0],x2[0])
        short = df.groupby('episode_id',sort=False).head(3)
        self.assertEqual(len(make_windows(short)[0]),0)

    def test_bad_training_does_not_replace_bundle(self):
        before = self.predictor.path.read_bytes()
        for frame in [generate_dataset(100), self.frame.assign(attack_stage='unknown'),
                      self.frame.assign(syn_ratio=2), self.frame.assign(timestamp='bad')]:
            with self.assertRaises(ValueError): self.predictor.train(frame)
            self.assertEqual(self.predictor.path.read_bytes(),before)
        duplicates = self.frame.copy()
        duplicates.loc[1,'timestamp'] = duplicates.loc[0,'timestamp']
        with self.assertRaises(ValueError): self.predictor.train(duplicates)

    def test_history_and_out_of_range_abstention(self):
        self.assertEqual(self.predictor.predict([])['status'],'insufficient_history')
        history=[dict(timestamp='2025-01-01T00:00:00Z',sample={f:1e9 for f in api_module.FEATURES})]*3
        prediction=self.predictor.predict(history)
        self.assertEqual(prediction['status'],'out_of_range')
        self.assertEqual(prediction['candidates'],[])

    def test_long_gaps_do_not_make_windows(self):
        frame=validate_frame(self.frame)
        frame['timestamp'] = [datetime(2025,1,1,tzinfo=timezone.utc)+timedelta(hours=i) for i in range(len(frame))]
        self.assertEqual(len(make_windows(frame)[0]),0)

class SequenceAPITest(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.previous_store=api_module.store
        self.previous_predictor=api_module.next_move
        api_module.store=EventStore(Path(self.temp.name)/'events.db')
        api_module.next_move=NextMovePredictor(Path(self.temp.name)/'models')
        self.client=api_module.app.test_client()

    def tearDown(self):
        api_module.store=self.previous_store
        api_module.next_move=self.previous_predictor
        self.temp.cleanup()

    def step(self,identity):
        return self.client.post('/api/prediction/demo-step',json={'investigation_id':identity})

    def test_demo_forecast_and_session_isolation(self):
        one=self.step('case-a');two=self.step('case-a');three=self.step('case-a')
        self.assertEqual(one.status_code,201)
        self.assertEqual(two.json['next_move']['status'],'insufficient_history')
        self.assertIn(three.json['next_move']['status'],{'forecast','uncertain'})
        self.assertEqual(len(three.json['next_move']['candidates']),3)
        self.assertEqual(three.json['next_move']['training_source'],'synthetic')
        self.assertFalse(three.json['next_move']['scores_calibrated'])
        self.assertEqual(self.step('case-b').json['next_move']['history_count'],1)
        saved=EventStore(api_module.store.path).list()['items']
        self.assertEqual(next(e for e in saved if e['id']==three.json['id'])['next_move'],three.json['next_move'])

    def test_sources_and_expired_history_do_not_mix(self):
        for _ in range(3): self.step('case-a')
        sample=stage_sample('reconnaissance',np.random.default_rng(1))
        response=self.client.post('/api/analyze',json=dict(sample,investigation_id='case-a'))
        self.assertEqual(response.json['next_move']['history_count'],1)
        with api_module.store.connect() as db:
            rows=db.execute('SELECT id,payload FROM events').fetchall()
            for key,payload in rows:
                event=json.loads(payload)
                event['timestamp']=(datetime.now(timezone.utc)-timedelta(hours=2)).isoformat()
                db.execute('UPDATE events SET payload=? WHERE id=?',(json.dumps(event),key))
        self.assertEqual(self.step('case-a').json['next_move']['history_count'],1)

    def test_training_endpoint_and_invalid_binary_csv(self):
        before=api_module.next_move.status()['version']
        r=self.client.post('/api/prediction/train',data={'file':(io.BytesIO(generate_dataset(100).to_csv(index=False).encode()),'binary.csv')})
        self.assertEqual(r.status_code,400)
        self.assertEqual(api_module.next_move.status()['version'],before)
        r=self.client.post('/api/prediction/train',data={'file':(io.BytesIO(sequence_fixture(24).to_csv(index=False).encode()),'sequences.csv')})
        self.assertEqual(r.status_code,200)
        self.assertEqual(r.json['next_move_model']['version'],before+1)
        self.assertEqual(r.json['next_move_model']['source'],'synthetic')

    def test_invalid_id_random_scenario_and_end_of_demo(self):
        self.assertEqual(self.step('../bad').status_code,400)
        r=self.client.post('/api/simulate',json={'scenario':'normal','investigation_id':'case-a'})
        self.assertEqual(r.status_code,201)
        self.assertEqual(r.json['next_move']['status'],'unrelated_samples')
        for _ in range(10): self.assertEqual(self.step('demo-end').status_code,201)
        self.assertEqual(self.step('demo-end').status_code,409)

if __name__=='__main__':unittest.main()
