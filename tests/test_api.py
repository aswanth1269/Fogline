import io
import tempfile
import unittest
from pathlib import Path
from backend import app as module
from backend.storage import EventStore
from backend.utils.sample_generator import generate_dataset
class APITest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory()
        module.store=EventStore(Path(cls.temp.name)/'test.db')
        cls.client=module.app.test_client()
    def test_scenarios_persistence(self):
        for scenario in ['normal','uncertain','attack']:
            r=self.client.post('/api/simulate',json={'scenario':scenario})
            self.assertEqual(r.status_code,201)
            self.assertEqual(len(r.json['edge_results']),3)
        self.assertGreaterEqual(EventStore(module.store.path).summary()['total'],3)
        self.assertEqual(len(self.client.get('/api/events?page_size=2').json['items']),2)
    def test_bad_inputs(self):
        for value in [[],None,{'scenario':'bad'}]:
            self.assertEqual(self.client.post('/api/simulate',json=value).status_code,400)
        self.assertEqual(self.client.post('/api/analyze',json={}).status_code,400)
        s=generate_dataset().iloc[0].drop('label').to_dict()
        for v in [-1,2,float('inf')]:
            s['syn_ratio']=v
            self.assertEqual(self.client.post('/api/analyze',json=s).status_code,400)
        self.assertEqual(self.client.get('/api/events?page=no').status_code,400)
    def test_csv_failure_preserves_model(self):
        before=module.edge.version
        df=generate_dataset(60);df['label']=2
        r=self.client.post('/api/train',data={'file':(io.BytesIO(df.to_csv(index=False).encode()),'bad.csv')})
        self.assertEqual(r.status_code,400)
        self.assertEqual(module.edge.version,before)
        self.assertEqual(len(module.edge.models),3)
    def test_cross_origin(self):
        self.assertEqual(self.client.post('/api/simulate',json={'scenario':'normal'},headers={'Origin':'https://example.com'}).status_code,403)
    def test_custom_filter(self):
        s=generate_dataset().iloc[0].drop('label').to_dict()
        self.assertEqual(self.client.post('/api/analyze',json=s).status_code,201)
        r=self.client.get('/api/events?risk=High&page=999')
        self.assertEqual(r.status_code,200)
        self.assertTrue(all(x['risk_level']=='High' for x in r.json['items']))
    def test_training_reload(self):
        import backend.services.edge_ids as ids
        old=ids.MODEL_DIR
        with tempfile.TemporaryDirectory() as directory:
            ids.MODEL_DIR=Path(directory)
            try:
                manager=ids.EdgeIDSManager()
                manager.train(generate_dataset(100))
                restored=ids.EdgeIDSManager()
                self.assertEqual(restored.version,manager.version)
                self.assertEqual(restored.metrics,manager.metrics)
            finally: ids.MODEL_DIR=old
if __name__=='__main__':unittest.main()
