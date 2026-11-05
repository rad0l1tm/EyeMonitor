import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import numpy as np
from eyemonitor.core import Calibration, Stabilizer, atomic_json, eye_features, region


class CoreTests(unittest.TestCase):
    def dataset(self):
        targets = np.array([(x, y) for y in (.1,.5,.9) for x in (.1,.5,.9)])
        features = np.zeros((9,6))
        features[:, :2] = (targets - .5) * .3
        features[:, 2:] = [.02,.1,.5,.4]
        return features, targets

    def test_calibration_generalizes_to_unseen_targets(self):
        x, y = self.dataset()
        model = Calibration().fit(x,y)
        for target in ([.22,.78],[.7,.3],[.5,.5]):
            v = np.array([*( (np.array(target)-.5)*.3 ), .02,.1,.5,.4])
            np.testing.assert_allclose(model.predict(v), target, atol=.025)

    def test_no_eye_motion_rejected(self):
        with self.assertRaises(ValueError):
            Calibration().fit(np.zeros((9,6)), np.zeros((9,2)))

    def test_nan_rejected(self):
        x,y = self.dataset()
        x[0,0] = np.nan
        with self.assertRaises(ValueError):
            Calibration().fit(x,y)

    def test_out_of_calibration_range_rejected(self):
        x,y = self.dataset()
        model = Calibration().fit(x,y)
        self.assertIsNone(model.predict([10,0,.02,.1,.5,.4]))

    def test_profile_roundtrip_and_context(self):
        x,y = self.dataset()
        m = Calibration().fit(x,y)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'profile.json'
            m.save(path, {'camera':'one'})
            loaded = Calibration.load(path, {'camera':'one'})
            np.testing.assert_allclose(m.predict(x[3]), loaded.predict(x[3]))
            with self.assertRaises(ValueError):
                Calibration.load(path, {'camera':'two'})
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)

    def test_corrupted_profile_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'profile.json'
            path.write_text(json.dumps(dict(version=1, context={}, mean=[0]*6,
                                           scale=[0]*6, weights=[[0,0]]*7)))
            with self.assertRaises(ValueError):
                Calibration.load(path,{})

    def test_grid_edges(self):
        self.assertEqual(region([1,1]), (2,2))
        self.assertEqual(region([0,0]), (0,0))
        self.assertEqual(region([.5,.5]), (1,1))
        self.assertEqual(region([-1,2]), (0,2))

    def test_dwell_and_tracking_loss(self):
        s = Stabilizer()
        self.assertIsNone(s.update([.5,.5], 0)[1])
        self.assertEqual(s.update([.5,.5], .21)[1], (1,1))
        self.assertEqual(s.update(None, .3), (None,None))
        self.assertIsNone(s.update([.5,.5], .4)[1])

    def test_region_changes_only_after_dwell(self):
        s = Stabilizer(tau=.001)
        s.update([.1,.1],0)
        s.update([.1,.1],.3)
        self.assertEqual(s.update([.9,.9], .4)[1], (0,0))
        self.assertEqual(s.update([.9,.9], .7)[1], (2,2))

    def test_atomic_latest_state(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/'nested/state.json'
            atomic_json(p, {'valid':True})
            atomic_json(p, {'valid':False})
            self.assertEqual(json.loads(p.read_text()), {'valid':False})
            self.assertFalse(p.with_name('state.json.tmp').exists())

    def face(self):
        p = [SimpleNamespace(x=.5,y=.5,z=0) for _ in range(478)]
        # Pixel positions at 640x480; both eye axes point to the right.
        positions = {33:(200,200),133:(260,200),159:(230,190),145:(230,210),468:(230,200),
                     362:(350,200),263:(410,200),386:(380,190),374:(380,210),473:(380,200),
                     234:(150,240),454:(460,240),1:(305,260)}
        for i,(x,y) in positions.items():
            p[i] = SimpleNamespace(x=x/640,y=y/480,z=0)
        return p

    def test_iris_translation_is_measured(self):
        p = self.face()
        before = eye_features(p,640,480)
        p[468].x += 6/640
        p[473].x += 6/640
        after = eye_features(p,640,480)
        self.assertAlmostEqual(after[0]-before[0], .1)
        self.assertAlmostEqual(after[1], 0)

    def test_closed_eye_rejected(self):
        p = self.face()
        p[159].y = p[145].y
        self.assertIsNone(eye_features(p,640,480))

    def test_missing_iris_rejected(self):
        self.assertIsNone(eye_features(self.face()[:468],640,480))

    def test_roll_preserves_eye_local_coordinates(self):
        p = self.face()
        p[468].x += 5/640
        p[473].x += 5/640
        before = eye_features(p,640,480)[:2]
        a = .3
        rotation = np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])
        for v in p:
            xy = rotation @ (np.array([v.x*640,v.y*480])-[320,240])+[320,240]
            v.x,v.y = xy/[640,480]
        np.testing.assert_allclose(before,eye_features(p,640,480)[:2],atol=1e-8)


if __name__ == '__main__':
    unittest.main()
