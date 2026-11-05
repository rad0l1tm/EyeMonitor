"""Offscreen integration checks with synthetic camera samples."""
import os
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
import unittest
import threading
import json
import numpy as np
try:
    import cv2
    for key in ('QT_QPA_PLATFORM_PLUGIN_PATH','QT_QPA_FONTDIR'):
        if 'cv2' in os.environ.get(key,''):
            os.environ.pop(key, None)
    from PySide6.QtWidgets import QApplication
    HAS_QT = True
except ImportError:
    HAS_QT = False
from eyemonitor.camera import Sample


@unittest.skipUnless(HAS_QT, 'PySide6 is not installed')
class AppTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        from eyemonitor.app import make_window
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.args = SimpleNamespace(profile=root/'profile.json', camera='test', width=640,
                                    output=root/'state.json', stdout=False)
        self.sample = Sample()
        self.camera = SimpleNamespace(snapshot=lambda:self.sample, enabled=threading.Event(),
                                      stop_event=threading.Event())
        self.window = make_window(self.args,self.app.screens()[0],self.camera)
        self.window.timer.stop()
        self.window.show()
        self.app.processEvents()

    def tearDown(self):
        self.window.close()
        self.temp.cleanup()

    def advance_target(self):
        w = self.window
        target = np.array(w.targets[w.target_index])
        feature = np.r_[(target-.5)*.3,.02,.1,.5,.4]
        self.sample = Sample(self.sample.sequence+1,time.monotonic(),feature,fps=15)
        w.samples = [feature]*12
        w.stage_started = time.monotonic()-3
        w.tick()
        self.app.processEvents()

    def test_full_calibration_validation_and_loss(self):
        w = self.window
        w.begin()
        for _ in range(9):
            self.advance_target()
        self.assertEqual(w.mode,'validate')
        for _ in range(5):
            self.advance_target()
        self.assertEqual(w.mode,'track')
        self.assertIn('5/5',w.note)
        self.assertTrue(self.args.profile.exists())
        self.sample = Sample(100,time.monotonic(),np.array([0,0,.02,.1,.5,.4]),fps=15)
        w.tick()
        w.emit_state(time.monotonic(),15)
        self.assertTrue(json.loads(self.args.output.read_text())['valid'])
        self.sample = Sample(101,time.monotonic(),None)
        w.tick()
        w.emit_state(time.monotonic(),15)
        state = json.loads(self.args.output.read_text())
        self.assertFalse(state['valid'])
        self.assertIsNone(state['normalized'])
        self.assertIsNone(state['cell'])

    def test_no_profile_and_shutdown_invalid(self):
        self.sample = Sample(1,time.monotonic(),np.zeros(6))
        self.window.tick()
        self.window.close()
        state = json.loads(self.args.output.read_text())
        self.assertFalse(state['valid'])
        self.assertEqual(state['mode'],'stopped')
        self.assertTrue(self.camera.stop_event.is_set())

    def test_stale_camera_result_not_reused(self):
        from eyemonitor.core import Calibration
        x = np.zeros((9,6))
        y = np.array([(a,b) for a in (.1,.5,.9) for b in (.1,.5,.9)])
        x[:,:2] = (y-.5)*.3
        self.window.model = Calibration().fit(x,y)
        self.sample = Sample(1,time.monotonic()-2,x[4])
        self.window.tick()
        self.assertIsNone(self.window.point)

    def test_tracking_gap_resets_calibration_target(self):
        w = self.window
        w.begin()
        w.samples = [np.zeros(6)]*6
        self.sample = Sample(1,time.monotonic(),None)
        w.tick()
        self.assertEqual(w.samples,[])
        self.assertEqual(w.target_index,0)


if __name__ == '__main__':
    unittest.main()
