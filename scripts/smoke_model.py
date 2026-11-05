"""CPU model smoke check on blank frames; not a gaze accuracy benchmark."""
import argparse
import os
import resource
import time
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ.setdefault(key,'2')
import numpy as np
import mediapipe as mp
from eyemonitor.download import default_model

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--model', default=str(default_model()))
a = p.parse_args()
opts = mp.tasks.vision.FaceLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path=a.model, delegate=mp.tasks.BaseOptions.Delegate.CPU),
    running_mode=mp.tasks.vision.RunningMode.VIDEO, num_faces=1)
with mp.tasks.vision.FaceLandmarker.create_from_options(opts) as model:
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.zeros((480,640,3),dtype=np.uint8))
    start = time.monotonic()
    for i in range(30):
        result = model.detect_for_video(image, i+1)
        assert not result.face_landmarks, 'Unexpected face on a blank frame'
    print(f'CPU inference OK, 30 blank frames: {time.monotonic()-start:.3f} seconds')
print(f'Process peak RSS: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024:.1f} MiB')
