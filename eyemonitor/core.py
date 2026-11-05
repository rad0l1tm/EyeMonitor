import json
import os
import zipfile
from collections import deque
from pathlib import Path

import cv2
import numpy as np

FEATURE_VERSION = 3
PATCH_W, PATCH_H = 10, 6               # final patch per eye (as in WebGazer)
UPSCALE = 4                            # equalise at 40x24, then shrink to 10x6
PIXELS = 2 * PATCH_W * PATCH_H         # 120 pixel features
IRIS = 4                               # (ex, ey) of each eye, in eye-widths
FEATURE_COUNT = PIXELS + IRIS          # layout: [pixels..., ex_a, ey_a, ex_b, ey_b]
EYE_MARGIN = 1.3                       # patch width / eye-corner distance
MIN_EYE_PX = 12                        # eye narrower than this (pixels) is rejected
MIN_OPENING = .12                      # blink / closed-eye rejection
WINDOW = 8000                          # max samples kept for training
MIN_SAMPLES = 30
LAMBDAS = np.logspace(-3, 1, 9)        # ridge strengths tried by CV (relative to one pixel's variance)
BOOSTS = (0., .5, 1., 2.)              # weight of the iris block tried by CV; 0 = pixels only
DEFAULT_LAM, DEFAULT_BOOST = .05, 1.

EYES = ((33, 133, 159, 145, 468), (362, 263, 386, 374, 473))


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    with temp.open('w') as f:
        os.chmod(temp, 0o600)
        json.dump(data, f, allow_nan=False)
    temp.replace(path)


# ----------------------------------------------------------------- features

LAST_REJECT = ''


def _reject(reason):
    global LAST_REJECT
    LAST_REJECT = reason
    return None


def eye_features(gray, landmarks, width, height):
    if len(landmarks) < 478 or gray.ndim != 2:
        return _reject('нет точек радужки (ожидается 478 landmarks)')

    def point(i):
        v = landmarks[i]
        return np.array([v.x * width, v.y * height])

    w2, h2 = PATCH_W * UPSCALE, PATCH_H * UPSCALE
    pixels, iris = [], []
    for a, b, top, bottom, iris_i in EYES:
        pa, pb, pt, pbm, pi = (point(i) for i in (a, b, top, bottom, iris_i))
        axis = pb - pa
        size = np.linalg.norm(axis)
        if not np.isfinite(size) or size < MIN_EYE_PX:
            return _reject(f'глаз слишком мал ({size:.0f} px < {MIN_EYE_PX}): подвиньтесь ближе или поднимите --width')
        u = axis / size
        n = np.array([-u[1], u[0]])
        opening = abs(np.dot(pbm - pt, n)) / size
        if opening < MIN_OPENING:
            return _reject(f'глаз закрыт или прищурен (раскрытие {opening:.2f} < {MIN_OPENING})')
        center = (pa + pb) / 2
        delta = (pi - center) / size
        ex, ey = float(np.dot(delta, u)), float(np.dot(delta, n))
        if not (np.isfinite(ex) and np.isfinite(ey)) or abs(ex) > .5 or abs(ey) > .5:
            return _reject('радужка вне контура глаза (ошибка трекинга)')
        iris += [ex, ey]
        bw = size * EYE_MARGIN
        bh = bw * PATCH_H / PATCH_W
        m = np.empty((2, 3))
        m[:, 0] = u * bw / w2
        m[:, 1] = n * bh / h2
        m[:, 2] = center + u * bw * (.5 / w2 - .5) + n * bh * (.5 / h2 - .5)
        if not np.isfinite(m).all():
            return _reject('некорректная геометрия глаза')
        big = cv2.warpAffine(gray, m, (w2, h2),
                             flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                             borderMode=cv2.BORDER_REPLICATE)
        big = cv2.equalizeHist(big)
        small = cv2.resize(big, (PATCH_W, PATCH_H), interpolation=cv2.INTER_AREA)
        pixels.append(small.ravel())
    global LAST_REJECT
    LAST_REJECT = ''
    return np.concatenate([np.concatenate(pixels).astype(np.float32) / 255.,
                           np.asarray(iris, np.float32)])


def _stats(x, w):
    sw = w.sum()
    mx = (w[:, None] * x).sum(0) / sw
    sd = np.sqrt((w[:, None] * (x - mx) ** 2).sum(0) / sw)
    for part in (slice(0, PIXELS), slice(PIXELS, None)):
        med = np.median(sd[part])
        sd[part] = np.maximum(sd[part], .1 * med if med > 0 else 1.)
    return mx, sd


def _block(boost):
    return np.concatenate([np.full(PIXELS, 1 / np.sqrt(PIXELS)), np.full(IRIS, boost / np.sqrt(IRIS))])


def _prep(x, y, w):
    mx, sd = _stats(x, w)
    sw = w.sum()
    my = (w[:, None] * y).sum(0) / sw
    r = np.sqrt(w / sw)[:, None]
    zr = ((x - mx) / sd) * r
    return mx, sd, my, zr.T @ zr, zr.T @ ((y - my) * r)


def _ridge(gz, xy, boost, lam):
    c = _block(boost)
    g = gz * c[:, None] * c[None, :]
    return c[:, None] * np.linalg.solve(g + lam / PIXELS * np.eye(len(c)), c[:, None] * xy)


def _select(x, y, w, g, k):
    ids = np.unique(g)
    folds = []
    for i in range(k):
        test = np.isin(g, ids[i::k])
        if not test.any() or test.all():
            continue
        mx, sd, my, gz, xy = _prep(x[~test], y[~test], w[~test])
        folds.append(((x[test] - mx) / sd, y[test], w[test], my, gz, xy))
    out = {}
    for boost in BOOSTS:
        for lam in LAMBDAS:
            se = sw = 0.
            for zt, yt, wt, my, gz, xy in folds:
                pred = zt @ _ridge(gz, xy, boost, lam) + my
                se += (wt * np.sum((pred - yt) ** 2, axis=1)).sum()
                sw += wt.sum()
            out[(boost, lam)] = np.sqrt(se / sw) if sw else np.inf
    return out


class Calibration:

    def __init__(self, window=WINDOW):
        self.x, self.y, self.w = deque(maxlen=window), deque(maxlen=window), deque(maxlen=window)
        self.lam, self.boost = DEFAULT_LAM, DEFAULT_BOOST
        self.mx = self.sd = self.my = self.W = None
        self.cv_error = None
        self.cv_by_boost = {}

    def __len__(self):
        return len(self.x)

    @property
    def ready(self):
        return self.W is not None

    def add(self, features, target, weight=1.):
        f = np.asarray(features, np.float32)
        t = np.asarray(target, np.float32)
        if (f.shape != (FEATURE_COUNT,) or t.shape != (2,)
                or not np.isfinite(f).all() or not np.isfinite(t).all()):
            return False
        self.x.append(f)
        self.y.append(t)
        self.w.append(float(weight))
        return True

    def fit(self, select=True):
        n = len(self.x)
        if n < MIN_SAMPLES:
            raise ValueError(f'Мало образцов: {n} из {MIN_SAMPLES}')
        x = np.asarray(self.x, np.float64)
        y = np.asarray(self.y, np.float64)
        w = np.asarray(self.w, np.float64)
        if select:
            if np.max(np.ptp(y, axis=0)) < .3:
                raise ValueError('Цели калибровки слишком близко друг к другу')
            g = (np.clip(y[:, 0] * 4, 0, 3).astype(int) * 4 + np.clip(y[:, 1] * 4, 0, 3).astype(int))
            k = min(5, len(np.unique(g)))
            if k >= 3:
                errors = _select(x, y, w, g, k)
                (self.boost, self.lam), self.cv_error = min(errors.items(), key=lambda kv: kv[1])
                self.lam, self.boost, self.cv_error = float(self.lam), float(self.boost), float(self.cv_error)
                self.cv_by_boost = {float(b): float(min(e for (bb, _), e in errors.items() if bb == b))
                                    for b in BOOSTS}
        mx, sd, my, gz, xy = _prep(x, y, w)
        self.mx, self.sd, self.my = mx, sd, my
        self.W = _ridge(gz, xy, self.boost, self.lam)
        return self

    def predict(self, features):
        if self.W is None or features is None:
            return None
        f = np.asarray(features, np.float64)
        if f.shape != (FEATURE_COUNT,) or not np.isfinite(f).all():
            return None
        raw = ((f - self.mx) / self.sd) @ self.W + self.my
        if np.any(raw < -.25) or np.any(raw > 1.25):
            return None
        return np.clip(raw, 0, 1)


    def save(self, path, context):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_name(path.name + '.tmp')
        x = np.asarray(self.x, np.float32).reshape(-1, FEATURE_COUNT)
        with temp.open('wb') as f:
            os.chmod(temp, 0o600)
            np.savez_compressed(
                f, version=FEATURE_VERSION, context=json.dumps(context, sort_keys=True),
                lam=self.lam, boost=self.boost,
                px=np.rint(x[:, :PIXELS] * 255).astype(np.uint8), iris=x[:, PIXELS:],
                y=np.asarray(self.y, np.float32).reshape(-1, 2), w=np.asarray(self.w, np.float32))
        temp.replace(path)

    @classmethod
    def load(cls, path, context):
        try:
            with np.load(path, allow_pickle=False) as d:
                version, ctx = int(d['version']), str(d['context'])
                lam, boost = float(d['lam']), float(d['boost'])
                px, iris, y, w = d['px'], d['iris'], d['y'], d['w']
        except (zipfile.BadZipFile, EOFError) as e:
            raise ValueError('Повреждённый профиль') from e
        if version != FEATURE_VERSION or ctx != json.dumps(context, sort_keys=True):
            raise ValueError('Профиль камеры/экрана изменился')
        n = len(px)
        if (px.ndim != 2 or px.shape[1] != PIXELS or iris.shape != (n, IRIS)
                or y.shape != (n, 2) or w.shape != (n,)):
            raise ValueError('Повреждённый профиль')
        obj = cls()
        obj.lam, obj.boost = lam, boost
        for p, i, t, wt in zip(px, iris, y, w):
            obj.add(np.concatenate([p.astype(np.float32) / 255., i]), t, wt)
        obj.fit(select=False)
        return obj


# ------------------------------------------------------------- stabilisation

class OneEuroFilter:
    def __init__(self, min_cutoff=1.0, beta=6.0, d_cutoff=1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.reset()

    @staticmethod
    def alpha(cutoff, dt):
        tau = 1.0 / (2.0 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def reset(self):
        self.x_prev = self.dx_prev = self.t_prev = None

    def update(self, value, t):
        x = np.asarray(value, float)
        if not np.isfinite(x).all():
            return None
        if self.x_prev is None:
            self.x_prev, self.dx_prev, self.t_prev = x.copy(), np.zeros_like(x), t
            return x.copy()
        dt = t - self.t_prev
        if dt <= 0:
            return self.x_prev.copy()
        dt = max(dt, 1e-4)
        dx = (x - self.x_prev) / dt
        a_d = self.alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev
        cutoff = self.min_cutoff + self.beta * np.linalg.norm(dx_hat)
        a = self.alpha(cutoff, dt)
        x_hat = a * x + (1 - a) * self.x_prev
        self.x_prev, self.dx_prev, self.t_prev = x_hat, dx_hat, t
        return x_hat.copy()


def region(point, columns=3, rows=3, prev=None, margin=.03):
    x, y = np.clip(point, 0, 1)
    c = min(int(x * columns), columns - 1)
    r = min(int(y * rows), rows - 1)
    if prev is not None:
        pc, pr = prev
        if abs(x - (pc + .5) / columns) < .5 / columns + margin: c = pc
        if abs(y - (pr + .5) / rows) < .5 / rows + margin: r = pr
    return c, r


class Stabilizer:
    def __init__(self, dwell=.2, min_cutoff=.5, beta=4.0, hold=.25, median=5, lock=.02):
        self.dwell, self.hold, self.lock = dwell, hold, lock
        self.filter = OneEuroFilter(min_cutoff, beta)
        self.buf = deque(maxlen=median)
        self.reset()

    def reset(self):
        self.point = self.current = self.candidate = self.last = self.anchor = None
        self.since = 0.
        self.buf.clear()
        self.filter.reset()

    def _deadband(self, p):
        if self.lock <= 0:
            return p
        if self.anchor is None:
            self.anchor = p.copy()
        else:
            d = p - self.anchor
            dist = np.linalg.norm(d)
            if dist > self.lock:
                self.anchor = p - d / dist * self.lock
        return self.anchor.copy()

    def update(self, point, now):
        if point is None:
            if self.last is not None and now - self.last > self.hold:
                self.reset()
            return (None if self.point is None else self.point.copy()), self.current
        self.buf.append(np.asarray(point, float))
        med = np.median(np.stack(self.buf), axis=0)
        filtered = self.filter.update(med, now)
        if filtered is None:
            return None, None
        self.point, self.last = self._deadband(filtered), now
        cell = region(self.point, prev=self.candidate)
        if cell != self.candidate:
            self.candidate, self.since = cell, now
        if now - self.since >= self.dwell:
            self.current = cell
        return self.point.copy(), self.current
