import argparse
import json
import os
from pathlib import Path
import random
import sys
import time


def main(argv=None):
    parser = argparse.ArgumentParser(description='EyeMonitor — приблизительная область взгляда, CPU')
    parser.add_argument('--camera', default='/dev/video0')
    parser.add_argument('--model', type=Path)
    parser.add_argument('--screen', type=int, default=0)
    parser.add_argument('--fps', type=int, choices=range(5, 31), default=30, metavar='5..30')
    parser.add_argument('--width', type=int, choices=(320, 640), default=640)
    parser.add_argument('--profile', type=Path)
    parser.add_argument('--output', type=Path, help='Atomic latest-state JSON; overwritten, not a log')
    parser.add_argument('--stdout', action='store_true', help='Stream JSON lines, at most 10 Hz')
    parser.add_argument('--lock', type=float, default=.02,
                        help='мёртвая зона указателя в долях экрана (0 — выкл.)')
    parser.add_argument('--cutoff', type=float, default=.5,
                        help='min_cutoff One Euro, Гц; меньше — сильнее сглаживание')
    parser.add_argument('--fly-interval', type=float, default=4.0,
                        help='начальный интервал появления мух, секунд')
    parser.add_argument('--fly-min-interval', type=float, default=.8,
                        help='минимальный интервал появления мух, секунд')
    parser.add_argument('--fly-ramp-seconds', type=float, default=120.0,
                        help='за сколько секунд игры интервал уменьшится до минимума')
    parser.add_argument('--fly-speed', type=float, default=125.0,
                        help='постоянная скорость мух, пикселей в секунду')
    parser.add_argument('--list-screens', action='store_true')
    parser.add_argument('--fly-image', type=Path, default = Path("assets/fly.png"), 
                        help='Путь к PNG изображению мухи')
    parser.add_argument('--fly-size', type=int, default=100,
                        help='Максимальный размер мухи в пикселях')
    parser.add_argument('--music', type=Path, help='Фоновая музыка (например, music.mp3)')
    parser.add_argument('--fly-sound', type=Path, default = Path("assets/fly.wav"), help='Зацикленный звук жужжания, пока на экране есть мухи')
    parser.add_argument('--pop-sound', type=Path, default = Path("assets/pop.wav"), help='Короткий звук при попадании во fly')
    parser.add_argument('--music-volume', type=float, default=.25, help='Громкость фоновой музыки: 0..1')
    parser.add_argument('--fly-volume', type=float, default=.22, help='Громкость жужжания: 0..1')
    parser.add_argument('--pop-volume', type=float, default=.75, help='Громкость хлопка: 0..1')
    args = parser.parse_args(argv)
    if args.fly_interval <= 0 or args.fly_min_interval <= 0:
        parser.error('--fly-interval и --fly-min-interval должны быть больше нуля')
    if args.fly_min_interval > args.fly_interval:
        parser.error('--fly-min-interval не может быть больше --fly-interval')
    if args.fly_ramp_seconds <= 0 or args.fly_speed <= 0:
        parser.error('--fly-ramp-seconds и --fly-speed должны быть больше нуля')
    for option, path in (
        ('--music', args.music),
        ('--fly-sound', args.fly_sound),
        ('--pop-sound', args.pop_sound),
    ):
        if path is not None and not path.is_file():
            parser.error(f'{option}: файл не найден: {path}')
    for option, volume in (
        ('--music-volume', args.music_volume),
        ('--fly-volume', args.fly_volume),
        ('--pop-volume', args.pop_volume),
    ):
        if not 0.0 <= volume <= 1.0:
            parser.error(f'{option} должен быть в диапазоне 0..1')

    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ.setdefault(key, '2')

    import cv2
    for key in ('QT_QPA_PLATFORM_PLUGIN_PATH', 'QT_QPA_FONTDIR'):
        if 'cv2' in os.environ.get(key, ''):
            os.environ.pop(key, None)

    from PySide6.QtWidgets import QApplication
    from .download import default_model

    app = QApplication(sys.argv[:1])
    screens = app.screens()
    if args.list_screens:
        for i, screen in enumerate(screens):
            g = screen.geometry()
            print(f'{i}: {screen.name()} {g.width()}x{g.height()} logical, scale={screen.devicePixelRatio()}')
        return 0
    if not 0 <= args.screen < len(screens):
        parser.error('Неверный --screen; используйте --list-screens')

    args.model = args.model or default_model()
    if not args.model.is_file():
        parser.error('Нет модели. Сначала выполните eyemonitor-model или задайте --model')

    from .camera import Camera
    device = int(args.camera) if args.camera.isdecimal() else args.camera
    camera = Camera(device, args.model, args.fps, args.width)
    window = make_window(args, screens[args.screen], camera)
    camera.start()
    app.aboutToQuit.connect(lambda: camera.stop_event.set())
    window.showFullScreen()
    result = app.exec()
    camera.stop_event.set()
    camera.join(timeout=3)
    return result


def make_window(args, screen, camera):
    import math
    from collections import deque
    import numpy as np
    from PySide6.QtCore import Qt, QTimer, QRectF, QPointF, QUrl
    from PySide6.QtGui import QColor, QPainter, QFont, QPen, QPixmap
    from PySide6.QtWidgets import QWidget
    from .core import Calibration, Stabilizer, atomic_json, region
    from PySide6.QtCore import QSize
    FLY_HIT_RADIUS = 50.0
    RETICLE_RADIUS = 19.0
    RETICLE_MARGIN = RETICLE_RADIUS + 9.0 
    FLY_SPAWN_INITIAL = args.fly_interval
    FLY_SPAWN_MIN = args.fly_min_interval
    FLY_SPAWN_RAMP_SECONDS = args.fly_ramp_seconds
    FLY_MAX_ACTIVE = 4
    FLY_SPEED = args.fly_speed
    FLY_TURN_MIN = .55             
    FLY_TURN_MAX = 1.35
    FLY_TRAIL_DURATION = .60      
    FLY_TRAIL_SAMPLES = 24
    POP_DURATION = .18
    CLICKS_PER_DOT = 5

    class Window(QWidget):
        def __init__(self):
            super().__init__()
            self.setWindowTitle('EyeMonitor — Fly target')
            self.resize(920, 620)
            self.setMinimumSize(640, 420)
            self.setAttribute(Qt.WidgetAttribute.WA_NativeWindow)
            self.windowHandle().setScreen(screen)

            self.context = self.screen_context()
            self.profile = args.profile or (
                Path(os.environ.get('XDG_CONFIG_HOME', Path.home() / '.config'))
                / 'eyemonitor/profile.npz'
            )
            self.model = Calibration()
            self.previous_model = self.model
            self.note = 'Нажмите C для калибровки. После неё наводите прицел взглядом на мух.'
            try:
                self.model = Calibration.load(self.profile, self.context)
                self.note = 'Профиль загружен. Наводите прицел взглядом на мух; C — новая калибровка.'
            except FileNotFoundError:
                pass
            except (ValueError, KeyError, TypeError, OSError) as e:
                self.note = f'Профиль не загружен: {e}. Нажмите C.'

            self.mode, self.paused = 'track', False
            self.sequence = -1
            self.smoother = Stabilizer(min_cutoff=args.cutoff, lock=args.lock)
            self.point = self.cell = None
            self.status = 'Камера запускается…'
            self.last_write = 0.

            self.recent = deque(maxlen=16)  # (monotonic timestamp, feature vector)
            self.samples, self.centers, self.targets = [], [], []
            self.target_index = self.clicks = 0
            self.learn_moves = False
            self.auto = False
            self.dirty, self.last_fit, self.last_move = False, 0., 0.
            self.stage_started = time.monotonic()

            self.score = 0
            self.flies = []
            self.pop_effects = []
            self.game_started = time.monotonic()
            self.next_fly_at = self.game_started + FLY_SPAWN_INITIAL
            self.pause_started = None

            self.timer = QTimer(self)
            self.timer.timeout.connect(self.tick)
            self.timer.start(33)
            screen.geometryChanged.connect(self.invalidate_screen)
            screen.logicalDotsPerInchChanged.connect(self.invalidate_screen)
            from PySide6.QtWidgets import QApplication
            QApplication.instance().screenRemoved.connect(self.screen_removed)
            self.fly_pixmap = QPixmap()

            self.music_player = None
            self.fly_audio_player = None
            self.pop_audio_player = None
            self.audio_outputs = []
            self.fly_audio_playing = False
            self.setup_audio()

            if args.fly_image:
                original = QPixmap(str(args.fly_image))

                if original.isNull():
                    raise RuntimeError(
                        f'Не удалось загрузить изображение: {args.fly_image}'
                    )

                size = max(8, args.fly_size)

                self.fly_pixmap = original.scaled(
                    QSize(size, size),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                )

        # ----------------------------- audio

        def setup_audio(self):
            paths = (args.music, args.fly_sound, args.pop_sound)
            if not any(paths):
                return

            try:
                from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
            except ImportError as exc:
                self.note += f' Звук отключён: PySide6.QtMultimedia недоступен ({exc}).'
                print(self.note, file=sys.stderr)
                return

            self._QMediaPlayer = QMediaPlayer

            def make_player(path, volume, loops, label):
                if path is None:
                    return None
                output = QAudioOutput(self)
                output.setVolume(float(np.clip(volume, 0.0, 1.0)))
                player = QMediaPlayer(self)
                player.setAudioOutput(output)
                player.setSource(QUrl.fromLocalFile(str(path.resolve())))
                player.setLoops(loops)

                def report_error(error, error_string, sound_name=label):
                    print(
                        f'[audio:{sound_name}] Ошибка воспроизведения: {error_string} ({error})',
                        file=sys.stderr,
                    )

                player.errorOccurred.connect(report_error)
                self.audio_outputs.append(output)
                return player

            infinite = QMediaPlayer.Loops.Infinite
            once = QMediaPlayer.Loops.Once
            self.music_player = make_player(
                args.music, args.music_volume, infinite, 'music'
            )
            self.fly_audio_player = make_player(
                args.fly_sound, args.fly_volume, infinite, 'fly'
            )
            self.pop_audio_player = make_player(
                args.pop_sound, args.pop_volume, once, 'pop'
            )

            if self.music_player is not None:
                self.music_player.play()

        def sync_fly_audio(self):
            """Play the looping fly ambience iff at least one fly is on screen."""
            should_play = (
                self.mode == 'track'
                and not self.paused
                and self.model.ready
                and bool(self.flies)
                and self.fly_audio_player is not None
            )
            if should_play == self.fly_audio_playing:
                return
            if should_play:
                self.fly_audio_player.play()
            else:
                if self.fly_audio_player is not None:
                    self.fly_audio_player.stop()
            self.fly_audio_playing = should_play

        def play_pop_sound(self):
            if self.pop_audio_player is None:
                return
            self.pop_audio_player.stop()
            self.pop_audio_player.setPosition(0)
            self.pop_audio_player.play()

        def stop_audio(self):
            for player in (
                self.music_player, self.fly_audio_player, self.pop_audio_player
            ):
                if player is not None:
                    player.stop()
            self.fly_audio_playing = False

        # ----------------------------- geometry / lifecycle

        def screen_context(self):
            g = screen.geometry()
            return dict(
                screen=screen.name(), width=g.width(), height=g.height(),
                scale=screen.devicePixelRatio(), camera=str(args.camera),
                capture_width=args.width,
            )

        def screen_removed(self, removed):
            if removed == screen:
                self.close()

        def invalidate_screen(self, *_):
            if self.screen_context() != self.context:
                self.context = self.screen_context()
                self.model = Calibration()
                self.mode = 'track'
                self.smoother.reset()
                self.point = self.cell = None
                self.flies.clear()
                self.pop_effects.clear()
                self.note = 'Параметры экрана изменились. Нужна калибровка C.'
                self.showFullScreen()

        def play_area(self):
            left, top, right, bottom = 18.0, 76.0, 18.0, 72.0
            return QRectF(
                left, top,
                max(1.0, self.width() - left - right),
                max(1.0, self.height() - top - bottom),
            )

        def gaze_to_widget(self, point):
            if point is None:
                return None
            x, y = np.clip(np.asarray(point, dtype=float), 0.0, 1.0)
            max_x = max(0.0, float(self.width() - 1))
            max_y = max(0.0, float(self.height() - 1))
            px = x * (self.width() - 1)
            py = y * (self.height() - 1)

            margin = RETICLE_RADIUS + 10

            px = np.clip(px, margin, self.width() - 1 - margin)
            py = np.clip(py, margin, self.height() - 1 - margin)
            return QPointF(px, py)

        # ----------------------------- calibration stages

        def begin(self, validation=False, auto=False):
            if self.paused:
                return
            if validation and not self.model.ready:
                return

            self.auto = auto and not validation
            self.previous_model = self.model
            self.mode = 'validate' if validation else 'calibrate'
            if validation:
                self.targets = [(.22, .22), (.78, .78), (.5, .5), (.78, .22), (.22, .78)]
            else:
                self.targets = [
                    (.5, .5), (.1, .1), (.9, .9), (.9, .1), (.1, .9),
                    (.5, .1), (.5, .9), (.1, .5), (.9, .5),
                ]
                self.model = Calibration()

            self.target_index = self.clicks = 0
            self.centers, self.samples = [], []
            self.recent.clear()
            self.stage_started = time.monotonic()
            self.smoother.reset()
            self.point = self.cell = None
            self.flies.clear()
            self.pop_effects.clear()
            self.showFullScreen()

        def end_stage(self):
            self.mode = 'track'
            self.showFullScreen()
            self.smoother.reset()
            self.point = self.cell = None
            self.flies.clear()
            self.pop_effects.clear()
            self.game_started = time.monotonic()
            self.next_fly_at = self.game_started + self.spawn_interval(self.game_started)

        def finish_calibration(self):
            try:
                self.model.fit(select=True)
            except ValueError as e:
                self.model = self.previous_model
                self.note = f'{e}. C — повторить.'
                self.end_stage()
                return

            cv, by = self.model.cv_error, self.model.cv_by_boost
            self.note = (
                f'Калибровка: {len(self.model)} образцов'
                + (
                    f', ошибка на отложенных областях ≈ {cv * 100:.1f}% '
                    f'(пиксели {by[0.] * 100:.1f}%, вес радужки {self.model.boost:g})'
                    if cv is not None else ''
                )
            )
            self.begin(validation=True)

        def finish_validation(self):
            guesses = [self.model.predict(v) for v, _ in self.centers]
            valid = [(g, t) for g, (_, t) in zip(guesses, self.centers) if g is not None]
            if len(valid) < .8 * len(guesses):
                self.note = 'Проверка не пройдена: выход за диапазон. C — повторить калибровку.'
            else:
                per_target = {}
                for g, t in valid:
                    per_target.setdefault(tuple(t), []).append(g)
                hit = sum(
                    region(np.median(gs, axis=0)) == region(t)
                    for t, gs in per_target.items()
                )
                error = np.mean([
                    np.linalg.norm(np.asarray(g) - np.asarray(t))
                    for g, t in valid
                ])
                self.note = (
                    f'Проверка: {hit}/{len(per_target)} областей; '
                    f'средняя ошибка кадра {error * 100:.1f}% экрана. '
                    'C — повторить, V — проверить.'
                )

                for v, t in self.centers:
                    self.model.add(v, t, 1.)
                self.refit()
                try:
                    self.model.save(self.profile, self.context)
                except OSError as e:
                    self.note += f' Профиль не сохранён: {e}'
            self.end_stage()

        def refit(self):
            try:
                self.model.fit(select=False)
            except ValueError:
                pass
            self.dirty, self.last_fit = False, time.monotonic()

        # ----------------------------- calibration input

        def recent_frames(self, now):
            return [features for stamp, features in self.recent if now - stamp <= .8]

        def screen_point(self, global_pos):
            g = screen.geometry()
            x = (global_pos.x() - g.x()) / max(g.width() - 1, 1)
            y = (global_pos.y() - g.y()) / max(g.height() - 1, 1)
            return (x, y) if 0 <= x <= 1 and 0 <= y <= 1 else None

        def mousePressEvent(self, event):
            if event.button() != Qt.MouseButton.LeftButton or self.paused:
                return

            now = time.monotonic()
            frames = self.recent_frames(now)
            if self.mode == 'calibrate' and not self.auto:
                tx, ty = self.targets[self.target_index]
                pos = event.position()
                if math.hypot(
                    pos.x() - tx * self.width(),
                    pos.y() - ty * self.height(),
                ) > 70:
                    return
                if len(frames) < 2:
                    self.status = 'Нет надёжного сигнала: клик не засчитан'
                    return
                for features in frames:
                    self.model.add(features, (tx, ty), 1.)
                self.clicks += 1
                if self.clicks >= 5:
                    self.clicks = 0
                    self.target_index += 1
                    self.recent.clear()
                    if self.target_index == len(self.targets):
                        self.finish_calibration()
                self.update()

            elif self.mode == 'track':
                target = self.screen_point(event.globalPosition())
                if target is None or len(frames) < 2 or not self.model.ready:
                    return
                for features in frames:
                    self.model.add(features, target, 2.)
                self.refit()
                self.note = f'Дообучение по клику: {len(self.model)} образцов.'

        def mouseMoveEvent(self, event):
            if not self.learn_moves or self.mode != 'track' or self.paused or not self.model.ready:
                return
            now = time.monotonic()
            if now - self.last_move < .25:
                return
            frames = self.recent_frames(now)
            target = self.screen_point(event.globalPosition())
            if target is None or not frames:
                return
            self.last_move = now
            self.model.add(frames[-1], target, .5)
            self.dirty = True

        # ----------------------------- fly game logic

        def spawn_interval(self, now):
            """Linearly shorten the spawn interval as game time accumulates."""
            elapsed = max(0.0, now - self.game_started)
            progress = min(1.0, elapsed / FLY_SPAWN_RAMP_SECONDS)
            return FLY_SPAWN_INITIAL + (FLY_SPAWN_MIN - FLY_SPAWN_INITIAL) * progress

        def fly_position(self, fly):
            return QPointF(fly['x'], fly['y'])

        def spawn_fly(self, now):
            area = self.play_area()
            margin = FLY_HIT_RADIUS + 18.0
            min_x, max_x = area.left() + margin, area.right() - margin
            min_y, max_y = area.top() + margin, area.bottom() - margin
            if max_x <= min_x or max_y <= min_y:
                return False

            gaze_pos = self.gaze_to_widget(self.point)
            x = y = 0.0
            for _ in range(16):
                x = random.uniform(min_x, max_x)
                y = random.uniform(min_y, max_y)
                if gaze_pos is None or math.hypot(x - gaze_pos.x(), y - gaze_pos.y()) > 110.0:
                    break

            heading = random.uniform(0.0, math.tau)
            trail = deque(maxlen=FLY_TRAIL_SAMPLES)
            trail.append((x, y, now))
            self.flies.append({
                'x': x,
                'y': y,
                'heading': heading,
                'speed': FLY_SPEED,
                'last_update': now,
                'next_turn_at': now + random.uniform(FLY_TURN_MIN, FLY_TURN_MAX),
                'trail': trail,
                'phase': random.uniform(0.0, math.tau),
            })
            return True

        def update_fly_motion(self, fly, now):
            """Integrate piecewise-random motion with constant speed."""
            cursor = fly['last_update']
            if now <= cursor:
                return

            area = self.play_area()
            margin = FLY_HIT_RADIUS + 18.0
            min_x, max_x = area.left() + margin, area.right() - margin
            min_y, max_y = area.top() + margin, area.bottom() - margin
            while cursor < now:
                if fly['next_turn_at'] <= cursor:
                    fly['heading'] = (
                        fly['heading'] + random.uniform(-1.35, 1.35)
                    ) % math.tau
                    fly['next_turn_at'] = cursor + random.uniform(
                        FLY_TURN_MIN, FLY_TURN_MAX
                    )

                segment_end = min(now, fly['next_turn_at'])
                dt = segment_end - cursor
                if dt > 0.0:
                    distance = fly['speed'] * dt
                    x = fly['x'] + math.cos(fly['heading']) * distance
                    y = fly['y'] + math.sin(fly['heading']) * distance
                    if x < min_x:
                        x = min_x + (min_x - x)
                        fly['heading'] = math.pi - fly['heading']
                    elif x > max_x:
                        x = max_x - (x - max_x)
                        fly['heading'] = math.pi - fly['heading']
                    if y < min_y:
                        y = min_y + (min_y - y)
                        fly['heading'] = -fly['heading']
                    elif y > max_y:
                        y = max_y - (y - max_y)
                        fly['heading'] = -fly['heading']
                    fly['x'] = min(max_x, max(min_x, x))
                    fly['y'] = min(max_y, max(min_y, y))
                    fly['heading'] %= math.tau

                cursor = segment_end
                if cursor >= fly['next_turn_at']:
                    fly['heading'] = (
                        fly['heading'] + random.uniform(-1.35, 1.35)
                    ) % math.tau
                    fly['next_turn_at'] = cursor + random.uniform(
                        FLY_TURN_MIN, FLY_TURN_MAX
                    )

            fly['last_update'] = now
            fly['trail'].append((fly['x'], fly['y'], now))

        def update_fly_game(self, now):
            self.pop_effects = [
                effect for effect in self.pop_effects
                if now - effect['started'] < FLY_TRAIL_DURATION
            ]

            if self.mode != 'track' or self.paused or not self.model.ready:
                return

            if now >= self.next_fly_at:
                if len(self.flies) < FLY_MAX_ACTIVE:
                    self.spawn_fly(now)
                self.next_fly_at = now + self.spawn_interval(now)

            gaze_pos = self.gaze_to_widget(self.point)
            hit_flies = []
            for fly in self.flies:
                self.update_fly_motion(fly, now)
                if gaze_pos is None:
                    continue

                fly_pos = self.fly_position(fly)
                distance = math.hypot(
                    gaze_pos.x() - fly_pos.x(),
                    gaze_pos.y() - fly_pos.y(),
                )

                if distance <= FLY_HIT_RADIUS:
                    hit_flies.append(fly)
                    self.pop_effects.append({
                        'x': fly_pos.x(),
                        'y': fly_pos.y(),
                        'started': now,
                        'phase': fly['phase'],
                        'trail': list(fly['trail']),
                    })

            if hit_flies:
                hit_ids = {id(fly) for fly in hit_flies}
                self.flies = [fly for fly in self.flies if id(fly) not in hit_ids]
                self.score += len(hit_flies)
                self.play_pop_sound()
                self.note = f'Попадание! Счёт: {self.score}. Продолжайте ловить мух взглядом.'

        # ----------------------------- keyboard / timer

        def keyPressEvent(self, event):
            key = event.key()
            if key == Qt.Key.Key_Escape:
                if self.mode != 'track':
                    self.model = self.previous_model
                    self.mode = 'track'
                    self.smoother.reset()
                    self.showFullScreen()
                    self.note = 'Калибровка/проверка отменена. C — начать заново.'
                elif self.isFullScreen():
                    self.showNormal()
                else:
                    self.close()
            elif key == Qt.Key.Key_C:
                self.begin()
            elif key == Qt.Key.Key_A:
                self.begin(auto=True)
            elif key == Qt.Key.Key_V:
                self.begin(validation=True)
            elif key in (Qt.Key.Key_BracketLeft, Qt.Key.Key_BracketRight) and self.mode == 'track':
                step = .005 if key == Qt.Key.Key_BracketRight else -.005
                self.smoother.lock = max(0., round(self.smoother.lock + step, 3))
                self.note = (
                    f'Мёртвая зона указателя: {self.smoother.lock * 100:.1f}% экрана '
                    '([ меньше, ] больше)'
                )
            elif key == Qt.Key.Key_M and self.mode == 'track':
                self.learn_moves = not self.learn_moves
                self.note = 'Обучение по движению мыши ' + ('включено' if self.learn_moves else 'выключено')
            elif key == Qt.Key.Key_F11 and self.mode == 'track':
                self.showNormal() if self.isFullScreen() else self.showFullScreen()
            elif key == Qt.Key.Key_Space and self.mode == 'track':
                self.paused = not self.paused
                camera.enabled.clear() if self.paused else camera.enabled.set()
                self.smoother.reset()
                self.point = self.cell = None
                now = time.monotonic()
                if self.paused:
                    self.pause_started = now
                else:
                    if self.pause_started is not None:
                        paused_for = now - self.pause_started
                        self.game_started += paused_for
                        self.next_fly_at += paused_for
                        for fly in self.flies:
                            fly['last_update'] = now
                            fly['next_turn_at'] += paused_for
                    self.pause_started = None

        def tick(self):
            now = time.monotonic()
            sample = camera.snapshot()
            fresh = (
                now - sample.time < .65
                and sample.features is not None
                and not sample.error
            )

            if sample.error:
                self.status = sample.error
            elif fresh:
                self.status = 'Глаза видны'
            elif now - sample.time >= .65:
                self.status = 'Нет надёжного сигнала: кадры не приходят (камера/инференс слишком медленные)'
            else:
                self.status = f'Нет сигнала: {sample.reason or "неизвестно"}'

            if self.paused:
                self.status = 'Пауза (камера открыта, анализ остановлен)'
                fresh = False

            if not fresh:
                self.smoother.reset()
                self.point = self.cell = None

            new = fresh and sample.sequence != self.sequence
            if new:
                self.recent.append((now, sample.features))

            if self.mode == 'validate' or (self.mode == 'calibrate' and self.auto):
                calibrating = self.mode == 'calibrate'
                elapsed = now - self.stage_started
                if not fresh:
                    self.stage_started = now
                    self.samples = []
                elif new and elapsed >= .9:
                    self.samples.append(sample.features)

                required_time = 3.5 if calibrating else 1.5
                required_samples = 70 if calibrating else 20
                if fresh and elapsed >= required_time and len(self.samples) >= required_samples:
                    target = self.targets[self.target_index]
                    if calibrating:
                        for features in self.samples:
                            self.model.add(features, target, 1.)
                    else:
                        self.centers.extend((features, target) for features in self.samples)
                    self.samples = []
                    self.target_index += 1
                    self.stage_started = now
                    if self.target_index == len(self.targets):
                        (self.finish_calibration if calibrating else self.finish_validation)()

            elif self.mode == 'track':
                if new and self.model.ready:
                    predicted = self.model.predict(sample.features)
                    self.point, self.cell = self.smoother.update(predicted, now)
                    if predicted is None:
                        self.status = 'Вне диапазона калибровки — верните голову или нажмите C'
                elif new and not self.model.ready:
                    self.status = 'Нет модели: нажмите C'

                if self.dirty and now - self.last_fit > 1.:
                    self.refit()

                self.update_fly_game(now)
            else:
                self.update_fly_game(now)

            self.sync_fly_audio()
            self.sequence = sample.sequence
            if now - self.last_write >= .1:
                self.emit_state(now, sample.fps)
                self.last_write = now
            self.update()

        # ----------------------------- state output

        def emit_state(self, now, fps):
            valid = self.mode == 'track' and not self.paused and self.point is not None
            g = screen.geometry()
            point = self.point if valid else None
            display_pos = self.gaze_to_widget(point) if point is not None else None
            data = dict(
                schema=1,
                timestamp=time.time(),
                monotonic=now,
                valid=valid,
                status=self.status,
                mode=self.mode,
                screen=screen.name(),
                coordinate_space='selected-screen-logical',
                screen_size=[g.width(), g.height()],
                fps=round(fps, 1),
                normalized=None if point is None else point.tolist(),
                logical=None if display_pos is None else [
                    float(display_pos.x()), float(display_pos.y())
                ],
                cell=None if not valid or self.cell is None else list(self.cell),
                grid=[3, 3],
                fly_score=self.score,
            )
            if args.output:
                try:
                    atomic_json(args.output, data)
                except OSError as e:
                    self.note = f'Не удалось записать JSON: {e}'
            if args.stdout:
                try:
                    print(json.dumps(data, ensure_ascii=False, allow_nan=False), flush=True)
                except BrokenPipeError:
                    self.close()

        # ----------------------------- drawing helpers

        @staticmethod
        def draw_bullseye(painter, center, radius=25.0, color='#6cf0bc'):
            painter.save()
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(QColor(color), 2.2))
            painter.drawEllipse(center, radius, radius)
            painter.drawEllipse(center, radius * .62, radius * .62)
            painter.drawEllipse(center, radius * .23, radius * .23)
            gap = radius * .34
            outer = radius + 9.0
            painter.drawLine(QPointF(center.x() - outer, center.y()),
                             QPointF(center.x() - gap, center.y()))
            painter.drawLine(QPointF(center.x() + gap, center.y()),
                             QPointF(center.x() + outer, center.y()))
            painter.drawLine(QPointF(center.x(), center.y() - outer),
                             QPointF(center.x(), center.y() - gap))
            painter.drawLine(QPointF(center.x(), center.y() + gap),
                             QPointF(center.x(), center.y() + outer))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(color))
            painter.drawEllipse(center, 2.4, 2.4)
            painter.restore()

        def draw_fly_trail(self, painter, trail, now):
            pass

        def draw_fly(self, painter, fly, now):
            self.draw_fly_trail(painter, fly['trail'], now)
            center = self.fly_position(fly)
            if self.fly_pixmap.isNull():
                return

            pixmap = self.fly_pixmap
            painter.save()
            painter.translate(center)
            painter.rotate(
                math.degrees(fly['heading']) + 90
            )
            painter.drawPixmap(
                QPointF(
                    -pixmap.width() / 2,
                    -pixmap.height() / 2
                ),
                pixmap
            )

            painter.restore()

        def draw_pop_effects(self, painter, now):
            for effect in self.pop_effects:
                self.draw_fly_trail(painter, effect['trail'], now)
                elapsed = now - effect['started']
                if elapsed >= POP_DURATION:
                    continue
                progress = min(1.0, max(0.0, elapsed / POP_DURATION))
                alpha = int(255 * (1.0 - progress))
                radius = 7.0 + progress * 28.0
                center = QPointF(effect['x'], effect['y'])

                painter.save()
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.setPen(QPen(QColor(255, 70, 80, alpha), 2.8))
                painter.drawEllipse(center, radius, radius)
                painter.setPen(QPen(QColor(255, 225, 225, alpha), 2.0))
                for i in range(8):
                    angle = effect['phase'] + i * math.tau / 8.0
                    inner = radius * .55
                    outer = radius + 8.0
                    painter.drawLine(
                        QPointF(center.x() + math.cos(angle) * inner,
                                center.y() + math.sin(angle) * inner),
                        QPointF(center.x() + math.cos(angle) * outer,
                                center.y() + math.sin(angle) * outer),
                    )
                painter.restore()

        def paintEvent(self, _):
            now = time.monotonic()
            painter = QPainter(self)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            painter.fillRect(self.rect(), QColor('#10151e'))

            if self.mode in ('calibrate', 'validate'):
                target = self.targets[self.target_index]
                center = QPointF(target[0] * self.width(), target[1] * self.height())
                ring = QColor('#6cf0bc')
                if self.mode == 'calibrate':
                    fill = QColor('#6cf0bc')
                    fill.setAlpha(int(255 * self.clicks / CLICKS_PER_DOT))
                    painter.setBrush(fill)
                else:
                    painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.setPen(QPen(ring, 3))
                painter.drawEllipse(center, 24, 24)
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QColor('#ffffff'))
                painter.drawEllipse(center, 5, 5)
                painter.setPen(QColor('#edf3fb'))
                if self.mode == 'calibrate' and self.auto:
                    text = (f'Автокалибровка {self.target_index+1}/{len(self.targets)} · смотрите на точку и '
                            f'медленно двигайте головой\n{self.status} · образцов {len(self.samples)}/{AUTO_MIN} · Esc — отмена')
                elif self.mode == 'calibrate':
                    text = (f'Калибровка {self.target_index+1}/{len(self.targets)} · смотрите на круг и кликайте '
                            f'по нему ({self.clicks}/{CLICKS_PER_DOT})\n'
                            f'Между кликами слегка поворачивайте голову · {self.status} · Esc — отмена')
                else:
                    text = (f'Проверка {self.target_index+1}/{len(self.targets)} · Смотрите в центр круга, не кликайте\n'
                            f'{self.status} · образцов {len(self.samples)}/20 · Esc — отмена')
                painter.drawText(QRectF(30, self.height()*.32, self.width()-60, self.height()*.10),
                           Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap, text)
                return

            area = self.play_area()
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor('#141e2a'))
            painter.drawRoundedRect(area, 18, 18)

            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(QColor('#2d4051'), 1.2))
            painter.drawRoundedRect(area, 18, 18)

            for fly in self.flies:
                self.draw_fly(painter, fly, now)
            self.draw_pop_effects(painter, now)

            painter.setPen(QColor('#edf3fb'))
            painter.setFont(QFont('Sans Serif', 20, QFont.Weight.Bold))
            painter.drawText(24, 38, 'EyeMonitor')
            painter.setFont(QFont('Sans Serif', 12, QFont.Weight.Bold))
            painter.setPen(QColor('#ffd166'))
            interval = self.spawn_interval(now)
            painter.drawText(
                self.width() - 285, 37,
                f'Мухи: {self.score} · интервал: {interval:.1f} с',
            )

            painter.setFont(QFont('Sans Serif', 10))
            painter.setPen(QColor('#a5b7ce'))
            moves = 'вкл' if self.learn_moves else 'выкл'
            painter.drawText(
                24, 61,
                f'{self.status}  ·  M — обучение мышью ({moves})  ·  C — калибровка  ·  V — проверка',
            )
            painter.setPen(QColor('#a5b7ce'))
            painter.drawText(
                QRectF(24, self.height() - 58, self.width() - 48, 48),
                Qt.TextFlag.TextWordWrap,
                self.note + '  ·  Прицел попадает в муху — она лопается сразу.',
            )

            gaze_pos = self.gaze_to_widget(self.point)
            if gaze_pos is not None:
                self.draw_bullseye(
                    painter, gaze_pos, radius=RETICLE_RADIUS, color='#7cf7d2'
                )

        def closeEvent(self, event):
            self.timer.stop()
            self.stop_audio()
            self.point = self.cell = None
            if self.mode == 'track' and self.model.ready:
                try:
                    self.model.save(self.profile, self.context)
                except OSError:
                    pass
            self.mode = 'stopped'
            self.status = 'Остановлено'
            self.emit_state(time.monotonic(), 0)
            camera.stop_event.set()
            event.accept()

    return Window()


if __name__ == '__main__':
    raise SystemExit(main())
