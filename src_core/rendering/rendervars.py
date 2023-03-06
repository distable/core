import random
from dataclasses import dataclass
from math import sqrt

from numpy import ndarray

from src_core.classes.printlib import pct
from src_core.rendering.hud import hud


@dataclass
class SessionVars:
    session: 'Session' = None

    # Allow defining new variables on demand
    def __getattr__(self, key):
        if key not in self.__dict__:
            self.__dict__[key] = 0

    def __setattr__(self, key, value):
        super().__setattr__(key, value)

    @property
    def image(self):
        return self.session.image

    @image.setter
    def image(self, value):
        self.session.set(value)

    @property
    def image_cv2(self):
        return self.session.image_cv2

    @image_cv2.setter
    def image_cv2(self, value):
        self.session.image_cv2 = value

    @property
    def fps(self):
        return self.session.fps

    @fps.setter
    def fps(self, value):
        self.session.fps = value

    @property
    def speed(self):
        # magnitude of x and y
        return sqrt(self.x ** 2 + self.y ** 2)

    @speed.setter
    def speed(self, value):
        # magnitude of x and y
        speed = self.speed
        if speed > 0:
            self.x = abs(self.x) / speed * value
            self.y = abs(self.y) / speed * value


@dataclass
class RenderVars(SessionVars):
    """
    Render variables supported by the renderer
    This provides a common interface for our libraries to use.
    """

    def __init__(self):
        self.len = 0
        self.prompt = ""
        self.negprompt = ""
        self.nprompt = None
        self.w = 640
        self.h = 448
        self.scalar = 0
        self.x = 0
        self.y = 0
        self.z = 0
        self.r = 0
        self.smear = 0
        self.hue = 0
        self.sat = 0
        self.val = 0
        self.steps = 20
        self.d = 1.1
        self.chg = 1
        self.cfg = 16.5
        self.seed = 0
        self.sampler = 'euler-a'
        self.nguide = 0
        self.nsp = 0
        self.nextseed = 0
        self.t = 0
        self.f = 0
        self.w2 = 0
        self.h2 = 0
        self.dt = 1 / 24
        self.ref = 1 / 12 * 24
        self.draft = 1
        self.dry = False
        self.signals = {}

        self.set_defaults()

    def set_defaults(self):
        v = self
        v.x, v.y, v.z, v.r = 0, 0, 0, 0
        v.hue, v.sat, v.val = 0, 0, 0
        v.d, v.chg, v.cfg, v.seed, v.sampler = 1.1, 1.0, 16.5, 0, 'euler-a'
        v.nguide, v.nsp = 0, 0
        v.smear = 0

    def reset(self, f, session):
        v = self
        s = session

        v.dry = False
        v.session = s
        v.nextseed = random.randint(0, 2 ** 32 - 1)
        v.f = int(f)
        v.t = f / v.fps
        if v.w: v.w2 = v.w / 2
        if v.h: v.h2 = v.h / 2
        v.dt = 1 / v.fps
        v.ref = 1 / 12 * v.fps
        v.tr = v.t * v.ref

    def set(self, **kwargs):
        protected_names = ['session', 'signals', 't', 'f', 'dt', 'ref', 'tr', 'w2', 'h2', 'len']
        for name, v in kwargs.items():
            self.__dict__[name] = v
            if isinstance(v, ndarray):
                if name in protected_names:
                    print(f"set_frame_signals: {name} is protected and cannot be set as a signal. Skipping...")
                    continue

                # print(f"SET {name} {v}")
                self.signals[name] = v
                self.__dict__[f'{name}s'] = v

    def set_frame_signals(self):
        dic = self.__dict__.copy()
        for name, value in dic.items():
            # if isinstance(value, ndarray):
            #     self.signals[name] = value

            if name in self.signals:
                signal = self.signals[name]
                try:
                    self.__dict__[name] = signal[self.f]
                    self.__dict__[f'{name}s'] = signal
                except IndexError:
                    print(f'IndexError: {name} {self.f} {len(signal)}')

    def hud(self):
        v = self
        hud(x=v.x, y=v.y, z=pct(v.z), rot=v.r)
        hud(hue=v.hue, sat=v.sat, val=v.val)
        hud(d=v.d, chg=pct(v.chg), cfg=v.cfg, nguide=pct(v.nguide), nsp=pct(v.nsp))
        hud(seed=v.seed, sampler=v.sampler)
        hud(force=v.scalar)
