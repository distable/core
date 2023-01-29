import random
from dataclasses import dataclass
from math import sqrt

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
    prompt: str = ""
    negprompt: str = ""
    nprompt = None
    w: int = 640
    h: int = 448
    scalar: int = 0
    x: float = 0
    y: float = 0
    z: float = 0
    r: float = 0
    smear: float = 0
    hue: float = 0
    sat: float = 0
    val: float = 0
    steps: int = 20
    d: float = 1.1
    chg: float = 1
    cfg: float = 16.5
    seed: int = 0
    sampler: str = 'euler-a'
    nguide: int = 0
    nsp: int = 0
    nextseed: int = 0
    t: float = 0
    f: int = 0
    w2: int = 0
    h2: int = 0
    dt: float = 1 / 24
    ref: float = 1 / 12 * 24
    draft: float = 1
    dry:bool = False

    def reset(self, f, session):
        v = self
        s = session

        v.dry = False
        v.session = s

        v.x, v.y, v.z, v.r = 0, 0, 0, 0
        v.hue, v.sat, v.val = 0, 0, 0
        v.d, v.chg, v.cfg, v.seed, v.sampler = 1.1, 1.0, 16.5, 0, 'euler-a'
        v.nguide, v.nsp = 0, 0
        v.smear = 0

        v.nextseed = random.randint(0, 2 ** 32 - 1)
        v.f = int(f)
        v.t = f / v.fps
        if v.w: v.w2 = v.w / 2
        if v.h: v.h2 = v.h / 2
        v.dt = 1 / v.fps
        v.ref = 1 / 12 * v.fps
        v.tr = v.t * v.ref

    def hud(self):
        v = self
        hud(x=v.x, y=v.y, z=pct(v.z), rot=v.r)
        hud(hue=v.hue, sat=v.sat, val=v.val)
        hud(d=v.d, chg=pct(v.chg), cfg=v.cfg, nguide=pct(v.nguide), nsp=pct(v.nsp))
        hud(seed=v.seed, sampler=v.sampler)
        hud(force=v.scalar)
