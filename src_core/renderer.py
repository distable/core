import os
import random
from dataclasses import dataclass
from pathlib import Path

from yachalk import chalk

from jargs import args
from src_core.classes import paths
from src_core.classes.logs import loglaunch_err
from src_core.classes.paths import get_script_file_path, parse_action_script
from src_core.classes.printlib import pct, trace
from src_core.hud import clear_hud, hud, save_hud


# TODO support a pygame or moderngl window to render to
# TODO support ryusig

@dataclass
class RenderVars:
    ctx: 'PipeData' = None
    gs: 'Session' = None
    prompt: str = ""
    negprompt: str = ""
    nprompt = None
    w: int = 640
    h: int = 448
    force: int = 0
    fps: int = 24
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
    chg: float = 0.5
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
    dt: float = 1 / fps
    ref: float = 1 / 12 * fps

    def reset(self, f):
        from src_core import core
        gs = core.gs
        ctx = core.gs.ctx
        v = self

        v.gs = gs
        v.ctx = ctx

        v.x, v.y, v.z, v.r = 0, 0, 0, 0
        v.hue, v.sat, v.val = 0, 0, 0
        v.d, v.chg, v.cfg, v.seed, v.sampler = 1.1, 0.5, 16.5, 0, 'euler-a'
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
        hud(x=v.x, y=v.y, z=pct(v.z), rot=v.r)
        hud(hue=v.hue, sat=v.sat, val=v.val)
        hud(d=v.d, chg=pct(v.chg), cfg=v.cfg, nguide=pct(v.nguide), nsp=pct(v.nsp))
        hud(seed=v.seed, sampler=v.sampler)
        hud(force=v.force)


script = None
v = RenderVars()


# A dataclass with the same properties as above


def safe_call(func, *kargs, failsleep=0.0, **kwargs):
    if func is None: return
    import time
    try:
        func(*kargs, **kwargs)
        return True
    except Exception as e:
        # Print the full stacktrace
        import traceback
        traceback.print_exc()
        loglaunch_err(e)
        time.sleep(failsleep)
        return False


def render_init():
    from src_core import core

    ses = core.gs

    with trace('Script loading'):
        if paths.is_session(args.action):
            with trace('core.open'): core.open(args.action or args.session)
        else:
            safe_call(load_script)
            safe_call(script.on_init, v)

    core.init(pluginstall=args.install)

    # In case the script set the session e.g. for standalone testing
    core.open(ses)
    core.opensub(args.subdir)


def render_frame(f, force=1):
    load_script()
    from src_core import core

    global v
    if v.w is None: v.w = core.ctx.w
    if v.h is None: v.h = core.ctx.h
    v.reset(f)
    v.force = force

    global last_frame_failed
    last_frame_failed = not safe_call(script.on_frame, v, failsleep=0.25)


last_frame_failed = False
script_time_cache = {}


def render_loop(lo=None, hi=None):
    from src_core import core
    from src_plugins.disco_party.globals import fps, max_duration

    if lo is not None:
        core.seek(lo)

    lprompt = ""

    i = 0
    hi = hi if hi is not None else max_duration * fps
    while core.f < hi:
        start_f = core.f

        # Iterate all files recursively in paths.script_dir
        if detect_script_modified():
            print(chalk.dim(chalk.magenta("Change detected in scripts, reloading")))
            safe_call(load_script())

        s = f'frame {v.f} | {v.t:.2f}s ----------------------'
        print(s)
        hud(s)
        yield core.f

        if not last_frame_failed:
            v.hud()

            prompt_changed = v.prompt != lprompt
            hud(p=v.prompt, tcolor=(255, 255, 255) if prompt_changed else (170, 170, 170))
            lprompt = v.prompt

            save_hud()
            i += 1
            if args.preview_every and i % args.preview_every == 0:
                # TODO video preview
                core.gs.make_video()
            if args.zip_every and i % args.zip_every == 0:
                # TODO zip frames
                core.gs.make_archive()
        else:
            # Restore the frame number
            core.gs.f = start_f
            clear_hud()


def detect_script_modified():
    modified = False
    for root, dirs, files in os.walk(paths.scripts):
        for file in files:
            file = Path(file)
            if file.suffix == ".py":
                file = root / file
                # Compare last modified time of the file with the cached time
                key = file.relative_to(paths.scripts).name
                if key not in script_time_cache:
                    modified = True
                elif script_time_cache[key] < file.stat().st_mtime:
                    modified = True
                if modified:
                    script_time_cache[key] = file.stat().st_mtime
    return modified


def load_script(name=None):
    global script
    with trace('renderer.load_script'):
        if name is None:
            a, sc = parse_action_script(args.action)
            name = sc

        oldglobals = None
        if script is not None:
            oldglobals = script.__dict__.copy()

        modpath = get_script_file_path(name)
        modname = 'imported_renderer_script'
        if os.path.exists(modpath):
            if script is None:
                import importlib
                script = importlib.import_module(
                        f'{paths.scripts.name}.{modpath.relative_to(paths.scripts).with_suffix("").as_posix().replace("/", ".")}',
                        package=modname)
            else:
                import importlib
                importlib.reload(script)
            # exec(open().read(), globals())
        if script is not None and oldglobals is not None:
            script.__dict__.update(oldglobals)


# def loopframes(duration):
#     for i in range(duration):
#         core.copyframe(name='1')
#         frame()
#
#
# def video_every(duration_s: int):
#     if core.f % duration_s * fps == 0:
#         core.gs.make_video(fps, bg=True)
