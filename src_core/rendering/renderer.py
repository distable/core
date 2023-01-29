"""
A renderer with a common interface to communicate with.
Runs a render script.

The script can have the following functions:

- on_init(v)  (optional)
- on_frame(v)  (required)
"""

import math
import os
import sys
import time
from pathlib import Path

from yachalk import chalk

import user_conf
from jargs import args, get_discore_session
from lib.corelib import invoke_safe
# from src_core.rendering.hobo import init
# from src_core.rendering.ryusig import ryusig_init
from src_core.classes import paths
from src_core.classes.logs import logdiscore_err
from src_core.classes.paths import get_script_file_path, parse_action_script
from src_core.classes.printlib import cpuprofile, trace, trace_decorator
from src_core.classes.Session import Session
from src_core.rendering.hud import clear_hud, draw_hud, hud, hud_rows, save_hud
from src_core.rendering.rendervars import RenderVars

initialized = False
callbacks = []
v = RenderVars()

session: None | Session = None  # Current session
script_name = ''  # Name of the script file
script_path = ''  # Path to the script file
devmode = False  # Are we in dev mode?
script = None  # The script module

start_f = 0  # Frame we started the renderer on
n_rendered = 0  # Number of frames rendered

paused = False  # Pause the playback/renderer
looping = False  # Enable looping
loop_start = 0  # Frame to loop back to
play_until = 0  # Frame to play until (auto-stop)
request_seek = []  # Frames to seek to
request_script_check = False  # Check if the script has been modified
request_render = False  # Render a frame
request_stop = False  # Stop the whole renderer
# audio = AudioPlayback()

enable_saving = True  # Enable saving the frames
detect_script_every = -1

is_rendering = False
was_paused = False
last_frame_prompt = ""
last_frame_time = 0
last_frame_dt = 1 / 24
script_time_cache = {}
invalidated = True

elapsed = 0
ltmp = []

def detect_script_modified():
    # TODO this is slow to do every frame
    def check_dir(path):
        modified = False

        # print(f'check_dir({path})')

        # Check all .py recursively
        for file in Path(path).rglob('*.py'):
            # Compare last modified time of the file with the cached time
            # print("Check file:", file.relative_to(path).as_posix())
            key = file.relative_to(path).as_posix()
            is_new = key not in script_time_cache
            if is_new or script_time_cache[key] < file.stat().st_mtime:
                script_time_cache[key] = file.stat().st_mtime
                if not is_new:
                    modified = True
                # print(key, file.stat().st_mtime)

        return modified

    return check_dir(paths.scripts) or check_dir(session.dirpath)

@trace_decorator
def emit(name):
    for cb in callbacks:
        cb(v, name)


def emit_register(cb):
    callbacks.append(cb)


def start_mainloop():
    """
    This handles the main renderer on the main thread.

    - Playback / Seeking
    """
    global invalidated, paused
    global request_stop

    while not request_stop:
        if initialized:
            update_playback()
            sleep_dt()

    request_stop = True


def pause_toggle():
    global request_render, paused, play_until, looping
    looping = False
    if is_rendering and request_render == 'toggle':
        request_render = False
    elif is_rendering and request_render == False:
        start_rendering('toggle')
    elif session.f == session.f_last + 1:
        start_rendering('toggle')
    else:
        paused = not paused

    if paused:
        play_until = 0


def pause_seek(f_target, manual_input=False):
    global paused, looping, invalidated
    global request_seek
    if is_rendering: return

    request_seek.append((f_target, manual_input))
    paused = True
    looping = False


def start_rendering(mode):
    global paused
    global invalidated, request_render
    if is_rendering: return
    pause_seek(session.f_last)
    pause_seek(session.f_last + 1)
    paused = True
    invalidated = True
    request_render = mode

def load_script(name=None):
    import importlib
    global script_name
    global script_path

    callbacks.clear()

    global script
    with trace('renderer.load_script'):
        script_name = script_name or name
        if script_name is None:
            a, sc = parse_action_script(args.action)
            script_name = sc

        if script_name is not None:
            fpath = get_script_file_path(script_name)
        else:
            fpath = session.res('script.py')
            paths.touch(fpath)

        script_path = fpath

        oldglobals = None
        if script is not None:
            oldglobals = script.__dict__.copy()

        # Reload all modules in the scripts folder
        mpath = paths.get_script_module_path(fpath)
        for m in [x for x in sys.modules if x.startswith('scripts.')]:
            if m != mpath:
                importlib.reload(sys.modules[m])

        if os.path.exists(fpath):
            if script is None:
                import importlib

                with trace(f'renderer.load_script.importlib.import_module({mpath})'):
                    script = importlib.import_module(mpath, package='imported_renderer_script')
            else:
                importlib.reload(script)
            # exec(open().read(), globals())
        if script is not None and oldglobals is not None:
            script.__dict__.update(oldglobals)

def render_init(s=None, scriptname='', dev=False):
    global initialized
    global session, script_name
    global devmode, paused
    global invalidated
    from src_core import core

    session = s or get_discore_session()
    v.reset(0, session)

    script_name = scriptname
    with trace('Script loading'):
        from PIL import Image
        invoke_safe(load_script)
        detect_script_modified()

        invoke_safe(emit, 'load')
        invoke_safe(emit, 'init')
        invoke_safe(emit, 'setup')
        invoke_safe(emit, 'prompt')
        invoke_safe(emit, 'ready')
        invoke_safe(emit, 'eval')

        session.width = v.w
        session.height = v.h

        # Default to black
        if session.image is None:
            session.set(Image.new('RGB', (v.w, v.h), (0, 0, 0)))

    core.init(pluginstall=args.install)

    if dev:
        from src_core.rendering import hobo, ryusig

        paused = True
        hobo.init()
        if args.ryusig:
            ryusig.ryusig_init()
        session.dev = True
        session.seek_min()
    else:
        session.seek_new()

    invalidated = True
    devmode = dev

    initialized = True
    return session


def render_loop(lo=None, hi=math.inf):
    global request_render, request_script_check, request_seek
    global invalidated, is_rendering, request_stop
    global paused, last_frame_dt, elapsed, last_frame_time

    if lo is not None:
        session.seek(lo)

    last_script_check = 0

    while session.f < hi and not request_stop:
        with trace("renderiter"):
            with trace("renderiter.script_reload"):
                # Iterate all files recursively in paths.script_dir
                script_check_elapsed = time.time() - last_script_check
                if request_script_check \
                        or 0 < detect_script_every < script_check_elapsed \
                        or not devmode:
                    if detect_script_modified():
                        print(chalk.dim(chalk.magenta("Change detected in scripts, reloading")))
                        invoke_safe(load_script)
                    last_script_check = time.time()
                    request_script_check = False

            with trace("renderiter.playback"):
                if not paused:
                    if last_frame_time is None:
                        last_frame_time = time.time()

                    last_frame_dt = time.time() - last_frame_time
                    last_frame_time = time.time()

                    elapsed += last_frame_dt
                else:
                    last_frame_time = None
                    elapsed = 0

                changed = update_playback()
                # sleep_dt()

            with trace("renderiter.seeking"):
                ltmp.clear()
                ltmp.extend(request_seek)
                for iseek, manual in ltmp:
                    session.f = iseek

                    # Clamping
                    if session.f_first is not None and session.f < session.f_first:
                        session.f = session.f_first
                    if session.f_last is not None and session.f > session.f_last + 1:
                        session.f = session.f_last + 1

                    session.load_f()
                    session.load_file()
                    invalidated = changed = True
                    request_seek.pop(0)
                    if manual:  # This is to process each frame even if there is multiple inputs buffered from pygame due to lag
                        break

                request_seek.clear()

            with trace("renderiter.render"):
                render = request_render == 'now' or request_render == 'toggle'
                if render:
                    # if changed:
                    #     # For some reason current_session.image loads async and everything is fucked up if we don't wait
                    #     time.sleep(0.05)

                    if request_render == 'now':
                        request_render = False

                    session.seek_new()
                    with cpuprofile(args.profile):
                        yield session.f
                elif changed:
                    require_dry_run = not session.has_frame_data('hud')
                    if require_dry_run:
                        render_frame(dry=True)

            # if invalidated:
            #     last_frame_dt = time.time() - start_time

    session.save_data()
    request_stop = True


def update_playback():
    global invalidated
    global play_until, was_paused
    global elapsed, paused

    if not paused:
        changed = False
        while elapsed >= 1 / session.fps:
            session.f += 1
            changed = True
            invalidated = True
            elapsed -= 1 / session.fps
    else:
        changed = False

    if changed:
        session.load_f()
        session.load_file()

    frame_exists = session.determine_current_frame_exists()
    catchedup_end = not frame_exists and not was_paused
    catchedup = play_until and session.f >= play_until
    if catchedup_end or catchedup:
        if looping:
            pause_seek(loop_start)
            paused = False
        else:
            paused = True
            play_until = None

    was_paused = paused
    return changed


def sleep_dt():
    if v.fps:
        time.sleep(1 / session.fps)
    else:
        time.sleep(1 / 24)


@trace_decorator
def render_frame(f=None, scalar=1, s=None, dry=False):
    global v
    global start_f, n_rendered, session
    global last_frame_prompt, n_rendered
    global invalidated
    global is_rendering, paused

    is_rendering = True

    # Update state
    s = s or session
    f = f or s.f
    f = int(f)

    s.f = f
    session = s

    # Prepare the rendervars
    if v.w is None: v.w = s.width
    if v.h is None: v.h = s.height
    v.reset(f, s)
    v.scalar = scalar

    # Start the HUD
    ss = f'frame {v.f} | {v.t:.2f}s ----------------------'
    if not user_conf.print_jobs:
        print("")
        print(ss)
    hud(ss)

    clear_hud()

    # Actual rendering with the render script
    s.disable_jobs = dry
    v.dry = dry

    # current_session.f = f
    start_f = s.f
    start_img = s.image
    last_frame_failed = not invoke_safe(emit, 'frame', failsleep=0.25)
    prompt_changed = v.prompt != last_frame_prompt or start_f == 1

    v.hud()
    hud(p=v.prompt, tcolor=(255, 255, 255) if prompt_changed else (170, 170, 170))

    if not last_frame_failed:
        last_frame_prompt = v.prompt
        s.set_frame_data('prompt_changed', prompt_changed)
        s.set_frame_data('hud', list(hud_rows))
        s.save_data()

    skip_save = last_frame_failed or dry
    if skip_save:
        # Restore the frame number
        s.seek(start_f)
        s.set(start_img)
        s.disable_jobs = False
    else:
        s.f = f
        if enable_saving:
            time.sleep(0.05)
            s.save()
        s.load_f(f + 1)

        # Save every... features
        if args.preview_every and n_rendered % args.preview_every == 0:
            # TODO video preview
            s.make_video()
        if args.zip_every and n_rendered % args.zip_every == 0:
            # TODO zip frames
            s.make_archive()

        # Flush the HUD
        if not devmode:
            save_hud(s, draw_hud(s))

        n_rendered += 1

    clear_hud()
    is_rendering = False
    invalidated = True
    if last_frame_failed:
        paused = True
