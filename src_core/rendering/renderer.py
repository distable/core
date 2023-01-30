"""
A renderer with a common interface to communicate with.
Runs a render script.

The script can have the following functions:

- on_init(v)  (optional)
- on_frame(v)  (required)


Devmode:
    - We will not check for script changes every frame.
"""

import math
import os
import signal
import sys
import time
from pathlib import Path

from yachalk import chalk

import user_conf
from jargs import args, get_discore_session
from src_core.lib.corelib import invoke_safe
from src_core.classes import paths
from src_core.classes.paths import get_script_file_path, parse_action_script
from src_core.classes.printlib import cpuprofile, trace, trace_decorator
from src_core.classes.Session import Session
from src_core.rendering.hud import clear_hud, draw_hud, hud, hud_rows, save_hud
from src_core.rendering.rendervars import RenderVars
from src_plugins.ryusig_calc.AudioPlayback import AudioPlayback

initialized = False
callbacks = []
v = RenderVars()

session: None | Session = None  # Current session
script_name = ''  # Name of the script file
script_path = ''  # Path to the script file
script = None  # The script module
is_dev = False  # Are we in dev mode?
is_cli = False

# Parameters (long)
detect_script_every = -1
enable_save = True  # Enable saving the frames
enable_save_hud = False # Enable saving the HUD frames

# Parameters (short)
paused = False
looping = False  # Enable looping
loop_start = 0  # Frame to loop back to
play_until = 0  # Frame to play until (auto-stop)
seeks = []  # Frames to seek to
request_script_check = False  # Check if the script has been modified
request_pause = False  # Pause the playback/renderer
request_render = False  # Render a frame
request_stop = False  # Stop the whole renderer
audio = AudioPlayback()

# State (short)
start_f = 0  # Frame we started the renderer on
n_rendered = 0  # Number of frames rendered

# State (internal)
is_rendering = False
was_paused = False
last_frame_prompt = ""
last_frame_time = 0
last_frame_dt = 1 / 24
script_time_cache = {}
invalidated = True
elapsed = 0

# Signals
on_frame_changed = []
on_t_changed = []

# Temporary
tmplist = [] # A list for temporary use


# region Emits
@trace_decorator
def emit(name):
    for cb in callbacks:
        cb(v, name)


def emit_register(cb):
    callbacks.append(cb)


# endregion

# region Script
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


# endregion

# region Core functionality

def init(s=None, scriptname='', cli=False):
    global initialized
    global session, script_name
    global is_dev, request_pause
    global invalidated
    global is_cli
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

    if not cli:
        from src_core.rendering import hobo, ryusig

        audio.init(v.wavs, root=session.dirpath)

        request_pause = True
        hobo.init()
        ryusig.ryusig_init(args.ryusig)
        session.seek_min()
    else:
        session.seek_new()

    is_cli = cli
    invalidated = True

    signal.signal(signal.SIGTERM, handle_exit)

    initialized = True
    return session


def handle_exit():
    print("hi")


def loop(lo=None, hi=math.inf):
    global request_render, request_script_check, seeks
    global invalidated, is_rendering, request_stop
    global paused, request_pause, was_paused, last_frame_dt, last_frame_time

    if lo is not None:
        session.seek(lo)

    last_script_check = 0

    while session.f < hi and not request_stop:
        with trace("renderiter"):
            with trace("renderiter.reload_script_check"):
                elapsed = time.time() - last_script_check
                if request_script_check \
                        or elapsed > detect_script_every and detect_script_every > 0\
                        or is_cli:
                    request_script_check = False
                    if detect_script_modified():
                        print(chalk.dim(chalk.magenta("Change detected in scripts, reloading")))
                        invoke_safe(load_script)
                    last_script_check = time.time()

            paused = request_pause

            with trace("renderiter.update_playback"):
                changed = update_playback()
                # sleep_dt()

            just_paused = paused and not was_paused
            just_unpaused = not paused and was_paused

            with trace("renderiter.flush_seeks"):
                changed = flush_seeks(changed, seeks)

            if changed:
                invoke_safe(on_frame_changed, session.f)
                invoke_safe(on_t_changed, session.t)


            with trace("renderiter.audio"):
                if just_unpaused or is_dev and request_render == 'toggle':
                    audio.play(session.t)
                elif just_paused:
                    audio.stop()

            with trace("renderiter.render"):
                render = request_render == 'now' or request_render == 'toggle'
                if render:
                    if request_render == 'now':
                        request_render = False

                    if session.f <= session.f_last:
                        session.seek_new()

                    with cpuprofile(args.profile):
                        yield session.f
                elif changed:
                    require_dry_run = not session.has_frame_data('hud')
                    if require_dry_run:
                        frame(dry=True)

            if paused and not request_render:
                time.sleep(0.1)

        was_paused = paused

    session.save_data()
    request_stop = True


@trace_decorator
def frame(f=None, scalar=1, s=None, dry=False):
    global v
    global start_f, n_rendered, session
    global last_frame_prompt, n_rendered
    global invalidated
    global is_rendering, paused, request_pause
    global request_render

    # The script has requested the maximum frame length
    if v.len > 0 and session.f >= v.len - 1:
        paused = True
        request_render = None
        return

    session.dev = is_dev

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
    restore = last_frame_failed or dry
    if restore:
        # Restore the frame number
        s.seek(start_f)
        s.set(start_img)
        s.disable_jobs = False
    else:
        prompt_changed = v.prompt != last_frame_prompt or start_f == 1

        v.hud()
        hud(p=v.prompt, tcolor=(255, 255, 255) if prompt_changed else (170, 170, 170))

        last_frame_prompt = v.prompt
        s.set_frame_data('prompt_changed', prompt_changed)
        s.set_frame_data('hud', list(hud_rows))

        if enable_save and not is_dev:
            time.sleep(0.05)
            s.save()
            s.save_data()

        if enable_save_hud and not is_dev:
            save_hud(s, draw_hud(s))

        n_rendered += 1

        # Handled by update_playback instead to keep framerate in sync with audio
        if not is_dev:
            s.load_f(f + 1)

    clear_hud()
    is_rendering = False
    invalidated = True
    if last_frame_failed:
        request_pause = True

# endregion

# region Internal
def update_playback():
    global invalidated
    global play_until
    global elapsed, paused
    global last_frame_time, last_frame_dt

    # Update delta time and elapsed
    if not paused:
        if last_frame_time is None:
            last_frame_time = time.time()

        last_frame_dt = time.time() - last_frame_time
        last_frame_time = time.time()

        elapsed += last_frame_dt
    else:
        last_frame_time = None
        elapsed = 0

    #  Update TODO
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
            seek(loop_start)
            paused = False
        else:
            play_until = None
            paused = True

    return changed


def flush_seeks(changed, seeks):
    global invalidated

    tmplist.clear()
    tmplist.extend(seeks)
    for iseek, manual in tmplist:
        session.f = iseek

        # Clamping
        if session.f_first is not None and session.f < session.f_first:
            session.f = session.f_first
        if session.f_last is not None and session.f > session.f_last + 1:
            session.f = session.f_last + 1

        session.load_f()
        session.load_file()
        invalidated = changed = True
        seeks.pop(0)
        # if manual:  # This is to process each frame even if there is multiple inputs buffered from pygame due to lag
        #     break

    seeks.clear()
    return changed


def sleep_dt():
    if v.fps:
        time.sleep(1 / session.fps)
    else:
        time.sleep(1 / 24)


# endregion

# region Controller Commands
def pause(set='toggle'):
    global request_render, request_pause, play_until, looping
    looping = False
    if is_rendering and request_render == 'toggle':
        request_render = False
    elif is_rendering and request_render == False:
        render('toggle')
    elif session.f >= session.f_last + 1:
        render('toggle')
    else:
        request_pause = not request_pause

    if request_pause:
        play_until = 0


def seek(f_target, manual_input=False, pause=True, clamp=True):
    global request_pause, looping, invalidated
    global seeks
    if is_rendering: return

    if clamp:
        if session.f_first is not None and f_target < session.f_first:
            f_target = session.f_first
        if session.f_last is not None and f_target >= session.f_last + 1:
            f_target = session.f_last + 1

    seeks.append((f_target, manual_input))
    request_pause = pause
    looping = False

    if not pause:
        print(f'NON-PAUSING SEEK MAY BE BUGGY')


def seek_t(t_target, manual_input=False, pause=True):
    f_target = int(session.fps * t_target)
    seek(f_target, manual_input, pause)


def render(mode):
    if is_rendering: return
    global paused
    global invalidated, request_render, request_pause

    # seek(session.f_last)
    if session.f < session.f_last + 1:
        seek(session.f_last + 1, clamp=False)
    request_pause = True
    invalidated = True
    request_render = mode


# endregion

def on_audio_playback_start(t):
    global request_pause

    seek_t(t)
    request_pause = False


def on_audio_playback_stop(t_start, t_end):
    global request_pause

    seek_t(t_start)
    request_pause = True


audio.on_playback_start.append(on_audio_playback_start)
audio.on_playback_stop.append(on_audio_playback_stop)
