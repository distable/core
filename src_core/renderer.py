"""
A renderer with a common interface to communicate with.
Runs a render script.

The script can have the following functions:

- on_init(v)  (optional)
- on_frame(v)  (required)

"""
import math
import os
import random
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import yappi
from yachalk import chalk

from jargs import args, get_discore_session
from src_core import core
from src_core.classes import paths
from src_core.classes.logs import logdiscore_err
from src_core.classes.paths import get_script_file_path, parse_action_script
from src_core.classes.printlib import cpuprofile, pct, trace
from src_core.classes.Session import Session
from src_core.hud import clear_hud, draw_hud, hud, save_hud
from src_core.rendervars import RenderVars


# TODO support a pygame or moderngl window to render to
# TODO support ryusig


current_script_name = ''
script = None
v = RenderVars()
script_time_cache = {}
start_f = 0
n_rendered = 0
current_session: None | Session = None
current_frame = -1
current_hud_pil = None
last_frame_prompt = ""
last_frame_pil = None
last_frame_time = 0.01
frame_changed = False


def detect_script_modified():
    # TODO this is slow to do every frame
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
    import importlib
    global current_script_name

    # Reload all modules in the scripts folder
    for m in [x for x in sys.modules if x.startswith('scripts.')]:
        importlib.reload(sys.modules[m])

    global script
    with trace('renderer.load_script'):
        current_script_name = current_script_name or name
        if current_script_name is None:
            a, sc = parse_action_script(args.action)
            current_script_name = sc

        oldglobals = None
        if script is not None:
            oldglobals = script.__dict__.copy()

        modpath = get_script_file_path(current_script_name)
        modname = 'imported_renderer_script'
        if os.path.exists(modpath):
            if script is None:
                import importlib
                script = importlib.import_module(
                        f'{paths.scripts.name}.{modpath.relative_to(paths.scripts).with_suffix("").as_posix().replace("/", ".")}',
                        package=modname)
            else:
                importlib.reload(script)
            # exec(open().read(), globals())
        if script is not None and oldglobals is not None:
            script.__dict__.update(oldglobals)


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
        logdiscore_err(e)
        time.sleep(failsleep)
        return False


def render_init(session=None, script_name='', dev=False):
    global current_session, current_script_name
    global current_dev
    from src_core import core

    current_session = session or get_discore_session()
    v.reset(0, current_session)

    current_script_name = script_name
    with trace('Script loading'):
        from PIL import Image
        safe_call(load_script)
        safe_call(getattr_safe(script, 'on_init'), v)

        current_session.width = v.w
        current_session.height = v.h
        current_session.set(Image.new('RGB', (v.w, v.h), 'black'))

    core.init(pluginstall=args.install)

    if dev:
        pygame_init()
        current_session.dev = True

    current_dev = dev
    return current_session



def pygame_loop():
    global last_frame_time
    global frame_changed
    import pygame
    pygame.init()
    screen = pygame.display.set_mode((v.w, v.h))
    pygame.display.set_caption("Discore Renderer")

    pil = None
    pilsurface = None
    i = 0
    acc = 0
    fps = 0
    dt = 1

    font = pygame.font.SysFont('Arial', 20)

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        dt = last_frame_time

        if frame_changed and last_frame_pil is not None:
            # New frame
            pil = last_frame_pil
            pilsurface = pygame.image.fromstring(pil.tobytes(), pil.size, pil.mode)
            i += 1
            acc += dt

            frame_changed = False

            # dt += acc

        if pilsurface is not None:
            screen.blit(pilsurface, (0, 0))

        # if acc > 1:
        #     acc -= 1
        #     fps = i
        #     i = 0

        fps = 1 / dt

        # Draw the fps in the upper left corner
        text = font.render(f'FPS: {fps:.2f}', True, (0, 0, 0))
        screen.blit(text, (2, 2))
        # screen.blit(text, (-2, -2))
        text = font.render(f'FPS: {fps:.2f}', True, (255, 255, 255))
        screen.blit(text, (0, 0))

        pygame.display.flip()


def pygame_init():
    threading.Thread(target=pygame_loop).start()


def getattr_safe(obj, attr, default=None):
    if obj is None: return default
    return getattr(obj, attr, default)


def renderloop(lo=None, hi=math.inf):
    global current_frame

    if lo is not None:
        current_session.seek(lo)

    while current_session.f < hi:
        current_frame = current_session.f

        # Iterate all files recursively in paths.script_dir
        # if detect_script_modified():
        #     print(chalk.dim(chalk.magenta("Change detected in scripts, reloading")))
        #     safe_call(load_script)

        with cpuprofile(args.profile):
            yield current_frame


def renderloop_frames(lo=None, hi=None):
    for f in renderloop(lo, hi):
        render_frame()


def render_frame(f=None, force=1, session=None):
    global v
    global start_f, n_rendered, current_session, current_frame
    global last_frame_prompt, last_frame_pil, last_frame_time, n_rendered
    global current_hud_pil
    global frame_changed

    # time_start = time.time()
    # dt = time_start - last_frame_time
    # fps = 1 / dt if dt > 0 else 0

    # Update state
    session = session or current_session
    f = f or current_frame
    f = int(f)

    current_session = session
    current_frame = f

    # Prepare the rendervars
    if v.w is None: v.w = session.width
    if v.h is None: v.h = session.height
    v.reset(f, session)
    v.force = force

    # Start the HUD
    ss = f'frame {v.f} | {v.t:.2f}s ----------------------'
    print("")
    print(ss)
    hud(ss)

    # Actual rendering with the render script
    start_f = session.f
    start_img = session.image
    start_time = time.time()
    last_frame_failed = not safe_call(script.on_frame, v, failsleep=0.25)
    if last_frame_failed:
        # Restore the frame number
        session.seek(start_f)
        session.set(start_img)
        clear_hud()
        return

    last_frame_pil = session.image
    last_frame_time = time.time() - start_time

    if not current_dev:
        session.save()
    session.seek_next()

    # Finalize the HUD stuff
    # v.hud()
    # prompt_changed = v.prompt != last_frame_prompt
    # hud(p=v.prompt, tcolor=(255, 255, 255) if prompt_changed else (170, 170, 170))
    # last_frame_prompt = v.prompt
    # current_hud_pil = draw_hud(session)

    if not current_dev:
        save_hud(session, current_hud_pil)

    # Save every... features
    if args.preview_every and n_rendered % args.preview_every == 0:
        # TODO video preview
        session.make_video()
    if args.zip_every and n_rendered % args.zip_every == 0:
        # TODO zip frames
        session.make_archive()

    n_rendered += 1
    frame_changed = True
