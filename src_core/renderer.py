"""
A renderer with a common interface to communicate with.
Runs a render script.

The script can have the following functions:

- on_init(v)  (optional)
- on_frame(v)  (required)

"""
import colorsys
import math
import os
import random
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import PIL
import pygame
import yappi
from PIL import Image
from skimage.util import random_noise
from yachalk import chalk

import user_conf
from jargs import args, get_discore_session
from src_core import core
from src_core.classes import paths
from src_core.classes.logs import logdiscore_err
from src_core.classes.paths import get_script_file_path, parse_action_script
from src_core.classes.printlib import cpuprofile, pct, trace, trace_decorator
from src_core.classes.Session import Session
from src_core.hud import clear_hud, draw_hud, hud, hud_rows, save_hud
from src_core.rendervars import RenderVars


# TODO support a pygame or moderngl window to render to
# TODO support ryusig

callbacks = []

initialized = False
current_script_name = ''
current_script_path = ''
script = None
v = RenderVars()
script_time_cache = {}
start_f = 0
n_rendered = 0
current_session: None | Session = None
current_hud_pil = None
current_dev = False
last_frame_prompt = ""
last_frame_dt = 0.01
last_frame_hudrows = []
frame_changed = False
request_script_check = False
request_stop = False
request_render = False
request_seek = []
detect_script_every = -1
paused = False
is_rendering = False
enable_saving = True
was_paused = False
enable_hud = False


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

    return check_dir(paths.scripts) or check_dir(current_session.dirpath)


def load_script(name=None):
    import importlib
    global current_script_name
    global current_script_path

    callbacks.clear()

    # Reload all modules in the scripts folder
    for m in [x for x in sys.modules if x.startswith('scripts.')]:
        importlib.reload(sys.modules[m])

    global script
    with trace('renderer.load_script'):
        current_script_name = current_script_name or name
        if current_script_name is None:
            a, sc = parse_action_script(args.action)
            current_script_name = sc

        if current_script_name is not None:
            fpath = get_script_file_path(current_script_name)
        else:
            fpath = current_session.res('script.py')
            paths.touch(fpath)

        current_script_path = fpath

        mpath = paths.get_script_module_path(fpath)

        oldglobals = None
        if script is not None:
            oldglobals = script.__dict__.copy()

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


def safe_call(func, *kargs, failsleep=0.0, **kwargs):
    if func is None: return
    import time
    try:
        with trace(f"safe_call({func.__name__ if hasattr(func, '__name__') else func.__class__.__name__})"):
            func(*kargs, **kwargs)
        return True
    except Exception as e:
        # Print the full stacktrace
        import traceback
        traceback.print_exc()
        logdiscore_err(e)
        time.sleep(failsleep)
        return False


# dst = main * r * 255

@trace_decorator
def emit(name):
    for cb in callbacks:
        cb(v, name)


def emit_register(cb):
    callbacks.append(cb)


def pygame_loop():
    global last_frame_dt
    global request_script_check, request_stop
    global current_session, current_hud_pil, current_dev
    global paused, frame_changed
    global request_render
    global enable_hud

    import pygame
    pygame.init()
    clock = pygame.time.Clock()
    screen = pygame.display.set_mode((v.w, v.h))
    pygame.display.set_caption("Discore Renderer")
    pygame.key.set_repeat(175, 50)

    pilsurface = None

    fontsize = 15
    font = pygame.font.Font((paths.plug_res / 'vt323.ttf').as_posix(), fontsize)

    def upfont(param):
        nonlocal font
        nonlocal fontsize
        fontsize += param
        font = pygame.font.Font((paths.plug_res / 'vt323.ttf').as_posix(), fontsize)

    def draw_text(s, x, y, col=(255, 255, 255), origin=(0, 0)):
        if s is None: return
        size = font.size(s)
        x -= size[0] * origin[0]
        y -= size[1] * origin[1]

        # Shadow
        text = font.render(s, False, (0, 0, 0))
        screen.blit(text, (x + -1, y + 0))
        screen.blit(text, (x + -1, y + -1))
        screen.blit(text, (x + 1, y + 1))
        screen.blit(text, (x + 0, y + -1))
        #
        text = font.render(s, False, col)
        screen.blit(text, (x, y))

    global detect_script_every

    f_displayed = None

    def pause_seek(f_target):
        global paused, frame_changed
        global request_seek
        if is_rendering: return

        paused = True
        request_seek.append(f_target)

    def start_rendering(mode):
        global paused
        global frame_changed, request_render
        if is_rendering: return
        pause_seek(current_session.f_last)
        pause_seek(current_session.f_last + 1)
        paused = True
        frame_changed = True
        request_render = mode

    def pygame_update():
        global enable_hud
        global last_frame_dt
        global request_script_check, request_stop
        global current_session, current_hud_pil, current_dev
        global paused, frame_changed
        global request_render
        global detect_script_every

        nonlocal f_displayed
        nonlocal pilsurface

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                request_stop = True
                return

            elif event.type == pygame.WINDOWFOCUSGAINED:
                request_script_check = True
                detect_script_every = -1
                continue

            elif event.type == pygame.WINDOWFOCUSLOST:
                detect_script_every = 1

            elif event.type == pygame.DROPFILE:
                paused = True
                current_session = Session(event.file)
                current_session.seek_min()

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    request_stop = True
                    return

                if event.key == pygame.K_SPACE:
                    if is_rendering and request_render == 'toggle':
                        request_render = False
                    elif current_session.f == current_session.f_last + 1:
                        start_rendering('toggle')
                    else:
                        paused = not paused
                if event.key == pygame.K_LEFT: pause_seek(current_session.f - 1)
                if event.key == pygame.K_RIGHT: pause_seek(current_session.f + 1)
                if event.key == pygame.K_UP: pause_seek(current_session.f - 10)
                if event.key == pygame.K_DOWN: pause_seek(current_session.f + 10)
                if event.key == pygame.K_PAGEUP: pause_seek(current_session.f - 100)
                if event.key == pygame.K_PAGEDOWN: pause_seek(current_session.f + 100)
                if event.key == pygame.K_HOME: pause_seek(current_session.f_first)
                if event.key == pygame.K_h: pause_seek(current_session.f_first)
                if event.key == pygame.K_0: pause_seek(current_session.f_first)
                if event.key == pygame.K_n or event.key == pygame.K_END:
                    pause_seek(current_session.f_last)
                    pause_seek(current_session.f_last + 1)

                if event.key == pygame.K_RETURN:
                    if is_rendering and request_render:
                        request_render = False

                if event.key == pygame.K_PERIOD:
                    upfont(1)
                if event.key == pygame.K_COMMA:
                    upfont(-1)
                if event.key == pygame.K_h:
                    enable_hud = not enable_hud

        dt = last_frame_dt

        changed = frame_changed
        if changed or f_displayed != current_session.f:
            # New frame
            pil = current_session.image
            if pil is None:
                pil = Image.new('RGB', (v.w, v.h), 'black')

            pilsurface = pygame.image.fromstring(pil.tobytes(), pil.size, pil.mode)

            # Create a tinted version of the surface

            frame_changed = False
            f_displayed = current_session.f

            # dt += acc

        if pilsurface is not None:
            surface = pilsurface
            if not current_session.f_exists:
                alpha = int(0.01 * 255)
                shadow = int(0.2 * 255)
                surface = pilsurface.copy()

                # Scrolling hue
                if is_rendering:
                    hue = (time.time() * 100) % 360
                    color = colorsys.hsv_to_rgb(hue / 360, 1, 0.2)
                    color = tuple(int(c * 255) for c in color)
                    color = (*color, alpha)
                    # Tinted surface
                    surface.fill(color, special_flags=pygame.BLEND_RGB_ADD)

                surface.fill((255-shadow, 255-shadow, 255-shadow, 255), special_flags=pygame.BLEND_RGB_MULT)

            screen.blit(surface, (0, 0))

        fps = 1 / dt

        playback_color = (255, 255, 255)
        if paused:
            playback_color = (0, 255, 255)

        # Draw the fps in the upper left corner
        ht = font.get_height() + 2
        if paused:
            draw_text(f"Paused", v.w, 0, playback_color, origin=(1, 0))
        else:
            draw_text(f"{int(fps)} FPS", v.w, 0, playback_color, origin=(1, 0))

        if is_rendering:
            color = (255, 255, 255)
            if request_render == 'toggle':
                color = (255, 0, 255)
                draw_text("-- Rendering --", v.w2, 32, color, origin=(0.5, 0))
            else:
                draw_text("-- Busy --", v.w2, 0, color, origin=(1, 0))

        draw_text(current_session.name, 0, v.h - ht * 1)
        draw_text(current_script_name, 0, v.h - ht * 2, col=(0, 255, 0))

        draw_text(f"{current_session.width}x{current_session.height}", 0, ht * 0)
        draw_text(f"{current_session.fps} fps", 0, ht * 1)
        if current_session.f <= current_session.f_last:
            draw_text(f"{current_session.f} / {current_session.f_last}", 0, ht * 2, playback_color)
        else:
            draw_text(f"{current_session.f}", 0, ht * 2, playback_color)

        # Draw a progress bar under the frame number
        if current_session.f_last > 0:
            progress = current_session.f / current_session.f_last
            thickness = 3

            pygame.draw.rect(screen, (0, 0, 0), (0, 0, v.w, thickness))
            pygame.draw.rect(screen, (255, 255, 255) if not paused else (0, 255, 255), (0, 0, v.w * progress, thickness))

        if is_rendering:
            lagindicator = np.random.randint(0, 255, (8, 8), dtype=np.uint8)

            lagindicator = random_noise(np.zeros((8, 8)), 's&p', amount=0.1)
            lagindicator = (lagindicator * 255).astype(np.uint8)
            lagindicator = np.stack([lagindicator, lagindicator, lagindicator], axis=2)
            lagindicator_pil = Image.fromarray(lagindicator)
            lagindicator_surface = pygame.image.fromstring(lagindicator_pil.tobytes(), lagindicator_pil.size, lagindicator_pil.mode)
            screen.blit(lagindicator_surface, (v.w - 8 - 2, v.h - 8 - 2))

        if enable_hud:
            padding = 12
            h = v.h
            ht = font.size("a")[1]
            x = padding
            y = ht * 4
            for i, row in enumerate(last_frame_hudrows):
                s = row[0]
                color = row[1]
                fragments = s.split('\n')
                for frag in fragments:
                    draw_text(frag, x, y, color)
                    y += ht

        pygame.display.flip()

        clock.tick(current_session.fps)

    while True:
        try:
            pygame_update()
        except InterruptedError:
            pygame.quit()


def pygame_init():
    threading.Thread(target=pygame_loop).start()


def ryusig_loop():
    import numpy as np
    from src_plugins.ryusig_calc.RyusigApp import RyusigApp
    RyusigApp(current_script_path, None, [
        'cam_x',
        'cam_y',
        'cam_z',
        'cam_rx',
        'cam_ry',
        'cam_rz',
        'cfg',
        'chg',
        'hue',
        'mprompt',
        'msectionsus',
        'brightness'
    ], project_variables=dict(
            session=current_session
    ))


def ryusig_init():
    threading.Thread(target=ryusig_loop).start()


def start_mainloop():
    """
    This handles the main renderer on the main thread.

    - Playback / Seeking
    """
    global frame_changed, last_frame_dt, paused

    while not request_stop:
        if initialized:
            update_playback()
            sleep_dt()


def sleep_dt():
    if v.fps:
        time.sleep(1 / v.fps)
    else:
        time.sleep(1 / 24)


def update_playback():
    global was_paused
    global frame_changed, last_frame_dt, paused

    if not paused:
        start_time = time.time()

        current_session.load_f()
        current_session.load_file()
        current_session.f += 1
        frame_changed = True

        last_frame_dt = time.time() - start_time
    frame_exists = current_session.determine_current_frame_exists()
    if not frame_exists and not was_paused:
        paused = True
    was_paused = paused


def render_init(session=None, script_name='', dev=False):
    global initialized
    global current_session, current_script_name
    global current_dev, paused
    global frame_changed
    from src_core import core

    # with cpuprofile():
    current_session = session or get_discore_session()
    v.reset(0, current_session)

    current_script_name = script_name
    with trace('Script loading'):
        from PIL import Image
        safe_call(load_script)
        detect_script_modified()

        safe_call(emit, 'init')

        current_session.width = v.w
        current_session.height = v.h

        # Default to black
        if current_session.image is None:
            current_session.set(Image.new('RGB', (v.w, v.h), (0, 255, 0)))

    core.init(pluginstall=args.install)

    if dev:
        paused = True
        pygame_init()
        if args.ryusig:
            ryusig_init()
        current_session.dev = True
        current_session.seek_min()
    else:
        current_session.seek_new()

    frame_changed = True
    current_dev = dev

    initialized = True
    return current_session


def renderloop(lo=None, hi=math.inf):
    global request_render, request_script_check, request_seek
    global frame_changed
    global is_rendering
    global paused
    global last_frame_dt

    if lo is not None:
        current_session.seek(lo)

    last_script_check = 0

    while current_session.f < hi and not request_stop:
        # Iterate all files recursively in paths.script_dir
        script_check_elapsed = time.time() - last_script_check
        if request_script_check \
                or 0 < detect_script_every < script_check_elapsed \
                or not current_dev:
            if detect_script_modified():
                print(chalk.dim(chalk.magenta("Change detected in scripts, reloading")))
                safe_call(load_script)
            last_script_check = time.time()
            request_script_check = False

        update_playback()
        sleep_dt()

        for seek in request_seek:
            current_session.f = seek

            # Clamping
            if current_session.f_first is not None and current_session.f < current_session.f_first:
                current_session.f = current_session.f_first
            if current_session.f_last is not None and current_session.f > current_session.f_last + 1:
                current_session.f = current_session.f_last + 1

            current_session.load_f()
            current_session.load_file()
            frame_changed = True

            # For some reason current_session.image loads async and everything is fucked up if we don't wait
            time.sleep(0.05)

        request_seek.clear()

        render = request_render == 'now' or request_render == 'toggle'
        if render:
            if request_render == 'now':
                request_render = False

            current_session.seek_new()
            with cpuprofile(args.profile):
                yield current_session.f


def render_frame(f=None, force=1, session=None):
    global v
    global start_f, n_rendered, current_session
    global last_frame_prompt, last_frame_dt, n_rendered
    global current_hud_pil
    global frame_changed
    global is_rendering

    is_rendering = True

    # time_start = time.time()
    # dt = time_start - last_frame_dt
    # fps = 1 / dt if dt > 0 else 0

    # Update state
    session = session or current_session
    f = f or session.f
    f = int(f)

    session.f = f
    current_session = session

    # Prepare the rendervars
    if v.w is None: v.w = session.width
    if v.h is None: v.h = session.height
    v.reset(f, session)
    v.force = force

    # Start the HUD
    ss = f'frame {v.f} | {v.t:.2f}s ----------------------'
    if not user_conf.print_jobs:
        print("")
        print(ss)
    hud(ss)

    clear_hud()

    # Actual rendering with the render script
    # current_session.f = f
    start_f = session.f
    start_img = session.image
    start_time = time.time()
    last_frame_failed = not safe_call(emit, 'frame', failsleep=0.25)
    if last_frame_failed:
        # Restore the frame number
        session.seek(start_f)
        session.set(start_img)
        is_rendering = False
        return

    last_frame_dt = time.time() - start_time

    session.f = f
    if enable_saving:
        session.save()

    session.load_f(f + 1)

    # Finalize the HUD stuff

    v.hud()
    prompt_changed = v.prompt != last_frame_prompt
    hud(p=v.prompt, tcolor=(255, 255, 255) if prompt_changed else (170, 170, 170))
    last_frame_prompt = v.prompt

    last_frame_hudrows.clear()
    last_frame_hudrows.extend(hud_rows)
    if not current_dev:
        save_hud(session, draw_hud(session))

    # Save every... features
    if args.preview_every and n_rendered % args.preview_every == 0:
        # TODO video preview
        session.make_video()
    if args.zip_every and n_rendered % args.zip_every == 0:
        # TODO zip frames
        session.make_archive()

    n_rendered += 1
    frame_changed = True
    is_rendering = False
