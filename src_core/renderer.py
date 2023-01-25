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
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.util import random_noise
from yachalk import chalk

import jargs
import user_conf
from jargs import args, get_discore_session
from src_core.classes import paths
from src_core.classes.logs import logdiscore_err
from src_core.classes.paths import get_script_file_path, parse_action_script
from src_core.classes.printlib import cpuprofile, trace, trace_decorator
from src_core.classes.Session import Session
from src_core.hud import clear_hud, draw_hud, hud, hud_rows, save_hud
from src_core.lib import corelib
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
invalidated = True
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
current_segment = -1
colors = corelib.generate_colors(8, v=1, s=0.765)
play_until = 0
looping = False
loop_start = 0


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


def toggle_pause():
    global request_render, paused, play_until, looping
    looping = False
    if is_rendering and request_render == 'toggle':
        request_render = False
    elif is_rendering and request_render == False:
        start_rendering('toggle')
    elif current_session.f == current_session.f_last + 1:
        start_rendering('toggle')
    else:
        paused = not paused

    if paused:
        play_until = 0


def pause_seek(f_target, manual_input=False):
    global paused, looping, invalidated
    global request_seek
    if is_rendering: return

    paused = True
    looping = False
    request_seek.append((f_target, manual_input))


def start_rendering(mode):
    global paused
    global invalidated, request_render
    if is_rendering: return
    pause_seek(current_session.f_last)
    pause_seek(current_session.f_last + 1)
    paused = True
    invalidated = True
    request_render = mode


def create_segment(off):
    global current_segment
    get_segments().append((current_session.f, current_session.f + off))
    current_segment = len(get_segments()) - 1
    current_session.save_data()


def pygame_loop():
    global last_frame_dt
    global request_script_check, request_stop
    global current_session, current_hud_pil, current_dev
    global paused, invalidated
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
    global detect_script_every

    f_displayed = None
    mode = 'main'
    action_page = 0
    actions = list(paths.iter_scripts())

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


    def pygame_update():
        global enable_hud
        global last_frame_dt
        global request_script_check, request_stop
        global current_session, current_hud_pil, current_dev
        global paused, looping, loop_start, invalidated, play_until
        global request_render
        global detect_script_every
        global current_segment

        nonlocal mode, action_page, actions
        nonlocal f_displayed
        nonlocal pilsurface

        w = current_session.w
        h = current_session.h
        f_last = current_session.f_last
        f_first = current_session.f_first
        f = current_session.f

        # pygame.display.update()
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

            if mode == 'main':
                if event.type == pygame.DROPFILE:
                    paused = True
                    current_session = Session(event.file, fixpad=True)
                    current_session.seek_min()
                    get_segments().clear()
                    pygame.display.set_mode((w, h))
                    invalidated = True

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        if is_rendering:
                            request_render = False
                            paused = True
                        return

                    if event.key == pygame.K_SPACE:
                        toggle_pause()
                    if event.key == pygame.K_LEFT: pause_seek(f - 1, True)
                    if event.key == pygame.K_RIGHT: pause_seek(f + 1, True)
                    if event.key == pygame.K_UP: pause_seek(f - v.fps, True)
                    if event.key == pygame.K_DOWN: pause_seek(f + v.fps, True)
                    if event.key == pygame.K_PAGEUP: pause_seek(f - f_last // 15, True)
                    if event.key == pygame.K_PAGEDOWN: pause_seek(f + f_last // 15, True)
                    if event.key == pygame.K_HOME: pause_seek(f_first, True)
                    if event.key == pygame.K_h: pause_seek(f_first, True)
                    if event.key == pygame.K_0: pause_seek(f_first, True)
                    if event.key == pygame.K_n or event.key == pygame.K_END:
                        pause_seek(f_last, True)
                        pause_seek(f_last + 1, True)

                    if event.key == pygame.K_RETURN:
                        if is_rendering and request_render:
                            request_render = False
                        elif is_rendering and not request_render:
                            start_rendering('toggle')
                        else:
                            start_rendering('now')

                    # if event.key == pygame.K_PERIOD:
                    #     upfont(1)
                    # if event.key == pygame.K_COMMA:
                    #     upfont(-1)
                    if event.key == pygame.K_f:
                        enable_hud = not enable_hud
                    if event.key == pygame.K_i:
                        if len(get_segments()) and not f > get_segments()[current_segment][1]:
                            get_segments()[current_segment] = (f, get_segments()[current_segment][1])
                            current_session.save_data()
                        else:
                            create_segment(50)
                    if event.key == pygame.K_o:
                        if len(get_segments()) and not f < get_segments()[current_segment][0]:
                            get_segments()[current_segment] = (get_segments()[current_segment][0], f)
                            current_session.save_data()
                        else:
                            create_segment(-50)
                    if event.key == pygame.K_COMMA:
                        indices = [i for s in get_segments() for i in s]
                        indices.sort()
                        # Find next value in indices that is less than current_session.f
                        for i in range(len(indices) - 1, -1, -1):
                            if indices[i] < f:
                                pause_seek(indices[i])
                                break
                    if event.key == pygame.K_PERIOD:
                        indices = [i for s in get_segments() for i in s]
                        indices.sort()
                        # Find next value in indices that is greater than current_session.f
                        for i in range(len(indices)):
                            if indices[i] > f:
                                pause_seek(indices[i])
                                break

                    if event.key == pygame.K_LEFTBRACKET:
                        current_session.fps = get_fps(current_session.fps, -1)
                    if event.key == pygame.K_RIGHTBRACKET:
                        current_session.fps = get_fps(current_session.fps, 1)
                    if event.key == pygame.K_LESS and len(get_segments()):
                        current_segment = max(0, current_segment - 1)
                        pause_seek(get_segments()[current_segment][0])
                    if event.key == pygame.K_GREATER and len(get_segments()):
                        current_segment = min(len(get_segments()) - 1, current_segment + 1)
                        pause_seek(get_segments()[current_segment][0])
                    if event.key == pygame.K_p:
                        lo, hi = get_segments()[current_segment]
                        current_session.seek(lo)
                        play_until = hi
                        paused = False
                        looping = True
                        looping_start = lo
                    if event.key == pygame.K_w:
                        mode = 'action'
            elif mode == 'action':
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE or event.key == pygame.K_w:
                        mode = 'main'
                    elif event.key == pygame.K_LEFT:
                        action_page = max(0, action_page - 10)
                    elif event.key == pygame.K_RIGHT:
                        action_page += 10
                        max_page = len(actions) // 10
                        action_page = min(action_page, max_page)

                    i = event.key - pygame.K_1
                    action_slice = actions[action_page:action_page + 10]

                    if 1 <= i <= pygame.K_9:
                        name, path = action_slice[i]
                        s = f"discore {current_session.dirpath} {name} "
                        if get_segments():
                            s += f"--frames {segments_to_frames()}"
                        os.popen(s)
                        mode = 'main'

        dt = last_frame_dt

        changed = invalidated
        if changed or f_displayed != f or is_rendering:
            # New frame
            pil = current_session.image
            # if pil is None:
            #     pil = Image.new('RGB', (v.w, v.h), 'black')

            im = current_session.image_cv2
            im = np.swapaxes(im, 0, 1)
            pilsurface = pygame.surfarray.make_surface(im)
            # pilsurface = pygame.image.fromstring(pil.tobytes(), pil.size, pil.mode)

            # Create a tinted version of the surface

            invalidated = False
            f_displayed = f

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

                surface.fill((255 - shadow, 255 - shadow, 255 - shadow, 255), special_flags=pygame.BLEND_RGB_MULT)

            screen.blit(surface, (0, 0))

        fps = 1 / dt

        playback_color = (255, 255, 255)
        if paused:
            playback_color = (0, 255, 255)

        # Draw the fps in the upper left corner
        ht = font.get_height() + 2
        if paused:
            draw_text(f"Paused", w, 0, playback_color, origin=(1, 0))
        else:
            draw_text(f"{int(fps)} FPS", w, 0, playback_color, origin=(1, 0))

        render_progressbar_y = 0
        if is_rendering:
            color = (255, 255, 255)
            if request_render == 'toggle':
                color = (255, 0, 255)
                draw_text("-- Rendering --", v.w2, 6, color, origin=(0.5, 0))
            else:
                draw_text("-- Busy --", v.w2, 0, color, origin=(0.5, 0))

            draw_text(f"{n_rendered} frames", v.w2, 6 + ht, color, origin=(0.5, 0.5))
            draw_text(f"{n_rendered / current_session.fps:.02}s", v.w2, 6 + ht + ht, color, origin=(0.5, 0.5))

            render_progressbar_y = 6 + ht + ht

        draw_text(current_session.name, 0, h - ht * 1)
        draw_text(current_script_name, 0, h - ht * 2, col=(0, 255, 0))

        draw_text(f"{current_session.width}x{current_session.height}", 0, ht * 0)
        draw_text(f"{current_session.fps} fps", 0, ht * 1)
        if f <= f_last:
            draw_text(f"{f} / {f_last}", 0, ht * 2, playback_color)
        else:
            draw_text(f"{f}", 0, ht * 2, playback_color)

        total_seconds = f / current_session.fps
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        draw_text(f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}", 0, ht * 3, playback_color)

        base_ul_offset = 5

        # Bars
        # ----------------------------------------
        playback_thickness = 3
        segment_thickness = 3
        segment_offset = 2

        pygame.draw.rect(screen, (0, 0, 0), (0, 0, w, playback_thickness + 2))

        # Draw a tiny bar under the rendering text
        if is_rendering and len(current_session.jobs):
            render_progress = current_session.jobs[-1].progress
            bw = 64
            bw2 = bw / 2
            yoff = 0
            pygame.draw.rect(screen, (0, 0, 0), (v.w2 - bw2 - 1, render_progressbar_y + ht + yoff, bw + 2, 2))
            pygame.draw.rect(screen, (0, 255, 255), (v.w2 - bw2, render_progressbar_y + ht + yoff, bw * render_progress, 2))

        # Draw segment bars on top of the progress bar
        for i, t in enumerate(get_segments()):
            lo, hi = t
            progress_lo = lo / f_last
            progress_hi = hi / f_last
            color = colors[i % len(colors)]
            yo = 0
            if i == current_segment:
                yo = playback_thickness

            x = w * progress_lo
            y = segment_offset + yo
            ww = w * (progress_hi - progress_lo)
            hh = segment_thickness

            pygame.draw.rect(screen, (0, 0, 0), (x + 1, y + 1, ww, hh))
            pygame.draw.rect(screen, (0, 0, 0), (x - 1, y + 1, ww, hh))
            pygame.draw.rect(screen, color, (x, y, ww, hh))

        # Draw a progress bar above the frame number
        if f_last > 0:
            progress = f / f_last

            pygame.draw.rect(screen, (0, 0, 0), (0, 0, w, playback_thickness))
            pygame.draw.rect(screen, (255, 255, 255) if not paused else (0, 255, 255), (0, 0, w * progress, playback_thickness))

            # Draw ticks
            major_ticks = 60 * current_session.fps
            minor_ticks = 15 * current_session.fps
            major_tick_height = playback_thickness - 1
            minor_tick_height = playback_thickness - 1
            major_tick_color = (193, 193, 193)
            minor_tick_color = (72, 72, 72)
            x = 0
            ppf = w / f_last
            while x < w:
                x = int(x)
                y = 0
                height = minor_tick_height
                color = minor_tick_color
                # print(current_session.w, current_session.f_last, ppf, x, minor_ticks * ppf, current_session.w // major_ticks, major_ticks, current_session.fps, int(major_ticks*ppf))
                if x % int(major_ticks * ppf) == 0:
                    height = major_tick_height
                    color = major_tick_color

                pygame.draw.line(screen, color, (x, y), (x, y + height))
                pygame.draw.line(screen, color, (x + 1, y), (x + 1, y + height))
                x += minor_ticks * ppf

        if is_rendering:
            lagindicator = np.random.randint(0, 255, (8, 8), dtype=np.uint8)

            lagindicator = random_noise(np.zeros((8, 8)), 's&p', amount=0.1)
            lagindicator = (lagindicator * 255).astype(np.uint8)
            lagindicator = np.stack([lagindicator, lagindicator, lagindicator], axis=2)
            lagindicator_pil = Image.fromarray(lagindicator)
            lagindicator_surface = pygame.image.fromstring(lagindicator_pil.tobytes(), lagindicator_pil.size, lagindicator_pil.mode)
            screen.blit(lagindicator_surface, (w - 8 - 2, h - 8 - 2))

        padding = 12
        ht = font.size("a")[1]
        if enable_hud:
            x = padding
            y = ht * base_ul_offset
            dhud = current_session.get_frame_data('hud')
            if dhud:
                for i, row in enumerate(dhud):
                    s = row[0]
                    color = row[1]
                    fragments = s.split('\n')
                    for frag in fragments:
                        draw_text(frag, x, y, color)
                        y += ht

        if mode == 'action':
            action_slice = actions[action_page:action_page + 9]
            x = padding
            y = ht * base_ul_offset
            for i, pair in enumerate(action_slice):
                name, path = pair
                color = (255, 255, 255)
                draw_text(f'({i + 1}) {name}', x, y, color)
                y += ht

        pygame.display.flip()

        clock.tick(current_session.fps)


    while not request_stop:
        try:
            pygame_update()
        except InterruptedError:
            pygame.quit()
            return
        except Exception as e:
            import traceback
            traceback.print_exc()


def pygame_init():
    threading.Thread(target=pygame_loop).start()


def ryusig_loop():
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
    global invalidated, last_frame_dt, paused

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
    global invalidated, last_frame_dt, paused
    global play_until

    changed = False
    if not paused:
        start_time = time.time()

        current_session.load_f()
        current_session.load_file()
        current_session.f += 1
        changed = True
        invalidated = True

        last_frame_dt = time.time() - start_time

    frame_exists = current_session.determine_current_frame_exists()
    catchedup = play_until and current_session.f >= play_until
    catchedup_end = not frame_exists and not was_paused
    if catchedup_end or catchedup:
        if looping:
            pause_seek(loop_start)
            paused = False
        else:
            paused = True
            play_until = None

    was_paused = paused
    return changed


def render_init(session=None, script_name='', dev=False):
    global initialized
    global current_session, current_script_name
    global current_dev, paused
    global invalidated
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

    invalidated = True
    current_dev = dev

    initialized = True
    return current_session


ltmp = []


def renderloop(lo=None, hi=math.inf):
    global request_render, request_script_check, request_seek
    global invalidated
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

        changed = update_playback()
        sleep_dt()

        ltmp.clear()
        ltmp.extend(request_seek)
        for iseek, manual in ltmp:
            current_session.f = iseek

            # Clamping
            if current_session.f_first is not None and current_session.f < current_session.f_first:
                current_session.f = current_session.f_first
            if current_session.f_last is not None and current_session.f > current_session.f_last + 1:
                current_session.f = current_session.f_last + 1

            current_session.load_f()
            current_session.load_file()
            invalidated = changed = True
            request_seek.pop(0)
            if manual:  # This is to process each frame even if there is multiple inputs buffered from pygame due to lag
                break

        request_seek.clear()

        render = request_render == 'now' or request_render == 'toggle'
        if render:
            if changed:
                # For some reason current_session.image loads async and everything is fucked up if we don't wait
                time.sleep(0.05)

            if request_render == 'now':
                request_render = False

            current_session.seek_new()
            with cpuprofile(args.profile):
                yield current_session.f
        elif changed:
            require_dry_run = not current_session.has_frame_data('hud')
            if require_dry_run:
                render_frame(dry=True)

    current_session.save_data()


def render_frame(f=None, scalar=1, session=None, dry=False):
    global v
    global start_f, n_rendered, current_session
    global last_frame_prompt, last_frame_dt, n_rendered
    global current_hud_pil
    global invalidated
    global is_rendering, paused

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
    v.scalar = scalar

    # Start the HUD
    ss = f'frame {v.f} | {v.t:.2f}s ----------------------'
    if not user_conf.print_jobs:
        print("")
        print(ss)
    hud(ss)

    clear_hud()

    # Actual rendering with the render script
    session.disable_jobs = dry
    v.dry = dry

    # current_session.f = f
    start_f = session.f
    start_img = session.image
    start_time = time.time()
    last_frame_failed = not safe_call(emit, 'frame', failsleep=0.25)
    prompt_changed = v.prompt != last_frame_prompt or start_f == 1
    last_frame_dt = time.time() - start_time

    v.hud()
    hud(p=v.prompt, tcolor=(255, 255, 255) if prompt_changed else (170, 170, 170))

    if not last_frame_failed:
        last_frame_prompt = v.prompt
        session.set_frame_data('prompt_changed', prompt_changed)
        session.set_frame_data('hud', list(hud_rows))
        session.save_data()

    if last_frame_failed or dry:
        # Restore the frame number
        session.seek(start_f)
        session.set(start_img)
        session.disable_jobs = False
    else:
        session.f = f
        if enable_saving:
            session.save()
        session.load_f(f + 1)

        # Save every... features
        if args.preview_every and n_rendered % args.preview_every == 0:
            # TODO video preview
            session.make_video()
        if args.zip_every and n_rendered % args.zip_every == 0:
            # TODO zip frames
            session.make_archive()

        # Flush the HUD
        if not current_dev:
            save_hud(session, draw_hud(session))

        n_rendered += 1

    clear_hud()
    is_rendering = False
    invalidated = True
    if last_frame_failed:
        paused = True


def get_fps(current, offset):
    stops = [12, 24, 30, 50, 60]
    pairs = list(zip(stops, stops[1:]))

    idx = stops.index(current)
    if idx >= 0:
        idx = max(0, min(idx + offset, len(stops) - 1))
        return stops[idx]

    for i, p in enumerate(pairs):
        a, b = p
        if a <= current <= b:
            return a if offset < 0 else b

    return current


def segments_to_frames():
    # example:
    # return '30:88,100:200,3323:4000'

    return '-'.join([f'{s[0]}:{s[1]}' for s in get_segments()])


def get_segments():
    dat = current_session.data
    if not 'segments' in dat:
        dat['segments'] = []

    return dat['segments']

# jargs.args.frames = ''
