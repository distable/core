"""
This is DreamStudio hobo!
A fun retro crappy PyGame interface to do
you work in.
"""

import subprocess
from pathlib import Path

import numpy as np
import pygame
from PIL import Image
from PyQt6 import QtCore
from pyqtgraph import *
from skimage.util import random_noise

from jargs import args

import jargs
import uiconf
from src_core.rendering import hud
from src_core.classes import paths
from src_core.classes.Session import Session
from src_core.lib import corelib
from src_core.rendering import renderer, ryusig

enable_hud = False
key_mode = 'main'
fps_stops = [1, 4, 6, 8, 10, 12, 24, 30, 50, 60]

discovered_actions = []
f_update = 0
f_displayed = None

sel_action_page = 0
sel_snapshot = -1

last_vram_reported = 0

surface = None
font = None
copied_frame = 0
current_segment = -1
invalidated = True
colors = corelib.generate_colors(8, v=1, s=0.765)
rv = None

def init(_rv):
    global enable_hud
    global font
    global discovered_actions
    global surface
    global rv

    rv = _rv

    discovered_actions = list(paths.iter_scripts())
    session = renderer.session

    # Setup pygame renderer
    ryusig.rv = rv
    pygame.init()
    surface = pygame.Surface((session.w, session.h))
    surface.fill((255, 0, 255))
    fontsize = 15
    font = pygame.font.Font((paths.plug_res / 'vt323.ttf').as_posix(), fontsize)


def update():
    global enable_hud
    global current_segment
    global copied_frame
    global sel_action_page, discovered_actions
    global f_displayed, last_vram_reported
    global f_update
    global surface

    if f_update % 60 == 0:
        # Update VRAM using nvidia-smi
        last_vram_reported = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits']).decode('utf-8').strip()
        last_vram_reported = int(last_vram_reported)

    if renderer.is_readonly and f_update % 120 == 0 and renderer.paused:
        refresh_session()

    draw()


def draw():
    global f_update
    global enable_hud
    global current_segment
    global copied_frame
    global sel_action_page, discovered_actions
    global f_displayed, pilsurface, last_vram_reported
    global frame_surface
    global invalidated

    session = renderer.session
    w = session.w
    h = session.h

    f_last = session.f_last
    f_first = session.f_first
    f = session.f

    fps = 1 / max(renderer.last_frame_dt, 1 / session.fps)
    pad = 12
    right = w - pad
    left = pad
    top = pad
    bottom = h - pad
    playback_color = (255, 255, 255)
    if renderer.request_pause:
        playback_color = (0, 255, 255)

    # BASE
    # ----------------------------------------
    surface.fill((0, 0, 0))

    changed = renderer.invalidated or invalidated
    if changed or f_displayed != f or renderer.is_rendering:
        im = None
        # New frame
        if sel_snapshot == -1:
            im = session.img
            if im is None:
                im = np.zeros((session.h, session.w, 3), dtype=np.uint8)
        elif sel_snapshot < len(hud.snaps):
            im = hud.snaps[sel_snapshot][1]

        if im is not None and im.shape >= (1, 1, 1):
            im = np.swapaxes(im, 0, 1)  # cv2 loads in h,w,c order, but pygame wants w,h,c
            frame_surface = pygame.surfarray.make_surface(im)
            renderer.invalidated = False
            invalidated = False
            f_displayed = f
        elif not renderer.is_rendering:
            im = np.zeros((session.w, session.h, 3), dtype=np.uint8)
            frame_surface = pygame.surfarray.make_surface(im)

    surface.blit(frame_surface, (0, 0))

    # UPPER LEFT
    # ----------------------------------------

    # Paused / FPS
    ht = font.get_height() + 2

    if not renderer.paused or (renderer.is_rendering and fps > 1):
        draw_text(f"{int(fps)} FPS", right, top, playback_color, origin=(1, 0))
    else:
        draw_text(f"Paused", right, top, playback_color, origin=(1, 0))

    # VRAM
    draw_text(f"{last_vram_reported} MB", right, top + ht, (255, 255, 255), origin=(1, 0))

    # Devmode
    draw_text("Dev" if renderer.is_dev else "", right, top + ht * 2, (255, 255, 255), origin=(1, 0))

    # Tracing
    draw_text("Trace" if jargs.args.trace else "", right, top + ht * 3, (255, 255, 255), origin=(1, 0))

    # UPPER RIGHT
    # ----------------------------------------
    top2 = top + 6
    render_progressbar_y = 0
    if renderer.is_rendering:
        color = (255, 255, 255)
        if renderer.request_render == 'toggle':
            color = (255, 0, 255)
            draw_text("-- Rendering --", rv.w2, top2, color, origin=(0.5, 0.5))
        else:
            draw_text("-- Busy --", rv.w2, top2, color, origin=(0.5, 0.5))

        draw_text(f"{renderer.n_rendered} frames", rv.w2, top2 + ht, color, origin=(0.5, 0.5))
        draw_text(f"{renderer.n_rendered / session.fps:.02f}s", rv.w2, top2 + ht + ht, color, origin=(0.5, 0.5))

        render_progressbar_y = top2 + ht + ht

    draw_text(renderer.script_name, left, bottom - ht * 2, col=(0, 255, 0))
    draw_text(f"{session.width}x{session.height}", left, top + ht * 0)
    draw_text(f"{session.fps} fps", left, top + ht * 1)
    if f <= f_last:
        draw_text(f"{f} / {f_last}", left, top + ht * 2, playback_color)
    else:
        draw_text(f"{f}", left, top + ht * 2, playback_color)

    total_seconds = f / session.fps
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    draw_text(f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}", left, top + ht * 3, playback_color)
    base_ul_offset = 5

    # LOWER LEFT
    # ----------------------------------------
    draw_text(session.name, left, bottom - ht * 1)

    # LOWER RIGHT
    # ----------------------------------------
    if -1 < sel_snapshot < len(hud.snaps):
        snap_str = f"Snapshot {sel_snapshot}"
        snapshot = hud.snaps[sel_snapshot]
        snap_str += f" ({snapshot[0]})"

        draw_text(snap_str, right, bottom - ht * 1, origin=(1, 0))

    # Bars
    # ----------------------------------------
    playback_thickness = 3
    segment_thickness = 3
    segment_offset = 2
    pygame.draw.rect(surface, (0, 0, 0), (0, 0, w, playback_thickness + 2))
    # Draw a tiny bar under the rendering text
    if renderer.is_rendering and len(session.jobs):
        render_steps = session.jobs[-1].progress_max
        render_progress = session.jobs[-1].progress
        if render_steps > 0:
            bw = 64
            bw2 = bw / 2
            bh = 3
            yoff = -2
            pygame.draw.rect(surface, (0, 0, 0), (rv.w2 - bw2 - 1, render_progressbar_y + ht + yoff, bw + 2, bh))
            pygame.draw.rect(surface, (0, 255, 255), (rv.w2 - bw2, render_progressbar_y + ht + yoff, bw * render_progress, bh))

    if f_last > 0:
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

            pygame.draw.rect(surface, (0, 0, 0), (x + 1, y + 1, ww, hh))
            pygame.draw.rect(surface, (0, 0, 0), (x - 1, y + 1, ww, hh))
            pygame.draw.rect(surface, color, (x, y, ww, hh))

        # Draw a progress bar above the frame number
        progress = f / f_last

        pygame.draw.rect(surface, (0, 0, 0), (0, 0, w, playback_thickness))
        pygame.draw.rect(surface, (255, 255, 255) if not renderer.request_pause else (0, 255, 255), (0, 0, w * progress, playback_thickness))

        # Draw ticks
        major_ticks = 60 * session.fps
        minor_ticks = 15 * session.fps
        major_tick_height = playback_thickness - 1
        minor_tick_height = playback_thickness - 1
        major_tick_color = (193, 193, 193)
        minor_tick_color = (72, 72, 72)
        x = 0
        ppf = w / f_last
        while x < w:
            y = 0
            height = minor_tick_height
            color = minor_tick_color
            # print(session.w, session.f_last, ppf, x, minor_ticks * ppf, session.w // major_ticks, major_ticks, session.fps, int(major_ticks*ppf))
            if int(x) % max(1, int(major_ticks * ppf)) == 0:
                height = major_tick_height
                color = major_tick_color

            pygame.draw.line(surface, color, (int(x), y), (int(x), y + height))
            pygame.draw.line(surface, color, (int(x) + 1, y), (int(x) + 1, y + height))
            x += minor_ticks * ppf

    if renderer.is_rendering:
        # lagindicator = np.random.randint(0, 255, (8, 8), dtype=np.uint8)

        lagindicator = random_noise(np.zeros((8, 8)), 's&p', amount=0.1)
        lagindicator = (lagindicator * 255).astype(np.uint8)
        lagindicator = np.stack([lagindicator, lagindicator, lagindicator], axis=2)
        lagindicator_pil = Image.fromarray(lagindicator)
        lagindicator_surface = pygame.image.frombuffer(lagindicator_pil.tobytes(), lagindicator_pil.size, 'RGB')
        surface.blit(lagindicator_surface, (w - 8 - 2, h - 8 - 2))
    ht = font.size("a")[1]

    if enable_hud:
        x = pad
        y = ht * base_ul_offset
        dhud = session.get_frame_data('hud', True)
        if dhud:
            for i, row in enumerate(dhud):
                value = row[0]
                color = row[1]
                changed = False
                if i > 0:
                    last_value = dhud[i - 1][0]
                    changed = value != last_value

                # This is a hack, the color is stored in file and i want to change it for existing files..
                # if changed:
                #     color = [000, 255, 000]

                fragments = value.split('\n')
                for frag in fragments:
                    draw_text(frag, x, y, color)
                    y += ht

    if key_mode == 'action':
        action_slice = discovered_actions[sel_action_page:sel_action_page + 9]
        x = pad
        y = ht * base_ul_offset
        for i, pair in enumerate(action_slice):
            name, path = pair
            color = (255, 255, 255)
            draw_text(f'({i + 1}) {name}', x, y, color)
            y += ht

    f_update += 1


def keydown(key, ctrl, shift, alt):
    global key_mode
    global sel_snapshot
    global invalidated

    qkeys = QtCore.Qt.Key
    s = renderer.session

    n_snapshots = len(hud.snaps)
    if n_snapshots > 1:
        if key == qkeys.Key_Right and shift:
            sel_snapshot = min(sel_snapshot + 1, n_snapshots - 1)
            invalidated = True
            return
        if key == qkeys.Key_Left and shift:
            sel_snapshot = max(sel_snapshot - 1, -1)
            invalidated = True
            return
    if key == qkeys.Key_ParenRight:
        s.f_last = s.f
        s.f_last_path = s.det_frame_path(s.f)
        return

    if key_mode == 'main':
        keydown_main(key, ctrl, shift, alt)
    elif key_mode == 'action':
        if key == qkeys.Key_Escape or key == qkeys.Key_W:
            key_mode = 'main'
        else:
            keydown_action(key, ctrl, shift, alt)


def keydown_action(key, ctrl, shift, alt):
    global key_mode, sel_action_page
    global invalidated

    qkeys = QtCore.Qt.Key
    session = renderer.session

    if key == qkeys.Key_Left:
        sel_action_page = max(0, sel_action_page - 10)
    elif key == qkeys.Key_Right:
        max_page = len(discovered_actions) // 10
        sel_action_page += 10
        sel_action_page = min(sel_action_page, max_page)

    i = key - qkeys.Key_1
    if i in range(1, 9):
        action_slice = discovered_actions[sel_action_page:sel_action_page + 10]
        if 1 <= i <= qkeys.Key_9:
            name, path = action_slice[i]
            s = f"discore {session.dirpath} {name} "

            if shift:
                s += f"--frames {segments_to_frames()}"

            os.popen(s)
            key_mode = 'main'


def keydown_main(key, ctrl, shift, alt):
    global current_segment, copied_frame, enable_hud, key_mode

    qkeys = QtCore.Qt.Key
    session = renderer.session
    w = session.w
    h = session.h

    f_last = session.f_last
    f_first = session.f_first
    f = session.f

    if key == qkeys.Key_F1:
        ryusig.toggle()
    if key == qkeys.Key_F2:
        renderer.is_dev = not renderer.is_dev
    if key == qkeys.Key_F3:
        args.trace = not args.trace

    if key == qkeys.Key_R and shift:
        s = Session(renderer.session.dirpath)
        s.f = renderer.session.f
        s.load_f()
        s.load_file()

        renderer.update_session(s)

    # Playback
    # ----------------------------------------
    if key == uiconf.key_pause: renderer.pause()
    if key == uiconf.key_seek_prev: renderer.seek(f - 1, True)
    if key == uiconf.key_seek_next: renderer.seek(f + 1, True)
    if key == uiconf.key_seek_prev_second: renderer.seek(f - session.fps, True)
    if key == uiconf.key_seek_next_second: renderer.seek(f + session.fps, True)
    if key == uiconf.key_seek_prev_percent: renderer.seek(f - int(f_last * uiconf.hobo_seek_percent), True)
    if key == uiconf.key_seek_next_percent: renderer.seek(f + int(f_last * uiconf.hobo_seek_percent), True)
    if key == uiconf.key_seek_first: renderer.seek(f_first, True)
    if key == uiconf.key_seek_first_2: renderer.seek(f_first, True)
    if key == uiconf.key_seek_first_3: renderer.seek(f_first, True)
    if key == uiconf.key_seek_last or key == uiconf.key_seek_last_2:
        renderer.seek(f_last, True)
        renderer.seek(f_last + 1, True)

    # Editing
    # ----------------------------------------
    if key == uiconf.key_fps_down:
        session.fps = get_fps_stop(session.fps, -1)
    if key == uiconf.key_fps_up:
        session.fps = get_fps_stop(session.fps, 1)
    if key == uiconf.key_select_segment_prev and len(get_segments()):
        current_segment = max(0, current_segment - 1)
        renderer.seek(get_segments()[current_segment][0])
    if key == uiconf.key_copy_frame:
        copied_frame = session.img
    if key == uiconf.key_paste_frame:
        if copied_frame is not None:
            session.img = copied_frame
            session.save()
            session.save_data()
            renderer.invalidated = True
    if key == uiconf.key_delete and shift:
        if session.delete_f():
            renderer.invalidated = True

    # Rendering
    # ----------------------------------------
    if key == uiconf.key_render:
        if renderer.is_rendering and renderer.request_render:
            renderer.request_render = False
        elif renderer.is_rendering and not renderer.request_render:
            renderer.render('toggle')
        else:
            renderer.render('now')
    if key == uiconf.key_reload_script:
        renderer.reload_script()
    if key == uiconf.key_toggle_hud:
        enable_hud = not enable_hud
    if key == uiconf.key_set_segment_start:
        if len(get_segments()) and not f > get_segments()[current_segment][1]:
            get_segments()[current_segment] = (f, get_segments()[current_segment][1])
            session.save_data()
        else:
            create_segment(50)
    if key == uiconf.key_set_segment_end:
        if len(get_segments()) and not f < get_segments()[current_segment][0]:
            get_segments()[current_segment] = (get_segments()[current_segment][0], f)
            session.save_data()
        else:
            create_segment(-50)
    if key == uiconf.key_seek_prev_segment:
        indices = [i for s in get_segments() for i in s]
        indices.sort()
        # Find next value in indices that is less than session.f
        for i in range(len(indices) - 1, -1, -1):
            if indices[i] < f:
                renderer.seek(indices[i])
                break
    if key == uiconf.key_seek_next_segment:
        indices = [i for s in get_segments() for i in s]
        indices.sort()
        # Find next value in indices that is greater than session.f
        for i in range(len(indices)):
            if indices[i] > f:
                renderer.seek(indices[i])
                break
    if key == uiconf.key_select_segment_next and len(get_segments()):
        current_segment = min(len(get_segments()) - 1, current_segment + 1)
        renderer.seek(get_segments()[current_segment][0])
    if key == uiconf.key_play_segment:
        lo, hi = get_segments()[current_segment]
        session.seek(lo)
        renderer.play_until = hi
        renderer.request_pause = False
        renderer.looping = True
        renderer.looping_start = lo
    if key == uiconf.key_toggle_action_mode:
        key_mode = 'action'


def focuslose():
    renderer.detect_script_every = 1


def focusgain():
    renderer.request_script_check = True
    renderer.detect_script_every = -1

    if renderer.is_readonly:
        refresh_session()


def dropenter(files):
    file = Path(files[0])
    if file.is_dir():
        session = Session(file, fixpad=True)
        session.seek_min()

        # TODO on_session_changed
        if session.w and session.h:
            pygame.display.set_mode((session.w, session.h))

        renderer.set_session(session)
    elif file.suffix in paths.image_exts:
        renderer.set_image(file)


# region API
def segments_to_frames():
    # example:
    # return '30:88,100:200,3323:4000'

    return '-'.join([f'{s[0]}:{s[1]}' for s in get_segments()])


def get_segments():
    dat = renderer.session.data
    if not 'segments' in dat:
        dat['segments'] = []

    return dat['segments']


def create_segment(off):
    global current_segment
    get_segments().append((renderer.session.f, renderer.session.f + off))
    current_segment = len(get_segments()) - 1
    renderer.session.save_data()


def get_fps_stop(current, offset):
    stops = fps_stops
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


def draw_text(s, x, y, col=(255, 255, 255), origin=(0, 0)):
    if s is None: return
    size = font.size(s)
    x -= size[0] * origin[0]
    y -= size[1] * origin[1]

    # Shadow
    text = font.render(s, False, (0, 0, 0))
    surface.blit(text, (x + -1, y + 0))
    surface.blit(text, (x + -1, y + -1))
    surface.blit(text, (x + 1, y + 1))
    surface.blit(text, (x + 0, y + -1))

    # Main
    text = font.render(s, False, col)
    surface.blit(text, (x, y))


# endregion

# region Controls
def upfont(param):
    global font, fontsize
    fontsize += param
    font = pygame.font.Font((paths.plug_res / 'vt323.ttf').as_posix(), fontsize)
# endregion

def refresh_session():
    ses = renderer.session
    nextfile = ses.det_frame_path(ses.f_last + 1)
    if os.path.exists(nextfile):
        tmp_f = ses.f
        was_last = ses.f >= ses.f_last
        renderer.session.load()
        renderer.session.f = tmp_f
        if was_last:
            ses.seek(ses.f_last)
