import os
import subprocess
import threading

import numpy as np
import pygame
from PIL import Image
from skimage.util import random_noise

from src_core.lib import corelib
from src_core.rendering import renderer
from src_core.classes import paths
from src_core.classes.printlib import trace_decorator
from src_core.classes.Session import Session
from src_core.rendering.renderer import script_name, is_rendering, n_rendered, pause_seek, start_rendering, pause_toggle, v

enable_hud = False
fps_stops = [1, 4, 6, 8, 10, 12, 24, 30, 50, 60]

screen = None
clock = None
font = None

actions = []
f_pygame = 0
f_displayed = None
mode = 'main'
action_page = 0
last_vram_reported = 0
pilsurface = None

copied_frame = 0
current_segment = -1
invalidated = True
colors = corelib.generate_colors(8, v=1, s=0.765)


@trace_decorator
def init():
    threading.Thread(target=loop).start()


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
    screen.blit(text, (x + -1, y + 0))
    screen.blit(text, (x + -1, y + -1))
    screen.blit(text, (x + 1, y + 1))
    screen.blit(text, (x + 0, y + -1))

    # Main
    text = font.render(s, False, col)
    screen.blit(text, (x, y))

def upfont(param):
    global font, fontsize
    fontsize += param
    font = pygame.font.Font((paths.plug_res / 'vt323.ttf').as_posix(), fontsize)

def loop():
    global enable_hud
    global screen, clock, font
    global actions

    pygame.init()
    clock = pygame.time.Clock()
    icon = pygame.image.load(paths.root / 'icon.png')
    pygame.display.set_icon(icon)

    screen = pygame.display.set_mode((v.w, v.h))
    pygame.display.set_caption("DreamStudio Hobo")
    pygame.key.set_repeat(175, 50)

    actions = list(paths.iter_scripts())
    fontsize = 15
    font = pygame.font.Font((paths.plug_res / 'vt323.ttf').as_posix(), fontsize)

    while not renderer.request_stop:
        try:
            pygame_update()
        except InterruptedError:
            pygame.quit()
            return
        except Exception:
            import traceback
            traceback.print_exc()


@trace_decorator
def pygame_update():
    global enable_hud
    global current_segment
    global copied_frame
    global mode, action_page, actions
    global f_displayed, pilsurface, last_vram_reported
    global f_pygame

    current_session = renderer.session

    w = current_session.w
    h = current_session.h
    f_last = current_session.f_last
    f_first = current_session.f_first
    f = current_session.f

    if f_pygame % 60 == 0:
        # Update VRAM using nvidia-smi
        last_vram_reported = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits']).decode('utf-8').strip()
        last_vram_reported = int(last_vram_reported)

    # pygame.display.update()
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            renderer.request_stop = True
            return

        elif event.type == pygame.WINDOWFOCUSGAINED:
            renderer.request_script_check = True
            renderer.detect_script_every = -1
            continue

        elif event.type == pygame.WINDOWFOCUSLOST:
            renderer.detect_script_every = 1

        if mode == 'main':
            if event.type == pygame.DROPFILE:
                renderer.paused = True
                current_session = Session(event.file, fixpad=True)
                current_session.seek_min()
                get_segments().clear()
                pygame.display.set_mode((w, h))
                renderer.invalidated = True

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    if is_rendering:
                        renderer.request_render = False
                        renderer.paused = True
                    return

                if event.key == pygame.K_SPACE:
                    pause_toggle()

                # Seeking
                # ----------------------------------------
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

                # Editing
                # ----------------------------------------
                if event.key == pygame.K_LEFTBRACKET:
                    current_session.fps = get_fps_stop(current_session.fps, -1)
                if event.key == pygame.K_RIGHTBRACKET:
                    current_session.fps = get_fps_stop(current_session.fps, 1)

                if event.key == pygame.K_LESS and len(get_segments()):
                    current_segment = max(0, current_segment - 1)
                    pause_seek(get_segments()[current_segment][0])

                if event.key == pygame.K_c:
                    copied_frame = current_session.image_cv2
                if event.key == pygame.K_v:
                    if copied_frame is not None:
                        current_session.image_cv2 = copied_frame
                        current_session.save()
                        current_session.save_data()
                        renderer.invalidated = True
                if event.key == pygame.K_DELETE:
                    if current_session.delete_f():
                        renderer.invalidated = True

                # Rendering
                # ----------------------------------------
                if event.key == pygame.K_RETURN:
                    if is_rendering and renderer.request_render:
                        renderer.request_render = False
                    elif is_rendering and not renderer.request_render:
                        start_rendering('toggle')
                    else:
                        start_rendering('now')

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

                if event.key == pygame.K_GREATER and len(get_segments()):
                    current_segment = min(len(get_segments()) - 1, current_segment + 1)
                    pause_seek(get_segments()[current_segment][0])
                if event.key == pygame.K_p:
                    lo, hi = get_segments()[current_segment]
                    current_session.seek(lo)
                    renderer.play_until = hi
                    renderer.paused = False
                    renderer.looping = True
                    renderer.looping_start = lo
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

    changed = renderer.invalidated
    if changed or f_displayed != f or is_rendering:
        # New frame
        im = current_session.image_cv2
        if im is None:
            im = np.zeros((current_session.h, current_session.w, 3), dtype=np.uint8)

        im = np.swapaxes(im, 0, 1)  # cv2 loads in h,w,c order, but pygame wants w,h,c
        pilsurface = pygame.surfarray.make_surface(im)
        # pilsurface = pygame.image.fromstring(pil.tobytes(), pil.size, pil.mode)

        renderer.invalidated = False
        f_displayed = f

        # dt += acc

    if pilsurface is not None:
        surface = pilsurface
        # if not current_session.f_exists:
        #     alpha = int(0.01 * 255)
        #     shadow = int(0.2 * 255)
        #     surface = pilsurface.copy()
        #
        #     # Rainbow scrolling hue
        #     render_steps = current_session.jobs[-1].progress_max
        #     if is_rendering and render_steps > 0:
        #         hue = (time.time() * 100) % 360
        #         color = colorsys.hsv_to_rgb(hue / 360, 1, 0.2)
        #         color = tuple(int(c * 255) for c in color)
        #         color = (*color, alpha)
        #         # Tinted surface
        #         surface.fill(color, special_flags=pygame.BLEND_RGB_ADD)
        #
        #     surface.fill((255 - shadow, 255 - shadow, 255 - shadow, 255), special_flags=pygame.BLEND_RGB_MULT)

        screen.blit(surface, (0, 0))

    fps = 1 / renderer.last_frame_dt
    pad = 12
    right = w - pad
    left = pad
    top = pad
    bottom = h - pad

    playback_color = (255, 255, 255)
    if renderer.paused:
        playback_color = (0, 255, 255)

    # Draw the fps in the upper left corner
    ht = font.get_height() + 2
    if renderer.paused:
        draw_text(f"Paused", right, top, playback_color, origin=(1, 0))
    else:
        draw_text(f"{int(fps)} FPS", right, top, playback_color, origin=(1, 0))

    # VRAM
    draw_text(f"{last_vram_reported} MB", right, top + ht, (255, 255, 255), origin=(1, 0))

    top2 = top + 6
    render_progressbar_y = 0
    if is_rendering:
        color = (255, 255, 255)
        if renderer.request_render == 'toggle':
            color = (255, 0, 255)
            draw_text("-- Rendering --", v.w2, top2, color, origin=(0.5, 0.5))
        else:
            draw_text("-- Busy --", v.w2, top2, color, origin=(0.5, 0.5))

        draw_text(f"{n_rendered} frames", v.w2, top2 + ht, color, origin=(0.5, 0.5))
        draw_text(f"{n_rendered / current_session.fps:.02}s", v.w2, top2 + ht + ht, color, origin=(0.5, 0.5))

        render_progressbar_y = top2 + ht + ht

    draw_text(current_session.name, left, bottom - ht * 1)
    draw_text(script_name, left, bottom - ht * 2, col=(0, 255, 0))

    draw_text(f"{current_session.width}x{current_session.height}", left, top + ht * 0)
    draw_text(f"{current_session.fps} fps", left, top + ht * 1)
    if f <= f_last:
        draw_text(f"{f} / {f_last}", left, top + ht * 2, playback_color)
    else:
        draw_text(f"{f}", left, top + ht * 2, playback_color)

    total_seconds = f / current_session.fps
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    draw_text(f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}", left, top + ht * 3, playback_color)

    base_ul_offset = 5

    # Bars
    # ----------------------------------------
    playback_thickness = 3
    segment_thickness = 3
    segment_offset = 2

    pygame.draw.rect(screen, (0, 0, 0), (0, 0, w, playback_thickness + 2))

    # Draw a tiny bar under the rendering text
    if is_rendering and len(current_session.jobs):
        render_steps = current_session.jobs[-1].progress_max
        render_progress = current_session.jobs[-1].progress
        if render_steps > 0:
            bw = 64
            bw2 = bw / 2
            bh = 3
            yoff = -2
            pygame.draw.rect(screen, (0, 0, 0), (v.w2 - bw2 - 1, render_progressbar_y + ht + yoff, bw + 2, bh))
            pygame.draw.rect(screen, (0, 255, 255), (v.w2 - bw2, render_progressbar_y + ht + yoff, bw * render_progress, bh))

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
        pygame.draw.rect(screen, (255, 255, 255) if not renderer.paused else (0, 255, 255), (0, 0, w * progress, playback_thickness))

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
        # lagindicator = np.random.randint(0, 255, (8, 8), dtype=np.uint8)

        lagindicator = random_noise(np.zeros((8, 8)), 's&p', amount=0.1)
        lagindicator = (lagindicator * 255).astype(np.uint8)
        lagindicator = np.stack([lagindicator, lagindicator, lagindicator], axis=2)
        lagindicator_pil = Image.fromarray(lagindicator)
        lagindicator_surface = pygame.image.frombuffer(lagindicator_pil.tobytes(), lagindicator_pil.size, 'RGB')
        screen.blit(lagindicator_surface, (w - 8 - 2, h - 8 - 2))

    ht = font.size("a")[1]
    if enable_hud:
        x = pad
        y = ht * base_ul_offset
        dhud = current_session.get_frame_data('hud', True)
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
        x = pad
        y = ht * base_ul_offset
        for i, pair in enumerate(action_slice):
            name, path = pair
            color = (255, 255, 255)
            draw_text(f'({i + 1}) {name}', x, y, color)
            y += ht

    pygame.display.flip()
    clock.tick(60)
    f_pygame += 1

def segments_to_frames():
    # example:
    # return '30:88,100:200,3323:4000'

    return '-'.join([f'{s[0]}:{s[1]}' for s in get_segments()])


def get_segments():
    dat = renderer.session.data
    if not 'segments' in dat:
        dat['segments'] = []

    return dat['segments']
