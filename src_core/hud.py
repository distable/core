from PIL import Image, ImageDraw, ImageFont

from src_core import core
from src_core.classes import paths
from src_core.classes.convert import save_png

hud_rows = []
work_rows = []

def hud(*args, tcolor=(255, 255, 255), **kwargs):
    # Turn args and kwargs into a string like 'a1 a2 x=1 y=2'
    # Format numbers to 3 decimal places (if they are number)
    s = ''
    for a in args:
        if isinstance(a, float):
            s += f'{a:.1f} '
        else:
            s += f'{a} '

    for k, v in kwargs.items():
        if isinstance(v, float):
            s += f'{k}={v:.1f} '
        else:
            s += f'{k}={v} '

    maxlen = 80
    s = '\n'.join([s[i:i + maxlen] for i in range(0, len(s), maxlen)])

    hud_rows.append((s, tcolor))

def clear_hud():
    """
    Clear the HUD
    """
    hud_rows.clear()

def draw_hud(session):
    """
    Add a HUD and save/edit current in hud folder for this frame
    """
    # Create a new black pil extended vertically to fit an arbitrary string
    work_rows.clear()
    work_rows.extend(hud_rows)
    hud_rows.clear()

    lines = len(work_rows)
    # count the number of \n in the work_rows (list of tuple[str,_])
    for row in work_rows:
        lines += row[0].count('\n')

    w = session.w
    h = session.h
    padding = 12
    font = ImageFont.truetype(str(paths.plug_res / 'vt323.ttf'), 15)
    tw, ht = font.getsize_multiline("foo")

    new_pil = Image.new('RGB', (w + padding * 2, h + ht * lines + padding * 2), color=(0, 0, 0))

    # Draw the old pil on the new pil at the top
    if session.image:
        new_pil.paste(session.image, (padding, padding))

    # Draw the arbitrary string on the new pil at the bottom
    draw = ImageDraw.Draw(new_pil)
    x = padding
    y = h + padding * 1.25
    for i, row in enumerate(work_rows):
        s = row[0]
        color = row[1]
        fragments = s.split('\n')
        for frag in fragments:
            draw.text((x, y), frag, font=font, fill=color)
            y += ht

    return new_pil

def save_hud(session, hud):
    save_png(hud,
             session.determine_current_frame_path('prompt_hud').with_suffix('.png'),
             with_async=True)
