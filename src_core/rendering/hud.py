from PIL import Image, ImageDraw, ImageFont

from classes.printlib import printkw
from rendering import renderer
from src_core.classes import paths
from src_core.classes.convert import save_png

snaps = []
rows = []  # list[tuple[str, tuple[int, int, int]]]
rows_tmp = []

def snap(name, img=None):
    if img is None:
        img = renderer.session.img.copy()
    snaps.append((name, img))

def hud(*args, tcolor=(255, 255, 255), **kwargs):
    # Turn args and kwargs into a string like 'a1 a2 x=1 y=2'
    # Format numbers to 3 decimal places (if they are number)
    s = ''
    for a in args:
        if isinstance(a, float):
            s += f'{a:.1f} '
        else:
            s += f'{a} '

    # TODO auto-snap if kwargs is ndarray hwc

    for k, v in kwargs.items():
        if isinstance(v, float):
            s += f'{k}={v:.2f} '
        else:
            s += f'{k}={v} '

    maxlen = 80
    s = '\n'.join([s[i:i + maxlen] for i in range(0, len(s), maxlen)])

    printkw(**kwargs)
    rows.append((s, tcolor))


def clear():
    """
    Clear the HUD
    """
    snaps.clear()
    rows.clear()

def save(session, hud):
    save_png(hud,
             session.det_current_frame_path('prompt_hud').with_suffix('.png'),
             with_async=True)


def to_pil(session):
    """
    Add a HUD and save/edit current in hud folder for this frame
    """
    # Create a new black pil extended vertically to fit an arbitrary string
    rows_tmp.clear()
    rows_tmp.extend(rows)
    rows.clear()

    lines = len(rows_tmp)
    # count the number of \n in the work_rows (list of tuple[str,_])
    for row in rows_tmp:
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
    for i, row in enumerate(rows_tmp):
        s = row[0]
        color = row[1]
        fragments = s.split('\n')
        for frag in fragments:
            draw.text((x, y), frag, font=font, fill=color)
            y += ht

    return new_pil
