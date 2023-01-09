import random

from src_core import core
from src_core.plugins import mod2dic


# List of renderer properties
#
# t: time in seconds
# f: frame number
# dt: time since last frame
# fps: frames per second
# w: width of the canvas (current session)
# h: height of the canvas (current session)
# w2: half of the width of the canvas (current session)
# h2: half of the height of the canvas (current session)
# tr: time ratio'd to 1/12th of a second
# ref: 1/12 * fps
#
# x, y, z, r: current position and rotation of the camera
# smear: smear value
#

def override(module):
    dic = mod2dic(module)

def on_init():
    pass


def on_frame(v):
    pass
