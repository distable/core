import random
from src_core.user_conf_base import *

# Values
# ----------------------------------------

precision = "full"
print_timing = False
print_more = False

# Plugins
# ----------------------------------------

install = ['distable/sd1111_plugin']
startup = ['sd1111']

aliases.dream = 'sd1111.txt2img'
aliases.imagine = 'sd1111.txt2img'
aliases.txt2img = 'sd1111.txt2img'

# Job Defaults
# ----------------------------------------

prompts = [
    "Beautiful painting of an ultra contorted landscape by Greg Ruktowsky and Salvador Dali. airbrushed, 70s prog rock album cover, psychedelic, elaborate, complex",
    "A kawaii alien monster horror by salvador dali, inspirationalism, surrealism, horrorealism",
    "A really crazy and weird art piece by salvador dali, picasso, and mark riddick",
    "Inside a strange contorted psychedelic cave with distance and horizon, airbrushed art, impressionism"
]

defaults.sd1111.txt2img = dict(p=lambda: random.choice(prompts), steps=22, sampler='euler-a', w=512, h=384)
defaults.sd1111.img2img = dict(chg=0.65)
