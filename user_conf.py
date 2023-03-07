from PyQt5 import QtCore

import jargs
from src_core.conf import *

share = False

# Core
# ----------------------------------------

precision = "full"
enable_print = jargs.args.print
print_timing = enable_print
print_trace = False
print_gputrace = enable_print
print_extended_init = enable_print
print_more2 = enable_print
print_jobs = enable_print

# Plugins
# ----------------------------------------

plugload('copyres')
plugload('distable/disco-party')
plugload('distable/ryusig-calc')

sd = plugdef('distable/sd1111_plugin')

# paella = plugload('paella')
# kup = plugdef('kupscale')
wc = plugload('wildcard')
mgk = plugload('magick')
m2d = plugload('math2d')
# m3d = plugdef('midas3d')
flo = plugload('opticalflow')
edgedet = plugload('edgedet')
noise = plugload('spnoise')
glsl = plugload('glsl')

# sd.attention = 4
sd.bit8 = False

aliasdef(dream='sd1111.txt2img',
         imagine='sd1111.txt2img')

forbidden_dev_jobs = [
    'txt2img',
    'img2img',
    'sd1111.txt2img',
    'sd1111.img2img',
]

# sd.res_ckpt = 'miniSD.ckpt'
# sd.res_ckpt = 'sd-v2-0-depth.ckpt'
# sd.res_ckpt = 'nouvisPsychedelicMod_15.ckpt'
sd.medvram = True
sd.lowvram = False
sd.lowram = False
sd.precision = 'full'
sd.no_half = True
sd.no_half_vae = True
sd.batch_cond_uncond = False

# Deployment
# ----------------------------------------
vastai_default_search = "gpu_name=RTX_3090"
vastai_sshfs = True  # Mount via sshfs
vastai_sshfs_path = "~/discore/mount/"

deploy_urls = {
    'sd-v1-5.ckpt': 'https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.ckpt',
    'vae.vae.pt'  : 'https://huggingface.co/stabilityai/sd-vae-ft-mse-original/resolve/main/vae-ft-mse-840000-ema-pruned.ckpt',
}

if jargs.args.remote:
    print("----------------------------------------")
    print("Activating remote arguments")
    print("----------------------------------------")
    sd.medvram = False
    sd.lowvram = False
    sd.precision = 'half'
    sd.no_half = False
    sd.no_half_vae = False
    sd.batch_cond_uncond = True
    # jargs.args.zip_every = 120

# GUI config
# ----------------------------------------

hobo_seek_percent = 1 / 15

qkeys = QtCore.Qt.Key
key_pause = qkeys.Key_Space

key_seek_prev = qkeys.Key_Left
key_seek_next = qkeys.Key_Right
key_seek_prev_second = qkeys.Key_Up
key_seek_next_second = qkeys.Key_Down
key_seek_prev_percent = qkeys.Key_PageUp
key_seek_next_percent = qkeys.Key_PageDown
key_seek_first = qkeys.Key_Home
key_seek_first_2 = qkeys.Key_H
key_seek_first_3 = qkeys.Key_0
key_seek_last = qkeys.Key_N
key_seek_last_2 = qkeys.Key_End

key_fps_down = qkeys.Key_BracketLeft
key_fps_up = qkeys.Key_BracketRight
key_copy_frame = qkeys.Key_C
key_paste_frame = qkeys.Key_V
key_delete = qkeys.Key_Delete
key_render = qkeys.Key_Return
key_toggle_hud = qkeys.Key_F
key_toggle_action_mode = qkeys.Key_W

key_select_segment_prev = qkeys.Key_Less
key_select_segment_next = qkeys.Key_Greater
key_set_segment_start = qkeys.Key_I
key_set_segment_end = qkeys.Key_O
key_seek_prev_segment = qkeys.Key_Comma
key_seek_next_segment = qkeys.Key_Period
key_play_segment = qkeys.Key_P



# # Plugin config
# # ----------------------------------------
#
# prompts = [
#     "Beautiful painting of an ultra contorted landscape by <artist> and <artist>. airbrushed, 70s prog rock album cover, psychedelic, elaborate, complex",
#     "A kawaii alien monster horror by <job>, inspirationalism, surrealism, horrorealism",
#     "A really crazy and weird art piece by <artist>, <artist>, and <artist>",
#     "Inside a strange <pow> contorted psychedelic cave with distance and horizon, airbrushed art, impressionism"
# ]

# sd.sd_job = dict(prompt=choice(prompts),
#                  steps=17,
#                  cfg=7.65,
#                  sampler='euler-a',
#                  w=512, h=384,
#                  chg=0.65)

# # wildcards.artist = ['salvador dali', 'picasso', 'mark riddick', 'greg ruktowsky']
# wc.artist = ['Aoshima Chiho', 'Arnegger Alois', 'shimoda hikari', 'terry redlin', 'Satoshi Kon', "hayao mizaki", 'rj palmer', 'alex grey', 'salvador dali']
# wc.scene = ['realms', 'skies', 'planetary sky']
# wc.distance = ['far away', 'in the distance', 'in the horizon']
# wc.painted = ['painted', 'drawn', ' inked', 'designed']
# wc.cloud = ['stratocumulus', 'altocumulus', 'cumulus', 'nimbostratus', 'cirrocumulus']
# wc.glow = ['shining', 'glowing', 'radiating', 'exploding']
# wc.blob = ['sun', 'moon', 'face', 'sunset', 'sunrise']
# wc.movement = ['surrealism', 'hyperrealism', 'eccentrism']
# wc.majestic = ['scenec', 'majestic', 'grandiose', 'picturesque', 'jawdropping']
# wc.scale = ['huge', 'big', 'wide', 'scenic', 'large', 'impressive', 'grandiose', 'stunning', 'picturesque']
# wc.cursed = ['cursed', 'twisted', 'contorted', 'strange', 'weird', 'rippled']
# wc.contort = ['contorted', 'twisted', 'bending', 'uneven', 'chaotic']
# wc.magic = ['magical', 'cursed', 'fantasy', 'mystical', 'enchanted']
# wc.elaborate = ['elaborate', 'complex', 'detailed']
# wc.view = ['shot', 'view']
# wc.detail = ['complex', 'intricate', 'detailed']
# wc.relate = ['rotating', 'sliding', 'moving', 'exploding', 'displacing', 'changing', 'transfmorphing', 'exchanging', 'expanding', 'stretching', 'condensing', 'tiling', 'alternating', 'juxtaposing', 'overlapping']
# wc.pow = ['slightly', 'super', 'very', 'greatly', 'ultra', 'extremely', 'intensely']
# wc._ = ["    ", "   ", "  ", " "]
#
# wc.energy = ['light', 'energy', 'glow', 'radiant']
# wc.shard = ["mirror", "gemstone", "diamond", 'ornate', "crystallite", "ice", "rock", "glass", "crystal", 'stained-glass', "quartz"]
# wc.coral = ["oceanic", "intricate", "brain", "colorful reef", "magical cursed reef"]
# wc.wavy = ["wavy", "zig-zaggy", "stretchy", "droopy", "twisty", "contorted wavy", "marbling", "cursed", "coral", 'uneven', 'deformed']
# wc.wave_t = ["huge", "separating", "ultra", "", "contorted"]
# wc.shape = ["tornado", "helix", "galaxy"]
# wc.texture = ["floral", "florally", "floraling", "inflorescent", "flowery"]
# wc.adjective = ['intricate', 'detailed', 'beautiful', 'picturesque', 'immense', 'sunny', 'rainy', "melting", "iridescent", "opalescent", "magical"]
# wc.related = ['', 'separating', 'repeating', 'alternating', 'overlapping']
# wc.tex_layout = ['asymmetric tiling', 'symmetric tiling', 'symmetric', 'asymmetric', 'tiling']
# wc.tex_shape = ['zig-zaggy', 'spiraling', 'contorting', 'stretching', "flower petal", "sunny", "coral", "crystalline water macro"]
