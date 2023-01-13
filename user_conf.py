from src_core.conf import *

# Core
# ----------------------------------------

precision = "full"
print_timing = False
print_trace = False
print_gputrace = False
print_extended_init = True
print_more2 = True
share = False

# Plugins
# ----------------------------------------

plugload('copyres')
plugload('distable/disco-party')

sd = plugdef('distable/sd1111_plugin')
# paella = plugload('paella')
# kup = plugdef('kupscale')
wc = plugload('wildcard')
mgk = plugload('magick')
m2d = plugload('math2d')
# m3d = plugload('math3d')
flo = plugload('opticalflow')
edgedet = plugload('edgedet')
noise = plugload('spnoise')
glsl = plugload('glsl')

# sd.attention = 4
sd.bit8 = False

aliasdef(dream='sd1111.txt2img',
         imagine='sd1111.txt2img')


# sd.res_ckpt = 'miniSD.ckpt'
# sd.res_ckpt = 'sd-v2-0-depth.ckpt'
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
vastai_sshfs = True # Mount via sshfs
vastai_sshfs_path = "~/discore/mount/"

# deploy_urls = {
#     'sd-v1-5.ckpt' : 'https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.ckpt',
#     'vae.vae.pt' : 'https://huggingface.co/stabilityai/sd-vae-ft-mse-original/resolve/main/vae-ft-mse-840000-ema-pruned.ckpt',
# }
deploy_urls = {}

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

# wildcards.artist = ['salvador dali', 'picasso', 'mark riddick', 'greg ruktowsky']
wc.artist = ['Aoshima Chiho', 'Arnegger Alois', 'shimoda hikari', 'terry redlin', 'Satoshi Kon', "hayao mizaki", 'rj palmer', 'alex grey', 'salvador dali']
wc.scene = ['realms', 'skies', 'planetary sky']
wc.distance = ['far away', 'in the distance', 'in the horizon']
wc.painted = ['painted', 'drawn', ' inked', 'designed']
wc.cloud = ['stratocumulus', 'altocumulus', 'cumulus', 'nimbostratus', 'cirrocumulus']
wc.glow = ['shining', 'glowing', 'radiating', 'exploding']
wc.blob = ['sun', 'moon', 'face', 'sunset', 'sunrise']
wc.movement = ['surrealism', 'hyperrealism', 'eccentrism']
wc.majestic = ['scenec', 'majestic', 'grandiose', 'picturesque', 'jawdropping']
wc.scale = ['huge', 'big', 'wide', 'scenic', 'large', 'impressive', 'grandiose', 'stunning', 'picturesque']
wc.cursed = ['cursed', 'twisted', 'contorted', 'strange', 'weird', 'rippled']
wc.contort = ['contorted', 'twisted', 'bending', 'uneven', 'chaotic']
wc.magic = ['magical', 'cursed', 'fantasy', 'mystical', 'enchanted']
wc.elaborate = ['elaborate', 'complex', 'detailed']
wc.view = ['shot', 'view']
wc.detail = ['complex', 'intricate', 'detailed']
wc.relate = ['rotating', 'sliding', 'moving', 'exploding', 'displacing', 'changing', 'transfmorphing', 'exchanging', 'expanding', 'stretching', 'condensing', 'tiling', 'alternating', 'juxtaposing', 'overlapping']
wc.pow = ['slightly', 'super', 'very', 'greatly', 'ultra', 'extremely', 'intensely']
wc._ = ["    ", "   ", "  ", " "]

wc.energy = ['light', 'energy', 'glow', 'radiant']
wc.shard = ["mirror", "gemstone", "diamond", 'ornate', "crystallite", "ice", "rock", "glass", "crystal", 'stained-glass', "quartz"]
wc.coral = ["oceanic", "intricate", "brain", "colorful reef", "magical cursed reef"]
wc.wavy = ["wavy", "zig-zaggy", "stretchy", "droopy", "twisty", "contorted wavy", "marbling", "cursed", "coral", 'uneven', 'deformed']
wc.wave_t = ["huge", "separating", "ultra", "", "contorted"]
wc.shape = ["tornado", "helix", "galaxy"]
wc.texture = ["floral", "florally", "floraling", "inflorescent", "flowery"]
wc.adjective = ['intricate', 'detailed', 'beautiful', 'picturesque', 'immense', 'sunny', 'rainy', "melting", "iridescent", "opalescent", "magical"]
wc.related = ['', 'separating', 'repeating', 'alternating', 'overlapping']
wc.tex_layout = ['asymmetric tiling', 'symmetric tiling', 'symmetric', 'asymmetric', 'tiling']
wc.tex_shape = ['zig-zaggy', 'spiraling', 'contorting', 'stretching', "flower petal", "sunny", "coral", "crystalline water macro"]
