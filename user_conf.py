

from src_core.conf import *

# Values
# ----------------------------------------

precision = "full"
print_timing = False
print_more = False

# Plugins
# ----------------------------------------

install = ['distable/sd1111_plugin']
startup = ['sd1111', 'wildcard']

aliases.dream = 'sd1111.txt2img'
aliases.imagine = 'sd1111.txt2img'

# Job Defaults
# ----------------------------------------

prompts = [
    "Beautiful painting of an ultra contorted landscape by Greg Ruktowsky and Salvador Dali. airbrushed, 70s prog rock album cover, psychedelic, elaborate, complex",
    "A kawaii alien monster horror by salvador dali, inspirationalism, surrealism, horrorealism",
    "A really crazy and weird art piece by salvador dali, picasso, and mark riddick",
    "Inside a strange contorted psychedelic cave with distance and horizon, airbrushed art, impressionism"
]

defaults.sd1111.txt2img = dict(p=lambda: random.choice(prompts), steps=22, cfg=7.65, sampler='euler-a', w=512, h=384)
defaults.sd1111.img2img = dict(chg=0.65)

# Plugin config
# ----------------------------------------

wildcard = Munch()
# wildcard.artist = ['salvador dali', 'picasso', 'mark riddick', 'greg ruktowsky']
wildcard.artist=['Aoshima Chiho', 'Arnegger Alois', 'shimoda hikari', 'terry redlin', 'Satoshi Kon', "hayao mizaki", 'rj palmer', 'alex grey', 'salvador dali']
wildcard.scene = ['realms', 'skies', 'planetary sky']
wildcard.distance = ['far away', 'in the distance', 'in the horizon']
wildcard.painted = ['painted', 'drawn', ' inked', 'designed']
wildcard.cloud = ['stratocumulus', 'altocumulus', 'cumulus', 'nimbostratus', 'cirrocumulus']
wildcard.glow = ['shining', 'glowing', 'radiating', 'exploding']
wildcard.blob = ['sun', 'moon', 'face', 'sunset', 'sunrise']
wildcard.movement = ['surrealism', 'hyperrealism', 'eccentrism']
wildcard.majestic = ['scenec', 'majestic', 'grandiose', 'picturesque', 'jawdropping']
wildcard.scale = ['huge', 'big', 'wide', 'scenic', 'large', 'impressive', 'grandiose', 'stunning', 'picturesque']
wildcard.cursed = ['cursed', 'twisted', 'contorted', 'strange', 'weird', 'rippled']
wildcard.contort = ['contorted', 'twisted', 'bending', 'uneven', 'chaotic']
wildcard.magic = ['magical', 'cursed', 'fantasy', 'mystical', 'enchanted']
wildcard.elaborate = ['elaborate', 'complex', 'detailed']
wildcard.view = ['shot', 'view']
wildcard.detail = ['complex', 'intricate', 'detailed']
wildcard.artist = ['Arnegger Alois']
wildcard.relate = ['rotating', 'sliding', 'moving', 'exploding', 'displacing', 'changing', 'transfmorphing', 'exchanging', 'expanding', 'stretching', 'condensing', 'tiling', 'alternating', 'juxtaposing', 'overlapping']
wildcard.pow = ['slightly', 'super', 'very', 'greatly', 'ultra', 'extremely', 'intensely']
wildcard._ = ["    ", "   ", "  ", " "]

wildcard.shard_t = ["mirror", "gemstone", "rock", "artistic"],
wildcard.coral_t = ["oceanic", "intricate", "brain", "colorful reef", "magical cursed reef"],
wildcard.mush_t = ["wavy", "droopy", "twisty", "contorted wavy", "marbling", "cursed", "coral"],
wildcard.wave_t = ["huge", "separating", "ultra", "", "contorted"],
wildcard.shape = ["tornado", "jungle", "helix", "galaxy"],
wildcard.texture = ["floral", "florally", "floraling", "inflorescent", "flowery"],
wildcard.adjective = ['intricate', 'detailed', 'beautiful', 'picturesque', 'immense', 'sunny', 'rainy', "melting", "iridescent", "opalescent", "magical"],
wildcard.symbol = ['separating', 'repeating', 'alternating', 'overlapping', 'contorting', 'flower', 'vegetation', 'overgrown', 'mechanical'],
wildcard.tex_layout = ['asymmetric tiling', 'symmetric tiling', 'symmetric', 'asymmetric', 'tiling'],
wildcard.tex_shape = ['zig-zaggy', 'spiraling', 'contorting', 'stretching', "flower petal", "sunny", "coral", "crystalline water macro"],
