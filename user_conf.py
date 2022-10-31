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

# Plugin config
# ----------------------------------------

prompts = [
    "Beautiful painting of an ultra contorted landscape by <artist> and <artist>. airbrushed, 70s prog rock album cover, psychedelic, elaborate, complex",
    "A kawaii alien monster horror by <job>, inspirationalism, surrealism, horrorealism",
    "A really crazy and weird art piece by <artist>, <artist>, and <artist>",
    "Inside a strange <pow> contorted psychedelic cave with distance and horizon, airbrushed art, impressionism"
]

defaults.sd1111.txt2img = dict(p=lambda: random.choice(prompts), steps=22, cfg=7.65, sampler='euler-a', w=512, h=384)
defaults.sd1111.img2img = dict(chg=0.65)

wildcards = Munch()
# wildcards.artist = ['salvador dali', 'picasso', 'mark riddick', 'greg ruktowsky']
wildcards.artist = ['Aoshima Chiho', 'Arnegger Alois', 'shimoda hikari', 'terry redlin', 'Satoshi Kon', "hayao mizaki", 'rj palmer', 'alex grey', 'salvador dali']
wildcards.scene = ['realms', 'skies', 'planetary sky']
wildcards.distance = ['far away', 'in the distance', 'in the horizon']
wildcards.painted = ['painted', 'drawn', ' inked', 'designed']
wildcards.cloud = ['stratocumulus', 'altocumulus', 'cumulus', 'nimbostratus', 'cirrocumulus']
wildcards.glow = ['shining', 'glowing', 'radiating', 'exploding']
wildcards.blob = ['sun', 'moon', 'face', 'sunset', 'sunrise']
wildcards.movement = ['surrealism', 'hyperrealism', 'eccentrism']
wildcards.majestic = ['scenec', 'majestic', 'grandiose', 'picturesque', 'jawdropping']
wildcards.scale = ['huge', 'big', 'wide', 'scenic', 'large', 'impressive', 'grandiose', 'stunning', 'picturesque']
wildcards.cursed = ['cursed', 'twisted', 'contorted', 'strange', 'weird', 'rippled']
wildcards.contort = ['contorted', 'twisted', 'bending', 'uneven', 'chaotic']
wildcards.magic = ['magical', 'cursed', 'fantasy', 'mystical', 'enchanted']
wildcards.elaborate = ['elaborate', 'complex', 'detailed']
wildcards.view = ['shot', 'view']
wildcards.detail = ['complex', 'intricate', 'detailed']
wildcards.relate = ['rotating', 'sliding', 'moving', 'exploding', 'displacing', 'changing', 'transfmorphing', 'exchanging', 'expanding', 'stretching', 'condensing', 'tiling', 'alternating', 'juxtaposing', 'overlapping']
wildcards.pow = ['slightly', 'super', 'very', 'greatly', 'ultra', 'extremely', 'intensely']
wildcards._ = ["    ", "   ", "  ", " "]

wildcards.shard_t = ["mirror", "gemstone", "rock", "artistic"],
wildcards.coral_t = ["oceanic", "intricate", "brain", "colorful reef", "magical cursed reef"],
wildcards.mush_t = ["wavy", "droopy", "twisty", "contorted wavy", "marbling", "cursed", "coral"],
wildcards.wave_t = ["huge", "separating", "ultra", "", "contorted"],
wildcards.shape = ["tornado", "jungle", "helix", "galaxy"],
wildcards.texture = ["floral", "florally", "floraling", "inflorescent", "flowery"],
wildcards.adjective = ['intricate', 'detailed', 'beautiful', 'picturesque', 'immense', 'sunny', 'rainy', "melting", "iridescent", "opalescent", "magical"],
wildcards.symbol = ['separating', 'repeating', 'alternating', 'overlapping', 'contorting', 'flower', 'vegetation', 'overgrown', 'mechanical'],
wildcards.tex_layout = ['asymmetric tiling', 'symmetric tiling', 'symmetric', 'asymmetric', 'tiling'],
wildcards.tex_shape = ['zig-zaggy', 'spiraling', 'contorting', 'stretching', "flower petal", "sunny", "coral", "crystalline water macro"],
