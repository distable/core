import user_conf
from src_core import core, plugins

if __name__ == "__main__":
    # assert user_conf.hasplug('sd1111'), "This test requires sd1111"

    core.init(autosave=False, restore=True)

    from src_plugins.disco_party.maths import *
    from src_plugins.disco_party.partyutils import *

    core.open('flow_testing')
    core.run('flow_init', init_video='bee.mov')


    # core.run0('txt2img', prompt="Beautiful contorted airbrushed landscape by salvador dali. 70s prog rock album cover, artistic, monochromatic, water reflections, green, poster print, acrylic, metallic, chrome, caustics, reflections, 90s cgi", sampler="dpmpp-2s-a")

    # Get base stats
    core.seek_min()
    base_comp = core.run('edgecomp')
    base_pal = core.image
    core.seek_max()

    # Configuration
    b_chg = (0.25, 0.75)
    b_cov = (0, 0.02)

    # Loopback for 200 frames
    for i in range(20000):
        # Target complexity & dynamic steps ----------------------------------------
        comp = core.run('edgecomp')
        chg = lerp(b_chg[1], b_chg[0], comp / base_comp)
        cov = lerp(b_cov[1], b_cov[0], comp / base_comp)
        kwprint(comp=comp, dist=dist, chg=chg)

        # Run the loopback ---------------------------------------------------------
        core.run('img2img', chg=chg, cfg=7.5)
        core.run('mat2d', zoom=0.01)
        core.add()
        core.run('spnoise', coverage=cov)
        core.run('cc',
                 hue=lerp(0.6, 0.75, perlin(core.f, 0.85)),
                 sat=lerp(0.6, 0.75, perlin(core.f, 1)),
                 val=lerp(0.6, 0.75, perlin(core.f, 0.25)),
                 speed=0.785)
        # core.run('palette', pal=base_pal)
