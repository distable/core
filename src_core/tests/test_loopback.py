import user_conf
from src_core import core, plugins

if __name__ == "__main__":
    assert user_conf.hasplug('sd1111'), "This test requires sd1111"

    core.init(autosave=False)

    p = "Beautiful contorted airbrushed landscape by salvador dali. 70s prog rock album cover, artistic, monochromatic, water reflections, green, poster print, acrylic, metallic, chrome, caustics, reflections, 90s cgi"

    if core.gsession.context.image is None:
        core.job('sd1111', p=p, chg=0.25)

    for i in range(200):
        core.job('img2img', p=p, chg=0.6, cfg=6.5)
        core.job('mat2d', zoom=0.01)
        core.save()
        core.job('spnoise')
        core.job('cc', brightness=0.85)
