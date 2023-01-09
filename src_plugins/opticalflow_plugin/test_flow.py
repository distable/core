import user_conf
from src_core import core, plugins

if __name__ == "__main__":
    assert user_conf.hasplug('sd1111'), "This test requires sd1111"
    assert user_conf.hasplug('opticalflow'), "This test requires optical flow"

    from src_plugins.disco_party.maths import *
    from src_plugins.disco_party.partyutils import *

    core.init(restore='flow_testing')
    core.run('copyframe', name='bee')
    core.run('maxsize', w=512)

    core.run('flow_init', name='bee')
    for i in range(20000):
        core.run('copyframe', name='bee')
        core.run('img2img', steps=24, chg=0.3, cfg=11.5, p="Orange Flowers, [[by van gogh]], by Salvador Dali and Greg Ruktowsky", sampler='dpmpp-2m-ka')
        core.run('flow')
        core.run('ccblend')
        # core.run('consistency', init_video='bee.mov')
        core.add()
        # core.run('consistency')
        # core.run('palette', pal=base_pal)
