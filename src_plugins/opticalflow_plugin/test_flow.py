import user_conf

from src_core import core
from src_core.classes.Session import Session

if __name__ == "__main__":
    assert user_conf.hasplug('sd1111'), "This test requires sd1111"
    assert user_conf.hasplug('opticalflow'), "This test requires optical flow"

    core.init()

    # TODO Merge PipeData into Session, we dont need that complexity
    # TODO override getattrib in order to get function call for run

    s = Session('flow_testing')
    s.maxsize(512)
    s.set(s.res_frame('bee'))
    s.flow_init(name='bee')
    s.run('flow_init', name='bee')
    for i in range(20000):
        s.set(s.res_frame('bee'))
        s.run('img2img', steps=24, chg=0.3, cfg=11.5, p="Orange Flowers, [[by van gogh]], by Salvador Dali and Greg Ruktowsky", sampler='dpmpp-2m-ka')
        s.run('flow')
        s.run('ccblend')
        # s.run('consistency', init_video='bee.mov')
        s.add()
        # core.run('consistency')
        # core.run('palette', pal=base_pal)
