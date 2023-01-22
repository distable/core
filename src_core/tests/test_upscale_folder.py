# This script upscales an entire folder by
# opening it as a session
# ----------------------------------------

import user_conf
from src_core import core, plugins

folder_name = '/home/nuck/discore/sessions/2022-11-04_20h01/'

if __name__ == "__main__":
    assert user_conf.hasplug('kupscale'), "This test requires kupscale"

    core.init()
    core.open0(folder_name)
    while core.gs.image is not None:
        # core.jrun('img2img', prompt="A beautiful painting from upclose.")
        # core.save()
        core.run('kup', prompt="A beautiful painting from upclose.")
        core.save()
        core.next()