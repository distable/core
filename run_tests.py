# this scripts installs necessary requirements and launches main program in webui.py

import core
from src_core import plugins, sessions

if __name__ == "__main__":
    core.init()

    from src_plugins.sd1111_plugin.sd_job import sd_txt

    for i in range(5):
        sessions.run(sd_txt(
                prompt="Beautiful painting of an ultra contorted landscape by Greg Ruktowsky and Salvador Dali. airbrushed, 70s prog rock album cover, psychedelic, elaborate, complex",
                cfg=7.75,
                steps=22,
                sampler='euler-a',
        ))
