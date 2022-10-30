# this scripts installs necessary requirements and launches main program in webui.py

from src_core import core, sessions

if __name__ == "__main__":
    core.init()
    sessions.job("dream", prompt="A <scale> <glow> galaxy painted by <artist>", n=8, cfg=7.75, steps=8, sampler='euler-a')