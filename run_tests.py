# this scripts installs necessary requirements and launches main program in webui.py

from src_core import core, plugins, sessions


def run_tests():
    # Note: tests on my machine (oxysoft) with my conf
    from src_plugins.math2d_plugin.Math2DPlugin import zoom_job

    core.install_core()
    core.download_plugins()
    core.create_plugins()

    assert len(plugins.get_jobs()) > 0

    core.install_plugins()
    core.launch_plugins()

    assert plugins.get_job("txt2img")
    assert plugins.get_job("dream").alias

    sessions.job("dream", prompt="A <scale> <glow> galaxy painted by <artist>", n=8, cfg=7.75, steps=8, sampler='euler-a')
    # sessions.job(zoom_job(2))
    # sessions.job("dream")


if __name__ == "__main__":
    core.init()
    run_tests()
