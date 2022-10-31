# this scripts installs necessary requirements and launches main program in webui.py

from src_core import core, plugins, sessions


def run_tests():
    # Note: tests on my machine (oxysoft) with my conf
    core.setup_annoying_logging()
    core.setup_args()

    core.install_core()
    core.download_plugins()
    core.create_plugins()

    assert len(plugins.get_jobs()) > 0
    assert plugins.get_job("txt2img")
    assert plugins.get_job("dream").alias

    from src_plugins.math2d_plugin.Math2DPlugin import zoom_job
    from src_plugins.sd1111_plugin.sd_job import sd_txt
    assert isinstance(plugins.new_params("dream"), sd_txt)
    sd_txt()

    core.install_plugins()
    core.launch_plugins()

    # sessions.job("dream", n=2)
    sessions.job("dream", prompt="A <scale> <glow> galaxy painted by <artist>", n=8, cfg=7.75, steps=8, sampler='euler-a')
    # sessions.job(zoom_job(2))
    # sessions.job("dream")


if __name__ == "__main__":
    # core.init()
    run_tests()
