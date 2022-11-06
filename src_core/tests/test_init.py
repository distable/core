import user_conf
from src_core import core, plugins

if __name__ == "__main__":
    assert user_conf.hasplug('sd1111'), "This test requires sd1111"

    core.init(1)


    # Verify plugins have loaded
    assert plugins.plugins

    # Verify plugins have loaded
    assert plugins.get_jobs()

    # Verify SD plugin
    assert plugins.get_job("txt2img")
    assert plugins.get_job("dream").alias

    # Verify SD plugin
    from src_plugins.sd1111_plugin.sd_job import sd_txt

    assert isinstance(plugins.new_args("dream"), sd_txt)
    sd_txt()
