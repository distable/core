import threading

from src_core.rendering import renderer
from src_core.rendering.renderer import script_path


def ryusig_loop():
    from src_plugins.ryusig_calc.RyusigApp import RyusigApp
    RyusigApp(callback_mod=renderer.script,
              callback_vars=renderer.v)


def ryusig_init():
    print("INIT")
    threading.Thread(target=ryusig_loop).start()
