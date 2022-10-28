import platform

from src_core.installing import pipargs
from src_core.plugins import Plugin


class XFormersPlugin(Plugin):
    def title(self):
        return "XFormers"

    def describe(self):
        return "Handle XFormers installation for other plugins."

    def install(self, args):
        if platform.python_version().startswith("3.10"):
            if platform.system() == "Windows":
                pipargs("install https://github.com/C43H66N12O12S2/stable-diffusion-webui/releases/download/c/xformers-0.0.14.dev0-cp310-cp310-win_amd64.whl", "xformers")
            elif platform.system() == "Linux":
                pipargs("install xformers", "xformers")