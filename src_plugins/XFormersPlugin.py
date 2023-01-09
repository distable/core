import platform

from src_core.installing import pipargs
from src_core.classes.Plugin import Plugin


class XFormersPlugin(Plugin):
    def title(self):
        return "XFormers"

    def describe(self):
        return "Handle XFormers installation for other plugins."

    def install(self, args):