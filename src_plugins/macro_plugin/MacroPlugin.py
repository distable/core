from src_core.jobs import JobPipe
from src_core.plugins import Plugin

class Macro:
    def __init__(self, name, jobs=None):
        self.name = name
        self.jobs = jobs or []


class MacroPlugin(Plugin):
    def init(self):
        self.macros = []

    def title(self):
        return "Macro Plugin"

    def save(self):
        # TODO save macros to a file
        pass

    def create(self):
        macro = Macro("New Macro")
        self.macros.append(macro)
        return macro

    def remove(self, macro):
        self.macros.remove(macro)

    def get(self, macro):
        if isinstance(macro, Macro):
            return macro
        elif isinstance(macro, str):
            for m in self.macros:
                if m.name == macro:
                    return m

    def exec(self, macro):
        macro = self.get(macro)
        pipe = JobPipe(macro)
