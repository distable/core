import random
import re

from munch import Munch

import userconf
from src_core.classes.prompt_job import prompt_job
from src_core.classes.JobArgs import JobArgs
from src_core.plugins import plugjob
from src_core.classes.Plugin import Plugin

all_wildcards: dict[str, list[str]] = Munch()

if hasattr(userconf, 'wildcards'):
    all_wildcards = userconf.wildcards

class add_params(JobArgs):
    def __init__(self, wname: str, pool: list[str], **kwargs):
        super().__init__(**kwargs)
        self.name = wname
        self.pool = pool

        if pool is None and 'csv' in kwargs:
            self.pool = kwargs['csv'].split(',')


class rem_params(JobArgs):
    def __init__(self, wname: str, **kwargs):
        super().__init__(**kwargs)
        self.name = wname



class WildcardPlugin(Plugin):
    def title(self):
        return "Wildcards"

    def describe(self):
        return "A wildcard system to add some variety to your outputs. Type <wildcards> in your prompts to use"

    def load(self):
        pass

    def unload(self):
        pass

    @plugjob
    def add(self, p: add_params):
        all_wildcards[p.name] = p.pool

    @plugjob
    def rem(self, p: rem_params):
        all_wildcards.pop(p.name)

    @plugjob
    def apply(self, p: prompt_job):
        # Use re.match to match all <word>
        # Replace with random word from all_wildcards[word]
        ret = p.prompt
        for match in re.finditer(r"<(\w+)>", ret):
            word = match.group(1)
            if word in all_wildcards:
                original = match.group(0)
                replacement = random.choice(all_wildcards[word])
                ret = ret.replace(original, replacement)

        return ret
