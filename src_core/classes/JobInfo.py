import inspect
import types

from src_core import paths
from src_core.classes import JobParams
from src_core.paths import split_jid
from src_core.lib.printlib import printerr


class JobInfo:
    def __init__(self, jid=None, jfunc=None, jplug=None, alias=False):
        self.jid = jid
        self.func = jfunc
        self.plug = jplug
        self.alias = alias

    @property
    def short_jid(self):
        plug,job = paths.split_jid(self.jid, True)
        return job

    def get_paramclass(self):
        """
        Get the parameters for a job query
        """
        for p in inspect.signature(self.func).parameters.values():
            if '_empty' not in str(p.annotation):
                ptype = type(p.annotation)
                if ptype == type:
                    return p.annotation
                elif ptype == types.ModuleType:
                    printerr("Make sure to use the type, not the module, when annotating jobparameters with @plugjob.")
                else:
                    printerr(f"Unknown jobparameter type: {ptype}")

    def get_jobentry(self, munch):
        """
        Find the entry for a plugin in a list of tuple (id, dict) where id is 'plug.job' or 'job'
        """
        jplug, jname = split_jid(self.jid, True)
        if self.plug.id in munch:
            v = munch[self.plug.id]
            if jname in v:
                return v[jname]
        return dict()

    def new_params(self, kwargs) -> JobParams:
        """
        Instantiate job parameters for a matching job.
        Args:
            kwargs: The parameters for the JobParams' constructor.

        Returns: A new JobParams of the matching type.
        """

        # Bake the job parameters
        for k, v in kwargs.items():
            if callable(v):
                kwargs[k] = v()

        return self.get_paramclass()(**kwargs)
