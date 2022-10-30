from pathlib import Path

from src_core import paths
from src_core.classes.JobInfo import JobInfo
from src_core.classes.PlugjobDeco import PlugjobDeco


class Plugin:
    def __init__(self, dirpath: Path = None, id: str = None):
        self.jobs = []

        # Determine our ID and directory
        if dirpath is not None and id is None:
            self.dir = Path(dirpath)
            self.id = id or Path(self.dir).stem
        elif id is not None:
            self.dir = None
            self.id = id
        else:
            raise ValueError("Either dirpath or id must be specified")

        # If the id ends with a suffix, remove it
        for suffix in paths.plugin_suffixes:
            if self.id.endswith(suffix):
                self.id = self.id[:-len(suffix)]

        # Discover PlugjobDecos to register our jobs
        for attr in dir(self):
            token = getattr(self, attr)
            if isinstance(token, PlugjobDeco):
                jname = attr
                jfunc = token.func

                # Register job
                # mprint(f"Registering {attr} job")
                self.jobs.append(JobInfo(f'{self.id}.{jname}', jfunc, self))

                # Register aliases
                # mprint(f"Registering aliases: {token.aliases}")
                if token.aliases is not None:
                    for alias in token.aliases:
                        self.jobs.append(JobInfo(alias, jfunc, self, alias=True))

                # Revert our function to the original decorated func
                setattr(self, attr, jfunc)

    def info(self):
        """
        Get a plugin's info by ID
        """
        return dict(id=self.id,
                    jobs=self.jobs,
                    title=self.title(),
                    description=self.describe())

    def res(self, join='') -> Path:
        """
        Returns: The resource directory for this plugin
        """
        return paths.plug_res / self.id / join

    def logs(self, join='') -> Path:
        """
        Returns: The log directory for this plugin
        """
        return paths.plug_logs / self.id / join

    def repos(self, join):
        """
        Returns: The git repo dependencies directory for this plugin
        """
        return paths.plug_repos / self.id / join

    def title(self):
        """
        Display title of the plugin, for UI purposes.
        """
        raise NotImplementedError()

    def describe(self):
        """Description of the plugin, for UI purposes"""
        return ""

    def init(self):
        """
        Perform some initialization, use this instead of __init__
        """
        pass

    def load(self):
        """
        Load the models and other things into memory, such that the plugin is ready for processng.
        If enabled on startup in user_conf, this runs right after the UI is launched.
        """
        pass
