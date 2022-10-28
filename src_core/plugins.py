import importlib
import inspect
import os
import sys
import traceback
import types
from pathlib import Path

# Constants
from bunch import Bunch

from src_core import paths, printlib
from src_core.jobs import Job, JobParams
from src_core.PipeData import PipeData
from src_core.printlib import print_bp

mprint = printlib.make_print("plugin")
mprinterr = printlib.make_printerr("plugin")


# region Library

class PlugjobToken:
    """
    A token return from a decorator to mark the function.
    It will be transformed in the constructor and replaced back
    with an actual function.
    """

    def __init__(self, func):
        self.func = func


def plugjob(func):
    """
    Decorate job functions with this to register them.
    """
    return PlugjobToken(func)


class Plugin:
    def __init__(self, dirpath: Path = None, id: str = None):
        if dirpath is not None and id is None:
            self.dir = Path(dirpath)
            self.id = id or Path(self.dir).stem
        elif id is not None:
            self.dir = None
            self.id = id
        else:
            raise ValueError("Either dirpath or id must be specified")

        self.jobs = Bunch()

        # Iterate all our attributes and transform JobTokens into functions
        for attr in dir(self):
            val = getattr(self, attr)

            if isinstance(val, PlugjobToken):
                mprint(f"Registering {attr} job")
                self.jobs[val.func.__name__] = val.func
                setattr(self, attr, val.func)

    # region API
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

    def gitclone(self, giturl, name, commithash):
        # TODO clone into temporary dir and move if successful

        from src_core.installing import run, git

        path = self.repos(name)

        if path.exists():
            if commithash is None:
                # Already installed
                return

            # Check if we have the right commit
            current_hash = run(f'"{git}" -C {path} rev-parse HEAD', None, f"Couldn't determine {name}'s hash: {commithash}").strip()
            if current_hash == commithash:
                return

            # Install the new commit
            run(f'"{git}" -C {path} fetch', f"Fetching updates for {name}...", f"Couldn't fetch {name}")
            run(f'"{git}" -C {path} checkout {commithash}', f"Checking out commint for {name} with hash: {commithash}...", f"Couldn't checkout commit {commithash} for {name}")
            return

        else:
            run(f'"{git}" clone "{giturl}" "{path}"', f"Cloning {name} into {path}...", f"Couldn't clone {name}")

            if commithash is not None:
                run(f'"{git}" -C {path} checkout {commithash}', None, "Couldn't checkout {name}'s hash: {commithash}")

            sys.path.append(path)

    def handles_job(self, job: JobParams | Job):
        if isinstance(job, Job):
            job = job.param

    def new_job(self, name, jobparams: JobParams):
        """
        Return a new job
        """
        from src_core import jobs
        j = jobs.new_job(self.id, name, jobparams)
        j.plugid = self.id
        return j

    # endregion

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

    def install(self):
        """
        Install the plugin, using the plugin's own API to do so.
        """
        pass

    def uninstall(self):
        """
        Uninstall the plugin.
        """
        pass

    def load(self):
        """
        Load the models and other things into memory, such that the plugin is ready for processng.
        If enabled on startup in user_conf, this runs right after the UI is launched.
        """
        pass


def fullid_to_short(plugid):
    if isinstance(plugid, Path):
        plugid = plugid.as_posix()

    if '/' in plugid:  # Convert "owner/id" to "id
        plugid = plugid.split('/')[-1]

    return plugid


# endregion


def list_ids():
    """
    Return a list of all plugins (string IDs only)
    """
    return [plug.id for plug in plugins]


def __call__(plugin):
    return get(plugin)


def get(plugid):
    """
    Get a plugin instance by ID.
    """

    if isinstance(plugid, Plugin):
        return plugid

    plugid = fullid_to_short(plugid)

    for plugin in plugins:
        if plugin.id == plugid:
            return plugin

    return None


def info(plugid):
    """
    Get a plugin's info by ID
    """
    plug = get(plugid)
    if plug:
        return dict(id=plug.id,
                    jobs=plug.jobs(),
                    title=plug.title(),
                    description=plug.describe())

    return None


def download(urls: list[str]):
    """
    Download plugins from GitHub into paths.plugindir
    """
    from src_core import installing

    for url in urls:
        url = Path("https://github.com/") / Path(url) / '.git '
        installing.gitclone(url, paths.plugins)


def load_path(path: Path):
    """
    Manually load a plugin at the given path
    TODO this function is probably shite
    """
    import inspect

    # Find classes that extend Plugin in the module
    try:
        # sys.path.append(path.as_posix())
        plugin_dirs.append(path)

        # from types import ModuleType
        # # this is missing a bunch of module files for some reason...
        # mod = importlib.import_module(f'modules.{path.stem}')
        #
        # for name, obj in inspect.getmembers(mod):
        #     if inspect.isclass(obj) and issubclass(obj, Plugin):
        #         mprint(f"Found plugin: {obj}")
        #         # Instantiate the plugin
        #         plugin = obj(dirpath=path)
        #         plugins.append(plugin)

        # TODO probably gonna have to detect by name instead (class & file name must be the same, and end with 'Plugin', e.g. StableDiffusionPlugin)

        if not any(['plugin' in Path(f).name.lower() for f in os.listdir(path)]):
            return

        for f in path.iterdir():
            if f.is_file() and f.suffix == '.py':
                mod = importlib.import_module(f'src_plugins.{path.stem}.{f.stem}')
                for name, member in inspect.getmembers(mod):
                    if inspect.isclass(member) and issubclass(member, Plugin) and not member == Plugin:
                        # Instantiate the plugin using __new__
                        # mprint(f"Loaded plugin: {obj}")
                        plugin = member(dirpath=path)
                        plugins.append(plugin)
                        plugin.init()

                        # create directories
                        plugin.res().mkdir(parents=True, exist_ok=True)
                        plugin.logs().mkdir(parents=True, exist_ok=True)

    except Exception as e:
        mprinterr(f"Couldn't load plugin {path.name}:")
        # mprint the exception e and its full stacktrace
        excmsg = ''.join(traceback.format_exception(None, e, e.__traceback__))
        mprinterr(excmsg)
        plugin_dirs.remove(path)


def load_urls(urls: list[str]):
    """
    Load plugins from a list of URLs
    """
    for url in urls:
        load_path(paths.plugins / fullid_to_short(url))


def load_dir(loaddir: Path):
    """
    Load all plugin directories inside loaddir.
    """
    if not loaddir.exists():
        return

    # Read the modules from the plugin directory
    for p in loaddir.iterdir():
        if p.is_dir() and not p.stem.startswith('__'):
            load_path(p)

    mprint(f"Loaded {len(plugins)} plugins:")
    for plugin in plugins:
        print_bp(f"{plugin.id} ({plugin.dir})")


def invoke(plugin, function, default=None, error=False, msg=None, *args, **kwargs):
    """
    Invoke a plugin, may return a job object.
    """
    try:
        plug = get(plugin)
        if not plug:
            if error:
                mprinterr(f"Plugin '{plugin}' not found")
            return default

        attr = getattr(plug, function, None)
        if not attr:
            if error:
                mprinterr(f"Plugin {plugin} has no attribute {function}")

            return default

        if msg:
            mprint(msg.format(id=plug.id))

        return attr(*args, **kwargs)
    except Exception:
        mprinterr(f"Error calling: {plugin}.{function}")
        mprinterr(traceback.format_exc())


def broadcast(name, msg=None, *args, **kwargs):
    """
    Dispatch a function call to all plugins.
    """
    for plugin in plugins:
        plug = get(plugin)
        if msg and plug:
            mprint(f"  - {msg.format(id=plug.id)}")

        invoke(plugin, name, None, False, None, *args, **kwargs)


plugin_dirs = []  # Plugin infos (script class, filepath)
plugins = []  # Loaded modules


def all_jobs():
    return [job for plugin in plugins for job in plugin.jobs.values()]


def make_jobparams_for_cmd(cmd, kwargs):
    """
    Instantiate job parameters for a matching job.
    Args:
        cmd: The name of the job.
        kwargs: The parameters for the JobParams' constructor.

    Returns: A new JobParams of the matching type.
    """
    for jfunc in all_jobs():
        if jfunc.__name__ == cmd:
            for p in inspect.signature(jfunc).parameters.values():
                if '_empty' not in str(p.annotation):
                    ptype = type(p.annotation)
                    if ptype == type:
                        return p.annotation(**kwargs)
                    elif ptype == types.ModuleType:
                        mprinterr("Make sure to use the type, not the module, when annotating jobparameters with @plugjob.")


def run(params: JobParams | None = None, cmd: str = None, print=None, **kwargs) -> PipeData | None:
    """
    Run a job, automatically dispatching to the plugin that can handle this job params.
    For string commands, we check for preferences in user_conf.
    """
    print = print if print is not None else mprint

    if params is None and cmd is not None:
        params = make_jobparams_for_cmd(cmd, kwargs)


    jobtype = type(params)

    func = None
    for plugin in plugins:
        members = inspect.getmembers(plugin, inspect.isfunction)
        for name, fn in members:
            # Check if any of the member's arguments is the same type as params
            for p in inspect.signature(fn).parameters.values():
                if p.annotation == jobtype:
                    # mprint("run ", p, params)
                        print(f" run {plugin.id} {params}")
                        return fn(plugin, params)

        print(members)

    mprinterr(f"Couldn't find a plugin to handle job: {params}")
