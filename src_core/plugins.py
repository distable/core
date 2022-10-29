import importlib
import inspect
import os
import traceback
import types
from pathlib import Path

from bunch import Bunch
from yachalk.ansi import Color, wrap_ansi_16

import user_conf
from src_core import installing, jobs, paths, printlib
from src_core.jobs import Job, JobParams
from src_core.PipeData import PipeData
from src_core.printlib import print_bp

# Constants

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

        if self.id.endswith("_plugin"):
            self.id = self.id[:-7]

        self.jobs = Bunch()

        # Iterate all our attributes and transform JobTokens into functions
        for attr in dir(self):
            val = getattr(self, attr)

            if isinstance(val, PlugjobToken):
                # mprint(f"Registering {attr} job")
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

    def handles_job(self, job: JobParams | Job):
        if isinstance(job, Job):
            job = job.params

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


def jid_to_jname(plugid):
    """
    Convert 'user/repository' to 'repository'
    """
    if isinstance(plugid, Path):
        plugid = plugid.as_posix()

    if '/' in plugid:
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


def get(query):
    """
    Get a plugin instance by ID.
    """

    if isinstance(query, Plugin):
        return query

    query = jid_to_jname(query)

    for plugin in plugins:
        if plugin.id.startswith(query):
            return plugin

    return None


def info(plugid):
    """
    Get a plugin's info by ID
    """
    plug = get(plugid)
    if plug:
        return dict(id=plug.id,
                    jobs=plug.jobs,
                    title=plug.title(),
                    description=plug.describe())

    return None


def download(urls: list[str]):
    """
    Download plugins from GitHub into paths.plugindir
    """
    from src_core import installing

    for uid in urls:
        url = f'https://{Path("github.com/") / uid}'
        installing.gitclone(url, paths.plugins)


def load_path(path: Path):
    """
    Manually load a plugin at the given path
    """
    import inspect

    if not path.exists():
        return

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
                installing.current_parent = path.stem
                if installing.current_parent.endswith('_plugin'):
                    installing.current_parent = installing.current_parent[:-7]

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
        load_path(paths.plugins / jid_to_jname(url))
        # _plugin is optional
        load_path(paths.plugins / f"{jid_to_jname(url)}_plugin")


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

        print(wrap_ansi_16(Color.gray.on), end="")
        invoke(plugin, name, None, False, None, *args, **kwargs)
        print(wrap_ansi_16(Color.gray.off), end="")


plugin_dirs = []  # Plugin infos (script class, filepath)
plugins = []  # Loaded modules


class JobInfo:
    def __init__(self, jid=None, jfunc=None, jplug=None, alias=False):
        self.jid = jid
        self.func = jfunc
        self.plug = jplug
        self.alias = alias


def resolve_and_split_jid(jid, allow_jobonly=False) -> tuple[str, str]:
    ifo = resolve_job(jid)
    return split_jid(ifo.jid, allow_jobonly)


def split_jid(uid, allow_jobonly=False) -> tuple[str, str]:
    """
    Split a plugin UID into a tuple of (owner, repo)
    """
    if '.' in uid:
        s = uid.split('.')
        return s[0], s[1]

    if allow_jobonly:
        return None, uid

    raise ValueError(f"Invalid plugin UID: {uid}")


def get_jobs() -> list[JobInfo]:
    """
    Return all jobs including aliases.
    e.g.:
    sd1111.txt2img
    sd1111.img2img
    dream
    imagine
    """
    ret = []
    for plug in plugins:
        ret.extend([JobInfo(f'{plug.id}.{jname}', jfunc, plug) for jname, jfunc in plug.jobs.items()])

    # Add user aliases
    for alias, uid in user_conf.aliases.items():
        # Find the original plugjob to point to
        alias_pname, alias_jname = split_jid(uid)  # sd1111,txt2img OR sd1111_plugin,txt2img
        for ifo in ret:
            if ifo.plug.id.startswith(alias_pname) and (ifo.jid == alias_jname or ifo.jid.endswith(f'.{alias_jname}')):
                ret.insert(0, JobInfo(alias, ifo.func, ifo.plug, alias=True))
                break

    return ret


# def all_jobs():
#     return [job for plugin in plugins for job in plugin.jobs.values()]


def get_jobentry(munch, ifo):
    """
    Find the entry for a plugin in a list of tuple (id, dict) where id is 'pluginid' or 'plugid.jobname'
    """
    jplug, jname = resolve_and_split_jid(ifo.jid, True)
    if ifo.plug.id in munch:
        v = munch[ifo.plug.id]
        if jname in v:
            return v[jname]
    return dict()


def get_job(query) -> JobInfo:
    for ifo in get_jobs():
        if query == ifo.jid:
            return ifo

    return None


def resolve_job(query):
    ifo = get_job(query)
    if not ifo.alias:
        return ifo
    else:
        for j in get_jobs():
            if not j.alias and ifo.func == j.func:
                return j

    return None


def get_jidparams(jid):
    ifo = resolve_job(jid)
    if ifo is None:
        return None

    for p in inspect.signature(ifo.func).parameters.values():
        if '_empty' not in str(p.annotation):
            ptype = type(p.annotation)
            if ptype == type:
                return p.annotation
            elif ptype == types.ModuleType:
                mprinterr("Make sure to use the type, not the module, when annotating jobparameters with @plugjob.")
            else:
                mprinterr(f"Unknown jobparameter type: {ptype}")


def make_jobparams_by_jid(jid, kwargs, user_defaults=True):
    """
    Instantiate job parameters for a matching job.
    Args:
        user_defaults:
        jid: The name of the job.
        kwargs: The parameters for the JobParams' constructor.

    Returns: A new JobParams of the matching type.
    """
    clas = get_jidparams(jid)
    ifo = resolve_job(jid)

    # Flatten kwargs onto the user defaults
    if user_defaults:
        entry = get_jobentry(user_conf.defaults, ifo)
        if entry:
            kwargs = {**entry, **kwargs}

    # Bake the job parameters
    for k, v in kwargs.items():
        if callable(v):
            kwargs[k] = v()
    return clas(**kwargs)


def get_job_function(params: JobParams | None = None, jid: str = None, print=None, **kwargs):
    print = print if print is not None else mprint

    if params is None and jid is not None:
        params = make_jobparams_by_jid(jid, kwargs)

    for j in get_jobs():
        for p in inspect.signature(j.func).parameters.values():
            if '_empty' not in str(p.annotation):
                if isinstance(params, p.annotation):
                    return j

    return None


def job_to_plugin(job: str | JobParams):
    if isinstance(job, JobParams):
        for plugin in plugins:
            for jname, jfunc in plugin.jobs.items():
                # If any parameter matches the jobparams
                sig = inspect.signature(jfunc)
                for p in sig.parameters.values():
                    if '_empty' not in str(p.annotation):
                        if isinstance(job, p.annotation):
                            return plugin
    elif isinstance(job, str):
        for plugin in plugins:
            for jname, jfunc in plugin.jobs.items():
                if job == jname:
                    return plugin

    pass


def new_job(params: JobParams | None = None, cmd: str = None, print=None, **kwargs) -> Job | None:
    j = get_job_function(params, cmd, print, **kwargs)
    return jobs.new_job(j.jid, params)


def run(params: JobParams | None = None, cmd: str = None, print=None, **kwargs) -> PipeData | None:
    j = get_job_function(params, cmd, print, **kwargs)
    return j.func(j.jid, params)
