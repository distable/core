import importlib
import inspect
import os
import traceback
from pathlib import Path

from yachalk.ansi import Color, wrap_ansi_16

import user_conf
from src_core import installing, jobs
from src_core.classes import paths
from src_core.classes.Job import Job
from src_core.classes.JobInfo import JobInfo
from src_core.classes.JobParams import JobParams
from src_core.classes.Plugin import Plugin
from src_core.classes.PlugjobDeco import PlugjobDeco
from src_core.logs import logplugin, logplugin_err
from src_core.classes.paths import short_pid, split_jid
from src_core.classes.printlib import print_bp

# STATE
# ----------------------------------------

plugin_dirs = []  # Plugin infos
plugins = []  # Loaded modules


# DEFINITIONS
#
# jid: a job ID, which can be either in the full form 'plug/job' or just 'job'
# pid: a plugin ID, which can be either in the full form 'user/repository' or just 'repository'
# jquery: a job query which can be either a jid,
# ----------------------------------------


def download(urls: list[str]):
    """
    Download plugins from GitHub into paths.plugindir
    """
    from src_core import installing

    for pid in urls:
        url = pid
        if 'http' not in pid and "github.com" not in pid:
            url = f'https://{Path("github.com/") / pid}'

        installing.gitclone(url, paths.plugins)
        logplugin(" -", url)


def plugjob(func, aliases=None):
    """
    Decorate your job functions with this to register them.
    """
    return PlugjobDeco(func, aliases)


def __call__(plugin):
    """
    Allows plugins.get(query) like this plugins(query) instead
    """
    return get_plug(plugin)


def get_plug(query, search_jobs=False):
    """
    Get a plugin instance by JobParams, pid, or jid
    """

    if isinstance(query, Plugin):
        return query

    if isinstance(query, JobParams):
        for plugin in plugins:
            for jname, jfunc in plugin.jobs.items():
                # If any parameter matches the jobparams
                sig = inspect.signature(jfunc)
                for p in sig.parameters.values():
                    if '_empty' not in str(p.annotation):
                        if isinstance(query, p.annotation):
                            return plugin

    if isinstance(query, str):
        # pid search
        for plugin in plugins:
            pid = short_pid(query)
            if plugin.id.startswith(pid):
                return plugin

        # job search
        if search_jobs:
            for plugin in plugins:
                for jname, jfunc in plugin.jobs.items():
                    if query == jname:
                        return plugin

    return None


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
        ret.extend(plug.jobs)

    # Add user aliases
    for alias, uid in user_conf.aliases.items():
        # Find the original plugjob to point to
        alias_pname, alias_jname = split_jid(uid)  # sd1111,txt2img OR sd1111_plugin,txt2img
        for ifo in ret:
            if ifo.plug.id.startswith(alias_pname) and (ifo.jid == alias_jname or ifo.jid.endswith(f'.{alias_jname}')):
                ret.insert(0, JobInfo(alias, ifo.func, ifo.plug, alias=True))
                break

    return ret


def load_plugin_at(path: Path):
    """
    Load a plugin at the given path
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

                        return plugin

    except Exception as e:
        logplugin_err(f"Couldn't load plugin {path.name}:")
        # mprint the exception e and its full stacktrace
        excmsg = ''.join(traceback.format_exception(None, e, e.__traceback__))
        logplugin_err(excmsg)
        plugin_dirs.remove(path)


def load_plugins_by_url(urls: list[str]):
    """
    Load plugins from a list of URLs
    """
    for pid in urls:
        if not load_plugin_at(paths.plugins / short_pid(pid)):
            for suffix in paths.plugin_suffixes:
                load_plugin_at(paths.plugins / (short_pid(pid) + suffix))


def load_plugins_in(loaddir: Path):
    """
    Load all plugin directories inside loaddir.
    """
    if not loaddir.exists():
        return

    # Read the modules from the plugin directory
    for p in loaddir.iterdir():
        if p.is_dir() and not p.stem.startswith('__'):
            load_plugin_at(p)

    logplugin(f"Loaded {len(plugins)} plugins:")
    for plugin in plugins:
        print_bp(f"{plugin.id} ({plugin.dir})")


def get_job(jquery, short=True, resolve=False) -> JobInfo | None:
    """
    Get a JobInfo from a job query, e.g. 'sd1111.img2img' or 'img2img' with partial is enabled.
        Args:
            jquery:
            short: Allow partial matches, e.g. 'img2img' will match 'sd1111.img2img'
            resolve: Resolve aliases to the actual job

        Returns:
    """

    ret = None
    def get():
        if isinstance(jquery, JobInfo):
            return jquery
        elif issubclass(type(jquery), JobParams):
            for ifo in get_jobs():
                for param in inspect.signature(ifo.func).parameters.values():
                    if '_empty' not in str(param.annotation):
                        if isinstance(jquery, param.annotation):
                            return ifo
                        elif issubclass(type(jquery), param.annotation):
                            return ifo
        elif isinstance(jquery, str):
            for ifo in get_jobs():
                if jquery == ifo.jid:
                    return ifo

            if ret is None and short:
                for ifo in get_jobs():
                    plug, job = split_jid(ifo.jid, True)
                    if job == jquery:
                        return ifo

    ret = get()
    if ret is not None and ret.alias and resolve:
        for j in get_jobs():
            if not j.alias and ret.func == j.func:
                return j

    return ret


def new_params(jquery: JobParams | str | None = None, uconf_defaults=True, **kwargs):
    """
    Create new JobParams for a job query or kwargs.
    Args:
        jquery: the job query to create for. If this is already a jobparams, return it.
        uconf_defaults: apply defaults from user_conf.
        **kwargs:

    Returns:

    """

    if isinstance(jquery, JobParams):
        return jquery
    else:
        ifo = get_job(jquery, short=True, resolve=True)

        # Flatten kwargs onto the user defaults
        if uconf_defaults:
            entry = ifo.get_jobentry(user_conf.defaults)
            if entry:
                kwargs = {**entry, **kwargs}

        return ifo.new_params(kwargs)


def new_job(jquery: JobParams | str | None = None, **kwargs) -> Job | None:
    """
    Create a new Job for a job query or kwargs.
    """
    ifo = get_job(jquery, short=True)
    params = new_params(jquery, partial=True, **kwargs)

    return jobs.new_job(ifo.jid, params)


def run(jquery: JobParams | str | None = None, **kwargs) -> object | None:
    """
    Run a job with the given query and kwargs.
    """
    ifo = get_job(jquery, short=True)
    params = new_params(jquery, partial=True, **kwargs)

    return ifo.func(ifo.jid, params)


def broadcast(name, msg=None, *args, **kwargs):
    """
    Dispatch a function call to all plugins.
    """
    ret = None
    for plugin in plugins:
        plug = get_plug(plugin)
        if msg and plug:
            logplugin(" -", msg.format(id=plug.id))

        print(wrap_ansi_16(Color.gray.on), end="")
        ret = invoke(plugin, name, None, False, None, *args, **kwargs) or ret
        print(wrap_ansi_16(Color.gray.off), end="")


def invoke(plugin, function, default=None, error=False, msg=None, *args, **kwargs):
    """
    Invoke a plugin, may return a job object.
    """
    try:
        plug = get_plug(plugin)
        if not plug:
            if error:
                logplugin_err(f"Plugin '{plugin}' not found")
            return default

        attr = getattr(plug, function, None)
        if not attr:
            if error:
                logplugin_err(f"Plugin {plugin} has no attribute {function}")

            return default

        if msg:
            plugin(msg.format(id=plug.id))

        return attr(*args, **kwargs)
    except Exception:
        logplugin_err(f"Error calling: {plugin}.{function}")
        logplugin_err(traceback.format_exc())
