# plugins.py
#
# The plugin management for the core.
# We load the plugins here and index their jobs.
# We also provide service features like downloading
# plugins from GitHub and installing them.

# DEFINITIONS
#
# jid: a job ID, which can be either in the full form 'plug/job' or just 'job'
# pid: a plugin ID, which can be either in the full form 'user/repository' or just 'repository'
# jquery: a job query which can be either a jid,
# ------------------------------------------------------------

import importlib
import inspect
import os
import threading
import time
import traceback
from pathlib import Path

from yachalk.ansi import Color, wrap_ansi_16

import user_conf
from src_core import installing
from src_core.classes import paths
from src_core.classes.Job import Job
from src_core.classes.JobArgs import JobArgs
from src_core.classes.JobInfo import JobInfo
from src_core.classes.logs import logplugin, logplugin_err
from src_core.classes.paths import short_pid, split_jid
from src_core.classes.Plugin import Plugin
from src_core.classes.PlugjobDeco import PlugjobDeco
from src_core.classes.printlib import print_bp, print

# STATE
# ----------------------------------------

plugin_dirs = []  # Plugin infos
plugins = []  # Loaded modules
num_loading = 0


def download(urls: list[str], log=False):
    """
    Download plugins from GitHub into paths.plugindir
    """
    from src_core import installing

    for pid in urls:
        url = pid
        if 'http' not in pid and "github.com" not in pid:
            url = f'https://{Path("github.com/") / pid}'

        installing.gitclone(url, paths.plugins)
        if log:
            logplugin(" -", url)


import functools


def plugjob(func=None, key=None, aliases=None):
    if func is None:
        return functools.partial(plugjob, key=key, aliases=aliases)

    return PlugjobDeco(func, key, aliases)


# def plugjob(func=None, key=None, aliases=None):
#     """
#     Decorate your job functions with this to register them.
#     """


def __call__(plugin):
    """
    Allows plugins.get_plug(query) like this plugins(query) instead
    """
    return get_plug(plugin)


def get_plug(query, search_jobs=False):
    """
    Get a plugin instance by JobArgs, pid, or jid
    """

    if isinstance(query, Plugin):
        return query

    if isinstance(query, JobArgs):
        for plugin in plugins:
            for jname, jfunc in plugin.jobs.items():
                # If any parameter matches the jobargs
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
                ret.insert(0, JobInfo(alias, ifo.func, ifo.plug, alias=True, key=ifo.key))
                break

    return ret


def create_plugin_at(path: Path):
    """
    Load a plugin, which is a python package/directory.
    Special files are expected:
        - __init__.py: the main plugin file
        - __install__.py: the install script
        - __uninstall__.py: the uninstall script
        - __conf__.py: the configuration options for the plugin
    """
    import inspect

    if not path.exists():
        return

    try:
        plugin_dirs.append(path)

        # We need a plugin file
        if not any(['plugin' in Path(f).name.lower() for f in os.listdir(path)]):
            return

        # Get the short pid of the plugin
        pid = path.stem
        for suffix in paths.plugin_suffixes:
            if pid.endswith(suffix):
                pid = pid[:-len(suffix)]

        # Import __install__
        installing.current_parent = pid
        try:
            importlib.import_module(f'src_plugins.{path.stem}.__install__')
        except:
            pass

        installing.current_parent = None

        # Unpack user_conf into __conf__
        try:
            confmod = importlib.import_module(f'src_plugins.{path.stem}.__conf__')
            for k, v in user_conf.plugins[pid].opt.items():
                setattr(confmod, k, v)
        except:
            pass

        classtype = None
        for f in path.iterdir():
            if f.is_file() and f.suffix == '.py':
                mod = importlib.import_module(f'src_plugins.{path.stem}.{f.stem}')
                for name, member in inspect.getmembers(mod):
                    if inspect.isclass(member) and issubclass(member, Plugin) and not member == Plugin:
                        classtype = member

        if classtype is None:
            logplugin_err(f'No plugin class found in {path}')
            return

        # Instantiate the plugin using __new__
        plugin = classtype(dirpath=path)
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


def create_plugins_by_url(urls: list[str]):
    """
    Load plugins from a list of URLs
    """
    for pid in urls:
        if not create_plugin_at(paths.plugins / short_pid(pid)):
            for suffix in paths.plugin_suffixes:
                create_plugin_at(paths.plugins / (short_pid(pid) + suffix))


def create_plugins_in(loaddir: Path, log=False):
    """
    Load all plugin directories inside loaddir.
    """
    if not loaddir.exists():
        return

    # Read the modules from the plugin directory
    for p in loaddir.iterdir():
        if p.is_dir() and not p.stem.startswith('__'):
            create_plugin_at(p)

    if log:
        logplugin(f"Loaded {len(plugins)} plugins:")
    for plugin in plugins:
        print_bp(f"{plugin.id} ({plugin.dir})")


def get_job(jquery, short=True, resolve=False) -> JobInfo | None:
    """
    Get a JobInfo from a job query, e.g. 'sd1111.img2img' or 'img2img' with short is enabled.
        Args:
            jquery:
            short: Allow short matches, e.g. 'img2img' will match 'sd1111.img2img'
            resolve: Resolve aliases to the actual job

        Returns:
    """

    ret = None

    def get():
        if isinstance(jquery, JobInfo):
            return jquery
        elif issubclass(type(jquery), JobArgs):
            for ifo in get_jobs():
                for pm in inspect.signature(ifo.func).parameters.values():
                    if '_empty' not in str(pm.annotation):
                        if type(jquery) == pm.annotation:
                            return ifo
            for ifo in get_jobs():
                for pm in inspect.signature(ifo.func).parameters.values():
                    if '_empty' not in str(pm.annotation):
                        if issubclass(type(jquery), pm.annotation):
                            return ifo
            for ifo in get_jobs():
                for pm in inspect.signature(ifo.func).parameters.values():
                    if '_empty' not in str(pm.annotation):
                        if isinstance(jquery, pm.annotation):
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


def new_args(jquery: JobArgs | str | None = None, uconf_defaults=True, **kwargs):
    """
    Create new JobArgs for a job query or kwargs.
    Args:
        jquery: the job query to create for. If this is already a jobargs, return it.
        uconf_defaults: apply defaults from user_conf.
        **kwargs:

    Returns:
    """
    if isinstance(jquery, JobArgs):
        return jquery
    else:
        ifo = get_job(jquery, resolve=True)
        kwargs = get_args(jquery, kwargs, uconf_defaults)

        return ifo.new_args(kwargs)


def get_args(jquery, kwargs, uconf_defaults):
    ifo = get_job(jquery, resolve=True)

    # Flatten kwargs onto the user defaults
    if uconf_defaults:
        mod = mod2dic(user_conf)
        pid = ifo.plug.id
        _, jid = split_jid(ifo.jid)

        opt = mod['plugins'].get(pid, None).opt
        kwargs = {**opt.get(jid, {}), **kwargs}

        if isinstance(ifo.key, str):
            kwargs = {**opt.get(ifo.key, None), **kwargs}
    return kwargs


def mod2dic(module):
    return {k: getattr(module, k) for k in dir(module) if not k.startswith('_')}


def new_job(jquery: JobArgs | str | None = None, **kwargs) -> Job | None:
    """
    Create a new Job for a job query.
    """
    ifo = get_job(jquery)
    jargs = new_args(jquery, **kwargs)
    jargs.job = Job(ifo.jid, jargs)

    return jargs.job


def run(jquery: JobArgs | str | None = None, require_loaded=False, ifo=None, **kwargs) -> object | None:
    """
    Run a job with the given query and kwargs.
    """
    ifo = ifo or get_job(jquery)
    jargs = new_args(jquery, **kwargs)

    if require_loaded and not ifo.plug.loaded:
        print("Waiting for plugin to load...")
        while not ifo.plug.loaded:
            pass

    return ifo.func(ifo.jid, jargs)


def broadcast(name, msg=None, threaded=False, on_before=None, on_after=None, *args, **kwargs):
    """
    Dispatch a function call to all plugins.
    """

    def _invoke(plug):
        global num_loading
        num_loading += 1
        if on_before: on_before(plug)
        invoke(plug, name, None, False, None, *args, **kwargs) or ret
        if on_after: on_after(plug)
        num_loading -= 1

    ret = None
    for plugin in plugins:
        plug = get_plug(plugin)
        if msg and plug:
            logplugin(" -", msg.format(id=plug.id))

        if threaded:
            threading.Thread(target=_invoke, args=(plug,)).start()
        else:
            print(wrap_ansi_16(Color.gray.on), end="")
            _invoke(plug)
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


def wait_loading():
    """
    Wait for all plugins to finish loading.
    """
    while num_loading > 0:
        time.sleep(0.1)
