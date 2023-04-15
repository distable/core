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

import userconf
from jargs import args
from src_core import installing
from src_core.classes import paths
from src_core.classes.Job import Job
from src_core.classes.JobArgs import JobArgs
from src_core.classes.JobInfo import JobInfo
from src_core.classes.logs import logplugin, logplugin_err
from src_core.classes.paths import short_pid, split_jid
from src_core.classes.Plugin import Plugin
from src_core.classes.PlugjobDeco import PlugjobDeco
from src_core.classes.printlib import cpuprofile, print_bp, print
from src_core.classes.prompt_job import prompt_job
from src_core.classes.printlib import trace
from src_core import installing

# STATE
# ----------------------------------------

plugin_dirs = []  # Plugin infos
alls = []  # Loaded plugins
all_jobs = []  # Loaded plugin jobs
all_jobs_jid = {}  # Loaded plugin jobs by jid (both full and short)
all_jobs_type = {}  # Loaded plugin jobs by arg type
num_loading = 0
loadings = []


def download_git_urls(urls: list[str], log=False):
    """
    Download plugins from GitHub into paths.plugindir
    """
    for pid in urls:
        url = pid
        if '/' in url:
            if 'http' not in pid and "github.com" not in pid:
                url = f'https://{Path("github.com/") / pid}'

            installing.gitclone(url, into_dir=paths.code_plugins)
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
    return get(plugin)


def get(query, search_jobs=False, instantiate=True):
    """
    Get a plugin instance by JobArgs, pid, or jid
    """

    if isinstance(query, Plugin):
        return query

    if isinstance(query, JobArgs):
        for plugin in alls:
            for jname, jfunc in plugin.jobs.items():
                # If any parameter matches the jobargs
                sig = inspect.signature(jfunc)
                for p in sig.parameters.values():
                    if '_empty' not in str(p.annotation):
                        if isinstance(query, p.annotation):
                            return plugin

    if isinstance(query, str):
        # pid search
        for plugin in alls:
            pid = short_pid(query)
            if plugin.id.startswith(pid):
                return plugin

        # job search
        if search_jobs:
            for plugin in alls:
                for jname, jfunc in plugin.jobs.items():
                    if query == jname:
                        return plugin

        if instantiate:
            for p in iter_plugins(paths.plugins):
                if p.stem == query:
                    plug = instantiate_plugin_at(p, True)
                    load(plug)
                    return plug


    return None


def instantiate_plugin_at(path: Path, install=True):
    """
    Create the plugin, which is a python package/directory.
    Special files are expected:
        - __init__.py: the main plugin file
        - __install__.py: the install script
        - __uninstall__.py: the uninstall script
        - __conf__.py: the configuration options for the plugin
    """
    import inspect

    if not path.exists():
        return

    # Get the short pid of the plugin
    pid = paths.short_pid(path.stem)

    # Check if it's already loaded.
    matches = [p for p in alls if p.id == pid]
    if len(matches) > 0:
        print(f'Plugin {pid} is already loaded.')
        return matches[0]

    try:
        plugin_dirs.append(path)

        # Install requirements
        # ----------------------------------------
        reqpath = (paths.code_plugins / pid / 'requirements.txt')
        if install and reqpath.exists():
            with trace(f'src_plugins.{path.stem}.requirements.txt'):
                print(f'Installing requirements for {pid}...')
                installing.pipreqs(reqpath)

        # Import __install__ -
        # Note: __install_ is still called even if we are not
        #       installing so that we can still declare our installations
        # ----------------------------------------
        installing.skip_installations = not install
        installing.default_basedir = paths.plug_repos / pid
        try:
            with trace(f'src_plugins.{path.stem}.__install__'):
                importlib.import_module(f'src_plugins.{path.stem}.__install__')
        except:
            pass

        installing.default_basedir = None

        # Unpack user_conf into __conf__ (timed)
        # ----------------------------------------
        try:
            with trace(f'src_plugins.{path.stem}.__conf__'):
                confmod = importlib.import_module(f'src_plugins.{path.stem}.__conf__')
                for k, v in userconf.plugins[pid].opt.items():
                    setattr(confmod, k, v)
        except:
            pass

        # NOTE:
        # We allow any github repo to be used as a discore plugin, they don't necessarily need to implement plugjobs
        # Hence we will now begin with real discore plugin instantiation
        if any(['plugin' in Path(f).name.lower() for f in os.listdir(path)]):
            classtype = None

            with trace(f'src_plugins.{path.stem}.find'):
                for f in path.iterdir():
                    if f.is_file() and f.suffix == '.py':
                        with trace(f'src_plugins.{path.stem}.{f.stem}'):
                            mod = importlib.import_module(f'src_plugins.{path.stem}.{f.stem}')
                            for name, member in inspect.getmembers(mod):
                                if inspect.isclass(member) and issubclass(member, Plugin) and not member == Plugin:
                                    classtype = member

            if classtype is None:
                logplugin_err(f'No plugin class found in {path}')
                return

            # Instantiate the plugin using __new__
            with trace(f'src_plugins.{path.stem}.instantiate'):
                plugin = classtype(dirpath=path)
                alls.append(plugin)
                plugin.init()

                # Add jobs
                all_jobs.extend(plugin.jobs)

                # Add user aliases
                for alias, uid in userconf.aliases.items():
                    # Find the original plugjob to point to
                    alias_pname, alias_jname = split_jid(uid)  # sd1111,txt2img OR sd1111_plugin,txt2img
                    for ifo in list(plugin.jobs):
                        if ifo.plugid.startswith(alias_pname) and (ifo.jid == alias_jname or ifo.jid.endswith(f'.{alias_jname}')):
                            all_jobs.insert(0, JobInfo(alias, ifo.func, get(ifo.plugid), is_alias=True, key=ifo.key))
                            break

                # Add long jid lookup
                for ifo in all_jobs:
                    all_jobs_jid[ifo.jid] = ifo

                # Add short jid lookup
                for ifo in all_jobs:
                    plug, job = split_jid(ifo.jid, True)
                    all_jobs_jid[job] = ifo

                # Add arg type lookup
                for ifo in all_jobs:
                    for pm in inspect.signature(ifo.func).parameters.values():
                        if '_empty' not in str(pm.annotation):
                            all_jobs_type[pm.annotation] = ifo

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


def instantiate_plugins_by_pids(urls: list[str], install=True):
    """
    Load plugins from a list of URLs
    """
    for pid in urls:
        instantiate_plugin_by_pid(pid, install)

def instantiate_plugin_by_pid(pid, install=True):
    if not instantiate_plugin_at(paths.plugins / short_pid(pid), install):
        for suffix in paths.plugin_suffixes:
            instantiate_plugin_at(paths.plugins / (short_pid(pid) + suffix), install)


def instantiate_plugins_in(loaddir: Path, log=False, install=True):
    """
    Load all plugin directories inside loaddir.
    """
    if not loaddir.exists():
        return

    # Read the modules from the plugin directory
    for p in iter_plugins(loaddir):
        instantiate_plugin_at(p, install)

    if log:
        logplugin(f"Loaded {len(alls)} plugins:")
        for plugin in alls:
            print_bp(f"{plugin.id} ({plugin._dir})")

def iter_plugins(loaddir):
    for p in loaddir.iterdir():
        if p.stem.startswith('.'):
            continue

        if p.is_dir() and not p.stem.startswith('__'):
            yield p


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
            v = all_jobs_type.get(type(jquery))
            if v is not None:
                return v

            # Try all superclasses
            for c in type(jquery).__mro__:
                v = all_jobs_type.get(c)
                if v is not None:
                    return v

        elif isinstance(jquery, str):
            return all_jobs_jid.get(jquery, None)


    ret = get()
    if ret is not None and ret.is_alias and resolve:
        for j in all_jobs:
            if not j.is_alias and ret.func == j.func:
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
        pid = ifo.plugid
        _, jid = split_jid(ifo.jid)

        opt = userconf.plugins.get(pid, None)
        if opt is not None:
            # Userconf defaults by exact jid
            opts = opt.get(jid, None)
            if opts:
                kwargs = {**opts, **kwargs}

        # Userconf defaults by key
        if isinstance(ifo.key, str):
            opts = opt.get(ifo.key, None)
            if opts:
                kwargs = {**opts, **kwargs}
    return kwargs


def mod2dic(module):
    return {k: getattr(module, k) for k in dir(module) if not k.startswith('_')}


def new_job(jquery: JobArgs | str | None = None, **kwargs) -> Job | None:
    """
    Create a new Job for a job query.
    """
    jargs = new_args(jquery, **kwargs)
    jargs.job = Job(get_job(jquery).jid, jargs)

    return jargs.job


def process_prompt(prompt):
    ret = run(prompt_job(prompt=prompt), required=False)
    if ret is not None:
        return ret
    else:
        return prompt


def run(jquery: JobArgs | str | None = None, require_loaded=False, ifo=None, required=True, **kwargs) -> object | None:
    """
    Run a job with the given query and kwargs.
    """
    ifo = ifo or get_job(jquery)
    jargs = new_args(jquery, **kwargs)

    if ifo is None:
        if not required:
            return None
        else:
            raise ValueError(f"Couldn't find job for {jquery}.")

    plug = get(ifo.plugid)
    if require_loaded and not plug.loaded:
        if plug not in loadings:
            load(plug)

        print("Waiting for plugin to load...")
        while not plug.loaded:
            pass

    from yachalk import chalk

    # jargs_str = {k: v for k, v in jargs.__dict__.items() if isinstance(v, (int, float, str))}
    # jargs_str = ' '.join([f'{chalk.white(k)}={chalk.grey(v)}' for k, v in jargs_str.items()])

    # logplugin("start", chalk.blue(ifo.jid), jargs_str)
    with cpuprofile(args.trace_jobs):
        ret = ifo.func(plug, jargs)
    # logplugin("end", chalk.lue(ifo.jid), jargs_str)
    return ret



def userconf_wants_loading(plugid):
    return plugid in userconf.plugins and userconf.plugins[plugid].load

def load(plug=None, userconf_only=True):
    def on_before(plug):
        loadings.append(plug)

    def on_after(plug):
        plug.loaded = True
        loadings.remove(plug)

    if plug is not None:
        # Load single plug
        if plug.loaded:
            logplugin_err(f"Plugin {plug.id} is already loaded!")
            return

        on_before(plug)
        invoke(plug, "load")
        on_after(plug)
    else:
        # Load all
        broadcast("load", "{id}",
                  threaded=False,
                  on_before=on_before,
                  on_after=on_after,
                  filter=lambda plug: not plug.loaded and (not userconf_only or userconf_wants_loading(plug.id)))


def broadcast(name,
              msg=None,
              threaded=False,
              on_before=None,
              on_after=None,
              filter=None,
              *args, **kwargs):
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
    for plugin in alls:
        if filter and not filter(plugin):
            continue

        plug = get(plugin)
        # if msg and plug:
        #     logplugin(" -", msg.format(id=plug.id))

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
        plug = get(plugin)
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
            plugin(msg.formagreyt(id=plug.id))

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
