import re
from pathlib import Path

root = Path(__file__).resolve().parent.parent

# Contains the user's downloaded plugins (cloned from github)
plugins = root / 'src_plugins'

# Contains the resources for each plugin, categorized by plugin id
plug_res = root / 'plug-res'

# Contains the logs output by each plugin, categorized by plugin id
plug_logs = root / 'plug-logs'

# Contains the repositories cloned by each plugin, categorized by plugin id
plug_repos = root / 'plug-repos'

# Image outputs are divied up into 'sessions'
# Session logic can be customized in different ways:
#   - One session per client connect
#   - Global session on a timeout
#   - Started manually by the user
sessions = root / 'sessions'

plug_res.mkdir(exist_ok=True)
plug_logs.mkdir(exist_ok=True)
plug_repos.mkdir(exist_ok=True)
sessions.mkdir(exist_ok=True)

# These suffixes will be stripped from the plugin IDs for simplicity
plugin_suffixes = ['_plugin']

# sys.path.insert(0, root.as_posix())


def format_session_id(name, num=None):
    if num is None:
        num = get_next_leadnum(directory=sessions)

    return f"{num:0>3}_{name}"


def get_next_leadnum(iterator=None, separator='_', directory=None):
    """
    Find the largest 'leading number' in the directory names and return it
    e.g.:
    23_session
    24_session
    28_session
    23_session

    return value is 28
    """
    iterator = iterator if iterator is not None else directory.iterdir()

    if isinstance(iterator, Path):
        if not iterator.exists():
            return 1
        iterator = iterator.iterdir()

    biggest = 0
    for path in iterator:
        if not path.is_dir():
            match = re.match(r"^(\d+)" + separator, path.name)
            if match is not None:
                num = int(match.group(1))
                if match:
                    biggest = max(biggest, num)

    return biggest + 1


def short_pid(pid):
    """
    Convert 'user/repository' to 'repository'
    """
    if isinstance(pid, Path):
        pid = pid.as_posix()

    if '/' in pid:
        pid = pid.split('/')[-1]

    # Strip suffixes
    for suffix in plugin_suffixes:
        if pid.endswith(suffix):
            pid = pid[:-len(suffix)]

    return pid


def split_jid(jid, allow_jobonly=False) -> tuple[str:None, str]:
    """
    Split a plugin jid into a tuple of (plug, job)
    """
    if '.' in jid:
        s = jid.split('.')
        return s[0], s[1]

    if allow_jobonly:
        return None, jid

    raise ValueError(f"Invalid plugin jid: {jid}")
