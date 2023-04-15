# --------------------------------------------------------------------------------
# This is a file intended to be the default configuration for stable-core.
# Important: user_conf.py must import * from this file
#            you may copy the line below
#
# from src_core.conf import *
# --------------------------------------------------------------------------------

from src_core.classes.Munchr import Munchr

# Do not remove these, we are making them available by default in user_conf.py
from munch import Munch
import random

from src_core.classes.paths import short_pid

# Core
# ------------------------------------------------------------
ip = '0.0.0.0'
port = 5000
share = False

precision = 'half'

print_timing = False
print_more = False

# Use the functions below to define plugins and aliases
# ------------------------------------------------------------
plugins = Munchr()
aliases = Munchr()

def choice(values):
    def func(*args, **kwargs):
        return random.choice(values)

    return func


def plugdef(url):
    from src_core.classes.paths import short_pid

    pid = short_pid(url)
    mdef = Munchr()
    mdef.url = url
    mdef.load = False

    globals()['plugins'][pid] = mdef
    return mdef


def plugload(url):
    opt = plugdef(url)
    pid = short_pid(url)
    opt.load = True
    return opt

def hasplug(pid):
    return pid in plugins

def aliasdef(**kwaliases):
    for k, v in kwaliases.items():
        globals()['aliases'][k] = v
