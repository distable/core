# --------------------------------------------------------------------------------
# This is a file intended to be the default configuration for stable-core.
# Important: user_conf.py must import * from this file
#            you may copy the line below
#
# from src_core.conf import *
# --------------------------------------------------------------------------------

from src_core.classes.Munch2 import Munch2

# Do not remove these, we are making them available by default in user_conf.py
from munch import Munch
import random



ip = '0.0.0.0'
port = 5000

precision = 'half'

print_timing = False
print_more = False

# Plugins to install
install = []

# Plugins to load on startup
startup = []

# Munch2 allows recursive dot notation assignment (all parents automatically created)
defaults = Munch2()
aliases = Munch2()
