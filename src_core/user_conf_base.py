from munch import Munch, munchify

ip = '0.0.0.0'
port = 5000

precision = 'half'

print_timing = False
print_more = False

# Plugins to install
install = []

# Plugins to load on startup
startup = []


class Munch2(Munch):
    def __getattr__(self, item):
        # Create missing
        if item not in self:
            self[item] = Munch2()
        return self[item]


defaults = Munch2()
aliases = Munch2()
