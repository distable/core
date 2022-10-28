print_timing = False

ip = '0.0.0.0'
port = 5000

# Plugins to install
plugins = [
    'stablecore-ai/sd1111_plugin',
]

# Plugins to load on startup
startup = [
    'sd1111_plugin',
]

defaults = [
    ('txt2img', 'sd1111_plugin'),
]
