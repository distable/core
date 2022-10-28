import sys
from pathlib import Path

# script_path = Path(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

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

sys.path.insert(0, root.as_posix())

# search for directory of stable diffusion in following places
# path_dirs = [
#     (sd_path, 'ldm', 'Stable Diffusion', []),
#     (os.path.join(sd_path, '../taming-transformers'), 'taming', 'Taming Transformers', []),
#     (os.path.join(sd_path, '../CodeFormer'), 'inference_codeformer.py', 'CodeFormer', []),
#     (os.path.join(sd_path, '../BLIP'), 'models/blip.py', 'BLIP', []),
#     (os.path.join(sd_path, '../k-diffusion'), 'k_diffusion/SDSampler.py', 'k_diffusion', ["atstart"]),
# ]
#
# paths = {}
#
# for d, must_exist, what, options in path_dirs:
#     path = Path(script_path, d, must_exist).resolve()
#
#     if not path.exists():
#         print(f"Warning: {what} not found at path {path}", file=sys.stderr)
#     else:
#         d = os.path.abspath(d)
#         if "atstart" in options:
#             sys.path.insert(0, d)
#         else:
#             sys.path.append(d)
#
#         paths[what] = d

# Convert above loop to a function (

