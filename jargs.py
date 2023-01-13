import argparse
import sys

argp = argparse.ArgumentParser()

# Add positional argument 'project' for the script to run. optional, 'project' by default
argp.add_argument("session", nargs="?", default=None, help="Session or script")
argp.add_argument("action", nargs="?", default=None, help="Script or action to run")
argp.add_argument("subdir", nargs="?", default='', help="Subdir in the session")

argp.add_argument('--run', action='store_true', help='Perform the run in a subprocess')
argp.add_argument("--recreate_venv", action="store_true")
argp.add_argument("--no_venv", action="store_true")
argp.add_argument('--upgrade', action='store_true', help='Upgrade to latest version')
argp.add_argument('--install', action='store_true', help='Install plugins requirements and custom installations.')

argp.add_argument("--dry", action="store_true")
argp.add_argument("--newplug", action="store_true", help="Create a new plugin with the plugin wizard")

# Script Arguments
argp.add_argument('--fps', type=int, default=30, help='FPS')
argp.add_argument('--frames', type=str, default=None, help='The frames to render in first:last format')
argp.add_argument('--w', type=int, default=None, help='The target width.')
argp.add_argument('--h', type=int, default=None, help='The target height.')
argp.add_argument('--music', type=str, default=None, help='Music file to play in video export')
argp.add_argument('--music_start', type=float, default=0, help='Music start time in seconds')
argp.add_argument('--mpv', action='store_true', help='Open the resulting video in MPV.')

# Deployment
argp.add_argument('--shell', action='store_true', default=None, help='Open a shell in the deployed remote.')
argp.add_argument('--local', action='store_true', help='Deploy locally. (test)')
argp.add_argument('--vastai', action='store_true', help='Deploy to VastAI.')
argp.add_argument('--vastai_search', type=str, default=None, help='Search for a VastAI server')
argp.add_argument('--vastai_continue', action='store_true', help='rm -rf the deployment and start anew.')
argp.add_argument('--vastai_copy', action='store_true', help='Copy files even with vastai_continue')

args = argp.parse_args()
original_args = sys.argv[1:]
spaced_args = ' '.join([f'"{arg}"' for arg in original_args])

# Eat up arguments
sys.argv = [sys.argv[0]]
