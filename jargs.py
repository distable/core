import argparse
import sys

argp = argparse.ArgumentParser()

# Add positional argument 'project' for the script to run. optional, 'project' by default
argp.add_argument("session", nargs="?", default=None, help="Session or script")
argp.add_argument("action", nargs="?", default=None, help="Script or action to run")
argp.add_argument("subdir", nargs="?", default='', help="Subdir in the session")

argp.add_argument('--run', action='store_true', help='Perform the run in a subprocess')
argp.add_argument('--cli', action='store_true', help='Run the renderer in CLI mode. (no gui)')
argp.add_argument('--remote', action='store_true', help='Indicates that we are running remotely.')
argp.add_argument('--dev', action='store_true', help='Use a development environment to test the setup.')
argp.add_argument('--ryusig', action='store_true', help='Use the ryusig calculator.')
argp.add_argument('--print', action='store_true', help='Enable printing.')
argp.add_argument('--trace', action='store_true', help='Enable tracing.')
argp.add_argument('--trace_jobs', action='store_true', help='Profile each job one by one.')
argp.add_argument('--trace_session_run', action='store_true', help='Profile session.run')
argp.add_argument('--trace_session_load', action='store_true', help='Profile session.load')
argp.add_argument('--trace_run_job', action='store_true', help='Profile jobs.run')
argp.add_argument("--recreate_venv", action="store_true")
argp.add_argument("--no_venv", action="store_true")
argp.add_argument('--upgrade', action='store_true', help='Upgrade to latest version')
argp.add_argument('--install', action='store_true', help='Install plugins requirements and custom installations.')

argp.add_argument("--dry", action="store_true")
argp.add_argument("--newplug", action="store_true", help="Create a new plugin with the plugin wizard")

# Renderer arguments
argp.add_argument('--zip_every', type=int, default=None, help='Create a zip of the frames every specified number of frames.')
argp.add_argument('--preview_every', type=int, default=None, help='Create a preview video every number of frames. (with ffmpeg)')
argp.add_argument('--preview_command', type=str, default='', help='The default ffmpeg command to use for preview videos.')

# Script Arguments
argp.add_argument('--fps', type=int, default=None, help='FPS to use for FFMPEG video. By default the session FPS is detected.')
argp.add_argument('--frames', '-f', type=str, default=None, help='The frames to render in first:last,first:last,... format')
argp.add_argument('--w', type=str, default=None, help='The target width for FFMPEG video. By default the session width is detected.')
argp.add_argument('--h', type=str, default=None, help='The target height for FFPMEG video. By default the session height is detected.')
argp.add_argument('--music', '-mu', type=str, default=None, help='Music to use for FFMPEG video. By default the session music is detected.')
argp.add_argument('--music_start', type=float, default=None, help='Music start time in seconds.')
argp.add_argument('--mpv', action='store_true', help='Open the resulting FFMPEG video in MPV.')

# Deployment
argp.add_argument('--shell', action='store_true', default=None, help='Open a shell in the deployed remote.')
argp.add_argument('--local', action='store_true', help='Deploy locally. (test)')
argp.add_argument('--vastai', '--vai', action='store_true', help='Deploy to VastAI.')
argp.add_argument('--vastai_continue', '--vaic', action='store_true', help='rm -rf the deployment and start anew.')
argp.add_argument('--vastai_copy', '--vaicp', action='store_true', help='Copy files even with vastai_continue')
argp.add_argument('--vastai_search', '--vais', type=str, default=None, help='Search for a VastAI server')
argp.add_argument('--vastai_no_download', '--vaindl', action='store_true', help='Prevent downloading during copy step.')

argv = sys.argv[1:]
args = argp.parse_known_args()
argvr = args[1]
args = args[0]

spaced_args = ' '.join([f'"{arg}"' for arg in argv])

# Eat up arguments
sys.argv = [sys.argv[0]]

is_vastai = args.vastai or args.vastai_continue


def get_discore_session(load=True, *, nosubdir=False):
    from src_core.classes.Session import Session
    s = Session(args.session or args.action or args.script, load=load)
    if not nosubdir:
        s = s.subsession(args.subdir)

    return s


def framerange():
    if args.frames:
        ranges = args.frames.split('-')
        for r in ranges:
            yield r
    else:
        yield args.frames
