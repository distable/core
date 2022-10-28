# this scripts installs necessary requirements and launches main program in webui.py
import os

import core
from src_core import server

if __name__ == "__main__":
    core.init()

    # Dry run, only install and exit.
    if core.cargs.dry:
        core.mprint("Exiting because of --dry argument")
        exit(0)

    # Start server
    server.run()
