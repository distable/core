import src_core.logs
from src_core import core, server

if __name__ == "__main__":
    core.init()

    # Dry run, only install and exit.
    if core.cargs.dry:
        src_core.logs.logcore("Exiting because of --dry argument")
        exit(0)

    # Start server
    server.run()
