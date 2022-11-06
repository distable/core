import argparse
import sys


cparse = argparse.ArgumentParser()
cparse.add_argument("--dry", action='store_true', help="Only install and test the core, do not launch server.")
cparse.add_argument("--newplug", action='store_true', help="Run the plugin creation wizard.")

cargs = cparse.parse_args(sys.argv[1:])
