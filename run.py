import re
import shutil
import subprocess

from art import text2art

from src_core.cargs import cargs
from src_core.classes import paths
from src_core.classes.logs import logcore, logwizard
from src_core import core, installing, server

# A run script to init the core and start the server.
# There is also an option to enter the plugin create wizard.
# ------------------------------------------------------------

if __name__ == "__main__":
    if cargs.newplug:
        PLUGIN_TEMPLATE_PREFIX = ".template"

        print(text2art("Plugin Wizard"))

        # Find template directories (start with .template)
        templates = []
        for d in paths.code_plugins.iterdir():
            if d.name.startswith(PLUGIN_TEMPLATE_PREFIX):
                templates.append(d)

        template = None

        if len(templates) == 0:
            print("No templates found.")
            exit(1)
        elif len(templates) == 1:
            template = templates[0]
        else:
            for i, path in enumerate(templates):
                s = path.name[len(PLUGIN_TEMPLATE_PREFIX):]
                while not s[0].isdigit() and not s[0].isalpha():
                    s = s[1:]
                print(f"{i + 1}. {s}")

            print()
            while template is None:
                try:
                    v = int(input("Select a template: ")) - 1
                    if v >= 0:
                        template = templates[v]
                except:
                    pass

        id = input("ID name: ")

        clsdefault = f"{id.capitalize()}Plugin"
        cls = input(f"Class name (default={clsdefault}): ")
        if not cls:
            cls = clsdefault

        plugdir = paths.code_plugins / f"{id}_plugin"
        clsfile = plugdir / f"{cls}.py"

        shutil.copytree(template.as_posix(), plugdir)
        shutil.move(plugdir / "TemplatePlugin.py", clsfile)

        # Find all {{word}} with regex and ask for a replacement
        regex = r'__(\w+)__'
        with open(clsfile, "r") as f:
            lines = f.readlines()
        for i, line in enumerate(lines):
            matches = re.findall(regex, line)
            if matches:
                vname = matches[0]

                # Default values
                vdefault = ''
                if vname == 'classname': vdefault = cls
                if vname == 'title': vdefault = id

                # Ask for a value
                if vdefault:
                    value = input(f"{vname} (default={vdefault}): ")
                else:
                    value = input(f"{vname}: ")

                if not value and vdefault:
                    value = vdefault

                # Apply the value
                lines[i] = re.sub(regex, value, line)

        # Write lines back to file
        with open(clsfile, "w") as f:
            f.writelines(lines)

        # Open plugdir in the file explorer
        installing.open_explorer(plugdir)
        print("Done!")
        input()
        exit(0)

    core.init()

    # Dry run, only install and exit.
    if cargs.dry:
        logcore("Exiting because of --dry argument")
        exit(0)

    # Start server
    server.run()
