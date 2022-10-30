import inspect
from pathlib import Path

import click
from click_shell import shell

# @click.group()  # no longer

bg_jobs = False


@click.group()
def cli():
    pass


@shell(prompt='\n> ')
def run():
    from src_core import plugins, sessions
    for plug in plugins.plugins:
        ctor = type(plug).__init__
        pluggroup = click.group(name=plug.id)
        for ifo in plugins.get_jobs():
            # The template function for every job
            def job_cmd(c):
                kw = dict()
                for a in c.args:
                    print(a)
                    kw.update([a.split('=')])

                # plugins.run(cmd='txt2img', **kw)
                sessions.job(plugins.make_params(ifo.jid, kw))

            # Annotate the function as we normally would with @
            job_command = click.pass_context(job_cmd)
            job_command = run.command(ifo.jid, context_settings=dict(ignore_unknown_options=True, allow_extra_args=True))(job_command)
    pass


@run.command("exit")
def exit_cmd():
    exit(0)


@run.command("bg")
def bg():
    global bg_jobs
    bg_jobs = True


# @run.command("server")
# @run.command(context_settings=dict(ignore_unknown_options=True, allow_extra_args=True))
# @click.pass_context
# def server(c, command):
#     from src_core import server
#     print("other args:", c.args)
#
#     # Send a command to the server
#     try:
#         if hasattr(server, command):
#             ret = getattr(server, command)()
#             print(ret)
#     except Exception as e:
#         print(e)


@run.command()
def new_plugin(name: str, classname: str):
    """
    Copy src_plugins/.template to src_plugins/{name}
    """

    src = Path('src_plugins') / '.template'
    dst = Path('src_plugins') / name

    import shutil
    shutil.copytree(src, dst)

    # Rename TemplatePlugin to {classname}
    import os
    classnamepy = dst / f'{classname}.py'
    os.rename(dst / 'TemplatePlugin.py', classnamepy)

    # Rename TemplatePlugin to {classname} in the file
    with open(classnamepy, 'r') as f:
        data = f.read()

    data = data.replace('TemplatePlugin', classname)
    with open(classnamepy, 'w') as f:
        f.write(data)
