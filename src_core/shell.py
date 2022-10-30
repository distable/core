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
    def flatten(l):
        return [item for sublist in l for item in sublist]

    short_jids = set([j.short_jid for j in plugins.get_jobs()])
    for jid in short_jids:
        ifo = plugins.get_job(jid, short=True)

        print(jid, ifo.jid)

        # The template function for every short_jid
        def job_cmd(c, **kwargs):
            kw = dict()
            for a in c.args:
                print(a)
                kw.update([a.split('=')])

            # plugins.run(cmd='txt2img', **kw)
            # print(ifo.jid)
            sessions.job(plugins.new_params(ifo.jid, kw))

        # Annotate the function as we normally would with @
        job_cmd = click.pass_context(job_cmd)
        job_cmd = run.command(jid, context_settings=dict(ignore_unknown_options=True, allow_extra_args=True))(job_cmd)

        pclasses = ifo.get_paramclass()
        pclasses = [pclasses, *inspect.getmro(ifo.get_paramclass())]

        args = flatten([inspect.signature(cls).parameters.values() for cls in pclasses])
        # print(args)
        for arg in args:
            job_cmd = click.option(f'--{arg.name}', required=False, type=arg.annotation)(job_cmd)


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
