import inspect

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

# @run.command(context_settings=dict(ignore_unknown_options=True, allow_extra_args=True))
# @click.pass_context
# def txt2img(c):
#     from src_plugins.sd1111_plugin.sd_job import sd_txt
#
#     kw = dict()
#     for a in c.args:
#         print(a)
#         kw.update([a.split('=')])
#
#     # plugins.run(cmd='txt2img', **kw)
#     sessions.run(sd_txt(job_repeats=kw.get("repeats", 1), **kw))
