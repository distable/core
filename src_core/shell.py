import click
from click_shell import shell

# @click.group()  # no longer
from src_core import sessions

bg_jobs = False


@shell(prompt='\n> ')
def run():
    pass


@run.command("exit")
def exit_cmd():
    exit(0)


@run.command("bg")
def bg():
    global bg_jobs
    bg_jobs = True


@run.command(context_settings=dict(ignore_unknown_options=True, allow_extra_args=True))
@click.pass_context
def txt2img(c):
    from src_plugins.sd1111_plugin.sd_job import sd_txt
    
    kw = dict()
    for a in c.args:
        print(a)
        kw.update([a.split('=')])

    # plugins.run(cmd='txt2img', **kw)
    sessions.run(sd_txt(job_repeats=kw['repeats'], **kw))
