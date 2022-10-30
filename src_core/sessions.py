from PIL.Image import Image

from src_core import jobs, plugins
from src_core.classes.prompt_job import prompt_job
from src_core.classes.JobParams import JobParams
from src_core.classes.Session import Session
from src_core.logs import logsession


def job(query: str | JobParams, **kwargs):
    """
    Run a job in the current session.
    Args:
        query:
        **kwargs:

    Returns:

    """
    def handler(dat):
        # Image output is moved into the context
        if isinstance(dat, Image):
            current.context.image = dat
        elif isinstance(dat, list) and isinstance(dat[0], Image):
            current.context.image = dat[0]

        # Prompt output is copied into the context
        dat = handler
        if isinstance(j.params, prompt_job) and isinstance(dat, str):
            current.context.prompt = j.params.prompt

        # and also saved to disk
        current.save_next(dat)

    j = plugins.new_job(query, **kwargs)

    # Prompt input is copied into the context
    j.handler = handler
    if hasattr(j.params, 'prompt') and j.params.prompt:
        current.context.prompt = j.params.prompt

    current.add_job(j)
    ret = jobs.enqueue(j)
    print("")


def run(query: JobParams | str | None = None, **kwargs):
    """
    Run a job in the current session context, meaning the output JobState data will be saved to disk
    """
    ret = plugins.run(query, print=logsession, **kwargs)
    current.save_next(ret)
    print("")


current = Session.timestamped_now()