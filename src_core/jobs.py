# jobs.py
#
# This file implements the job management for the core.
# As of 2022 november 1st, this is a very simple implementation.
# Jobs are queued and run on each their own thread.
# ------------------------------------------------------------


import threading
from copy import deepcopy
from datetime import datetime

import src_core.classes.common
import src_core.core
from src_core.classes import printlib
from src_core.classes.common import extract_dict
from src_core.classes.Job import Job
from src_core.classes.logs import logjob, logjob_err
from src_core.classes.MockServer import MockServer
from src_core.classes.PipeData import PipeData

jobs: list[Job] = []  # All jobs currently managed (queued or processing)
queued: list[Job] = []  # jobs currently in the queue
running: list[Job] = []  # jobs currently processing
server = MockServer()  # to receive events


def get(uid):
    """
    Get a job by id. (queued or processing)
    """
    if isinstance(uid, Job):
        return uid

    for job in jobs:
        if job.uid == uid:
            return job
    raise Exception("Job not found")


def enqueue(job):
    """
    Add a job to the queue.
    """
    if job.queued:
        raise Exception("Job already queued")

    job.queued = True
    jobs.append(job)
    queued.append(job)

    server.emit('job_queued', job)

    # TODO hardcoded to one job at a time currently
    if len(queued) == 1:
        start(job)


def start(job: Job = None):
    """
    Start a queued job, or the next in the queue if null.
    """
    if job is None:
        if len(queued) == 0:
            return

        job = queued[0]

    running.append(job)
    if job.queued:
        queued.remove(job)

    job.queued = False
    threading.Thread(target=run, args=(job,)).start()


def run(job):
    """
    Execution loop for a single job
    """
    from src_core import plugins

    job.running = True
    job.timestamp_run = datetime.now().isoformat()
    server.emit('job_started', job.uid)

    # Run the job for N repeats  ----------------------------------------
    args = deepcopy(job.args)
    args.job = job
    args.input = job.input

    for i in range(job.args.job_repeats):
        if job.aborting: break
        ret = plugins.run(args, require_loaded=True)
        if job.aborting: break

        # TODO merge ret into input PipeData to get output PipeData

        dat = PipeData.automatic(ret)

        if job.on_output is not None:
            job.on_output(dat)

    # Finish the job ----------------------------------------
    update(job, 1)

    if job.aborting:
        logjob("Aborted: ", job)
        server.emit('job_aborted', job.uid)
    else:
        logjob("Finished: ", job)
        server.emit('job_finished', job.uid)

    remove(job)

    if len(running) > 0:
        start(running[0])

    job.running = False
    job.aborting = False


def update(job: Job, progress: float):
    job.state.progress_norm = progress
    server.emit('job_updated', extract_dict(job, 'uid', 'state'))


def abort(job: str | Job):
    """
    Abort a running job, removing it entirely from the job manager.
    """
    job = get(job)

    if job.running:
        # It's already running so we will mark the abort and let run() take care of the rest
        job.aborting = True
    elif job.queued:
        queued.remove(job)
        remove(job)
    else:
        logjob_err("Cannot abort a job that is not running: ", job)
        return


def remove(job):
    if job in queued: queued.remove(job)
    if job in running: running.remove(job)
    if job in jobs: jobs.remove(job)

    server.emit('job_removed', job.uid)


def await_run():
    """
    Do nothing until all jobs have finished.
    """
    while len(jobs) > 0:
        pass


def hijack_ctrlc():
    # Setup CTRL-C to cancel the job
    def ctrlc():
        if len(running) > 0:
            for j in running:
                abort(j)
        else:
            exit(0)

    src_core.classes.common.setup_ctrl_c(ctrlc)
