# IDEAS
# - We could have other stable-core nodes and dispatch to them (probably better ways to do this lol)
from tqdm import tqdm

from src_core.classes.Job import Job
from src_core.classes.JobQueue import JobQueue
from src_core.lib.printlib import progress_print_out


def get(jobid):
    """
    Get a job by id.
    The job must be queued or processing.
    """
    if isinstance(jobid, Job):
        return jobid

    for job in queue.all:
        if job.jobid == jobid:
            return job
    raise Exception("Job not found")


def finish(jobid):
    job = get(jobid)

    job.update(1)
    queue.finish(job)
    start_next()


def start_next():
    """
    Start the next job in the queue
    """
    if len(queue.queued) == 0:
        return

    job = queue.queued[0]
    queue.run(job)


def enqueue(job):
    """
    Add a job to the queue.
    """
    # Run it immediately
    # TODO we need a job thread with proper dispatching
    queue.enqueue(job)


def run(job):
    queue.run(job)


def new_job(name, jobparams):
    global total_created
    total_created += 1
    j = Job(name, jobparams)

    return j


# We need to do the job's queue in global functions to keep the api clean
queue = JobQueue()
total_created = 0


def is_running(id):
    """
    Check if a job is processing (by ID)
    """
    j = get(id)
    if j is not None:
        return j in queue.processing


def any_running():
    return len(queue.processing) > 0


# This is an old zombie from sd1111, idk if we still need it
# I kept it mostly for reference
class JobTQDM:
    def __init__(self):
        self.tqdm = None

    def _create(self):
        self.tqdm = tqdm(desc="Total progress",
                         total=queue.all,
                         position=1,
                         file=progress_print_out)

    def update(self):
        from src_core.core import cargs
        from src_core.options import opts
        if not opts.multiple_tqdm or cargs.disable_console_progressbars:
            return

        if self.tqdm is None:
            self._create()

        self.tqdm.update()

    def update_total(self, new_total):
        from src_core.options import opts
        if not opts.multiple_tqdm or cargs.disable_console_progressbars:
            return
        if self.tqdm is None:
            self._create()
        self.tqdm.total = new_total

    def hide(self):
        if self.tqdm is not None:
            self.tqdm.close()
            self.tqdm = None
