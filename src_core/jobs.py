import uuid
from datetime import datetime

from tqdm import tqdm

# IDEAS
# - We could have other stable-core nodes and dispatch to them (probably better ways to do this lol)
from src_core import server
from src_core.core import cargs
from src_core.PipeData import PipeData
from src_core.printlib import progress_print_out


class JobParams:
    """
    Parameters for a job
    """
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.job_repeats = 0


class Job:
    def __init__(self, plugin_id: str, plugin_func: str, parameters: JobParams):
        self.jobid = str(uuid.uuid4())
        self.plugid = plugin_id
        self.plugfunc = plugin_func

        # State
        self.param: JobParams = parameters
        self.data = PipeData()
        self.run_count = 0
        if self.param is not None:
            self.param.job = self
        self.state_text: str = ""
        self.progress_norm: float = 0
        self.progress_i: int = 0
        self.progress_max: int = 0
        self.request_abort: bool = False
        self.request_skip: bool = False
        self.callback = None
        self.timestamp: str = datetime.now().strftime("%Y%m%d%H%M%S")  # shouldn't this return job_timestamp?

    @property
    def plugin(self):
        import src_core.plugins
        return src_core.plugins.get(self.plugid)

    @property
    def done(self):
        return self.progress_norm == 1

    def update(self, progress):
        self.progress_norm = progress
        server.emit('updated_job', self.jobid, self.progress_norm)

    def update_step(self, num=None):
        if num is None:
            num = self.progress_i + 1

        self.progress_i = num
        self.progress_norm = self.progress_i / self.progress_max

        tqdm_total.update()
        # if opts.show_progress_every_n_steps > 0:

    def __repr__(self):
        return f"Job({self.jobid}, {self.plugid}, {self.plugfunc}, {self.progress_norm})"


class JobQueue:
    def __init__(self):
        self.all = []  # job objects
        self.queued = []  # job objects
        self.processing = []  # job objects

    def list(self):
        return dict(all=self.all, queud=self.queued, processing=self.processing)

    def enqueue(self, job):
        self.all.append(job)
        self.queued.append(job)
        job.run_count += 1

        server.emit('added_job', job.jobid)

        # Immediately start next
        if len(self.queued) == 1:
            self.process(job)

    def process(self, job):
        """
        Must already be enqueued
        """
        if job not in self.queued:
            raise Exception("Job not in queue")

        self.queued.remove(job)
        self.processing.append(job)

        job.param.on_start(job)

        server.emit('started_job', job.jobid)

    def cancel(self, job):
        if job not in self.queued:
            raise Exception("Job not in queue")

        self.queued.remove(job)
        self.all.remove(job)

        server.emit('cancelled_job', job.jobid)

    def finish(self, job):
        if job not in self.processing:
            raise Exception("Job not in processing")

        if job.run_count < job.p.job_repeats:
            self.enqueue(job)

        self.processing.remove(job)
        self.all.remove(job)

        server.emit('finished_job', job.jobid)

    def abort(self, job):
        if job not in self.processing:
            raise Exception("Job not in processing")

        self.processing.remove(job)
        self.all.remove(job)

        job.request_abort = True

        server.emit('aborted_job', job.jobid)

    def remove(self, job):
        if job not in self.all:
            raise Exception("Job not in queue")

        if job in self.queued:
            self.cancel(job)

        if job in self.processing:
            self.abort(job)

        self.all.remove(job)

        server.emit('removed_job', job.jobid)


class JobTQDM:
    def __init__(self):
        self.tqdm = None

    def _create(self):
        self.tqdm = tqdm(desc="Total progress",
                         total=queue.all,
                         position=1,
                         file=progress_print_out)

    def update(self):
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


class JobPipe:
    def __init__(self, *jobs):
        self.jobs = jobs

    def exec(self, dat: PipeData):
        for job in self.jobs:
            job.exec(dat)


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
    queue.process(job)


def enqueue(job):
    """
    Add a job to the queue.
    """
    queue.enqueue(job)


def new_job(plugid, name, jobparams):
    global total_created
    total_created += 1
    j = Job(plugid, name, jobparams)
    enqueue(j)

    return j


# We need to do the job's queue in global functions to keep the api clean
queue = JobQueue()
total_created = 0
tqdm_total = JobTQDM()


def is_running(id):
    """
    Check if a job is processing (by ID)
    """
    j = get(id)
    if j is not None:
        return j in queue.processing


def any_running():
    return len(queue.processing) > 0