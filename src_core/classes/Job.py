import uuid
from datetime import datetime

from src_core.classes.JobParams import JobParams
from src_core.classes.PipeData import PipeData


class Job:
    def __init__(self, jid: str, parameters: JobParams):
        self.jobid = str(uuid.uuid4())
        self.jid = jid

        # State
        self.handler: None = None
        self.params: JobParams = parameters
        self.data = PipeData()
        if self.params is not None:
            self.params.job = self
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
        return src_core.plugins.get_plug(self.plugid)

    @property
    def done(self):
        return self.progress_norm == 1

    def update(self, progress):
        self.progress_norm = progress
        from src_core import server
        server.emit('updated_job', self.jobid, self.progress_norm)

    def update_step(self, num=None):
        if num is None:
            num = self.progress_i + 1

        self.progress_i = num
        self.progress_norm = self.progress_i / self.progress_max

        # tqdm_total.update()
        # if opts.show_progress_every_n_steps > 0:

    def __repr__(self):
        return f"Job({self.jobid}, {self.plugid}, {self.plugfunc}, {self.progress_norm})"