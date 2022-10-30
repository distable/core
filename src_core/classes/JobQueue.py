from src_core.classes.Job import Job


class JobQueue:
    def __init__(self):
        class MockServer:
            def emit(self, message, *args, **kwargs):
                pass

        self.all = []  # job objects
        self.queued = []  # job objects
        self.processing = []  # job objects
        self.server = MockServer()

    def list(self):
        return dict(all=self.all, queud=self.queued, processing=self.processing)

    def enqueue(self, job):
        self.all.append(job)
        self.queued.append(job)

        self.server.emit('added_job', job.jobid)

        # Immediately start next
        if len(self.queued) == 1:
            self.run(job)

    def run(self, job: Job):
        """
        Must already be enqueued
        """
        if job in self.queued:
            self.queued.remove(job)

        self.processing.append(job)
        self.server.emit('started_job', job.jobid)

        from src_core import plugins

        for i in range(job.params.job_repeats):
            ret = plugins.run(job.params)
            if job.handler is not None:
                job.handler(ret)

        self.finish(job)

    def cancel(self, job):
        if job in self.all:
            self.all.remove(job)
        if job in self.queued:
            self.queued.remove(job)
            self.server.emit('cancelled_job', job.jobid)

    def finish(self, job):
        if job in self.all:
            self.all.remove(job)
        if job in self.processing:
            self.processing.remove(job)
            self.server.emit('finished_job', job.jobid)

    def abort(self, job):
        if job in self.all:
            self.all.remove(job)
        if job in self.processing:
            self.processing.remove(job)
            job.request_abort = True
            self.server.emit('aborted_job', job.jobid)

    def remove(self, job):
        if job not in self.all:
            raise Exception("Job not in queue")

        if job in self.queued:
            self.cancel(job)

        if job in self.processing:
            self.abort(job)

        self.all.remove(job)

        self.server.emit('removed_job', job.jobid)
