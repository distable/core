from src_core.classes.PipeData import PipeData


class JobPipe:
    def __init__(self, *jobs):
        self.jobs = jobs

    def exec(self, dat: PipeData):
        for job in self.jobs:
            job.exec(dat)