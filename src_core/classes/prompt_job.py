from src_core.classes.JobParams import JobParams


class prompt_job(JobParams):
    def __init__(self, prompt: str = None, p: str = None, **kwargs):
        super().__init__(**kwargs)
        self.prompt = prompt or p or ''