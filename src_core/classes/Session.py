from datetime import datetime
from pathlib import Path

from PIL.Image import Image

from src_core import paths
from src_core.classes.PipeData import PipeData
from src_core.paths import get_next_leadnum
from src_core.logs import logsession, logsession_err


class Session:
    def __init__(self, name=None, path: Path | str = None):
        self.context = PipeData()
        self.jobs = []

        if name is not None:
            self.name = name
            self.path = paths.sessions / name
        elif path is not None:
            self.path = Path(path)
            self.name = Path(path).stem
        else:
            self.valid = False
            logsession_err("Cannot create session! No name or path given!")
            return

        if not self.path.exists():
            logsession("New session:", self.name)
        else:
            self.load_if_exists()

    @staticmethod
    def timestamped_now():
        """
        Returns: A new session which is timestamped, e.g.
        """
        return Session(paths.format_session_id(datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))

    def create_if_missing(self):
        if not self.path.exists():
            self.path.mkdir(parents=True)

    def load_if_exists(self):
        if self.path.exists():
            logsession(f"Loading session: {self.name}")
            # TODO load a session metadata file

    def save_next(self, dat):
        path = self.path / str(get_next_leadnum(self.path, ''))
        self.save(dat, path)

    def save(self, dat, path):
        if isinstance(dat, list):
            for d in dat:
                self.save_next(d)
        elif isinstance(dat, Image):
            self.create_if_missing()
            dat.save(path.with_suffix(".png"))
            print(f"Saved pil/image to {path.with_suffix('.png')}")
        elif dat is None:
            pass
        else:
            logsession_err("Cannot save! unknown data type:", type(dat))

    def add_job(self, j):
        self.jobs.append(j)

    def rem_job(self, j):
        self.jobs.remove(j)

    # def run(self, query: JobParams | str | None = None, **kwargs):
    #     """
    #     Run a job in the current session context, meaning the output JobState data will be saved to disk
    #     """
    #     ret = plugins.run(query, print=logsession, **kwargs)
    #     current.save_next(ret)
    #     print("")
