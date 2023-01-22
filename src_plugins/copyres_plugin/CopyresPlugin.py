import PIL

from src_core.classes.convert import load_pil
from src_core.classes.JobArgs import JobArgs
from src_core.classes.Plugin import Plugin
from src_core.classes.printlib import printerr
from src_core.plugins import plugjob

class res_job(JobArgs):
    def __init__(self,
                 name='',
                 subdir='',
                 *kargs, **kwargs):
        super().__init__(*kargs, **kwargs)
        self.name = name
        self.subdir = subdir

class initframes_job(res_job):
    def __init__(self, *kargs, **kwargs):
        super().__init__(*kargs, **kwargs)

class copyres_job(res_job):
    def __init__(self,
                 *kargs, **kwargs):
        super().__init__(*kargs, **kwargs)


class copyframe_job(res_job):
    def __init__(self,
                 *kargs, **kwargs):
        super().__init__(*kargs, **kwargs)


class maxsize_job(JobArgs):
    def __init__(self,
                 w=None,
                 h=None,
                 grid=64,
                 *kargs, **kwargs):
        super().__init__(*kargs, **kwargs)
        self.w = w
        self.h = h
        self.grid = grid

class CopyresPlugin(Plugin):
    def title(self):
        return "copyres"

    def describe(self):
        return "A simple plugin to copy a session resource into the context buffer."

    @plugjob
    def maxsize(self, j: maxsize_job):
        cw = j.session.width
        ch = j.session.height

        if j.w and j.h:
            j.session.width = j.w
            j.session.height = j.h
        elif j.w:
            # Set the width to the given value, and scale the height accordingly to preserve aspect ratio
            aspect = ch / cw
            j.session.width = j.w
            j.session.height = j.w * aspect
        elif j.h:
            # Set the height to the given value, and scale the width accordingly to preserve aspect ratio
            aspect = cw / ch
            j.session.height = j.h
            j.session.width = j.h * aspect

        j.session.width = int(j.session.width // j.grid * j.grid)
        j.session.height = int(j.session.height // j.grid * j.grid)

        return j.session.image.resize((j.session.width, j.session.height), PIL.Image.BICUBIC)

    @plugjob
    def initframes(self, j: initframes_job):
        """
        Initialize the frames for an init video. (extract them)
        """
        j.session.extract_frames(j.name)
        return self.copyframe(self, j)

    @plugjob
    def copyframe(self, j: copyframe_job):
        """Copy a session resource into the context buffer."""
        path = j.session.res_frame(j.name, j.subdir) #, loop: j.loop)
        if path is not None:
            image = load_pil(path)
            return image
        else:
            printerr(f"copyres: No such session resource frame: {j.name}")
            return None

    @plugjob
    def copyres(self, j: copyres_job):
        """Copy a session resource into the context buffer."""
        path = j.session.res_framepil(j.name, j.subdir)

        if path is not None and path.is_file():
            image = PIL.Image.open(path)
            return image
        else:
            printerr("copyres: no such session resource:", j.name)
            return None
