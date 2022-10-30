from src_core.classes.JobParams import JobParams
from src_core.plugins import plugjob
from src_core.classes.Plugin import Plugin


class colorcorrect_job(JobParams):
    def __init__(self, hue=100, saturation=100, brightness=100, contrast=100, **kwargs):
        super().__init__(**kwargs)
        self.hue = hue
        self.saturation = saturation
        self.brightness = brightness
        self.contrast = contrast

class MagickPlugin(Plugin):
    def title(self):
        return "Magick Plugin"

    def describe(self):
        return "Run some color corrections with image magick"

    def init(self):
        pass

    def install(self):
        pass

    def uninstall(self):
        pass

    def load(self):
        pass

    def unload(self):
        pass

    @plugjob
    def colorcorrect(self: colorcorrect_job):
        # TODO we need an image input pil to work with
        pass
        # cmd = "+sigmoidal-contrast 5x-3%"
        # pil.save('tmp.png')
        #
        # run("\"convert\" tmp.png $command tmp_out.png")
        #
        # pil = Image.open('tmp_out.png')
        # os.remove('tmp.png')
        # os.remove('tmp_out.png')
        # return pil