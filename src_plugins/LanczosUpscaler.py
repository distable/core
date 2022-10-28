from old.upscaler import LANCZOS, UpscalerData
from src_core.plugins import Plugin


class LanczosUpscaler(Plugin):
    scalers = []

    def load_model(self, _):
        pass

    def postprocess(self, img, selected_model=None):
        return img.resize((int(img.width * self.scale), int(img.height * self.scale)), resample=LANCZOS)

    def __init__(self, dirname=None):
        super().__init__(False)
        self.name = "Lanczos"
        self.scalers = [UpscalerData("Lanczos", None, self)]