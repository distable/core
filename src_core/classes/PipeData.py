from pathlib import Path

from PIL import Image

from src_core.lib.printlib import printerr


class PipeData:
    """
    This is semantic data to be piped between jobs and plugins, used as output, can be saved, etc.
    """

    def __init__(self):
        self.prompt: str = ""
        self.image = None

    def save(self, path):
        path = Path(path)
        if isinstance(self.image, Image.Image):
            self.image.save(path)

            from image import DrawImage
            image = DrawImage.from_file(path)
            image.draw_image()
        else:
            printerr(f"Cannot save {self.image} to {path}")
