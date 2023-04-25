from PyQt6 import QtGui
from PyQt6.QtWidgets import QWidget

class ImageWidget(QWidget):
    def __init__(self, surf, parent=None):
        super(ImageWidget, self).__init__(parent)
        self.surface = surf
        self.image = None
        self.update_image()

    def update_image(self):
        data = self.surface.get_buffer().raw
        w = self.surface.get_width()
        h = self.surface.get_height()
        self.image = QtGui.QImage(data, w, h, QtGui.QImage.Format.Format_RGB32)

    def paintEvent(self, event):
        # Update the image data without creating a new QImage
        self.update_image()

        # Paint the image
        qp = QtGui.QPainter()
        qp.begin(self)
        qp.drawImage(0, 0, self.image)
        qp.end()
