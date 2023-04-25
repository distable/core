from PyQt6 import QtCore
from PyQt6.QtWidgets import QMainWindow

from rendering.ImageWidget import ImageWidget
from src_plugins.ryusig_calc.QtUtils import get_keypress_args

class HoboWindow(QMainWindow):
    def __init__(self, surf, parent=None):
        super(HoboWindow, self).__init__(parent)
        self.setCentralWidget(ImageWidget(surf))

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.on_timer_timeout)
        self.timer.start(int(1000 / 60))
        self.timeout_handlers = []
        self.key_handlers = []
        self.dropenter_handlers = []
        self.dropleave_handlers = []
        self.dropfile_handlers = []
        self.focusgain_handlers = []
        self.focuslose_handlers = []
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self.setAcceptDrops(True)

    def on_timer_timeout(self):
        for hnd in self.timeout_handlers:
            hnd()
        self.update()
        self.centralWidget().repaint()


    def keyPressEvent(self, event):
        for hnd in self.key_handlers:
            hnd(*get_keypress_args(event))

    def dropEvent(self, event):
        for hnd in self.dropfile_handlers:
            hnd(event.mimeData().urls())

    def dragEnterEvent(self, event):
        for hnd in self.dropenter_handlers:
            qurls = event.mimeData().urls()
            strs = [qurl.toLocalFile() for qurl in qurls]
            hnd(strs)

    def dragLeaveEvent(self, event):
        for hnd in self.dropleave_handlers:
            hnd(event.mimeData().urls())

    def focusInEvent(self, event):
        for hnd in self.focusgain_handlers:
            hnd()

    def focusOutEvent(self, event):
        for hnd in self.focuslose_handlers:
            hnd()
