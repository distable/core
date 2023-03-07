import threading

from PyQt5 import QtCore
from PyQt5.QtCore import QCoreApplication

from src_core.rendering import renderer
from src_plugins.ryusig_calc.RyusigApp import RyusigApp

initialized = False
app:RyusigApp|None = None

def toggle():
    if not initialized:
        init()

    w = app.win
    if w.isVisible():
        app.win.hide()
    else:

        app.win.show()
        app.win.raise_()


def on_t_selected_ryusig(t):
    renderer.seek_t(t, False)


def on_t_selected_renderer(t):
    app.on_update_playback_t(t)


def on_script_loaded():
    app.request_reload = True


def init():
    global initialized
    from src_plugins.ryusig_calc.RyusigApp import RyusigApp
    global app

    renderer.on_t_changed.append(on_t_selected_renderer)
    renderer.on_script_loaded.append(on_script_loaded)

    app = RyusigApp(
            callback_mod=renderer.script,
            callback_vars=renderer.v,
            audio=renderer.audio)

    app.init_qapp()
    app.init_qwindow()

    app.keypress_handlers.append(on_keypress)

    app.win.hide()  # hidden by default on startup
    app.win.update()

    app.on_t_selected.append(on_t_selected_ryusig)
    app.key_hide_window = QtCore.Qt.Key.Key_F1

    initialized = True


def on_keypress(key, ctrl, shift, alt):
    if key == QtCore.Qt.Key.Key_F1:
        toggle()
