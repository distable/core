import threading

from PyQt5.QtCore import QCoreApplication

from src_core.rendering import renderer

app = None

def ryusig_loop(start_visible=False):
    from src_plugins.ryusig_calc.RyusigApp import RyusigApp
    global app

    renderer.on_t_changed.append(on_t_selected_renderer)

    app = RyusigApp(callback_mod=renderer.script,
              callback_vars=renderer.v,
              audio=renderer.audio)

    app.on_t_selected.append(on_t_selected_ryusig)
    if not start_visible:
        app.win.hide()
    app.exec()

def toggle():
    w = app.win
    if w.isVisible():
        app.win.hide()
    else:
        app.win.show()

def on_t_selected_ryusig(t):
    renderer.seek_t(t, False)

def on_t_selected_renderer(t):
    app.on_update_playback_t(t)

def kill_loop():
    import time
    while True:
        time.sleep(1)
        if renderer.request_stop:
            app.quit = True
            break

def ryusig_init(start_visible=False):
    threading.Thread(target=ryusig_loop, args=[start_visible]).start()
    threading.Thread(target=kill_loop).start()
