import sys
import threading

import flask_socketio as fsock
from flask import Flask, request
from jsonic import serialize

import src_core.core
import user_conf
from src_core import jobs, plugins, shell
from src_core.classes.logs import logserver
from src_core.classes.Session import Session
from src_core.jobs import running


class _Client:
    def __init__(self, sid):
        self.sid = sid
        self.session = Session.now(prefix=sid)

    def emit(self, event, data, *args, **kwargs):
        sock.emit(event, serialize(data), room=self.sid, *args, **kwargs)

    def __repr__(self):
        return f"Client({self.sid})"

    def __str__(self):
        return self.__repr__()


# noinspection PyPep8Naming
def Client(sid=None):
    """
    A client functoin which returns a class for the current request.sid or a specific SID.
    """
    if not user_conf.share:
        return _Client("local")
    if sid is None:
        sid = request.sid
    if sid not in clients:
        clients[sid] = _Client(sid)
    return clients[sid]


def serialized(fn):
    """
    A decorator which wraps the return value of fn with serialize(...)
    """

    def wrapper(*args, **kwargs):
        return serialize(fn(*args, **kwargs))

    return wrapper


queue_lock = threading.Lock()

app = Flask(__name__)
sock = fsock.SocketIO(app)

app.config['SECRET_KEY'] = 'supersecretidkwtfthisisfor!'
clients = {}


def emit(event, data, *args, **kwargs):
    with queue_lock:
        sock.emit(event, serialize(data), *args, **kwargs)


@app.route('/')
def index():
    return "Hello from stable-core!"


@sock.on
def connect():
    logserver('Client connected')
    Client()
    emit('handshake', dict(plugins=plugins.plugins, jobs=plugins.get_jobs()))


@sock.on
def disconnect():
    clients.pop(request.sid)
    logserver('Client disconnected')


# API
# ----------------------------------------

@sock.on
@serialized
def abort_job(uid):
    jobs.abort(uid)


@sock.on
@serialized
def start_job(jargs):
    """
    Start a job
    """
    src_core.core.job(jargs, session=Client().session)


@sock.on
@serialized
def any_running():
    return len(running) > 0


def run():
    def serve():
        # import waitress
        # waitress.serve(app, host=user_conf.ip, port=user_conf.port)
        logserver(f"Serving on {user_conf.ip}:{user_conf.port}")
        sock.run(app)

    # Serve with waitress on a separate thread
    t = threading.Thread(target=serve)
    t.start()

    # Set the server module on the jobqueue
    modname = globals()['__name__']
    module = sys.modules[modname]

    jobs.server = module
    jobs.hijack_ctrlc()

    # Launch the shell to write commands.
    plugins.wait_loading()
    shell.run()

    # from waitress import serve
    #
    # serve(app, host="0.0.0.0", port=8080)
