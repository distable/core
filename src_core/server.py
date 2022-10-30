import sys
import threading

import flask_socketio as fsock
from flask import Flask, jsonify

from src_core import jobs, shell
from src_core.logs import logserver
from src_core.plugins import plugins

queue_lock = threading.Lock()

app = Flask(__name__)
sock = fsock.SocketIO(app)

app.config['SECRET_KEY'] = 'supersecretidkwtfthisisfor!'


def emit(event, *args, **kwargs):
    with queue_lock:
        sock.emit(event, *args, **kwargs)


@app.route('/')
def index():
    return "Hello from stable-core!"


@sock.on('connect')
def connect():
    logserver('Client connected')


@sock.on('disconnect')
def disconnect():
    logserver('Client disconnected')


# API
# ----------------------------------------

@app.route('/plugins')
def list_plugins():
    return jsonify(list_plugin_ids())


@sock.on('list_plugins')
def list_plugins():
    """ Send an array of plugin IDs """
    import src_core.plugins
    data = [extract_dict(x, 'id') for x in src_core.plugins.plugins]
    emit('list_plugins', jsonify(data))


@sock.on('list_plugin_ids')
def list_plugin_ids():
    """
    Return a list of all plugins (string IDs only)
    """
    return [plug.id for plug in plugins]


# @socketio.on('plugin_call')
# def plugin_call(js):
#     """
#     An API message with socketio to call a plugin and optionally add a job
#     """
#     import src_core.plugins
#
#     msg = json.loads(js)
#     pid = msg['plugin_id']
#     fname = msg['plugin_func']
#     args = msg['args']
#     kwargs = msg['kwargs']
#
#     src_core.plugins.invoke(pid, fname, *args, **kwargs)


@sock.on('list_jobs')
def list_jobs():
    return jsonify(list_plugin_ids())


# @socketio.on('list_running_jobs')
# def list_running_jobs():
#     return

@sock.on('abort_job')
def abort_job(job):
    if jobs.is_running(job):
        jobs.queue.abort(job)


@sock.on('any_running')
def any_running():
    return jobs.any_running()


def extract_dict(obj, *names):
    return {x: getattr(obj, x) for x in names}


def run():
    def serve():
        import waitress
        waitress.serve(app, host='0.0.0.0', port=5000)

    # Serve with waitress on a separate thread
    logserver("Starting ...")
    t = threading.Thread(target=serve)
    t.start()

    # Set the server module on the jobqueue
    modname = globals()['__name__']
    module = sys.modules[modname]
    jobs.queue.server = module

    # Launch the shell to write commands.
    shell.run()

    # from waitress import serve
    #
    # serve(app, host="0.0.0.0", port=8080)
