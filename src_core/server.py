import json
import threading

import flask_socketio as fsock
from flask import Flask, jsonify

from src_core import jobs, printlib, shell

queue_lock = threading.Lock()

app = Flask(__name__)
app.config['SECRET_KEY'] = 'supersecretidkwtfthisisfor!'

socketio = fsock.SocketIO(app)

mprint = printlib.make_print("server")
mprinterr = printlib.make_printerr("server")


def emit(event, *args, **kwargs):
    with queue_lock:
        socketio.emit(event, *args, **kwargs)


@app.route('/')
def index():
    return "Hello from stable-core!"


# A route to list all plugins
@app.route('/plugins')
def list_plugins():
    import src_core.plugins
    return jsonify(src_core.plugins.list_ids())


@socketio.on('connect')
def connect():
    print('Client connected')


@socketio.on('disconnect')
def disconnect():
    print('Client disconnected')


@socketio.on('list_plugins')
def list_plugins():
    """ Send an array of plugin IDs """
    import src_core.plugins
    data = [extract_dict(x, 'id') for x in src_core.plugins.plugins]
    emit('list_plugins', jsonify(data))


@socketio.on('call_plugin')
def call_plugin(js):
    """
    An API message with socketio to call a plugin and optionally add a job
    """
    import src_core.plugins

    msg = json.loads(js)
    pid = msg['plugin_id']
    fname = msg['plugin_func']
    args = msg['args']
    kwargs = msg['kwargs']

    src_core.plugins.invoke(pid, fname, *args, **kwargs)


@socketio.on('list_jobs')
def list_jobs():
    return jsonify(jobs.queue.list_ids())


@socketio.on('is_job_running')
def is_job_running(id):
    return jsonify(jobs.is_running(id))


@socketio.on('abort_job')
def abort_job(id):
    if jobs.is_running(id):
        jobs.queue.abort(id)


@socketio.on('any_running')
def any_running():
    return jobs.any_running()


def extract_dict(obj, *names):
    return {x: getattr(obj, x) for x in names}


def run():
    def serve():
        import waitress
        waitress.serve(app, host='0.0.0.0', port=5000)

    # Serve with waitress on a separate thread
    mprint("Starting ...")
    t = threading.Thread(target=serve)
    t.start()

    # Launch the shell to write commands.
    shell.run()

    # from waitress import serve
    #
    # serve(app, host="0.0.0.0", port=8080)
