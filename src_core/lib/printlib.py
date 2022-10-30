import sys
import traceback

import user_conf

enable_timing = True
last_time = False


def print_bp(msg, *args, **kwargs):
    print(f' - {msg}', *args, **kwargs)


def printerr(msg, *args, **kwargs):
    import sys
    print(msg, file=sys.stderr, *args, **kwargs)


def printerr_bp(msg, *args, **kwargs):
    print(f' - {msg}', file=sys.stderr, *args, **kwargs)


def run(code, task):
    try:
        code()
    except Exception as e:
        print(f"{task}: {type(e).__name__}", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)


def make_print(module_name):
    def ret(msg, *args, **kwargs):
        if not enable_timing:
            print(f"[{module_name}] {msg}", *args, **kwargs)
        else:
            # Print the elapsed time since the last call to this function
            import time
            global last_time
            if last_time and user_conf.print_timing:
                print(f"[{module_name}] ({time.time() - last_time:.2f}s) {msg}", *args, **kwargs)
            else:
                print(f"[{module_name}] {msg}", *args, **kwargs)
            last_time = time.time()

    return ret


def make_printerr(module_name):
    def ret(msg, *args, **kwargs):
        printerr(f"[{module_name}] {msg}", *args, **kwargs)

    return ret


def print_info():
    from src_core.installing import git
    try:
        commit = run(f"{git} rev-parse HEAD").strip()
    except Exception:
        commit = "<none>"
    print(f"Python: {sys.version}")
    print(f"Revision: {commit}")


progress_print_out = sys.stdout
