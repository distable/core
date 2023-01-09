def to_dict(f):
    return dict((key, value) for key, value in f.__dict__.items() if not callable(value) and not key.startswith('__'))


