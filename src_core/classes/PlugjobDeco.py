class PlugjobDeco:
    """
    A token return from a decorator to mark the function.
    It will be transformed in the constructor and replaced back
    with an actual function.
    """

    def __init__(self, func, aliases=None):
        self.func = func
        self.aliases = aliases