class JobParams:
    """
    Parameters for a job
    """

    def __init__(self, __getdefaults__=False, **kwargs):
        self.__dict__.update(kwargs)

        self.job_repeats = 1
        if 'n' in kwargs: self.job_repeats = int(kwargs['n'])
        elif 'repeats' in kwargs: self.job_repeats = int(kwargs['repeats'])

        if not __getdefaults__:
            self.defaults = self.__class__(__getdefaults__=True)

    def __str__(self):
        # Print the values that have changed from the defaults
        s = ''
        for k, v in self.__dict__.items():
            if k == 'defaults':
                continue
            if v != self.defaults.__dict__.get(k, None):
                s += f'{k}={v} '
        return s