import warnings


def ignore_warnings(func):
    """ Ignore warnings """

    def inner(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return func(*args, **kwargs)

    return inner
