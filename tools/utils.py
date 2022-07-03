import warnings


def ignore_warnings(func):
    """ Wrap generator to be reusable """

    def inner(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return func(*args, **kwargs)

    return inner
