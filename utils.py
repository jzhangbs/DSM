from datetime import datetime
import time
from collections import OrderedDict


class ClassProperty(property):
    """For dynamically obtaining system time"""
    def __get__(self, cls, owner):
        return classmethod(self.fget).__get__(None, owner)()


class Notify(object):
    """Colorful printing prefix.
    A quick example:
    print(Notify.INFO, YOUR TEXT, Notify.ENDC)
    """

    def __init__(self):
        pass

    @ClassProperty
    def HEADER(cls):
        return str(datetime.now()) + ': \033[95m'

    @ClassProperty
    def INFO(cls):
        return str(datetime.now()) + ': \033[92mI'

    @ClassProperty
    def OKBLUE(cls):
        return str(datetime.now()) + ': \033[94m'

    @ClassProperty
    def WARNING(cls):
        return str(datetime.now()) + ': \033[93mW'

    @ClassProperty
    def FAIL(cls):
        return str(datetime.now()) + ': \033[91mF'

    @ClassProperty
    def BOLD(cls):
        return str(datetime.now()) + ': \033[1mB'

    @ClassProperty
    def UNDERLINE(cls):
        return str(datetime.now()) + ': \033[4mU'
    ENDC = '\033[0m'


def info(*msg):
    print(Notify.INFO, *msg, Notify.ENDC)


def fail(*msg):
    print(Notify.FAIL, *msg, Notify.ENDC)


class TimeMan:

    def __init__(self):
        self.start_time = None
        self.tic_time = None
        self.history = []

    def start(self):
        self.start_time = time.time()

    def tic(self):
        self.tic_time = time.time()

    def toc(self):
        if self.tic_time is None:
            raise Exception('toc before tic.')
        duration = time.time() - self.tic_time
        self.history.append(duration)
        return duration

    def remaining(self, num_iter, format=True):
        if len(self.history) < 20:
            average = sum(self.history) / len(self.history)
        else:
            average = sum(self.history[10:]) / len(self.history[10:])
        remaining = average * num_iter
        if not format:
            return remaining
        else:
            t = OrderedDict()
            t['d'] = remaining // 86400
            t['h'] = remaining % 86400 // 3600
            t['m'] = remaining % 3600 // 60
            t['s'] = remaining % 60
            return ''.join([f'{int(v)}{k}' for k, v in t.items() if v != 0])

    def end(self, num_iter, format=True):
        end = time.time() + self.remaining(num_iter, False)
        if not format:
            return end
        else:
            end = str(datetime.fromtimestamp(end))
            end = end.split('.')[0]
            return end
