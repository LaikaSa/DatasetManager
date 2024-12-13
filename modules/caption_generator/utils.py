import sys

class ProgressBar:
    def __init__(self, total, prefix='', length=20):
        self.total = total
        self.prefix = prefix
        self.length = length
        self.current = 0

    def update(self, current):
        self.current = current
        filled_length = int(self.length * current // self.total)
        bar = '=' * filled_length + '-' * (self.length - filled_length)
        sys.stdout.write(f'\r{self.prefix}[{bar}] {current}/{self.total}')
        sys.stdout.flush()
        if current == self.total:
            sys.stdout.write('\n')
            sys.stdout.flush()