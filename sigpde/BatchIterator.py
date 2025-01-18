from sigpde.utils import ceil_div

class BatchIterator:
    def __init__(self, batch_size, mb_size):
        self.batch_size = batch_size
        self.mb_size = mb_size
        self.mb_mult = ceil_div(batch_size, mb_size)

    def __iter__(self):
        for i in range(self.mb_mult):
            mb_size_i = (
                self.batch_size - self.mb_size * (self.mb_mult - 1)
                if i == self.mb_mult - 1
                else self.mb_size
            )
            start = i * self.mb_size
            stop = start + mb_size_i
            yield i, start, stop