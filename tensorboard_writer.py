from torch.utils import tensorboard
from torch.multiprocessing import Queue


class Writer:
    """Writer that runs in a separate process and consumes data to write produced by the workers.
    """

    def __init__(self, queue):
        self.queue = queue
        self.step = 0
        self.on = True
        self.summary_writer = tensorboard.SummaryWriter()
        self.test_queue = Queue()

    def run(self):
        """Loop consuming items from the queue and writing them in tensorboard.

        Elements in the queue must be tuples (tag, step, values), where:
            tag (str): The tag of the data.
            values (list): List of scalars to write.
            steps (list): For each value, the corresponding duration in steps. I.e. if values
                represents episode rewards, steps must contain the length of each episode. For
                single-ste data (e.g. losses), leave as None or pass a list of ones.
        """
        while self.on:
            tag, values, steps = self.test_queue.get(block=True)
            for i, v in enumerate(values):
                if steps is not None:
                    self.step += steps[i]
                else:
                    self.step += 1
                self.summary_writer.add_scalar(tag, v, self.step)

    def stop(self):
        self.on = False
