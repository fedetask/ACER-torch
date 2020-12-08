from torch.utils import tensorboard


WRITER_TIMEOUT = 2  # Used to avoid writer getting stuck on empty queue when workers terminate.
WRITER_FLUSH_STEPS = 2


class Writer:
    """Writer that runs in a separate process and consumes data to write produced by the workers.
    """

    def __init__(self, queue, processes=None):
        """Create the writer.

        Args:
            queue (torch.multiprocessing.Queue): Shared queue from which to read the data.
            processes (list): List of torch.multiprocessing.Process running the workers. If this
                list is provided, the writer will stop once all processes are dead.
        """
        self.queue = queue
        self.processes = processes
        self.step = 0
        self.summary_writer = tensorboard.SummaryWriter(flush_secs=WRITER_FLUSH_STEPS)
        self.c = 0

    def run(self):
        """Loop consuming items from the queue and writing them in tensorboard.

        Elements in the queue must be tuples (tag, step, values), where:
            tag (str): The tag of the data.
            values (list): List of scalars to write.
            steps (list): For each value, the corresponding duration in steps. I.e. if values
                represents episode rewards, steps must contain the length of each episode. For
                single-ste data (e.g. losses), leave as None or pass a list of ones.
        The function stops when all known workers are dead. If no workers were passed at
        initialization, runs forever.
        """
        while self.processes is None or self.workers_alive():
            try:
                tag, values, steps = self.queue.get(block=True, timeout=WRITER_TIMEOUT)
                for i, v in enumerate(values):
                    self.step += 1 if steps is None else steps[i]
                    self.summary_writer.add_scalar(tag, v, self.step)
            except:
                pass  # Avoid writer stuck in queue when workers terminate.

    def workers_alive(self):
        """Return True if at leas a worker process is alive.
        """
        return any(p.is_alive() for p in self.processes)
