import time
import queue

from torch.utils import tensorboard

WRITER_TIMEOUT = 2  # Used to avoid writer getting stuck on empty queue when workers terminate.
WRITER_FLUSH_STEPS = 2
DEFAULT_DIR = 'runs'


class Writer:
    """Writer that runs in a separate process and consumes data to write produced by the workers.
    """

    def __init__(self, summary_queue, exp_name, processes=None):
        """Create the writer.

        Args:
            summary_queue (torch.multiprocessing.Queue): Shared queue from which to read the data.
            exp_name (str): Name of the experiments. Data will be logged in runs/exp_name.
            processes (list): List of torch.multiprocessing.Process running the workers. If this
                list is provided, the writer will stop once all processes are dead.
        """
        self.summary_queue = summary_queue
        self.processes = processes
        self.step = {}
        logdir = f'{DEFAULT_DIR}/{exp_name}'
        self.summary_writer = tensorboard.SummaryWriter(flush_secs=WRITER_FLUSH_STEPS,
                                                        log_dir=logdir)
        self.c = 0

    def run(self):
        """Loop consuming items from the queue and writing them in tensorboard.

        Elements in the queue must be a dictionary {tag: (values, steps)}. Each dictionary may
        contain multiple tags corresponding to different plots.
            tag (str): Tag for the data.
            values (list): List of scalars to write.
            steps (list): For each value, the corresponding duration in steps. I.e. if values
                represents total episode rewards, steps must contain the length of each episode.
                For single-ste data (e.g. losses), leave as None or pass a list of ones.
        The function stops when all known workers are dead. If no workers were passed at
        initialization, runs forever.
        """
        while self.processes is None or self.workers_alive():
            try:
                data_dict = self.summary_queue.get(block=True, timeout=WRITER_TIMEOUT)
                for tag, (values, steps) in data_dict.items():
                    for i, v in enumerate(values):
                        if tag not in self.step:
                            self.step[tag] = -1
                        self.step[tag] += 1 if steps is None else steps[i]
                        self.summary_writer.add_scalar(tag, v, self.step[tag], time.time_ns())
            except queue.Empty:
                pass  # Avoid writer stuck in queue when workers terminate.

    def workers_alive(self):
        """Return True if at leas a worker process is alive.
        """
        return any(p.is_alive() for p in self.processes)
