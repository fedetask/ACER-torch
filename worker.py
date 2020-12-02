"""This module contain the workers code, that takes care of running the policy in the given
environment and collecting data used by the manager process for training.

The code structure related to the A2C algorithm is strongly taken from
https://github.com/grantsrb/PyTorch-A2C
"""

import torch
from torch import multiprocessing


class Worker:

    def __init__(self, data, n_steps, df, start_queue, stop_queue):
        """Create the worker, that will collect n_steps of data from the environment.

        The manager process sends messages in start_queue containing the index at which the
        worker must start to fill the data. Once n_steps of data are collected, the worker puts
        the index back into the queue, notifying the manager that the worker finished.

        Args:
            data (dict): Dictionary with the shared data that the worker will fill. The
                dictionary contains:
                    {
                        'states':    # list to be filled with states (torch.Tensor)
                        'deltas':    # list to be filled with GAE deltas (float)
                        'rewards':   # list to be filled with rewards (float)
                        'dones':     # list to be filled with dones (bool)
                        'actions':   # list to be filled with actions (torch.Tensor)
                        'h_states':  # list to be filled with hidden states (torch.Tensor) if
                                       using recurrent architecture.
                    }
            n_steps (int):  Number of steps that the worker will collect in the environment.
            df (float): Discount factor.
            start_queue (multiprocessing.Queue): Queue in which the manager will put the index at
                which the worker should start to fill the data.
            stop_queue (multiprocessing.Queue): Queue in which the worker will put the index back
                after collecting n_steps.
        """
        self.data = data
        self.n_steps = n_steps
        self.df = df
        self.start_queue = start_queue
        self.stop_queue = stop_queue
