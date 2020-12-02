import collections

import numpy as np


class Trajectory:
    """Container class for a trajectory.

    A trajectory is a sequence (s_0, a_0, p_a_0, r_0, s_1, ...). A trajectory ends with a next
    state s_n if the episode is not over. The episode_ended() method can be used to check whether
    the last state in states corresponds to a terminal state or not.
    """

    def __init__(self):
        self.states = []  # Note that states[1:] corresponds to the next states
        self.actions = []
        self.p_actions = []
        self.rewards = []

    def episode_ended(self):
        """Return True if the this trajectory ended because of an episode termination,
        false otherwise.

        If True, then states has the same length as the other attributes, as there is no next state.
        If False, states has an additional state corresponding to the state at which the
        trajectory was cut.
        """
        return self.states[-1] is None

    def get_length(self):
        """Returns the length of the trajectory, without counting the possible additional final
        state.
        """
        return len(self.actions)

    def truncate(self, start, end):
        """Return a copy of this trajectory, truncated from the given start and end indices.

        The states list will contain the additional next state unless the trajectory ends at
        index end due to episode termination.

        Args:
            start (int): Index from which to keep transitions.
            end (int): Last index (inclusive) of the kept transitions.
        """
        new_t = Trajectory()
        last_state_idx = end if end == self.get_length() - 1 and self.episode_ended() else end + 1
        new_t.states = self.states[:last_state_idx]
        new_t.actions = self.actions[start: end + 1]
        new_t.p_actions = self.p_actions[start: end + 1]
        new_t.rewards = self.rewards[start: end + 1]
        return new_t


class EpisodicReplayBuffer:
    """Implementation of a replay buffer that stores Trajectory elements.
    """

    def __init__(self, maxlen):
        self.maxlen = maxlen
        self.buffer = collections.deque(maxlen=maxlen)
        self._cur_trajectory = Trajectory()

    def append_transition(self, transition):
        """Append a transition in the form (s, a, p_a, r, done) to the current cached
        trajectory.

        If done is True, s' is added to the cached trajectory, which is then added to the buffer
        and a new empty one is instantiated.

        Args:
            transition (tuple): A tuple (s, a, p_a, r, done).
        """
        self._cur_trajectory.states.append(transition[0])
        self._cur_trajectory.actions.append(transition[1])
        self._cur_trajectory.p_actions.append(transition[2])
        self._cur_trajectory.rewards.append(transition[3])
        if transition[-1]:  # If done
            self.buffer.append(self._cur_trajectory)
            self._cur_trajectory = Trajectory()

    def cutoff(self, next_state):
        """Signal the replay buffer that the current cached trajectory has been cut by the
        algorithm.

        The given next state is added to the cached trajectory, which is then stored in the
        replay buffer. Do not call this if the episode terminates, as add_transition() already
        deals with it.

        Args:
            next_state (torch.Tensor): A Tensor containing the state at which the trajectory
                was cut.
        """
        self._cur_trajectory.states.append(next_state)
        self.buffer.append(self._cur_trajectory)
        self._cur_trajectory = Trajectory()

    def sample(self, batch_size, random_start=True, same_length=True):
        """Return a list of batch_size Trajectory objects sampled uniformly from the buffer and
        truncated to have the same length.

        Args:
            batch_size (int): Number of trajectories to sample.
            same_length (bool): Whether to cut trajectories to have the same length.
            random_start (bool): Whether the initial step of each trajectory is sampled randomly.
        """
        assert len(self.buffer) >= batch_size, \
            f'Cannot sample {batch_size} trajectories from buffer of length {len(self.buffer)}.'
        indices = np.random.choice(range(len(self.buffer)), size=batch_size, replace=False)
        trajectories = [self.buffer[i] for i in indices]
        if not random_start and not same_length:
            return trajectories

        if random_start:
            start_indices = [np.random.choice(range(int(t.get_length()))) for t in trajectories]
        else:
            start_indices = [0] * len(trajectories)
        if same_length:
            min_len = min(t.len() - start_indices[i] for i, t in enumerate(trajectories))
            end_indices = [start_indices[i] + min_len - 1 for i, t in enumerate(trajectories)]
        else:
            end_indices = [t.get_length() - 1 for t in trajectories]
        res_trajectories = [
            t.truncate(start_indices[i], end_indices[i]) for i, t in enumerate(trajectories)]
        return res_trajectories

    def n_steps(self):
        """Returns the sum of lengths of trajectories in the buffer.
        """
        return sum(t.get_length() for t in self.buffer)
