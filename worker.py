"""This module contain the workers code, that takes care of running the policy in the given
environment and performing asynchronous updates on the shared model.

Code is freely taken or inspired from https://github.com/Kaixhin/ACER or
https://github.com/dchetelat/acer

"""

import copy
import math
import random
import torch
from torch import multiprocessing
from typing import List

import memory
import gym

import utils

# Helper functions from https://github.com/Kaixhin/ACER/blob/master/train.py


def _poisson(lmbd):
    # Knuth's algorithm for generating Poisson samples
    l, k, p = math.exp(-lmbd), 0, 1
    while p > l:
        k += 1
        p *= random.uniform(0, 1)
    return max(k - 1, 0)


class EpisodeTrainingData:
    """Container class that keeps data used for training.
    """

    def __init__(self):
        self.actions = []
        self.policies = []
        self.q_values = []
        self.values = []
        self.rewards = []
        self.avg_policies = []
        self.old_policies = None
        self.last_state = None

    def append(self, action, policy, q_values, values, reward, average_policy, last_state=None,
               old_policies=None):
        # NOTE: policy is always the policy for the current model, old_policy the one in the
        # replay buffer.
        self.actions.append(action)
        self.policies.append(policy)
        self.q_values.append(q_values)
        self.values.append(values)
        self.rewards.append(reward)
        self.avg_policies.append(average_policy)
        self.last_state = last_state
        self.old_policies = old_policies

    def length(self):
        return len(self.actions)


class Worker:

    def __init__(self, env_name, n_steps, max_steps, shared_model, shared_avg_model,
                 shared_optimizer, shared_counter, df, c, entropy_weight, tr_decay, buffer_len,
                 start_train_at, grad_norm_clip, shared_model_lock, use_lock_update):
        """Create the worker, that collects trajectories from the environment, stores them in the
        replay buffer, performs local training and updates the shared model parameters.

        Args:
            env_name (str): Name of the environment to create.
            n_steps (int): Maximum length of trajectories collected at each iteration.
            max_steps (int): Maximum number of steps to perform across all threads.
            shared_model (torch.nn.Module): Shared model that will be updated by the worker.
            shared_avg_model (torch.nn.Module): Shared running average of shared_model.
            shared_counter (utils.Counter): Shared counter that keeps track of the total step.
            df (float): Discount factor.
            c (float): Maximum value for importance weights.
            entropy_weight (float): Weight of the entropy term in policy loss.
            tr_decay (float): Decay parameter for average policy network update.
            buffer_len (int): Maximum capacity (in trajectories) of the episodic replay buffer.
            start_train_at (int): Number of steps to take before starting to train.
            grad_norm_clip (float): Clipping gradient norm value. None for no clip.
            shared_model_lock (torch.multiprocessing.Lock): If not None, the lock is used to
                safely pass gradients to the shared model.
            use_lock_update(bool): Whether to perform the shared model update inside the lock block.
        """
        self.n_steps = n_steps
        self.max_steps = max_steps
        self.shared_model = shared_model
        self.shared_avg_model = shared_avg_model
        self.shared_optimizer = shared_optimizer
        self.shared_counter = shared_counter
        self.df = df
        self.c = c
        self.entropy_weight = entropy_weight
        self.tr_decay = tr_decay
        self.start_train_at = start_train_at
        self.grad_norm_clip = grad_norm_clip
        self.shared_model_lock = shared_model_lock
        self.use_lock_update = use_lock_update
        if self.use_lock_update and self.shared_model_lock is None:
            raise ValueError('Lock is not passed but use_lock_update is True.')
        self.env = gym.make(env_name)
        self.replay_buffer = memory.EpisodicReplayBuffer(maxlen=buffer_len)
        self.model = copy.deepcopy(self.shared_model)
        self.cur_state = None
        self.done = True
        self.rewards = []  # Each element is the total reward of an episode
        self.episode_rewards = []  # Each element is a reward of the current episode

    def run(self):
        while self.shared_counter.value() <= self.max_steps:
            self.model.load_state_dict(self.shared_model.state_dict())
            training_data = self.on_policy()  # Collect n_steps on policy
            self._train(training_data)

    def on_policy(self):
        """Perform n_steps on-policy, and return the data necessary for on-policy update,
        and updates shared_counter.

        Returns:
            training_data (list): A list of EpisodeTrainingData objects, one for each episode run.
                Only the last object may contain a last_state attribute corresponding to the
                state at which the last episode was cut.
        """
        t = 0
        training_data = []
        while t < self.n_steps:
            if self.done:  # Re-initialize objects for new episode
                self.cur_state = utils.state_to_tensor(self.env.reset())
                self.done = False
                training_data.append(EpisodeTrainingData())
                self.rewards.append(sum(self.episode_rewards))
                self.episode_rewards = []

            # Compute policy and q_values. Note that we do not detach elements used in training,
            # as this saves us computations in _train()
            policy, q_values = self.model(self.cur_state)
            values = (policy * q_values).sum(dim=1, keepdim=True)
            avg_policy, _ = self.shared_avg_model(self.cur_state)
            action = torch.multinomial(policy, num_samples=1)[0, 0]

            next_state, reward, done, _ = self.env.step(action.item())
            next_state = utils.state_to_tensor(next_state)

            # Save transition in replay buffer
            self.replay_buffer.append_transition(
                (self.cur_state, action, policy.detach(), reward, done))
            # Save data for training (all tensors have first dimension 1)
            training_data[-1].append(
                action=torch.LongTensor([[action]]), policy=policy, q_values=q_values,
                values=values, reward=torch.Tensor([[reward]]), average_policy=avg_policy)

            # Update loop data
            t += 1
            self.done = done
            self.cur_state = next_state
            self.episode_rewards.append(reward)

        if not self.done:
            training_data[-1].last_state = self.cur_state
            self.replay_buffer.cutoff(self.cur_state)  # Notify termination to the replay buffer
        self.shared_counter.increment(t)
        return training_data

    def _train(self, training_data: List[EpisodeTrainingData], off_policy=None):
        # Perform training step. Code mainly inspired from
        # https://github.com/Kaixhin/ACER/blob/c711b911baf34b7acf1dbaf0cfeccc6d78277134/train.py
        # NOTE: training_data contains data from trajectories that do not have the same length
        act_dim = training_data[0].policies[0].size(1)
        policy_loss, value_loss = 0., 0.

        # Iterate all training data from last step backwards, so that they are aligned w.r.t. the
        # backward retrace recursive target computation even when they have different lengths.
        n_episodes = len(training_data)
        max_length = max(len(data.rewards) for data in training_data)
        q_rets = None  # retrace action values for each trajectory, shape (n_episodes, 1)
        for t in range(1, max_length):
            # Indices of trajectories that have more than t steps, used as mask
            valid_indices = [i for i in range(n_episodes) if training_data[i].length() - t >= 0]
            actions, policies, rewards, q_values, values, rhos, avg_p = self._extract(
                training_data, valid_indices, t, off_policy, act_dim)

            if t == 1:  # Last time step of trajectories, initialize q_rets
                q_rets = self._initial_q_ret(training_data)
            q_rets = q_rets[valid_indices]
            q_rets = rewards + self.df * q_rets
            adv = q_rets - values
            log_prob = policies.gather(1, actions).log()
            step_policy_loss = -(rhos.gather(1, actions).clamp(max=self.c) * log_prob *
                                 adv.detach()).mean(0)  # Average over batch
            if off_policy:  # Applying bias correction (second term of Eq. 9 in the paper)
                # Multiply by policies and sum(1) in the end = expected value
                bias_weight = (1 - self.c / rhos).clamp(min=0) * policies
                step_policy_loss -= (bias_weight * policies.log() *
                                     (q_values.detach() - values.expand_as(q_values).detach())
                                     ).sum(1).mean(0)  # Average over batch
            # TODO: Trust region
            policy_loss += step_policy_loss
            # Sum over probabilities (H = E[log p]), average over batch
            policy_loss += self.entropy_weight * (policies.log() * policies).sum(1).mean(0)

            # Value update
            q = q_values.gather(1, actions)
            value_loss += ((q_rets - q) ** 2 / 2).mean(0)  # Least squares loss

            # Update the retrace target
            truncated_rho = rhos.gather(1, actions).clamp(max=self.c)
            q_rets = truncated_rho * (q_rets - q.detach()) + values.detach()
        # Transfer gradients to shared model and update
        self._update_networks(policy_loss + value_loss)

    def _initial_q_ret(self, training_data):
        # Compute the initial retrace action-value.
        q_rets = torch.empty(len(training_data), 1)
        for i in range(len(training_data)):
            if training_data[i].last_state is None:
                q_rets[i, 0] = 0.
            else:
                with torch.no_grad():
                    policy, q_values = self.model(training_data[i].last_state)
                    q_rets[i, 0] (policy * q_values).sum(dim=1)
        return q_rets

    def _extract(self, training_data, valid_indices, t, off_policy, act_dim):
        # Extract elements at position -t from training data
        actions = torch.cat(tuple(training_data[i].actions[-t] for i in valid_indices), dim=0)
        policies = torch.cat(tuple(training_data[i].policies[-t] for i in valid_indices), dim=0)
        q_values = torch.cat(tuple(training_data[i].q_values[-t] for i in valid_indices), dim=0)
        rewards = torch.cat(tuple(training_data[i].rewards[-t] for i in valid_indices), dim=0)
        values = torch.cat(tuple(training_data[i].values[-t] for i in valid_indices), dim=0)
        avg_p = torch.cat(tuple(training_data[i].avg_policies[-t] for i in valid_indices), dim=0)
        if off_policy:
            old_policies = torch.cat(
                tuple(training_data[i].old_policies[-t] for i in valid_indices), dim=0)
            rhos = policies.detach() / old_policies
        else:
            rhos = torch.ones(1, act_dim)
        return actions, policies, rewards, q_values, values, rhos, avg_p

    def _update_networks(self, loss):
        def update_avg_policy():
            # Update shared_average_model
            for param, avg_param in zip(self.shared_model.parameters(),
                                        self.shared_avg_model.parameters()):
                avg_param.data.copy_(self.tr_decay * avg_param + (1 - self.tr_decay) * param)
        loss.backward()
        if self.grad_norm_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm_clip)
        if self.shared_model_lock is not None:
            with self.shared_model_lock:
                self._transfer_grads_to_shared_model(overwrite_check=False)
                if self.use_lock_update:
                    self.shared_optimizer.step()
                    update_avg_policy()
            if not self.use_lock_update:  # Perform updates outside lock
                self.shared_optimizer.step()
                update_avg_policy()
        else:  # Do updates without lock
            self._transfer_grads_to_shared_model(overwrite_check=True)
            self.shared_optimizer.step()
            update_avg_policy()

    def _transfer_grads_to_shared_model(self, overwrite_check):
        # Transfers gradients from thread-specific model to shared model
        for param, shared_param in zip(self.model.parameters(), self.shared_model.parameters()):
            if shared_param.grad is not None and overwrite_check:
                return
            shared_param.grad = param.grad
