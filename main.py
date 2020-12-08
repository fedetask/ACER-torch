import argparse
import copy

import gym
from torch import multiprocessing as mp

import networks
import optimizers
import utils
import worker
import tensorboard_writer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env-name', action='store', type=str, required=True,
                        help='Gym environment full name.')
    parser.add_argument('--num-workers', action='store', type=int, default=4, required=False,
                        help='Number of workers to be spawned (one process for each).')
    parser.add_argument('--t-max', action='store', type=int, default=int(1e6), required=False,
                        help='Total number of steps to run in the environment.')
    parser.add_argument('--worker-steps', action='store', type=int, default=int(1e3),
                        required=False, help='Number of steps that each worker will collect in '
                                             'the environment before performing on and off '
                                             'policy updates.')
    parser.add_argument('--batch-size', action='store', type=int, default=32, required=False,
                        help='Size of trajectory batches sampled in off-policy training.')
    parser.add_argument('--replay-ratio', action='store', type=int, default=4, required=False,
                        help='Expected value of off-policy trainings for each on-policy training.')
    parser.add_argument('--buffer-len', action='store', type=int, default=int(1e4), required=False,
                        help='Capacity of the replay buffer measured in trajectories. The total '
                             'number of transitions is therefore worker steps * buffer len.')
    parser.add_argument('--start-train-at', action='store', type=int, default=0, required=False,
                        help='How many steps to perform before to start off-policy training. '
                             'On-policy training is always performed from the beginning.')
    parser.add_argument('--discount', action='store', type=float, default=0.99, required=False,
                        help='Discount factor.')
    parser.add_argument('--c', action='store', type=float, default=1,
                        help='Retrace importance weight truncation.')
    parser.add_argument('-trust-region', action='store_true', default=True,
                        help='Use trust region constraint in policy updates.')
    parser.add_argument('--tau', action='store', type=float, default=0.99, required=False,
                        help='Weight of current average policy when updating it with the new '
                             'policy. avg_p = tau * avg_p + (1 - tau) p')
    parser.add_argument('--trust-region-threshold', action='store', type=float, default=1,
                        help='Trust region threshold value (delta in Eq. 12 of ACER paper).')
    parser.add_argument('--lr', action='store', type=float, default=5e-4, required=False,
                        help='Learning rate')
    parser.add_argument('--reward-clip', action='store_true', default=False,
                        help='Clip rewards in range [-1, 1].')
    parser.add_argument('--entropy-weight', action='store', type=float, default=1e-4,
                        required=False, help='Entropy regularisation weight')
    parser.add_argument('--max-grad-norm', action='store', type=float, default=40,
                        required=False, help='Gradient L2 normalisation')
    parser.add_argument('--debug', action='store_true', default=False,
                        help='Run a single thread and without tensorboard.')
    parser.add_argument('--no-lock', action='store_true', default=False, required=False,
                        help='Do not use the Lock mechanism when transferring gradients to the '
                             'shared model. Potentially allows workers to overwrite each other\'s '
                             'work, but may improve speed. Default is to use Lock.')
    parser.add_argument('--no-lock-update', action='store_true', default=False, required=False,
                        help='Do not use the Lock mechanism when performing the optimization step '
                             'on the shared model. Potentially allows worker to perform '
                             'optimization at the same time, but may improve speed. Default is to '
                             'use Lock both in gradient transfer and in optimization.')
    args = parser.parse_args()

    # Get environment information
    env = gym.make(args.env_name)
    state_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    env.close()

    # Create shared models
    value_net = networks.LinearNetwork(inputs=state_dim, outputs=act_dim,
                                       n_hidden_layers=3, n_hidden_units=128)
    act_net = networks.LinearNetwork(inputs=state_dim, outputs=act_dim,
                                     n_hidden_layers=3, n_hidden_units=128)
    shared_model = networks.DiscreteActorCriticSplit(actor=act_net, critic=value_net,
                                                     add_softmax=True)
    shared_average_model = copy.deepcopy(shared_model)
    shared_average_model.no_grads()  # Set requires_grad to false for all parameters
    shared_model.share_memory()
    shared_average_model.share_memory()
    shared_opt = optimizers.SharedAdam(shared_model.parameters(), lr=args.lr)

    # Create shared variables
    shared_counter = utils.Counter()
    shared_model_lock = mp.Lock() if not args.no_lock else None
    summary_queue = mp.Queue()

    processes = []
    workers = []
    for i in range(args.num_workers):
        w = worker.Worker(worker_id=i, env_name=args.env_name, n_steps=args.worker_steps,
                          max_steps=args.t_max, shared_model=shared_model,
                          shared_avg_model=shared_average_model, shared_optimizer=shared_opt,
                          shared_counter=shared_counter, df=args.discount, c=args.c,
                          entropy_weight=args.entropy_weight, tau=args.tau,
                          replay_ratio=args.replay_ratio, buffer_len=args.buffer_len,
                          batch_size=args.batch_size, start_train_at=args.start_train_at,
                          grad_norm_clip=args.max_grad_norm,
                          shared_model_lock=shared_model_lock,
                          use_lock_update=(not args.no_lock_update), summary_queue=summary_queue)
        if args.debug:
            w.run()
            exit()
        p = mp.Process(target=w.run)
        processes.append(p)
        workers.append(w)
        p.start()

    # Start tensorboard writer in the main thread
    writer = tensorboard_writer.Writer(summary_queue, processes=processes)
    writer.run()

    print('Finished')
