import os
# os.environ["CUDA_VISIBLE_DEVICES"] = ""  # hides all GPUs
# import jax
# print(jax.devices())  # should list only CPU

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


import json
import sys
import jax.numpy as jnp
import jax
import chex
import numpy as np
from flax import struct
import abc
from jax import lax
import flax.linen as nn
import optax
import time
from pathlib import Path
from tensorboardX import SummaryWriter

import argparse

from dataclasses import dataclass
# from asw_jax import MLP, SubEnv

# from asw_jax import make_train, SubEnv, save_model_and_optimizer, load_model_and_optimizer, save_model, load_model
# from asw_jax import load_all_params_from_checkpoints, MLP, SubEnv

# from asw_jax_utils import GetActionHistoryFromDict, nonzero_prob_dict, GoThroughAllStates, ComputeJaxPolicy


# Add ASW/utils to path
UTILS_PATH = Path(__file__).parent / "ASW_toy/utils"
if str(UTILS_PATH) not in sys.path:
    sys.path.insert(0, str(UTILS_PATH))


from learner import make_train
from asw_toy_env import SubState, Transition, Target, SubEnv
from ActorCriticNetwork import MLP
from to_tabular_policy_utils import ComputeJaxPolicy
from train_utils import create_config, save_model, save_model_and_optimizer, load_model, load_model_and_optimizer, log_metrics

def main(args):

    np.set_printoptions(
        precision=3,       # number of decimal places
        threshold=100000,    # max number of elements before summarizing (use np.inf for full)
        linewidth=200,     # max line width before wrapping
        suppress=True      # don't use scientific notation for small numbers
    )


    config = create_config(args)

    
    key = jax.random.PRNGKey(10)

    env = SubEnv( )
    key, _rng = jax.random.split(key)
    dummy_input, env_state_0 = env.reset(1, 1, _rng)

    network = MLP(hidden_dim=128)
    key, _rng = jax.random.split(key)
    model_params = network.init(_rng, dummy_input)

    key, _rng = jax.random.split(key)
    reg_params_0 = network.init(_rng, dummy_input)


    optimizer = optax.chain(   optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                        optax.adam(learning_rate=config["LR"], eps=1e-5))

    opt_state = optimizer.init(model_params)

    log_path = args.save_path + "/log/"
    writer = SummaryWriter(log_dir=log_path)
    train_jit = jax.jit(make_train(config, env, network, optimizer))


    ## save a .txt file with all checkpoints 
    stored_checkpoint_indices = []
    for i in range(args.N_train_steps+1):
        if (i%args.cpt_freq==0) or i in [2, 4, 8, 16, 32]:
            stored_checkpoint_indices.append(int(i))

    save_folder_path_ = Path(args.save_path + "/target_model")
    save_folder_path_.mkdir(parents=True, exist_ok=True)

    save_folder_path_ = Path(args.save_path + "/target_model/jax_trained_policy")
    save_folder_path_.mkdir(parents=True, exist_ok=True)
    with open(args.save_path + "/target_model/jax_trained_policy" + "/stored_checkpoint_indices.txt", "w") as f:
        f.write(" ".join(map(str, stored_checkpoint_indices)))

    save_folder_path_ = Path(args.save_path + "/searcher_model")
    save_folder_path_.mkdir(parents=True, exist_ok=True)

    save_folder_path_ = Path(args.save_path + "/searcher_model/jax_trained_policy")
    save_folder_path_.mkdir(parents=True, exist_ok=True)
    with open(args.save_path + "/searcher_model/jax_trained_policy" + "/stored_checkpoint_indices.txt", "w") as f:
        f.write(" ".join(map(str, stored_checkpoint_indices)))


    with open(args.save_path + "/config.json", "w") as f:
        json.dump(config, f, indent=4)

    print(f"args.train_model: {args.train_model}")


    def copy_params(params):
        return jax.tree_util.tree_map(lambda x: jnp.array(x, copy=True), params)

    # reg_params = [reg_params_0, reg_params_0, reg_params_0]
    # params_target_new = reg_params_0
    reg_params = [copy_params(reg_params_0) for _ in range(3)]
    params_target_new = copy_params(reg_params_0)
    recent_checkpoint_to_tabular_policy = -1

    with open(f"./ASW_toy/utils/game_state.json", "r") as f:
        game_state_dict = json.load(f)

    if (args.train_model):
        time_training_start = time.time()
        time_start = time.time()
        # save_model_and_optimizer(model_params, opt_state, save_checkpoints_path, 0)

        save_model(params_target_new, args.save_path + "/target_model/target_model_params", 0)
        save_model_and_optimizer(model_params, opt_state, args.save_path + "/searcher_model/searcher_model_params", 0)
        

        # alpha_kl = args.alpha_kl
        # alpha_kl_list = np.linspace(0.01, args.alpha_kl, n).tolist()
        for i in range(args.N_train_steps+1):
            out = train_jit(key, model_params, params_target_new, opt_state, config, reg_params[0], reg_params[1])

            # runner_state = (model_params_, optimizer_state_, env_state, obsv, _rng)     "runner_state"
            (trained_params, params_target_new, optimizer_state_, key, current_reg_step) = out['runner_state']

            model_params = trained_params
            opt_state = optimizer_state_

            reg_params[2] = copy_params(params_target_new)
            if (current_reg_step>=config['max_reg_step']):
                print(f"New reg-params. current_reg_step: {current_reg_step}/{config['max_reg_step']}")
                reg_params[0] = copy_params(reg_params[1])
                reg_params[1] = copy_params(reg_params[2])
                current_reg_step=0
                

            config['current_reg_step'] = current_reg_step

            # metrics = out['metrics']
            # (losses, transition, targets) = metrics
            # {"runner_state": runner_state, "metric_losses": metric_losses, "metric_outcome": metric_outcome}
            metric_losses = out['metric_losses']
            metric_outcome = out['metric_outcome']
            # log_metrics(writer, i, config, args, metric_outcome, metric_losses, metric_clip_norm_ratio)    # function to log training progress to 'writer'
            log_metrics(writer, i, config, args, metric_outcome, metric_losses)    # function to log training progress to 'writer'



            if i==0:
                time_start = time.time()

            # if (i%args.cpt_freq==0 and i>0) or i in [2, 4, 8, 16, 32]:
            if (i%args.cpt_freq==0 or (i in stored_checkpoint_indices)) and i>0:
                time_end = time.time()
                print(f"training-step: {i}/{args.N_train_steps}    time: {int(time_end-time_start)} / {int( (time_end-time_start)*args.N_train_steps/i) } s   current_reg_step: {current_reg_step}/{config['max_reg_step']}")
                
                save_model(params_target_new, args.save_path + "/target_model/target_model_params", i)
                save_model_and_optimizer(trained_params, opt_state, args.save_path + "/searcher_model/searcher_model_params", i)
                # save_model_and_optimizer(trained_params, opt_state, save_checkpoints_path, i)
                # save_model_and_optimizer(params_target_new, opt_state, save_checkpoints_path, i)
            if i%512==0 or i>=args.N_train_steps:
                key, key_ = jax.random.split(key)
                recent_checkpoint_to_tabular_policy_ = ComputeJaxPolicy(network, env, key_, stored_checkpoint_indices, args.save_path + "/target_model/target_model_params", args.save_path + "/target_model", game_state_dict, recent_checkpoint_to_tabular_policy)

                key, key_ = jax.random.split(key)
                recent_checkpoint_to_tabular_policy = ComputeJaxPolicy(network, env, key_, stored_checkpoint_indices, args.save_path + "/searcher_model/searcher_model_params", args.save_path + "/searcher_model", game_state_dict, recent_checkpoint_to_tabular_policy)
                print(f"New 'recent_checkpoint_to_tabular_policy': {recent_checkpoint_to_tabular_policy}")
    else:
        key, key_ = jax.random.split(key)
        recent_checkpoint_to_tabular_policy_ = ComputeJaxPolicy(network, env, key_, stored_checkpoint_indices, args.save_path + "/target_model/target_model_params", args.save_path + "/target_model", game_state_dict, recent_checkpoint_to_tabular_policy)

        key, key_ = jax.random.split(key)
        recent_checkpoint_to_tabular_policy = ComputeJaxPolicy(network, env, key_, stored_checkpoint_indices, args.save_path + "/searcher_model/searcher_model_params", args.save_path + "/searcher_model", game_state_dict, recent_checkpoint_to_tabular_policy)


## uv run asw_jax_train.py --save_path "~dev/AIBLS/asw-planning-foi/ASW_Pub_AMG/retrain_2025_10_28" --N 64 --n 2048 --lr 0.001 --N_JIT_STEPS 32 --N_steps 32768 --alpha_kl 0.1 --gamma_averaging 0.001 --train_model
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to load dependencies and process a file.")


    parser.add_argument('--save_path', type=str, nargs='?', const="test", default="test", help="Path to the file to process")
    parser.add_argument('--path_folder_save_model', type=str, help="Path to save checkpoints during training")
    parser.add_argument('--N', type=int, nargs='?', const=64, default=64, help="Training batch size")
    parser.add_argument('--n', type=int, nargs='?',  const=16, default=16, help="Training sub-batch size")
    parser.add_argument('--lr', type=float, nargs='?',  const=0.0001, default=0.0001, help="learning rate")
    parser.add_argument('--N_JIT_STEPS', type=int, nargs='?',  const=64, default=64, help="number of compiled update-steps each python step/loop")
    parser.add_argument('--N_train_steps', type=int, nargs='?',  const=512+1, default=512+1, help="Number training steps")
    parser.add_argument('--num_minibatches', type=int, nargs='?',  const=1, default=1, help="Number of minibatches to split up training iterations in")
    parser.add_argument('--num_epochs', type=int, nargs='?',  const=1, default=1, help="Number of epochs for each transition")

    parser.add_argument('--max_grad_norm', type=float, nargs='?',  const=0.5, default=0.5, help="Max grad norm. {0.1: Low and stable}, {1.0: standard PPO}, {5.0+: usually unstable in RL}")
    parser.add_argument('--clip_eps', type=float, nargs='?',  const=0.2, default=0.2, help="PPO clipping surrogate objective. {0.1: low and stable but slow}, {0.2: standard PPO}, {0.3: High and unstable}")


    parser.add_argument('--alpha_kl', type=float, nargs='?',  const=0.001, default=0.001, help="learning rate")
    parser.add_argument('--gamma_averaging', type=float, nargs='?',  const=0.01, default=0.01, help="learning rate")
    parser.add_argument('--sv_ent_loss', type=float, nargs='?',  const=0.001, default=0.001, help="Surface vehicle entropy loss")
    parser.add_argument('--sub_ent_loss', type=float, nargs='?',  const=0.001, default=0.001, help="Submarine entropy loss")


    parser.add_argument('--train_model', type=bool, nargs='?',  const=False, default=False, help="Train model parameters or it is already done")
    # parser.add_argument('--train_model', action='store_true', help="Train model parameters or it is already done")

    parser.add_argument('--reg_freq', type=int, nargs='?',  const=128, default=128, help="Number training steps")
    parser.add_argument('--cpt_freq', type=int, nargs='?',  const=64, default=64, help="Number training steps")

    


    # parser.add_argument('--AC_COEF', type=float, help="Actor loss")
    # parser.add_argument('--NUM_ENVS', type=int, help="Number of parallel environments")
    # parser.add_argument('--N_steps', type=int, help="Number training steps")
    # parser.add_argument('--N_y', type=int, help="board height")
    # parser.add_argument('--N_x', type=int, help="board width")

    # parser.add_argument('--N_JIT_STEPS', type=int, help="number of compiled update-steps each python step/loop")
    # parser.add_argument('--lr', type=float, help="learning rate")
    # parser.add_argument('--cpt_freq', type=int, help="Checkpoint frequency")
    # parser.add_argument('--layer_width', type=int, help="network layer width")
    
    # parser.add_argument('--resume_with_new_settings', type=bool, help='if resuming, change some parameters e.g., learning-rates, batch-size, etc. ')

    args = parser.parse_args()
    print(f"Start training with parameters")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")
    print(f"\n")

    main(args)

# uv run Train_asw_toy.py --save_path "./ASW_toy/checkpoints/train_2026_02_16" --N 512 --n 256 --lr 0.00001 --N_JIT_STEPS 32 --N_train_steps 256 --alpha_kl 0.01 --gamma_averaging 0.001 --train_model True --N_JIT_STEPS 16 --num_minibatches 1 --num_epochs 1 --max_grad_norm 0.5 --clip_eps 0.2 --reg_freq 64 --cpt_freq 4