import os
import sys
import json
import time
import gc
import numpy as np
import jax
import jax.numpy as jnp
import flax
from flax.core import FrozenDict
import optax
from pathlib import Path
from typing import List, Dict, Any, Optional
from tensorboardX import SummaryWriter
import argparse

# Add ASW/utils to path
UTILS_PATH = Path(__file__).parent / "ASW/utils"
if str(UTILS_PATH) not in sys.path:
    sys.path.insert(0, str(UTILS_PATH))

# Import local modules
from train_utils import (
    resume_with_two_priors, copy_log, copy_stored_checkpoint_indices,
    load_stored_checkpoint_indices, save_model, save_model_and_optimizer,
    copy_params, load_config, parse_with_defaults, add_with_default,
    set_config_from_provided_flags, filter_jax_dict, log_metrics, create_config
)
from asw_env import (
    SubEnv, get_transition_mask_MaskedHexesHexCord,
    create_transition_kernel_HexCord, CreateDipMatrix_SonEkv_Topography
)
from learner import make_train
from ActorCriticNetworks import ActorCritic

# === CONFIGURATION ===
np.set_printoptions(
    precision=3,
    threshold=100000,
    linewidth=200,
    suppress=True
)

## Example Run training
# uv run Train_asw_large.py --save_path "./ASW/checkpoints/train_2026_02_09/" --N 32 --n 1024 --lr 0.0001 --N_train_steps 16385 --N_JIT_STEPS 16 --alpha_kl 0.001 --gamma_averaging 0.001 --reg_freq 32 --cpt_freq 32 --sv_ent_loss 0.001 --sub_ent_loss 0.001 --a 1.0 --b 1.0 --c 0.008 --r 5 --SL_loss 40 --RL_loss 40 --SL 130 --max_grad_norm 1.0 --clip_eps 0.2 --num_minibatches 4 --num_epochs 4
# uv run Train_asw_large.py --save_path "./ASW/checkpoints/train_2026_02_10_3/" --N 128 --n 1024 --lr 0.0001 --N_train_steps 16385 --N_JIT_STEPS 16 --alpha_kl 0.001 --gamma_averaging 0.001 --reg_freq 32 --cpt_freq 32 --sv_ent_loss 0.001 --sub_ent_loss 0.001 --a 1.0 --b 1.0 --c 0.008 --r 5 --SL_loss 40 --RL_loss 40 --SL 130 --max_grad_norm 1.0 --clip_eps 0.2 --num_minibatches 4 --num_epochs 1

def InterpolateConfigs(config_i, config1, config2, i_step, keys, N_train_steps):
    config_i_ = config_i.copy().unfreeze()
    def is_valid_element(element):
        return (isinstance(element, int) or isinstance(element, float) or isinstance(element, np.float32) or isinstance(element, np.float64))
    def InterpolateEXP(vi, v1, v2, i, N_train_steps):
        alpha = np.exp( (np.log(v2)-np.log(v1))/N_train_steps )
        vi = vi*alpha
        return vi
    def Interpolate(vi, v1, v2, i, N_train_steps):
        dv = (v2-v1)/N_train_steps
        return v1 + dv*i
    for i in range(len(keys)):
        if keys[i] in config1.keys and keys[i] in config2.keys:
            if is_valid_element(config1[keys[i]]) and is_valid_element(config2[keys[i]]):
                if (config1[keys[i]]>0 and config2[keys[i]]>0):
                    config_i_[keys[i]] = InterpolateEXP(config_i[keys[i]], config1[keys[i]], config2[keys[i]], i_step, N_train_steps)
                else:
                    config_i_[keys[i]] = Interpolate(config_i[keys[i]], config1[keys[i]], config2[keys[i]], i_step, N_train_steps)

    return FrozenDict(config_i_)


def main(args, provided, available_attributes_to_change_when_resuming):
    np.set_printoptions(
        precision=3,       # number of decimal places
        threshold=100000,  # max number of elements before summarizing (use np.inf for full)
        linewidth=200,     # max line width before wrapping
        suppress=True      # don't use scientific notation for small numbers
    )
    
    key = jax.random.PRNGKey(args.seed)

    SCRIPT_DIR = Path(__file__).resolve().parent
    csv_path = SCRIPT_DIR / "ASW/utils/normalized_topography.csv"
    print(f"csv_path: {csv_path}")

    if args.resume:
        ## Resuming from prior checkpoint. Here you can change some parameters for training (everything that is not explicitly changed use the values from the config and default flag-values)
        print(f"resuming...")

        config = load_config(args.path_to_config)
        if config is None:
            return
        
        config_ = set_config_from_provided_flags(config, provided, args, available_attributes_to_change_when_resuming)
        print(f"\nconfig_: {config_}")

        load_success, extra = resume_with_two_priors(config, args.path_to_config)
        print(f"load_success: {load_success}")
        if not load_success:
            print(f"    Could not resume from config and/or checkpoints")
            return
        
        (config, env, network, board, params_reg_2, params_reg_1, params_target, searcher_params, opt_state, optimizer, i, j, k) = extra
        
        print(f"params_target: {type(params_target)}")
        print(f"params_reg_1: {type(params_reg_1)}")
        print(f"params_reg_2: {type(params_reg_2)}")
        print(f"searcher_params: {type(searcher_params)}")
        print(f"board: {type(board)}")
        config['resume_path'] = args.path_to_config
        # config['']    set alpha etc.

        N_train_steps = args.N_train_steps + config['i'] + 1 # the +1 is to make sure that it saves the checkpoint for the last iteration
        i_start = config['i'] + 1
        print(f"i_start: {i_start}")
        reg_params = [params_reg_1, params_reg_2, params_target]
        

        if args.save_path==None or args.save_path=="":
            save_path = Path(args.path_to_config).parent
            stored_checkpoint_indices = load_stored_checkpoint_indices(save_path / Path("stored_checkpoint_indices.txt") )
        else:
            save_path = Path(args.save_path)
            old_log = Path(args.path_to_config).parent / Path('log')
            new_log = SCRIPT_DIR / Path(args.save_path) / Path('log')
            copy_log(old_log, new_log, overwrite=True)
            # stored_checkpoint_indices = copy_stored_checkpoint_indices(Path(args.path_to_config).parent / Path('stored_checkpoint_indices.txt'), SCRIPT_DIR / Path(args.save_path))
            stored_checkpoint_indices = load_stored_checkpoint_indices(old_log.parent / Path('stored_checkpoint_indices.txt') )

            save_model(params_target, args.save_path + "/target_model/target_model_params", config['i'])
            save_model_and_optimizer(searcher_params, opt_state, args.save_path + "/searcher_model/searcher_model_params", config['i'])

            if j<i:
                save_model(params_reg_1, args.save_path + "/target_model/target_model_params", j)
            if k<j:
                save_model(params_reg_2, args.save_path + "/target_model/target_model_params", k)
        

        for i in range(stored_checkpoint_indices[-1]+1, N_train_steps):
            if (i%args.cpt_freq==0) or i in [2, 4, 8, 16, 32, 64]:
                stored_checkpoint_indices.append(int(i))
        with open(save_path / "stored_checkpoint_indices.txt", "w") as f:
            f.write(" ".join(map(str, stored_checkpoint_indices)))
        print(f"stored_checkpoint_indices: {stored_checkpoint_indices}")
        
        with open(save_path / "config.json", "w") as f:
            json.dump(config, f, indent=4)      # Save config

        log_path = SCRIPT_DIR / save_path / Path('log')
        print(f"log_path: {log_path}")


        # set some environment variables
        board_mask = jnp.logical_and(board<0.5, board>0).astype(jnp.int32)  ## {==0: Maskes out the sides of the Axial coordinate representation to create a square board}, {>0.5: Masks out islands, everything that is above 0.5 is above the minimum sub-depth}
        transition_mask, transition_mask_B = get_transition_mask_MaskedHexesHexCord(config, board_mask)     ## 
        transition_kernel = create_transition_kernel_HexCord()      ## Creates a (D,C,3,3) = (7,7,3,3) kernel that is used to update the sub position from its policy-profile in the environment step-function. It is applied as the kernel of a 2D convolution over the policy*belief_distribution 
        dip_matrix = CreateDipMatrix_SonEkv_Topography(config['N_y'], config['N_x'], config['p_min'], config['tile_size'], config['SL'], config['TS'], config['DT'], config['NL'], config['DI'], config['sig'], board, config['SL_loss'], config['RL_loss'])


    else:
        ## Start training from scratch (not resuming from other checkpoint)
        save_path = Path(args.save_path)
        # return
        board = np.loadtxt(csv_path, delimiter=",")
        board_mask = jnp.logical_and(board<0.5, board>0).astype(jnp.int32)  ## {==0: Maskes out the sides of the Axial coordinate representation to create a square board}, {>0.5: Masks out islands, everything that is above 0.5 is above the minimum sub-depth}
        N_y, N_x = board.shape
        config = create_config(args, N_y, N_x)

        transition_mask, transition_mask_B = get_transition_mask_MaskedHexesHexCord(config, board_mask)     ## 
        transition_kernel = create_transition_kernel_HexCord()      ## Creates a (D,C,3,3) = (7,7,3,3) kernel that is used to update the sub position from its policy-profile in the environment step-function. It is applied as the kernel of a 2D convolution over the policy*belief_distribution 

        config['SL_loss'] = args.SL_loss    ## {0: if the sonar efficiency is spatial independent (same detection properties for all positions)}, {>0, e.g 40: higher detection probability in deeper water (less reverberation)}
        config['RL_loss'] = args.RL_loss    ## {0: if the sonar efficiency is spatial independent (same detection properties for all positions)}, {>0, e.g 40: higher detection probability in deeper water (less reverberation)}
        config['SL'] = args.SL              ## standard is 120 but should be increased to e.g 130 if SL_loss/RL_loss is set since they decrease sonar detection.

        print(f"\nconfig: {config}\n")

        ## Create a (N_y, N_x, N_y, N_x) look-up table sonar detection. Based on seabed topography if SL_loss/RL_loss is set
        dip_matrix = CreateDipMatrix_SonEkv_Topography(config['N_y'], config['N_x'], config['p_min'], config['tile_size'], config['SL'], config['TS'], config['DT'], config['NL'], config['DI'], config['sig'], board, config['SL_loss'], config['RL_loss'])

        env = SubEnv(N_y=int(config["N_y"]), N_x=int(config["N_x"]), C=int(config["C"]), D=int(config['D']), tile_size=config['tile_size'], sv_speed=config["sv_speed"], dip_time=config["dip_time"], time_to_first_dip=config['time_to_first_dip'], dt=config["dt"], T=config["T"], a=config["a"], b=config["b"], c=config["c"], r=config['r'],transition_mask=transition_mask_B, board_mask=board_mask, dip_matrix=dip_matrix, transition_kernel=transition_kernel, p_is_done=0.001)

        ## Create Neural Network for policy and value
        key, _rng = jax.random.split(key)
        _, eg_state = env.reset(1, 1, _rng)     ## Create a (N=1,n=1) batch of environment states to initiate the neural-network layer/kernel-sizes
        # network = ActorCritic(in_channels=config['D']+2, out_channels=config["C"], grid_height=config["N_y"], grid_width=config["N_x"], out_directions=config['D'], layer_width=config['layer_width'] )
        network = ActorCritic(in_channels=6, out_channels=config["C"], grid_height=config["N_y"], grid_width=config["N_x"], out_directions=config['D'], layer_width=config['layer_width'] )
        init_x = env.get_obs(eg_state)      ## Create an example observation to initiate the neural net
        key, _rng = jax.random.split(key)
        searcher_params = network.init(_rng, init_x)

        optimizer = optax.chain(   optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                            optax.adam(learning_rate=config["LR"], eps=1e-5))   ## Create the optimizer
        opt_state = optimizer.init(searcher_params)     ## set the optimizer state (size) to match the neural network parameters

        ## Create regulizer and target parameters for the KL-divergence reference policy
        key, _rng = jax.random.split(key)
        reg_params_0 = network.init(_rng, init_x)
        
        reg_params = [copy_params(reg_params_0) for _ in range(3)]  # copy 3 set of the reg_params_0 to be used for two regularization networks and one target network [reg0, reg1, target]
        params_target = copy_params(reg_params_0)

        ## Create folder for the path to save the data
        save_folder_path_ = Path(args.save_path)
        save_folder_path_.mkdir(parents=True, exist_ok=True)

        save_folder_path_ = Path(args.save_path + "/target_model")
        save_folder_path_.mkdir(parents=True, exist_ok=True)

        save_folder_path_ = Path(args.save_path + "/searcher_model")
        save_folder_path_.mkdir(parents=True, exist_ok=True)

        save_folder_path_ = Path(args.save_path + "/log")
        save_folder_path_.mkdir(parents=True, exist_ok=True)

        ## save a .txt file with all checkpoints. Always save the first checkpoints in higher frequency 
        stored_checkpoint_indices = []
        for i in range(args.N_train_steps):
            if (i%args.cpt_freq==0) or i in [2, 4, 8, 16, 32, 64]:
                stored_checkpoint_indices.append(int(i))

        with open(save_path / "config.json", "w") as f:
            json.dump(config, f, indent=4)      # Save config
        with open(save_path / "stored_checkpoint_indices.txt", "w") as f:
            f.write(" ".join(map(str, stored_checkpoint_indices)))


        save_model(params_target, args.save_path + "/target_model/target_model_params", 0)
        save_model(searcher_params, args.save_path + "/searcher_model/searcher_model_params", 0)


        N_train_steps = args.N_train_steps+1
        log_path = args.save_path + "/log/"
        i_start = 0


    writer = SummaryWriter(log_dir=log_path)
    train_jit = jax.jit(make_train(config, env, network, optimizer))    ## create a jit-compiled train-function for self-play and learning-updates
    
    current_reg_step = config["current_reg_step"]
    # return 
    ## Train the model loop
    time_training_start = time.time()
    for i in range(i_start, N_train_steps):     # starting at i_start which is important when resuming (otherwise set to 0)
        # print(f"    i: {i}/{N_train_steps} ({i-i_start}/{N_train_steps-i_start})  time: {time_training_start-time_end}   reg-index: {current_reg_step}/{config['max_reg_step']}  m: {config['m']}")
        
        key, _rng = jax.random.split(key)

        config_for_jit = filter_jax_dict(config)    # filter out elements that is not of types (int, float, bool, jnp.ndarray, np.ndarray).
        out = train_jit(_rng, searcher_params, params_target, opt_state, config_for_jit, reg_params[0], reg_params[1])       ## Run self-play and learning updates config['N_JIT_STEPS'] times (usually 32)

        (searcher_params, params_target, opt_state, key, current_reg_step) = out['runner_state']
        current_reg_step = int(current_reg_step)
        config['current_reg_step'] = current_reg_step   ## Update the reg_step used to computethe KL reference policy --> alpha=reg_step/max_reg_step --> reference_policy = reg_policy_1*alpha + reg_policy_0*(1-alpha)

        reg_params[2] = params_target   # copy_params(params_target)
        if (current_reg_step>=config['max_reg_step']):
            print(f"New reg-params. current_reg_step: {current_reg_step}/{config['max_reg_step']}")
            reg_params[0] = copy_params(reg_params[1])
            reg_params[1] = copy_params(reg_params[2])
            current_reg_step=0
            config["m"] += 1
            config["current_reg_step"] = current_reg_step
        
        ## Log data with tensorboard
        metric_outcome = out['metric_outcome']        # (R_tT_, R_tTi_, Pd_tT_, Pd_tTi_, Ps_tT_, Ps_tTi_, Ps_vec_tT_, R_tT_std, Ps_tT_std, Pd_tT_std)
        metric_policy = out["metric_policy"]                # (training_loss, loss_v, loss_vi, loss_actor_sub, loss_actor_sv)
        metric_losses = out["metric_losses"]                        # (loss_kl_sub_i, loss_kl_sv, KL_dist_sub_i_max, KL_dist_sv_max, KL_dist_sub_i_mean, KL_dist_sv_mean)
        metric_clip_norm_ratio = out["metric_clip_norm_ratio"]              # (sub_i_entropy_loss_mean, sub_i_entropy_loss_max, sv_entropy_loss_mean, sv_entropy_loss_max, global_grad_norm)
        # {"runner_state": runner_state, "metric_outcome": metric_outcome, "metric_policy": metric_policy, "metric_losses": metric_losses, "metric_clip_norm_ratio": metric_clip_norm_ratio}

         

        time_end = time.time()
        time_ = time_end-time_training_start
        log_metrics(writer, i, config, args, time_, metric_outcome, metric_policy, metric_losses, metric_clip_norm_ratio)    # function to log training progress to 'writer'

        

        def coeff_func(i_, c_1=0.008, c_2=0.001, n=1024, lower_value=0.0, i_start=0):    ## Exponentially decreasing function: return f(i) = c_1*alpha^i, where alpha is set to fit f(0)=c_1, f(n)=c_2
            if (i_<=i_start):
                return c_1
            i_ = i_ - i_start
            alpha = np.exp(np.log(c_2/c_1)/n)
            new_c = c_1*np.pow(alpha, i_) + lower_value
            if (new_c<1e-14):
                new_c = 0
            return new_c
        
        if (i%32==0):# and i>0):   # Delete and clear memory for unused variables and jit-compiled functions. Necessary because of an out-of-memory bug. After this is done, all jax functions is recompiled which add computation-time slightly.
            # print(f"del out, clear cache and gc.collect...  i: {i}")
            del out
            jax.clear_caches()
            gc.collect()

            if config['c']>0:
                config['c'] = coeff_func(i, i_start=512)

            env = SubEnv(N_y=int(config["N_y"]), N_x=int(config["N_x"]), C=int(config["C"]), D=int(config['D']), tile_size=config['tile_size'], sv_speed=config["sv_speed"], dip_time=config["dip_time"], time_to_first_dip=config['time_to_first_dip'], dt=config["dt"], T=config["T"], a=config["a"], b=config["b"], c=config["c"], r=config['r'],transition_mask=transition_mask_B, board_mask=board_mask, dip_matrix=dip_matrix, transition_kernel=transition_kernel, p_is_done=0.001)
            train_jit = jax.jit(make_train(config, env, network, optimizer))


        if i%args.show_progress_interval==0 or (i in stored_checkpoint_indices):
            if (i-i_start==0):
                print(f"training-step: {i-i_start}/{args.N_train_steps}  current_reg_step: {current_reg_step}/{config['max_reg_step']}   m: {config['m']}")
            else:
                print(f"training-step: {i-i_start}/{args.N_train_steps}    time: {int(time_end-time_training_start)} / {int( (time_end-time_training_start)*args.N_train_steps/(i-i_start) ) } s   current_reg_step: {current_reg_step}/{config['max_reg_step']}   m: {config['m']}")

        if ( (i%args.cpt_freq==0 or (i in stored_checkpoint_indices))):
            save_model(params_target, save_path / "target_model/target_model_params", i)
            save_model_and_optimizer(searcher_params, opt_state, save_path / "searcher_model/searcher_model_params", i)
            
            config["i"] = i
            with open(save_path / "config.json", "w") as f: 
                # print(f"config: {config}")
                json.dump(config, f, indent=4)      # Save config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to load dependencies and process a file.")

    add_with_default(parser, '--save_path', type=str, nargs='?', const=None, default=None, help="Path to save checkpoints during training and other data")
    add_with_default(parser, '--path_to_config', type=str, nargs='?', const="", default="", help="Path to checkpoint config")
    add_with_default(parser, '--resume', type=bool, nargs='?', const=False, default=False, help="Resume or not")
    add_with_default(parser, '--show_progress_interval',  type=int, nargs='?',  const=32, default=32, help="How often to print progress in the terminal")
    add_with_default(parser, '--N', type=int, nargs='?', const=64, default=64, help="Training batch size")
    add_with_default(parser, '--n', type=int, nargs='?',  const=16, default=16, help="Training sub-batch size")
    add_with_default(parser, '--lr', type=float, nargs='?',  const=0.0001, default=0.0001, help="learning rate")
    add_with_default(parser, '--N_JIT_STEPS', type=int, nargs='?',  const=64, default=64, help="number of compiled update-steps each python step/loop-iteration")        ## Old: 'NUM_UPDATES'
    add_with_default(parser, '--N_train_steps', type=int, nargs='?',  const=512+1, default=512+1, help="Number outer python training steps")                             ## Old: 'N_steps'
    add_with_default(parser, '--cpt_freq', type=int, nargs='?',  const=64, default=64, help="Number outer training steps between saving checkpoints")
    # add_with_default(parser, '--state_ent_loss', type=float, nargs='?',  const=0.00, default=0.00, help="Entropy loss for the sub-position. Makes the submarine spread out. But this is replaced by the 'c' coefficient in the reward")
    add_with_default(parser, '--seed', type=int, nargs='?',  const=10, default=10, help="Seed for random sampling in jax")

    ## MMD Regularization
    add_with_default(parser, '--alpha_kl', type=float, nargs='?',  const=0.001, default=0.001, help="coefficient of how much to regularize the policy on prior regularization policies. Required to converge and dampen policy-oscillation")
    add_with_default(parser, '--gamma_averaging', type=float, nargs='?',  const=0.01, default=0.01, help="learning rate for the target-w. Controlls how fast the target moves to the searcher each step. targ_w = targ_w*(1-gamma_averaging) + search_w*gamma_averaging")
    add_with_default(parser, '--sub_ent_loss', type=float, nargs='?',  const=0.001, default=0.001, help="Policy entropy loss for the submarine")
    add_with_default(parser, '--sv_ent_loss', type=float, nargs='?',  const=0.01, default=0.01, help="Policy entropy loss for the surface vehicle")
    add_with_default(parser, '--reg_freq', type=int, nargs='?',  const=128, default=128, help="Number of outer training steps before updating the regularization models (reg_0 and reg_1)")

    ## Neural Network learning and optimizer parameters.
    add_with_default(parser, '--max_grad_norm', type=float, nargs='?',  const=0.5, default=0.5, help="Max grad norm. {0.1: Low and stable}, {1.0: standard PPO}, {5.0+: usually unstable in RL}")
    add_with_default(parser, '--clip_eps', type=float, nargs='?',  const=0.2, default=0.2, help="PPO clipping surrogate objective. {0.1: low and stable but slow}, {0.2: standard PPO}, {0.3: High and unstable}")
    add_with_default(parser, '--num_epochs', type=int, nargs='?',  const=1, default=1, help="Number of Epochs to learn from each simulation")
    add_with_default(parser, '--num_minibatches', type=int, nargs='?',  const=1, default=1, help="Number of minibatches")
    
    ## Sonar and environment
    add_with_default(parser, '--SL_loss', type=float, nargs='?',  const=0.0, default=0.0, help="How much to decrease sonar source level based on depth (dB)")   # This makes the sonar detection probability depend on the seabed-topography. This limits the source level in shallow waters --> worse when dipping in shallow water.
    add_with_default(parser, '--RL_loss', type=float, nargs='?',  const=0.0, default=0.0, help="How much shallow seabed disrupts/reflect sonar signal (dB)")    # This makes the sonar detection probability depend on the seabed-topography. This reduce reflection signature if the UV is in shallow water --> worse if UV is in shallow water.
    add_with_default(parser, '--SL', type=float, nargs='?',  const=120.0, default=120.0, help="Sonar source level (dB)")
    add_with_default(parser, '--dip_time', type=float, nargs='?',  const=0.25, default=0.25, help="Time it takes for the surface vehicle to move (max 'r' hexes away) to a new position and use the sonar")
    add_with_default(parser, '--time_to_first_dip', type=float, nargs='?',  const=1.0, default=1.0, help="time until searcher can make its first dip")
    add_with_default(parser, '--r', type=int, nargs='?',  const=5, default=5, help="How many hexagons away the searcher can move between dips")
    ## Environment reward 
    add_with_default(parser, '--a', type=float, nargs='?',  const=1.0, default=0.25, help="constant 'a' used in reward function: r = a*ps - b*pd - (r_p_i*c)")
    add_with_default(parser, '--b', type=float, nargs='?',  const=1.0, default=0.25, help="constant 'b' used in reward function: r = a*ps - b*pd - (r_p_i*c)")
    add_with_default(parser, '--c', type=float, nargs='?',  const=0.008, default=0.25, help="variabble 'c' used in reward function: r = a*ps - b*pd - (r_p_i*c)")   # set to 0.008 and then it is exponentially decreasing during training.

    args, provided = parse_with_defaults(parser)
    available_attributes_to_change_when_resuming = {
        "N": "NUM_ENVS",
        "n": "NUM_SUB_ENVS",
        "N_JIT_STEPS": "N_JIT_STEPS",
        "sv_ent_loss": "SV_ENT_LOSS",
        "sub_ent_loss": "SUB_ENT_LOSS",
        # "state_ent_loss": "STATE_ENT_LOSS",
        "lr": "LR",
        "alpha_kl": "alpha_kl",
        "gamma_averaging": "gamma_averaging",
        "reg_freq": "reg_freq",
        "cpt_freq": "cpt_freq",
        "num_epochs": "UPDATE_EPOCHS",
        "max_grad_norm": "MAX_GRAD_NORM",
        "clip_eps": "CLIP_EPS",
        "a": "a", 
        "b": "b", 
        "c": "c",
        "r": "r",
        "time_to_first_dip": "time_to_first_dip",
        "num_minibatches": "NUM_MINIBATCHES"
    }

    print("=== vars(args) ===")
    for k, v in vars(args).items():
        print(f"{k} = {v}")

    print("\n=== provided ===")
    for k in sorted(provided):
        print(k)

    print("\n=== per-flag status ===")  ## Print what values each flag has and if it was explicitly set by the user
    for k in vars(args):
        print(f"{k}: value={getattr(args, k)}, explicitly_set={k in provided}")
    print(f"\n")


    main(args, provided, available_attributes_to_change_when_resuming)
    # main(args)
