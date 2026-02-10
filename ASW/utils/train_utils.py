from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Callable, Optional, Tuple, List
import shutil
import json

import numpy as np
import jax
import optax

_STEP_RE = re.compile(r"^params_step(\d+)\.flax$")

from flax.serialization import from_bytes, msgpack_restore
import flax.serialization
# from asw_utils import create_env
from asw_env  import create_env
from ActorCriticNetworks import ActorCritic


# from flax.serialization import msgpack_restore
from flax.core import unfreeze
from flax.traverse_util import flatten_dict

import jax.numpy as jnp




def create_config(args, N_y, N_x):

    config = {}
    config["N_JIT_STEPS"] = args.N_JIT_STEPS

    config['NUM_ENVS'] = args.N
    config['NUM_SUB_ENVS'] = args.n

    config['cpt_freq'] = args.cpt_freq
    config['reg_freq'] = args.reg_freq

    config['N_y'] = N_y     ## 31
    config['N_x'] = N_x     ## 47

    # print(f"args.n: {args.n}")
    # print(f"args.N: {args.N}")
    # print(f"args.N_JIT_STEPS: {args.N_JIT_STEPS}")

    config['LR'] = args.lr
    config['MAX_GRAD_NORM'] = 0.1
    config["CLIP_EPS"] = args.clip_eps
    config["UPDATE_EPOCHS"] = 1
    config['layer_width'] = 128

    sub_speed = 7
    tile_size = 0.8
    board_height = N_y*tile_size   # 24.8
    dt = tile_size/sub_speed        # ~0.114 ~6.8min
    max_steps = N_x+N_y           # 78
    T_max = max_steps*dt            # ~8.9h

    config['C'] = 7
    config['D'] = 7
    config['tile_size'] = tile_size
    config['Board_Height'] = board_height
    config['dt'] = dt
    config['T'] = T_max
    config['sv_speed'] = 150
    config['dip_time'] = args.dip_time              # 0.25 h
    config['time_to_first_dip'] = args.time_to_first_dip  # 1.0 h
    config['a'] = args.a    ## reward after the sub reaches one target objective
    config['b'] = args.b    ## reward after the SV detects the sub
    config['c'] = args.c    ## Exploration coefficient (higher is more exploration). Applied on the reward (not as a policy regularization), higher reward if the sub sample is in a position where the belief-probability is low.
    config['r'] = args.r    ## distance, in hexagons, that the SV can move between dips.  
    config['NUM_ENV_STEPS'] = max_steps
    # config['STATE_ENT_LOSS'] = args.state_ent_loss
    config['SUB_ENT_LOSS'] = args.sub_ent_loss
    config['SV_ENT_LOSS'] = args.sv_ent_loss

    config['p_min'] = 0.01
    config['SL'] = 120
    config['TS'] = -20
    config['DT'] = 15
    config['NL'] = 80    # config['NL'] = 90

    config['DI'] = 15
    config['sig'] = 9.4

    config["TOTAL_TIMESTEPS"] = config["N_JIT_STEPS"] * config["NUM_ENV_STEPS"] * config["NUM_ENVS"]

    if config["NUM_ENVS"]>8:
        config["NUM_MINIBATCHES"] = args.num_minibatches
    else:
        config["NUM_MINIBATCHES"] = 1

    config["MINIBATCH_SIZE"] = (config["NUM_ENVS"] * config["NUM_ENV_STEPS"] // config["NUM_MINIBATCHES"])

    config['current_reg_step'] = 0
    config['m'] = 0
    config['max_reg_step'] = config['N_JIT_STEPS'] * config['reg_freq']
    config['alpha_kl'] = args.alpha_kl
    config['gamma_averaging'] = args.gamma_averaging

    return config


####
#### Functions to save and load the model and optimizer
def save_model(params, save_dir, step=None):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    step_str = f"_step{step}" if step is not None else ""

    with (save_dir / f"params{step_str}.flax").open("wb") as f:
        f.write(flax.serialization.to_bytes(params))

def save_model_and_optimizer(params, opt_state, save_dir, step=None):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    step_str = f"_step{step}" if step is not None else ""

    with (save_dir / f"params{step_str}.flax").open("wb") as f:
        f.write(flax.serialization.to_bytes(params))

    with (save_dir / f"opt_state{step_str}.flax").open("wb") as f:
        f.write(flax.serialization.to_bytes(opt_state))

def load_model(model, save_dir, key, dummy_input, step=None):
    save_dir = Path(save_dir)
    step_str = f"_step{step}" if step is not None else ""

    with (save_dir / f"params{step_str}.flax").open("rb") as f:
        param_bytes = f.read()

        def param_names_from_bytes(param_bytes, only_trainable=True):
            state = msgpack_restore(param_bytes)             # nested dict/FrozenDict with arrays
            tree = state.get("params", state) if only_trainable and isinstance(state, dict) else state
            flat = flatten_dict(unfreeze(tree))              # keys are tuples like ("Dense_0","kernel")

            names = []
            for k_tuple, v in flat.items():
                names.append("/".join(k_tuple))
            return names
        
        names = param_names_from_bytes(param_bytes)
        print(f"Params-names: {names}")

    param_template = model.init(key, dummy_input)
    params = flax.serialization.from_bytes(param_template, param_bytes)

    return params

def load_model_and_optimizer(model, tx, save_dir, key, dummy_input, step=None):
    save_dir = Path(save_dir)
    step_str = f"_step{step}" if step is not None else ""

    with (save_dir / f"params{step_str}.flax").open("rb") as f:
        param_bytes = f.read()
    with (save_dir / f"opt_state{step_str}.flax").open("rb") as f:
        opt_bytes = f.read()

    param_template = model.init(key, dummy_input)
    opt_template = tx.init(param_template)

    params = flax.serialization.from_bytes(param_template, param_bytes)
    opt_state = flax.serialization.from_bytes(opt_template, opt_bytes)

    return params, opt_state
#### Functions to save and load the model and optimizer
####

def copy_params(params):
    return jax.tree_util.tree_map(lambda x: jnp.array(x, copy=True), params)

def make_tx(config):
    return optax.chain(
        optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
        optax.adam(learning_rate=config["LR"], eps=1e-5),
    )

## 
def resume_with_two_priors(
    config: dict,
    config_path: str | Path,
) -> Tuple[bool, Optional[object], Optional[object], Optional[object]]:
    config_path = Path(config_path).resolve()

    if 'i' not in config.keys():
        config["i"] = 0
    i = config["i"]

    ckpt_dir_target = config_path.parent / Path("target_model/target_model_params/")
    ckpt_dir_searcher = config_path.parent / Path("searcher_model/searcher_model_params/")
    if not ckpt_dir_target.is_dir():
        return False, None
    if not ckpt_dir_searcher.is_dir():
        return False, None
    
    csv_path = "utils/normalized_topography.csv"
    board = np.loadtxt(csv_path, delimiter=",")
    env = create_env(config, board)
    key = jax.random.PRNGKey(10)    ## seed does not matter, this parameters will be overwritten
    key, _rng = jax.random.split(key)
    _, eg_state = env.reset(1, 1, _rng)

    # create neural-net architecture
    network = ActorCritic(in_channels=config["D"]+2, out_channels=config["C"], grid_height=config["N_y"], grid_width=config["N_x"], out_directions=config['D'], layer_width=config['layer_width'] )
    init_x = env.get_obs(eg_state)
    key, _rng = jax.random.split(key)
    param_template = network.init(_rng, init_x)

    # Collect available checkpoint steps
    steps = []
    for p in ckpt_dir_target.iterdir():
        m = _STEP_RE.match(p.name)
        if m:
            steps.append(int(m.group(1)))
    steps.sort()

    if i not in steps:
        return False, None

    # Determine j and k
    priors = [s for s in steps if s < i]
    j = priors[-1] if len(priors) >= 1 else i
    k = priors[-2] if len(priors) >= 2 else j
    print(f"loading checkpoints  i: {i}  j: {j}  k: {k}")

    def load_params(param_template, path: str | Path):
        print(f"    load {path}")
        with path.open("rb") as f:
            param_bytes = f.read()
        return flax.serialization.from_bytes(param_template, param_bytes)
    
    try:
        params_searcher = load_params(param_template, ckpt_dir_searcher / f"params_step{i}.flax" )

        # 4) Re-create optimizer and template opt_state (structure)
        optimizer_tx = make_tx(config)
        opt_state_template = optimizer_tx.init(params_searcher)

        # 5) Load opt_state into the template
        opt_state = load_params(opt_state_template, ckpt_dir_searcher / f"opt_state_step{i}.flax" )
    except Exception:
        return False, None

    try:
        params_i = load_params(param_template, ckpt_dir_target / f"params_step{i}.flax")
        params_j = load_params(param_template, ckpt_dir_target / f"params_step{j}.flax") if j != i else copy_params(params_i)
        params_k = load_params(param_template, ckpt_dir_target / f"params_step{k}.flax") if k != j else copy_params(params_j)
    except Exception:
        return False, None

    return True, (config, env, network, board, params_k, params_j, params_i, params_searcher, opt_state, optimizer_tx, i, j, k)

def make_flax_load_fn(params_template):
    def _load(path: Path):
        return from_bytes(params_template, path.read_bytes())
    return _load



def copy_log(old_log: str | Path, new_log: str | Path, *, overwrite: bool = False):
    old_log = Path(old_log).expanduser().resolve()
    new_log = Path(new_log).expanduser().resolve()

    if not old_log.is_dir():
        raise FileNotFoundError(f"Old log dir does not exist: {old_log}")

    if new_log.exists():
        if overwrite:
            shutil.rmtree(new_log)
        else:
            raise FileExistsError(
                f"New log dir already exists: {new_log}. "
                f"Use overwrite=True to replace it."
            )

    shutil.copytree(old_log, new_log)



def load_stored_checkpoint_indices(path):
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    return [int(x) for x in text.split()]

def copy_stored_checkpoint_indices(old_path, new_path):
    # --- copy stored_checkpoint_indices.txt ---
    old_indices = old_path.parent / "stored_checkpoint_indices.txt"
    new_indices = new_path / "stored_checkpoint_indices.txt"

    print(f"old_indices: {old_indices}")
    print(f"new_indices: {new_indices}")

    if old_indices.is_file():
        new_indices.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(old_indices, new_indices)
    else:
        print(f"Warning: no stored_checkpoint_indices.txt found at {old_indices}")

    return load_stored_checkpoint_indices(new_indices)

def load_config(path: str) -> dict:
    config_path = Path(path)
    if not config_path.is_file():
        print(f"    config is not a file")
        return None
    print(f"    config is a file {config_path}")

    try:
        with config_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    # except (json.JSONDecodeError, OSError):
    except (json.JSONDecodeError, OSError) as e:
        print(f"Failed to load config {config_path}: {e!r}")
        return None


import argparse
_UNSET = object()
def add_with_default(parser: argparse.ArgumentParser, *args, default, **kwargs):
    kwargs["default"] = _UNSET
    action = parser.add_argument(*args, **kwargs)
    action.real_default = default
    return action

def parse_with_defaults(parser: argparse.ArgumentParser, argv=None):
    args = parser.parse_args(argv)

    # Provided = options whose dest is present and not UNSET
    provided = set()
    for action in parser._actions:
        dest = action.dest
        val = getattr(args, dest, _UNSET)
        if val is not _UNSET:
            provided.add(dest)

    # Apply real defaults for ones not provided
    for action in parser._actions:
        if not hasattr(action, "real_default"):
            continue
        dest = action.dest
        if getattr(args, dest, _UNSET) is _UNSET:
            setattr(args, dest, action.real_default)

    return args, provided


def set_config_from_provided_flags(config, provided, args, available_attributes_to_change_when_resuming):
    for name in provided:
        if name in available_attributes_to_change_when_resuming.keys():
            conf_name = available_attributes_to_change_when_resuming[name]
            print(f"    Add to config: {conf_name}: {getattr(args, name)}")
            config[conf_name] = getattr(args, name)
        else:
            print(f"False: {name}")
        
    if 'path_to_config' in provided:
        config['path_to_config'] = getattr(args, 'path_to_config')

    config["TOTAL_TIMESTEPS"] = config["N_JIT_STEPS"] * config["NUM_ENV_STEPS"] * config["NUM_ENVS"]
    if config["NUM_ENVS"]>8:
        config["NUM_MINIBATCHES"] = args.num_minibatches
    else:
        config["NUM_MINIBATCHES"] = 1

    config["MINIBATCH_SIZE"] = (config["NUM_ENVS"] * config["NUM_ENV_STEPS"] // config["NUM_MINIBATCHES"])


    return config


def is_jax_compatible(x):
    return isinstance(x, (int, float, bool, jnp.ndarray, np.ndarray))

def filter_jax_dict(d):
    return {k: v for k, v in d.items() if is_jax_compatible(v)}



# def log_metrics(writer, i, config, args, time_, metric_trajectory, metric_losses, metric_kl, metric_entropy, metric_clip_norm_ratio):
def log_metrics(writer, i, config, args, time_, metric_outcome, metric_policy, metric_losses, metric_clip_norm_ratio):
    (R_vec, R_i_vec, Ps_vec, Pd_vec, Ps1_vec, Ps2_vec, R_tT_std, Ps_tT_std, Pd_tT_std) = metric_outcome   # (Njs). Ps_vec_tT_: (Njs, 2)
    (sv_KL_distance_vec, sub_KL_distance_vec, sub_entropy_vec, sv_entropy_vec) = metric_policy                   # (Njs, E, MbN)
    (training_loss, loss_v, loss_vi, loss_actor_sub, loss_actor_sv, loss_kl_sub_i, loss_kl_sv, sv_entropy_loss, sub_entropy_loss) = metric_losses       # (Njs, E, MbN)
    (global_grad_norm, sub_ratio_dist, sv_ratio_dist, sv_ratio_clip_rate, sub_ratio_clip_rate, sv_Adv_vec, sub_Adv_vec) = metric_clip_norm_ratio     # (Njs, E, MbN)

    global_grad_norm_mean = jnp.mean(global_grad_norm)
    global_grad_norm_vec = jnp.array([ jnp.min(global_grad_norm, initial=1e+10),
                                    jnp.mean(global_grad_norm, where=global_grad_norm<global_grad_norm_mean),
                                    global_grad_norm_mean,
                                    jnp.mean(global_grad_norm, where=global_grad_norm>=global_grad_norm_mean),
                                    jnp.max(global_grad_norm, initial=-1e+10)])

    writer.add_scalars(main_tag="Trajectory outcomes/R_tT",
        tag_scalar_dict={
            "min(R_tT) ":  jnp.mean(R_vec[:,0]) ,
            "mean(R_tT<mean) ":  jnp.mean(R_vec[:,1]) ,
            "mean(R_tT) ":  jnp.mean(R_vec[:,2]) ,
            "mean(R_tT>=mean) ":  jnp.mean(R_vec[:,3]) ,
            "max(R_tT) ":  jnp.mean(R_vec[:,4]) ,
        }, global_step=i)
    writer.add_scalars(main_tag="Trajectory outcomes/R_i_vec",
        tag_scalar_dict={
            "min(R_i_vec) ":  jnp.mean(R_i_vec[:,0]) ,
            "mean(R_i_vec<mean) ":  jnp.mean(R_i_vec[:,1]) ,
            "mean(R_i_vec) ":  jnp.mean(R_i_vec[:,2]) ,
            "mean(R_i_vec>=mean) ":  jnp.mean(R_i_vec[:,3]) ,
            "max(R_i_vec) ":  jnp.mean(Ps_vec_tT_R_i_vec[:,4]) ,
        }, global_step=i)
    writer.add_scalars(main_tag="Trajectory outcomes/Ps",
        tag_scalar_dict={
            "min(Ps) ":  jnp.mean(Ps_vec[:,0]) ,
            "mean(Ps<mean) ":  jnp.mean(Ps_vec[:,1]) ,
            "mean(Ps) ":  jnp.mean(Ps_vec[:,2]) ,
            "mean(Ps>=mean) ":  jnp.mean(Ps_vec[:,3]) ,
            "max(Ps) ":  jnp.mean(Ps_vec[:,4]) ,
        }, global_step=i)
    writer.add_scalars(main_tag="Trajectory outcomes/Pd",
        tag_scalar_dict={
            "min(Pd) ":  jnp.mean(Pd_vec[:,0]) ,
            "mean(Pd<mean) ":  jnp.mean(Pd_vec[:,1]) ,
            "mean(Pd) ":  jnp.mean(Pd_vec[:,2]) ,
            "mean(Pd>=mean) ":  jnp.mean(Pd_vec[:,3]) ,
            "max(Pd) ":  jnp.mean(Pd_vec[:,4]) ,
        }, global_step=i)
    
    writer.add_scalars(main_tag="Trajectory outcomes/Ps1",
        tag_scalar_dict={
            "min(Ps1) ":  jnp.mean(Ps1_vec[:,0]) ,
            "mean(Ps1<mean) ":  jnp.mean(Ps1_vec[:,1]) ,
            "mean(Ps1) ":  jnp.mean(Ps1_vec[:,2]) ,
            "mean(Ps1>=mean) ":  jnp.mean(Ps1_vec[:,3]) ,
            "max(Ps1) ":  jnp.mean(Ps1_vec[:,4]) ,
        }, global_step=i)
    writer.add_scalars(main_tag="Trajectory outcomes/Ps2",
        tag_scalar_dict={
            "min(Ps2) ":  jnp.mean(Ps2_vec[:,0]) ,
            "mean(Ps2<mean) ":  jnp.mean(Ps2_vec[:,1]) ,
            "mean(Ps2) ":  jnp.mean(Ps2_vec[:,2]) ,
            "mean(Ps2>=mean) ":  jnp.mean(Ps2_vec[:,3]) ,
            "max(Ps2) ":  jnp.mean(Ps2_vec[:,4]) ,
        }, global_step=i)
    writer.add_scalars(main_tag="Trajectory outcomes/Standard deviation",
        tag_scalar_dict={
            "R_tT_std":  jnp.sqrt( jnp.mean(R_tT_std*R_tT_std, axis=0) ),
            "Ps_tT_std": jnp.sqrt( jnp.mean(Ps_tT_std*Ps_tT_std, axis=0) ),
            "Pd_tT_std": jnp.sqrt( jnp.mean(Pd_tT_std*Pd_tT_std, axis=0) ),
        }, global_step=i)
    writer.add_scalars(main_tag="Trajectory outcomes/Ps_vec_tT_",
        tag_scalar_dict={
            "Ps_1": jnp.mean(jnp.mean(Ps1_vec[:,2])),    ## probability that the sub reaches the left objective
            "Ps_2": jnp.mean(jnp.mean(Ps2_vec[:,2])),    ## probability that the sub reaches the right objective
        }, global_step=i)
    writer.add_scalars(main_tag="Trajectory outcomes/(Pd,Ps,P0)",
        tag_scalar_dict={
            "Pd": jnp.mean(jnp.mean(Pd_vec[:,2])),
            "Ps": jnp.mean(jnp.mean(Ps_vec[:,2])),
            "P0": jnp.mean(1.0-(jnp.mean(Ps_vec[:,2])+jnp.mean(Pd_vec[:,2]))),
        }, global_step=i)
    

    writer.add_scalars(main_tag="Policy metrics/sv_KL_distance_vec",
        tag_scalar_dict={
            "min(KL) ":  jnp.mean(sv_KL_distance_vec[:,0]) ,
            "mean(KL<mean) ":  jnp.mean(sv_KL_distance_vec[:,1]) ,
            "mean(KL) ":  jnp.mean(sv_KL_distance_vec[:,2]) ,
            "mean(KL>=mean) ":  jnp.mean(sv_KL_distance_vec[:,3]) ,
            "max(KL) ":  jnp.mean(sv_KL_distance_vec[:,4]) ,
        }, global_step=i)
    writer.add_scalars(main_tag="Policy metrics/sub_KL_distance_vec",
        tag_scalar_dict={
            "min(KL) ":  jnp.mean(sub_KL_distance_vec[:,0]) ,
            "mean(KL<mean) ":  jnp.mean(sub_KL_distance_vec[:,1]) ,
            "mean(KL) ":  jnp.mean(sub_KL_distance_vec[:,2]) ,
            "mean(KL>=mean) ":  jnp.mean(sub_KL_distance_vec[:,3]) ,
            "max(KL) ":  jnp.mean(sub_KL_distance_vec[:,4]) ,
        }, global_step=i)
    
    writer.add_scalars(main_tag="Policy metrics/sv_entropy_vec",
        tag_scalar_dict={
            "min(entropy) ":  jnp.mean(sv_entropy_vec[:,0]) ,
            "mean(entropy<mean) ":  jnp.mean(sv_entropy_vec[:,1]) ,
            "mean(entropy) ":  jnp.mean(sv_entropy_vec[:,2]) ,
            "mean(entropy>=mean) ":  jnp.mean(sv_entropy_vec[:,3]) ,
            "max(entropy) ":  jnp.mean(sv_entropy_vec[:,4]) ,
        }, global_step=i)
    writer.add_scalars(main_tag="Policy metrics/sub_entropy_vec",
        tag_scalar_dict={
            "min(entropy) ":  jnp.mean(sub_entropy_vec[:,0]) ,
            "mean(entropy<mean) ":  jnp.mean(sub_entropy_vec[:,1]) ,
            "mean(entropy) ":  jnp.mean(sub_entropy_vec[:,2]) ,
            "mean(entropy>=mean) ":  jnp.mean(sub_entropy_vec[:,3]) ,
            "max(entropy) ":  jnp.mean(sub_entropy_vec[:,4]) ,
        }, global_step=i)
    


    writer.add_scalars(main_tag="Clip Norm Ration/Grad-norm",
        tag_scalar_dict={
            "min(grad_norm) ":  jnp.mean(global_grad_norm_vec[0]) ,
            "mean(grad_norm<mean) ":  jnp.mean(global_grad_norm_vec[1]) ,
            "mean(grad_norm) ":  jnp.mean(global_grad_norm_vec[2]) ,
            "mean(grad_norm>=mean) ":  jnp.mean(global_grad_norm_vec[3]) ,
            "max(grad_norm) ":  jnp.mean(global_grad_norm_vec[4]) ,
        }, global_step=i)
    
    writer.add_scalars(main_tag="Clip Norm Ration/Sub-ratio",
        tag_scalar_dict={
            "min(ratio) ":  jnp.mean(sub_ratio_dist[:,0]) ,
            "mean(ratio<1) ":  jnp.mean(sub_ratio_dist[:,1]) ,
            "mean(ratio) ":  jnp.mean(sub_ratio_dist[:,2]) ,
            "mean(ratio>=1) ":  jnp.mean(sub_ratio_dist[:,3]) ,
            "max(ratio) ":  jnp.mean(sub_ratio_dist[:,4]) ,
        }, global_step=i)
    writer.add_scalars(main_tag="Clip Norm Ration/SV-ratio",
        tag_scalar_dict={
            "min(ratio) ":  jnp.mean(sv_ratio_dist[:,0]) ,
            "mean(ratio<1) ":  jnp.mean(sv_ratio_dist[:,1]) ,
            "mean(ratio) ":  jnp.mean(sv_ratio_dist[:,2]) ,
            "mean(ratio>=1) ":  jnp.mean(sv_ratio_dist[:,3]) ,
            "max(ratio) ":  jnp.mean(sv_ratio_dist[:,4]) ,
        }, global_step=i)
    writer.add_scalars(main_tag="Clip Norm Ration/SV-Adv",
        tag_scalar_dict={
            "min(adv) ":  jnp.mean(sv_Adv_vec[:,0]) ,
            "mean(adv<mean) ":  jnp.mean(sv_Adv_vec[:,1]) ,
            "mean(adv) ":  jnp.mean(sv_Adv_vec[:,2]) ,
            "mean(adv>=mean) ":  jnp.mean(sv_Adv_vec[:,3]) ,
            "max(adv) ":  jnp.mean(sv_Adv_vec[:,4]) ,
        }, global_step=i)
    writer.add_scalars(main_tag="Clip Norm Ration/Sub-Adv",
        tag_scalar_dict={
            "min(adv) ":  jnp.mean(sub_Adv_vec[:,0]) ,
            "mean(adv<0) ":  jnp.mean(sub_Adv_vec[:,1]) ,
            "mean(adv) ":  jnp.mean(sub_Adv_vec[:,2]) ,
            "mean(adv>=0) ":  jnp.mean(sub_Adv_vec[:,3]) ,
            "max(adv) ":  jnp.mean(sub_Adv_vec[:,4]) ,
        }, global_step=i)
    

    writer.add_scalars( main_tag="Clip Norm Ration/ratio_clip_rate", tag_scalar_dict={"SV ratio clip-rate": jnp.mean(sv_ratio_clip_rate) }, global_step=i)
    writer.add_scalars( main_tag="Clip Norm Ration/ratio_clip_rate", tag_scalar_dict={"Sub ratio clip-rate": jnp.mean(sub_ratio_clip_rate) }, global_step=i)

    
    writer.add_scalar("losses/training loss", jnp.mean(training_loss), global_step=i)
    writer.add_scalar("losses/value-i loss", jnp.mean(loss_vi), global_step=i)
    writer.add_scalar("losses/value loss", jnp.mean(loss_v), global_step=i)
    writer.add_scalars(main_tag="losses/policy-loss",
        tag_scalar_dict={
            "loss_policy_sub": jnp.mean(loss_actor_sub),         # Mean sub policy loss
            "loss_policy_sv": jnp.mean(loss_actor_sv),           # Mean SV policy loss
        }, global_step=i)
    writer.add_scalar("losses/Sub KL-loss", jnp.mean(loss_kl_sub_i), global_step=i)
    writer.add_scalar("losses/SV KL-loss", jnp.mean(loss_kl_sv), global_step=i)
    writer.add_scalar("losses/SV Entropy-loss", jnp.mean(sv_entropy_loss), global_step=i)
    writer.add_scalar("losses/Sub Entropy-loss", jnp.mean(sub_entropy_loss), global_step=i)    
    
    writer.add_scalar("Training scheme/c", config['c'], global_step=i)
    writer.add_scalar("Training scheme/update-dt", (time_), global_step=i)
    

    return