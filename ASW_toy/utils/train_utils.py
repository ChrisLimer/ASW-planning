from __future__ import annotations

import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import re
from pathlib import Path
from typing import Any, Callable, Optional, Tuple, List
import shutil
import json

import numpy as np
import jax
import optax
import flax.linen as nn


_STEP_RE = re.compile(r"^params_step(\d+)\.flax$")

from flax.serialization import from_bytes, msgpack_restore
import flax.serialization
# from asw_utils import create_env
# from asw_env  import create_env
# from ActorCriticNetwork import MLP


# from flax.serialization import msgpack_restore
from flax.core import unfreeze
from flax.traverse_util import flatten_dict

import jax.numpy as jnp



def create_config(args):
    config = {}

    N = args.N
    n = args.n

    config['NUM_ENVS'] = N
    config['NUM_SUB_ENVS'] = n
    config['NUM_STEPS'] = 8
    # config['LR'] = 0.00001  # _lr0_00001
    config['LR'] = args.lr

    config['MAX_GRAD_NORM'] = args.max_grad_norm
    config["CLIP_EPS"] = args.clip_eps
    config["NUM_EPOCHS"] = args.num_epochs

    config["N_JIT_STEPS"] = args.N_JIT_STEPS
    config["TOTAL_TIMESTEPS"] = config["N_JIT_STEPS"] * config["NUM_STEPS"] * config["NUM_ENVS"]

    if config["NUM_ENVS"]>8:
        config["NUM_MINIBATCHES"] = args.num_minibatches
    else:
        config["NUM_MINIBATCHES"] = 1

    config["MINIBATCH_SIZE"] = (config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"])

    config['current_reg_step'] = 0
    config['max_reg_step'] = config['N_JIT_STEPS'] * args.reg_freq
    config['alpha_kl'] = args.alpha_kl
    config['gamma_averaging'] = args.gamma_averaging

    config['SUB_ENT_LOSS'] = args.sv_ent_loss
    config['SV_ENT_LOSS'] = args.sub_ent_loss
    
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


def load_all_params_from_checkpoints(model: nn.Module, key, checkpoint_dir: str) -> List[dict]:
    checkpoint_dir = Path(checkpoint_dir)
    param_files = sorted(
        checkpoint_dir.glob("params*.flax"),
        key=lambda p: int(re.search(r"step(\d+)", p.name).group(1)) if re.search(r"step(\d+)", p.name) else -1
    )

    dummy_input = model.get_dummy_input()

    # Initialize a template for deserialization
    template_params = model.init(key, dummy_input)

    param_list = []
    for file in param_files:
        with file.open("rb") as f:
            params = flax.serialization.from_bytes(template_params, f.read())
            param_list.append(params)

    return param_list