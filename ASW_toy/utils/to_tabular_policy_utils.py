import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import json
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
import pathlib

import argparse
import os

from dataclasses import dataclass

from train_utils import load_all_params_from_checkpoints




def GetActionHistoryFromDict(sub_dict, key=""):
    sub_base_history = [[], [1], [1,3], [1,3], [1,3,6], [1,3,6], [1,3,6,10], [1,3,6,10]]
    # fallback = jnp.array([2, 2, 6, 5, 10], dtype=jnp.int32)
    fallback = jnp.array([1, 3, 4, 6, 7, 10], dtype=jnp.int32)

    SV = sub_dict['SV']
    Sub = sub_dict['Sub']

    if (sub_dict['pl']==0):
        Sub = sub_dict.get("Sub", [])
    else:
        Sub = sub_base_history[sub_dict['turn']]

    SV = sub_dict.get("SV", [])
    # Sub_trimmed = Sub[1:] if len(Sub) > 0 else []  # Exclude the first element in "Sub" if there are any elements
    Sub_trimmed = Sub

    # Convert to jax arrays and concatenate
    sub = jnp.array(Sub_trimmed, dtype=jnp.int32)
    sv = jnp.array(SV, dtype=jnp.int32)

    output = []
    output.append(sub[0])
    # if sub.shape[0] > 1:
    #     output.append(sub[1])

    sub_tail = sub[1:]
    sv_tail = sv
    max_len = max(sub_tail.shape[0], sv_tail.shape[0])

    # print(f"sub_tail.shape[0]: {sub_tail.shape[0]}  sv_tail.shape[0]: {sv_tail.shape[0]}")
    # print(f"min_len: {max_len}")
    for i in range(max_len):
        if (sub_tail.shape[0]>i):
            output.append(sub_tail[i])
        if(sv_tail.shape[0]>i):
            output.append(sv_tail[i])
    
    # print(f"output-1: {output}")
    # print(f"output-2: {output}")

    # Pad or truncate to shape (6,)
    output = jnp.array(output, dtype=jnp.int32)
    pad_len = 6 - output.shape[0]
    if pad_len > 0:
        output = jnp.concatenate([output, fallback[output.shape[0]:6]])
    else:
        output = output[:6]
    
    # print(f"sv: {sv}  sub: {sub}")
    # print(f"{sub_dict}: {output} {output.dtype}")
    return output


def nonzero_prob_dict(prob_array: jnp.ndarray) -> dict:
    """
    Converts a 1D JAX array of probabilities to a Python dict {index: prob}
    including only entries with non-zero probability.
    """
    # Convert to numpy for indexing and iteration
    array_np = jnp.asarray(prob_array)
    
    # Get indices where probability is non-zero
    nonzero_indices = jnp.nonzero(array_np)[0]

    # Construct the dictionary
    # return {int(i): float(array_np[i]) for i in nonzero_indices}
    return {str(i): float(array_np[i]) for i in nonzero_indices}


def GoThroughAllStates(action_history, turn_bool, env_state, key, model_params, env, network ):
    # jax.debug.print("'GoThroughAllStates' action_history: {}  ", action_history)
    # jax.debug.print("'GoThroughAllStates' turn_bool: {}  ", turn_bool)



    # (action_history, key) = input
    def env_step(runner_state, data):



        # (model_params, env_state, rng) = runner_state
        (action_hist, turn_bool) = data
        (env_state, rng) = runner_state
        obs = env.get_obs(env_state)

        # action_hist = unused
        sub_pos_samples = jnp.array(env_state.sub_pos_samples)
        (logits, v, v_map) = network.apply(model_params, obs)
        dip_mask = env.available_sv_dips[jnp.maximum(env_state.turn//2-1, 0)] # (N, 3,13) --> (N, 13)

        # print(f"env_state.sub_pos_samples.dtype: {env_state.sub_pos_samples.dtype}")
        # jax.debug.print("env_state.sub_pos_samples.dtype: {}", env_state.sub_pos_samples.dtype)
        # jax.debug.print("env_state.sub_pos_samples.: {}\n{}", env_state.sub_pos_samples.shape, env_state.sub_pos_samples)

        sub_logit = logits[:, :13, :]
        dip_logit = logits[:, 13, :]
        sub_policy, sampled_sub_policy, sv_policy = env.mask_logit_to_policy(sub_logit, dip_logit, env_state.sub_pos_samples, dip_mask) # env_step
        rng, _rng = jax.random.split(rng)
        action_sub_samples, action_dip = env.sample_action(_rng, sv_policy, sampled_sub_policy)

        action_dip = jnp.where( (env_state.turn%2==0), jnp.expand_dims(action_hist, axis=0), action_dip)

        action_sub_samples = jnp.where( jnp.logical_or((env_state.turn%2==1), (env_state.turn==0)), jnp.expand_dims(jnp.expand_dims(action_hist, axis=0), axis=0), action_sub_samples)

        (new_state, r, r_i, done, done_i, done2, done_i2, p_prior, p_i_prior, p, p_i, pd, ps, ps_i, pd_i) = env.step(env_state, sub_policy, action_dip, action_sub_samples)
        obs = env.get_obs(new_state)
        # env_state = new_state


        def use_new_state(_):
            return new_state

        def keep_old_state(_):
            return env_state
        env_state = lax.cond(turn_bool, use_new_state, keep_old_state, operand=None)
        

        # runner_state = (env_state, model_params, rng)
        # runner_state = (model_params, env_state, rng)
        runner_state = (env_state, rng)
        return runner_state, action_dip

    
    key, _rng = jax.random.split(key)
    # obsv, env_state = env.reset(N, n, _rng)

    # runner_state = (model_params, env_state, key)
    runner_state = (env_state, key)


    # print(f"env_state.turn: {env_state.turn.shape}")
    # print(f"turn_bool: {turn_bool.shape}")

    # jax.debug.print("", )

    runner_state, action_dip = jax.lax.scan(
        # env_step, runner_state, action_history, action_history.shape[0]
        env_step, runner_state, (action_history, turn_bool), 6
    )

    # (env_state, model_params, rng) = runner_state
    (env_state, key) = runner_state

    obs = env.get_obs(env_state)
    (logits, v, v_map) = network.apply(model_params, obs)
    dip_mask = env.available_sv_dips[jnp.maximum(env_state.turn//2-1, 0)] # (N, 3,13) --> (N, 13)

    sub_logit = logits[:, :13, :]
    dip_logit = logits[:, 13, :]
    sub_policy, sampled_sub_policy, sv_policy = env.mask_logit_to_policy(sub_logit, dip_logit, env_state.sub_pos_samples, dip_mask) # end state
    
    sub_policy = sub_policy.astype(jnp.float64)
    sampled_sub_policy = sampled_sub_policy.astype(jnp.float64)
    sv_policy = sv_policy.astype(jnp.float64)

    sub_policy = sub_policy / jnp.sum(sub_policy, axis=(-1), keepdims=True)
    sampled_sub_policy = sampled_sub_policy / jnp.sum(sampled_sub_policy, axis=(-1), keepdims=True)
    sv_policy = sv_policy / jnp.sum(sv_policy, axis=(-1), keepdims=True)
    # print(f"sampled_sub_policy: {sampled_sub_policy.shape}")
    # print(f"sv_policy: {sv_policy.shape}")
    # print(f"sub_policy: {sub_policy.shape}")

    
    
    def GetSVPolicy(env_state, sub_policy, sv_policy, sampled_sub_policy):
        return sv_policy

    def GetSubPolicy(env_state, sub_policy, sv_policy, sampled_sub_policy):
        return sampled_sub_policy[0]

    policy = lax.cond(
        env_state.turn[0]%2==0,
        GetSVPolicy,
        GetSubPolicy,
        env_state, sub_policy, sv_policy, sampled_sub_policy
    )

    # print(f"policy: {policy.shape}")

    return policy




def ComputeJaxPolicy(network, env, key, stored_checkpoint_indices, model_params_path, model_training_path, game_state_dict, recent_checkpoint_to_tabular_policy):
    # model_params_path = args.save_path + "target_model/target_model_params"
    # model_training_path = args.save_path + "target_model"
    list_of_params = load_all_params_from_checkpoints(network, key, model_params_path)
    print(f"list_of_params: {len(list_of_params)}")


    N = 1
    n = 1

    rng = jax.random.PRNGKey(0)
    # rng, _rng = jax.random.split(rng)
    # network = MLP(hidden_dim=128)
    # dummy_input = jnp.ones((N, 2, 13))
    # model_params = network.init(_rng, dummy_input)
    # model_params = trained_params
    turn_arange = jnp.arange(0, 6)
    print(f"turn_arange: {turn_arange}")

    action_histories = {}
    action_histories_list = []
    state_string_list = []
    turn_bool_list = []
    for state_string in game_state_dict.keys():
        state_string_list.append(state_string)
        action_hist = GetActionHistoryFromDict(game_state_dict[state_string], state_string)
        action_histories_list.append(action_hist)

        turn_bool_ = turn_arange<game_state_dict[state_string]['turn']
        turn_bool_list.append(turn_bool_)


        # action_histories[state_string] = action_hist
    print(f"state_string_list:\n{state_string_list}")

    action_histories = jnp.stack(action_histories_list, axis=0)
    print(f"action_histories: {action_histories.shape}")
    turn_bools= jnp.stack(turn_bool_list, axis=0)



    ## Alt-2: jax-vmap over the model-params
    time_start = time.time()
    params_batch = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *list_of_params)
    keys = jax.random.split(rng, action_histories.shape[0])
    # def per_param_fn(params):
    #     return jax.vmap(GoThroughAllStates, in_axes=(0,0,0,0,None,None,None))(
    #         action_histories, turn_bools, env_state, keys, params, env, network)

    print(f"GoThroughAllStates...")
    
    batched_reset_fn = jax.vmap(lambda key: env.reset(N, n, key))
    obsv, env_state = batched_reset_fn(keys)
    all_policies = jax.vmap(
        lambda params: jax.vmap(GoThroughAllStates, in_axes=(0,0,0,0,None,None,None))(action_histories, turn_bools, env_state, keys, params, env, network)
    )(params_batch)
    time_end = time.time()

    print(f"all_policies: {all_policies.shape}")
    print(f"Policy extraction time: {time_end-time_start} s ")

    for cp_i in range(len(list_of_params)):
        if recent_checkpoint_to_tabular_policy < stored_checkpoint_indices[cp_i]:

            policy = all_policies[cp_i]
            policy = jax.device_put(policy, device=jax.devices("cpu")[0])
            # print(f"policies: {policies.shape}")
            policy = policy[:,0,:]
            # print(f"policy[:,0,:]: {policy.shape}")

            policy_dicts = {}
            for i in range(policy.shape[0]):
                policy_dict = nonzero_prob_dict(policy[i])
                policy_dicts[state_string_list[i]] = policy_dict

                # print(f"{state_string_list[i]}:")
                # for key in policy_dict.keys():
                #     print(f"    {key}: {policy_dict[key]}")

            # Optional: convert inner keys to strings if needed
            policy_dicts_float = {
                str(k): {str(inner_k): v for inner_k, v in inner_dict.items()}
                for k, inner_dict in policy_dicts.items()
            }

            # Save to JSON file
            save_dir = pathlib.Path(model_training_path + "/jax_trained_policy")
            save_dir.mkdir(parents=True, exist_ok=True)

            # stored_checkpoint_indices
            with open(model_training_path + "/jax_trained_policy/jax_policy_step" + str(int(stored_checkpoint_indices[cp_i])) + ".json", "w") as f:
                json.dump(policy_dicts_float, f, indent=2)
                
            recent_checkpoint_to_tabular_policy = int(stored_checkpoint_indices[cp_i])

    time_end2 = time.time()

    print(f"Save policy time: {time_end2-time_start} s ")
    return recent_checkpoint_to_tabular_policy