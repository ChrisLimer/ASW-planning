import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax.numpy as jnp
import jax
import chex
import numpy as np
from flax import struct
import abc
from jax import lax
import flax.linen as nn
import optax
from typing import NamedTuple

from jax import tree_util
import jax.tree_util as jtu

from dataclasses import dataclass

import flax.serialization
import pathlib
from typing import List
import re


@chex.dataclass
# sub_pos_samples, sub_distribution, sampled_sub_detected, turn, sv_dip_history, done, done_i, p, p_i
class SubState:
    sub_pos_samples: jnp.array
    sub_distribution: jnp.array     # (13)
    sampled_sub_detected: jnp.array
    turn: jnp.array
    sv_dip_history: jnp.array
    sub_samples_history: jnp.array
    
    done: jnp.array
    done_i: jnp.array

    p: jnp.array
    p_i: jnp.array
    



class Transition(NamedTuple):
    done: jnp.ndarray       # true if the new state after applying the actions ('action_sub_samples' and 'action_dip') is terminal
    done_i: jnp.ndarray     # true if the new state after applying the actions ('action_sub_samples' and 'action_dip') is terminal
    done2: jnp.ndarray      # true if the state was terminal --> the actions are invalid in the current invalid state
    done_i2: jnp.ndarray    # true if the state was terminal --> the actions are invalid in the current invalid state

    r: jnp.ndarray          # expected reward in the next state after playing action a. E(R(S)|a): next state reward multiplied by the transition-probability 
    r_i: jnp.ndarray        # expected future reward from the current state S(h)
    v: jnp.ndarray          # expected future value (reward) from the current state S(h)
    v_map: jnp.ndarray      # expected future value (reward) from the current state S(h). Map over all sub positions 
    sub_distribution: jnp.ndarray
    p_prior: jnp.ndarray    # reach probability of current state
    p_i_prior: jnp.ndarray  # reach probability of current state
    p: jnp.ndarray          # reach probability of the next state
    p_i: jnp.ndarray        # reach probability of the next state

    sub_pos_samples: jnp.ndarray    # position of the sub samples position
    dip_mask: jnp.ndarray           # mask of available dip positions

    obs: jnp.ndarray        # observation of the current state
    
    sub_logit: jnp.ndarray
    dip_logit: jnp.ndarray
    sub_policy: jnp.ndarray
    sv_policy: jnp.ndarray
    action_sub_samples: jnp.ndarray
    action_dip: jnp.ndarray

    pd: jnp.ndarray
    ps: jnp.ndarray
    ps_i: jnp.ndarray
    pd_i: jnp.ndarray

    player: jnp.ndarray

class Target(NamedTuple):
    R_tT: jnp.ndarray
    R_tTi: jnp.ndarray
    r: jnp.ndarray
    r_i: jnp.ndarray






# class SubEnv(nn.Module):
class SubEnv(nn.Module):
    N_nodes: int =13
    a: float =1.0
    b: float =0.0

                                            #0 1 2 3 4 5 6 7 8 9 101112
    available_sub_destinations = jnp.array([[0,1,1,0,0,0,0,0,0,0,0,0,0],  # 0
                                            [0,0,0,1,1,0,0,0,0,0,0,0,0],  # 1
                                            [0,0,0,0,1,1,0,0,0,0,0,0,0],  # 2
                                            [0,0,0,0,0,0,1,1,0,0,0,0,0],  # 3
                                            [0,0,0,0,0,0,0,1,1,0,0,0,0],  # 4
                                            [0,0,0,0,0,0,0,0,1,1,0,0,0],  # 5
                                            [0,0,0,0,0,0,0,0,0,0,1,0,0],  # 6
                                            [0,0,0,0,0,0,0,0,0,0,1,1,0],  # 7
                                            [0,0,0,0,0,0,0,0,0,0,0,1,1],  # 8
                                            [0,0,0,0,0,0,0,0,0,0,0,0,1],  # 9
                                            [0,0,0,0,0,0,0,0,0,0,1,0,0],  # 10
                                            [0,0,0,0,0,0,0,0,0,0,0,1,0],  # 11
                                            [0,0,0,0,0,0,0,0,0,0,0,0,1]], # 12
                                            dtype=jnp.bool_)  # jnp.bool_, jnp.float32, jnp.int32
    
                                            #0 1 2 3 4 5 6 7 8 9 1011   turn/2-1
    available_sv_dips = jnp.array(       [[0,0,0,1,1,1,0,0,0,0,0,0,0],  # 0
                                            [0,0,0,0,0,0,1,1,1,1,0,0,0],  # 1
                                            [0,0,0,0,0,0,0,0,0,0,1,1,1]], # 2
                                            dtype=jnp.bool_)  # jnp.bool_, jnp.float32, jnp.int32
    
    

    def get_obs(self, state: SubState):
        # obs = jnp.stack([state.sub_distribution, state.sv_dip_history], axis=1)  # shape: (N, 2, 13)
        # obs = jnp.stack([state.sub_distribution], axis=1)  # shape: (N, 2, 13)

        obs = state.sub_distribution
        return obs


    def reset(self, N: int, n: int, key: jax.random.PRNGKey):
        print(f"N: {N}  n: {n}  self.N_nodes: {self.N_nodes}")
        sub_distribution = jnp.zeros((N, self.N_nodes))
        sub_distribution = sub_distribution.at[:,0].set(1.0)
        sub_pos_samples = jnp.zeros((N, n), dtype=jnp.int32)
        sampled_sub_detected = jnp.zeros((N, n), dtype=jnp.bool_)
        sv_dip_history = jnp.zeros((N, self.N_nodes))
        sub_samples_history = jnp.zeros((N, n, self.N_nodes))

        # turn = jnp.array([1], dtype=jnp.int32)
        turn = jnp.zeros((N), dtype=jnp.int32)

        done = jnp.zeros((N), dtype=jnp.bool_)
        done_i = jnp.zeros((N, n), dtype=jnp.bool_)
        p = jnp.ones((N), dtype=jnp.float32)
        p_i = jnp.ones((N, n), dtype=jnp.float32)



        # sub_pos_samples, sub_distribution, sampled_sub_detected, turn, sv_dip_history, done, done_i, p, p_i
        # state = SubState(sub_pos_samples=sub_pos_samples, sub_distribution=sub_distribution, sampled_sub_detected=sampled_sub_detected, turn=turn, sv_dip_history=sv_dip_history)
        state = SubState(sub_pos_samples=sub_pos_samples, 
                         sub_distribution=sub_distribution, 
                         sampled_sub_detected=sampled_sub_detected, 
                         turn=turn, 
                         sv_dip_history=sv_dip_history,
                         sub_samples_history=sub_samples_history,
                         done=done,
                         done_i=done_i,
                         p=p,
                         p_i=p_i)

        obs = self.get_obs(state)
        return obs, state
    


    
    # def mask_logit_to_policy(self, logit: jnp.array, sub_pos_samples, dip_mask):
        # print(f"    logit: {logit.shape}")
        # dip_logit = logit[:,13,:]   # (N, 14,13) --> (N, 13)
        # sub_logit = logit[:, :13,:]
        # print(f"    dip_logit: {dip_logit.shape}")
        # print(f"    sub_logit: {sub_logit.shape}")

    def mask_logit_to_policy(self, sub_logit: jnp.array, dip_logit: jnp.array, sub_pos_samples, dip_mask):


        # sub_logit = sub_logit.at[:,0,:].set(-1e15)
        # sub_logit = sub_logit.at[:,0,1].set(1)
        sub_logit = sub_logit.at[:,0,:].set(1.0)

        logit_exp = jnp.exp(dip_logit)
        logit_exp = jnp.where(dip_mask[:], logit_exp, 0.0)
        sv_policy = logit_exp/jnp.sum(logit_exp, axis=1, keepdims=True)

        N = sub_pos_samples.shape[0]
        batch_idx = jnp.arange(N)

        # jax.debug.print("sub_pos_samples: {}", sub_pos_samples.dtype)
        print(f"'mask_logit_to_policy' sub_pos_samples: {sub_pos_samples.dtype}")

        sampled_mask = self.available_sub_destinations[sub_pos_samples]     # (N, n)
        sampled_logit = sub_logit[batch_idx[:,None], sub_pos_samples, :]
        sampled_masked_logit = jnp.where(sampled_mask, sampled_logit, 0.0)
        sampled_masked_exp_logit = jnp.exp(sampled_masked_logit)
        sampled_masked_exp_logit = jnp.where(sampled_mask, sampled_masked_exp_logit, 0.0)
        sampled_sub_policy = sampled_masked_exp_logit/jnp.sum(sampled_masked_exp_logit, axis=2, keepdims=True )

    

        sub_logit = jnp.where(self.available_sub_destinations[None,:], sub_logit, 1e-15)   # (N, 13, 13)
        sub_exp_logit = jnp.where(self.available_sub_destinations[None,:], jnp.exp(sub_logit), 0.0)
        sub_policy = sub_exp_logit/jnp.sum(sub_exp_logit, where=self.available_sub_destinations[None,:], axis=2, keepdims=True)

        return sub_policy, sampled_sub_policy, sv_policy
    
    
    def sample_action(self, key, sv_policy, sampled_sub_policy):

        dip_sampled_action = jax.random.categorical(key, jnp.log(sv_policy+1e-15), axis=-1)

        def sample_n(key, logits_n):
            return jax.random.categorical(key, logits_n, axis=-1)

        keys = jax.random.split(key, sampled_sub_policy.shape[0])
        sub_samples_action = jax.vmap(sample_n)(keys, jnp.log(sampled_sub_policy+1e-15))

        return sub_samples_action, dip_sampled_action



    def step(self, state: SubState, sub_policy: jnp.array, sv_action: jnp.array, sampled_sub_action: jnp.array):
        
        print(f"step...")
        print(f"    sub_policy: {sub_policy.shape}")
        print(f"    sv_action: {sv_action.shape}")
        print(f"    sampled_sub_action: {sampled_sub_action.shape}")

        p_prior = state.p
        p_i_prior = state.p_i
        # sv_dip_history: jnp.array
        # sub_samples_history: jnp.array
        def ApplyDip(state: SubState, sub_policy: jnp.array, sv_action: jnp.array, sampled_sub_action: jnp.array):
            
            print(f"    ApplyDip. sv_action: {sv_action.shape}")
            batch_idx = jnp.arange(sv_action.shape[0])
            
            pd = state.sub_distribution[batch_idx, sv_action]
            state.sub_distribution = state.sub_distribution.at[batch_idx, sv_action].set(0.0)
            state.sv_dip_history = state.sv_dip_history.at[batch_idx, sv_action].set(1.0)


            ps_i_mask = jnp.isin(state.sub_pos_samples, jnp.array([10, 11, 12]))     # Get a mask of sub_samples that reached the terminal nodes

            rs_i = jnp.isin(state.sub_pos_samples, jnp.array([10, 12])).astype(jnp.float32)
            rs_i+= jnp.isin(state.sub_pos_samples, jnp.array([11])).astype(jnp.float32) * 2.0


            pd_i_mask = (state.sub_pos_samples==sv_action[:, None])               # Get a mask of sub_samples that got detected
            state.sampled_sub_detected = jnp.logical_or(state.sampled_sub_detected, pd_i_mask)
            # state.sampled_sub_detected = jnp.logical_or(state.sampled_sub_detected, ps_i_mask)

            # jax.debug.print("'ApplyDip' state.sub_pos_samples: {}  sv_action: {}  pd_i_mask: {}  state.sampled_sub_detected: {}  rs_i: {}", state.sub_pos_samples, sv_action, pd_i_mask, state.sampled_sub_detected, rs_i)
 
            ps_i_mask = jnp.logical_and(ps_i_mask, jnp.logical_not(pd_i_mask))


            rs_i = jnp.where(jnp.logical_not(state.sampled_sub_detected), rs_i, 0.0)
            # jax.debug.print("'ApplyDip' state.sub_pos_samples: {}  sv_action: {}  pd_i_mask: {}  state.sampled_sub_detected: {}  rs_i: {}", state.sub_pos_samples, sv_action, pd_i_mask, state.sampled_sub_detected, rs_i)
            

            # ps = jnp.sum(state.sub_distribution[:, (10,11,12)], axis=(1))                # Get probability of subs at pos (9, 10, 11)
            # rs = state.sub_distribution[:, 10]
            # rs+= state.sub_distribution[:, 11]*2.0
            # rs+= state.sub_distribution[:, 12]
            ps1 = state.sub_distribution[:, 10]
            ps2 = state.sub_distribution[:, 11]
            ps3 = state.sub_distribution[:, 12]
            rs = ps1*1.0 + ps2*2.0 + ps3*1.0
            ps = ps1*1.0 + ps2*1.0 + ps3*1.0
            # state.sub_distribution = state.sub_distribution.at[:, (9, 10, 11)].set(0.0) # Set the probability of the sub at (9, 10, 11) to 0.0%
            

            print(f"    pd_i_mask: {pd_i_mask.shape}")
            print(f"    ps_i_mask: {ps_i_mask.shape}")
            # return state, pd, ps, rs, rs_i, ps_i_mask.astype(jnp.float32), pd_i_mask.astype(jnp.float32)
            return state, pd, ps, rs, rs_i, ps_i_mask.astype(jnp.float32), pd_i_mask.astype(jnp.float32)

        def MoveSub(state: SubState, sub_policy: jnp.array, sv_action: jnp.array, sampled_sub_action: jnp.array):
            state.sub_pos_samples = sampled_sub_action
            state.sub_distribution = jnp.sum(state.sub_distribution[:,:,None] * sub_policy, axis=1)   # (N, 13)[:,:,None] * (N, 13, 13) --> (N, 13)

            batch_idx = jnp.arange(sampled_sub_action.shape[0])
            sub_batch_idx = jnp.arange(sampled_sub_action.shape[1])
            print(f"state.sub_samples_history: {state.sub_samples_history.shape}  sampled_sub_action: {sampled_sub_action.shape}")
            state.sub_samples_history = state.sub_samples_history.at[batch_idx[:,None], sub_batch_idx[None,:], sampled_sub_action].set(1.0)

            pd = jnp.zeros(sampled_sub_action.shape[0])
            ps = jnp.zeros(sampled_sub_action.shape[0])
            # ps_i = jnp.zeros(state.sampled_sub_detected.shape[0], dtype=jnp.float32)
            # pd_i = jnp.zeros(state.sampled_sub_detected.shape[0], dtype=jnp.float32)
            ps_i = jnp.zeros_like(state.sampled_sub_detected, dtype=jnp.float32)
            pd_i = jnp.zeros_like(state.sampled_sub_detected, dtype=jnp.float32)

            rs_i = jnp.zeros_like(ps_i)
            rs = jnp.zeros_like(ps)

            return state, pd, ps, rs, rs_i, ps_i, pd_i


        def apply_action(state, sub_policy, sv_action, sampled_sub_action):
            new_state, pd, ps, rs, rs_i, ps_i, pd_i = lax.cond(
                (state.turn[0]%2==0) & (state.turn[0]!=0),
                ApplyDip,
                MoveSub,
                state, sub_policy, sv_action, sampled_sub_action
            )
            new_state.turn += 1
            # return new_state, pd, ps, ps_i_mask.astype(jnp.float32), pd_i_mask.astype(jnp.float32)
            return new_state, pd, ps, rs, rs_i, ps_i, pd_i
        


        # new_state, pd, ps, ps_i, pd_i = jax.vmap(apply_action)(state, sub_policy, sv_action, sampled_sub_action)
        new_state, pd, ps, rs, rs_i, ps_i, pd_i = apply_action(state, sub_policy, sv_action, sampled_sub_action)

        # r = rs-pd
        # r_i = rs_i-pd_i
        r = rs * self.a - pd*self.b
        r_i = rs_i * self.a - pd_i*self.b

        done = new_state.turn>6
        done_i = new_state.sampled_sub_detected
        done_i = jnp.logical_or(done_i, jnp.expand_dims(done, axis=1))

        done2 = jnp.logical_and(done, new_state.done)
        done_i2 = jnp.logical_and(done_i, new_state.done_i)
        new_state.done = done
        new_state.done_i = done_i

        # new_state.p = 1.0 - (1.0-state.p) * (1.0-ps) * (1.0-pd)
        # new_state.p_i = 1.0 - (1.0-state.p_i) * (1.0-ps_i) * (1.0-pd_i)

        new_state.p = p_prior * (1.0-ps) * (1.0-pd)
        new_state.p_i = p_i_prior * (1.0-ps_i) * (1.0-pd_i)

        # new_state.p = 1.0 - (1.0-p_prior) * (1.0-ps) * (1.0-pd)
        # new_state.p_i = 1.0 - (1.0-p_i_prior) * (1.0-ps_i) * (1.0-pd_i)

        new_state.sub_distribution = new_state.sub_distribution/jnp.sum(new_state.sub_distribution, axis=(1), keepdims=True)

        # return new_state, r, r_i, done, done_i, done2, done_i2, p_prior, p_i_prior, state.p, state.p_i
        return new_state, r, r_i, done, done_i, done2, done_i2, p_prior, p_i_prior, new_state.p, new_state.p_i, pd, ps, ps_i, pd_i

    
    # r, r_i, done, done_i, p)