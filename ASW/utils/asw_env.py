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
from pathlib import Path
import json


class Target(NamedTuple):
    R_tT: jnp.ndarray
    R_tTi: jnp.ndarray
    r: jnp.ndarray
    r_i: jnp.ndarray

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
    sv_pos: jnp.ndarray
    action_sub_samples: jnp.ndarray
    action_dip: jnp.ndarray

    pd: jnp.ndarray
    ps: jnp.ndarray
    ps_i: jnp.ndarray
    pd_i: jnp.ndarray

    player: jnp.ndarray

    ## Extra metrics for logging
    ps_vec: jnp.ndarray
    # ps_i_vec: jnp.ndarray


def axial_distance(q1, r1, q2, r2):
    return (jnp.abs(q1 - q2) + jnp.abs(q1 + r1 - q2 - r2) + jnp.abs(r1 - r2) ) / 2

def axial_to_cartesian_Pup(q, r, hex_hor=1, hex_vert=np.sqrt(3)/2):    # pointy up and flat at the sides
    x = hex_hor * q + r*hex_hor/2
    y = -hex_vert * r
    return x, y


## This function takes in a mask of islands (N_y, N_x) and returns a policy mask of possible sub actions (can't move onto an island or outside the map)
def get_transition_mask_MaskedHexesHexCord(config, board_mask): # (D, C, N_y, N_x)
    offsets = jnp.array([
        [0, 0],      # Placeholder for transition_mask[0] (can be left as zero)
        [-1, 1],     # NE
        [0, 1],      # E
        [1, 0],      # SE
        [1, -1],     # SW
        [0, -1],     # W
        [-1, 0]      # NW
    ])
    N_y = board_mask.shape[0]   # = config['N_y']
    N_x = board_mask.shape[1]   # = config['N_x']
    def get_directional_mask(offset, padded_board_mask):
        dy, dx = offset
        return lax.dynamic_slice(padded_board_mask, ((1+dy), (1+dx)), (N_y, N_x) )
    
    padded_board_mask = jnp.pad(board_mask, pad_width=1, mode='constant')   # pad zeros around the mask
    transition_mask = jax.vmap(get_directional_mask, in_axes=(0, None))(offsets, padded_board_mask)
    transition_mask = transition_mask.at[0].set(jnp.ones((N_y, N_x))) # set first channel to one (can stay in any hex, including masked hexes)
    transition_mask = transition_mask.at[1:].multiply(board_mask)   # set all transitions (except to stay/first channel) to zero/false if that hex is unplayable

    transition_mask = jnp.expand_dims(transition_mask, axis=0)  # shape (1, C, N_y, N_x)
    transition_mask = jnp.broadcast_to(transition_mask, (config['D'],) + transition_mask.shape[1:]) # (1, C, N_y, N_x) --> (D, C, N_y, N_x)

    transition_mask_B = transition_mask > 0.5
    return transition_mask, transition_mask_B
#### Transition Mask

def CreateDipMatrix_SonEkv_Topography(N_y, N_x, p_min, tile_size, SL, TS, DT, NL, DI, sig, board, SL_loss, RL_loss):
    Q = jnp.arange(N_x)  # Shape: (N_x) [0, 1, ..., N_x-1]
    R = jnp.arange(N_y)  # Shape: (N_y) [0, 1, ..., N_y-1]
    Q = jnp.repeat(jnp.expand_dims(Q, axis=(0)), repeats=N_y, axis=(0)) # (N_x) -> (N_y, N_x)
    R = jnp.repeat(jnp.expand_dims(R, axis=(1)), repeats=N_x, axis=(1)) # (N_y) -> (N_y, N_x)

    Q_ = Q.reshape(-1)      # (N_y, N_x) -> (N_y * N_x)
    R_ = R.reshape(-1)      # (N_y, N_x) -> (N_y * N_x)

    board_ = board.reshape(-1)      # (N_y, N_x) -> (N_y * N_x)

    hex_hor = 1
    hex_vert = np.sqrt(3)/2

    x, y = axial_to_cartesian_Pup(Q_, R_, hex_hor, hex_vert)        # Dip-position
    X, Y, = axial_to_cartesian_Pup(Q, R, hex_hor, hex_vert)         # Potential sub-position
    distance_map = jnp.sqrt((Y[None] - y[:, None, None]) ** 2 +
                            (X[None] - x[:, None, None]) ** 2)*tile_size

    def Compute_Probability(SE, sig):
        SE = SE/sig
        a1 =  0.254829592
        a2 = -0.284496736
        a3 =  1.421413741
        a4 = -1.453152027
        a5 =  1.061405429
        p  =  0.3275911

        # Handle negative values
        sign = jnp.where( (SE<0), -1, 1)
        SE = jnp.abs(SE)/jnp.sqrt(2)

        # Approximation formula
        t = 1.0 / (1.0 + p * SE)    # 0.0->1.0
        y = 1.0 - ((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t * jnp.exp(-SE * SE)  # # 0.0->1.0

        return 0.5 * (1.0 + sign * y)


    TL = 20*jnp.log(distance_map+1e-3)  # (1457, 31, 47)
    SL = SL - SL_loss*board_            # (1457,)
    RL = board*RL_loss                  # (31, 47)

    # SE = SL+DI+TS-2*TL-NL-DT
    SE = SL[:,None,None]+DI+TS-2*TL-NL-DT - RL[None, :,:]
    P = Compute_Probability(SE, sig)
    P = jnp.where(P>=p_min, P, jnp.zeros_like(P))

    P = jnp.expand_dims(P, axis=1)              # (N_y*N_x, N_y, N_x) --> (N_y*N_x, 1, N_y, N_x)
    # P = jnp.repeat(P, repeats=7, axis=(1) )     # (N_y*N_x, 1, N_y, N_x) --> (N_y*N_x, 7, N_y, N_x)

    P = P.reshape(N_y, N_x, *P.shape[1:])   # (N_y*N_x, 1, N_y, N_x) --> (N_y, N_x, 1, N_y, N_x)
    P = 1.0 - P
    return P



adj_kernel = jnp.array([[0, 1, 1],
                    [1, 1, 1],
                    [1, 1, 0]], dtype=jnp.float32)

def create_transition_kernel_HexCord():
    # [0,   NW,   NE]
    #    [W,  stay,  E]
    #       [SW,  SE,   0]
    # Now we flip the direction since we want the adjacent hexes actions that add to this hex. The East adjacent hex needs to go West to get to this hex
    transition_kernel = jnp.array([
    [                                               # Stay  --> set stay (only this hex will get to this hex if it stay)
        [[0, 0, 0], [0, 1, 0], [0, 0, 0]],  # stay
        [[0, 0, 0], [0, 0, 0], [0, 0, 0]],  # NE
        [[0, 0, 0], [0, 0, 0], [0, 0, 0]],  # E
        [[0, 0, 0], [0, 0, 0], [0, 0, 0]],  # SE
        [[0, 0, 0], [0, 0, 0], [0, 0, 0]],  # SW
        [[0, 0, 0], [0, 0, 0], [0, 0, 0]],  # W
        [[0, 0, 0], [0, 0, 0], [0, 0, 0]]   # NW
    ],
    [                                               # NE    --> set SW in row 1
        [[0, 0, 0], [0, 0, 0], [0, 0, 0]],  # stay
        [[0, 0, 0], [0, 0, 0], [1, 0, 0]],  # NE
        [[0, 0, 0], [0, 0, 0], [0, 0, 0]],  # E
        [[0, 0, 0], [0, 0, 0], [0, 0, 0]],  # SE
        [[0, 0, 0], [0, 0, 0], [0, 0, 0]],  # SW
        [[0, 0, 0], [0, 0, 0], [0, 0, 0]],  # W
        [[0, 0, 0], [0, 0, 0], [0, 0, 0]]   # NW
    ],
    [                                               # E     --> set W in row 2
        [[0, 0, 0], [0, 0, 0], [0, 0, 0]],  # stay
        [[0, 0, 0], [0, 0, 0], [0, 0, 0]],  # NE
        [[0, 0, 0], [1, 0, 0], [0, 0, 0]],  # E
        [[0, 0, 0], [0, 0, 0], [0, 0, 0]],  # SE
        [[0, 0, 0], [0, 0, 0], [0, 0, 0]],  # SW
        [[0, 0, 0], [0, 0, 0], [0, 0, 0]],  # W
        [[0, 0, 0], [0, 0, 0], [0, 0, 0]]   # NW
    ],
    [                                               # SE    --> set NW in row 3
        [[0, 0, 0], [0, 0, 0], [0, 0, 0]],  # stay
        [[0, 0, 0], [0, 0, 0], [0, 0, 0]],  # NE
        [[0, 0, 0], [0, 0, 0], [0, 0, 0]],  # E
        [[0, 1, 0], [0, 0, 0], [0, 0, 0]],  # SE
        [[0, 0, 0], [0, 0, 0], [0, 0, 0]],  # SW
        [[0, 0, 0], [0, 0, 0], [0, 0, 0]],  # W
        [[0, 0, 0], [0, 0, 0], [0, 0, 0]]   # NW
    ],
    [                                               # SW    --> set NE in row 4
        [[0, 0, 0], [0, 0, 0], [0, 0, 0]],  # stay
        [[0, 0, 0], [0, 0, 0], [0, 0, 0]],  # NE
        [[0, 0, 0], [0, 0, 0], [0, 0, 0]],  # E
        [[0, 0, 0], [0, 0, 0], [0, 0, 0]],  # SE
        [[0, 0, 1], [0, 0, 0], [0, 0, 0]],  # SW
        [[0, 0, 0], [0, 0, 0], [0, 0, 0]],  # W
        [[0, 0, 0], [0, 0, 0], [0, 0, 0]]   # NW
    ],
    [                                               # W     --> set E in row 5
        [[0, 0, 0], [0, 0, 0], [0, 0, 0]],  # stay
        [[0, 0, 0], [0, 0, 0], [0, 0, 0]],  # NE
        [[0, 0, 0], [0, 0, 0], [0, 0, 0]],  # E
        [[0, 0, 0], [0, 0, 0], [0, 0, 0]],  # SE
        [[0, 0, 0], [0, 0, 0], [0, 0, 0]],  # SW
        [[0, 0, 0], [0, 0, 1], [0, 0, 0]],  # W
        [[0, 0, 0], [0, 0, 0], [0, 0, 0]]   # NW
    ],
    [                                               # NW    --> set SE in row 6
        [[0, 0, 0], [0, 0, 0], [0, 0, 0]],  # stay
        [[0, 0, 0], [0, 0, 0], [0, 0, 0]],  # NE
        [[0, 0, 0], [0, 0, 0], [0, 0, 0]],  # E
        [[0, 0, 0], [0, 0, 0], [0, 0, 0]],  # SE
        [[0, 0, 0], [0, 0, 0], [0, 0, 0]],  # SW
        [[0, 0, 0], [0, 0, 0], [0, 0, 0]],  # W
        [[0, 0, 0], [0, 0, 0], [0, 1, 0]]   # NW
    ]], dtype=jnp.float32)  # (7, 7, 3, 3) (D, C, 3, 3)
    ## D is the direction that the submarine have and C is its possible actions (which adjacent hex to move to). The subs new direction is the direction of its previous decision. 
    return transition_kernel


@chex.dataclass
class SubState:
    sub_distribution: chex.Array    # (N, D, N_y, N_x)
    sub_pos_samples: chex.Array    # (N,n,3) [x,y,d]
    sv_pos: chex.Array           # (N, 2)
    dip_mask: chex.Array
    t: float
    time_to_next_dip: float

    done: chex.Array        # if the game is over or not
    done_i: chex.Array      # if the submarine has been detected or not (sample_done)
    p: jnp.array        # remaining probability of not being detected   
    p_i: jnp.array      # probability of not have been detected (p_samples)
    player: jnp.array

    def get_bytes(self):
        n_bytes = self.sub_distribution.nbytes
        n_bytes += self.sub_pos_samples.nbytes
        n_bytes += self.sv_pos.nbytes
        n_bytes += self.t.nbytes
        n_bytes += self.time_to_next_dip.nbytes
        n_bytes += self.done.nbytes
        n_bytes += self.done_i.nbytes
        n_bytes += self.p.nbytes
        n_bytes += self.p_i.nbytes
        return n_bytes


class SubEnv(nn.Module):
    N_y: int
    N_x: int
    C: int
    D: int
    # d_max: float
    tile_size: float
    sv_speed: float   # 150km/h
    dip_time: float     # 1.0/12 (5 min)
    time_to_first_dip: float   # 1.0 h
    dt: float
    T: float
    a: float
    b: float
    c: float
    r: float
    transition_mask: jnp.array  # (D, C, N_y, N_x)
    board_mask: jnp.array       # (N_y, N_x)
    dip_matrix: jnp.array       # (N_y, N_x, D, N_y, N_x) or (N_y, N_x, N_y, N_x)
    transition_kernel: jnp.array
    # N_samples: int
    p_is_done: float

    def get_obs(self, state: SubState):

        def add_coord_channels(x: jnp.ndarray) -> jnp.ndarray: ## Add two new channels to the observation to give it more spatial awareness.
            """
            x: (N, Ny, Nx, C). Returns (N, Ny, Nx, C+2) with normalized coord channels.
            """
            N, _, Ny, Nx = x.shape
            yy = jnp.linspace(-1.0, 1.0, Ny)
            xx = jnp.linspace(-1.0, 1.0, Nx)
            Y, X = jnp.meshgrid(yy, xx, indexing="ij")
            XY = jnp.stack([X, Y], axis=0)                      # (2, Ny, Nx)
            XY = jnp.broadcast_to(XY, (N, 2, Ny, Nx))
            return jnp.concatenate([x, XY], axis=1)


        N, D, N_y, N_x = state.sub_distribution.shape                           # (N, D, N_y, N_x)
        t_obs = jnp.expand_dims(state.t, axis=(1, 2, 3))                          #  (N,) --> (N, 1, 1, 1)
        time_to_next_dip_obs = jnp.expand_dims(state.time_to_next_dip, axis=(1, 2, 3))    #  (N,) --> (N, 1, 1, 1)

        t_obs = jnp.broadcast_to(t_obs, (N, 1, N_y, N_x))
        time_to_next_dip_obs = jnp.broadcast_to(time_to_next_dip_obs, (N, 1, N_y, N_x))

        dip_pos = jnp.zeros_like(time_to_next_dip_obs)
        dip_pos = dip_pos.at[:,0,state.sv_pos[:,0], state.sv_pos[:,1]].set(1.0)

        sub_distribution_obs = jnp.sum(state.sub_distribution, axis=(1), keepdims=False)    # (N, D, N_y, N_x) --> (N, N_y, N_x)
        sub_distribution_obs = jnp.expand_dims(sub_distribution_obs, axis=(1))              # (N, N_y, N_x) --> (N, 1, N_y, N_x)

        obs = jnp.concatenate([sub_distribution_obs, t_obs, time_to_next_dip_obs, dip_pos], axis=1)     # (N, 4, N_y, N_x)
        obs = add_coord_channels(obs)       # (N, 4, N_y, N_x) --> (N, 4+2, N_y, N_x) = (N, 6, N_y, N_x)

        return obs

    def reset(self, N, N_samples, rng):
        state_distribution = jnp.zeros((N, self.D, self.N_y, self.N_x))

        sv_init_pos_x = 18
        sv_init_pos_y = 25

        init_pos_x = 15
        init_pos_y = 29

        state_distribution = state_distribution.at[:, 0, init_pos_y, init_pos_x].set(1.0)       # first location bottom middle (init_pos_y, init_pos_x)
        # state_distribution = state_distribution.at[:, 0, :, :].set(1.0)                       # uniform initial distribution (0,0)
        state_distribution = state_distribution / jnp.sum(state_distribution, axis=(-1,-2,-3), keepdims=True)

        time_to_next_dip = jnp.ones((N), dtype=jnp.float32) * self.time_to_first_dip
        t = jnp.ones((N), dtype=jnp.float32) * 0.0

        N, D, N_y, N_x = state_distribution.shape                                       # (N, D, N_y, N_x)

        def sample_indices_n(key, probs, n, order="dyx"):
            N, D, Ny, Nx = probs.shape
            K = D * Ny * Nx

            flat = probs.reshape(N, K)
            logits = jnp.log(jnp.clip(flat, 1e-20, 1.0))

            cls = jax.random.categorical(key, logits, axis=-1, shape=(n, N))  # (n, N)

            d, y, x = jnp.unravel_index(cls, (D, Ny, Nx))     # each (n, N)
            idx = jnp.stack([d, y, x], axis=-1)
            idx = jnp.stack([y, x, d], axis=-1).transpose(1, 0, 2)  # (N, n, 3)
            return idx
      

        rng, key = jax.random.split(rng)
        sub_pos_samples =  sample_indices_n(key, state_distribution, N_samples)    # (N, n, 3) [y, x, dir]

        p_i = jnp.ones((N, N_samples), dtype=jnp.float32)
        done_i = jnp.zeros((N, N_samples), dtype=jnp.bool_)
        done = jnp.zeros((N), dtype=jnp.bool_)

        p = jnp.ones((N), dtype=jnp.float32)

        sv_pos = jnp.expand_dims(jnp.array([sv_init_pos_y, sv_init_pos_x]), axis=0)
        sv_pos = jnp.broadcast_to(sv_pos, (N,2))
        player = jnp.ones((N), dtype=jnp.int32) * 0

        dip_distance_map = self.dip_distance_map(sv_pos)
        dip_mask = jnp.logical_and(self.board_mask[None,:,:], dip_distance_map<100)


        state = SubState(sub_distribution=state_distribution, sub_pos_samples=sub_pos_samples, sv_pos=sv_pos, dip_mask=dip_mask, t=t, time_to_next_dip=time_to_next_dip, done=done, done_i=done_i, p=p, p_i=p_i, player=player)
        obs = self.get_obs(state)

        return obs, state      

    def observation_size(self):
        return jnp.array([1, self.N_y, self.N_x], dtype=jnp.int32)

    
    def compute_entropy(self, state_distribution):
        p = state_distribution + 1e-10  # Avoid log(0) by adding a small epsilon
        entropy = -jnp.sum(p * jnp.log(p), axis=tuple(range(1, p.ndim)))  # Sum over all but the first dimension
        
        return entropy
    
    # @nn.compact    
    def apply_fixed_conv(self, transition_weight, transition_kernel):
        convolved = lax.conv_general_dilated(
            lhs=transition_weight,         # The input (state distribution)
            # rhs=kernel,                     # The fixed kernel
            rhs=transition_kernel,                     # The fixed kernel
            window_strides=(1, 1),          # Stride of 1
            padding="SAME",                 # Padding to match the PyTorch padding=1 behavior
            lhs_dilation=None,
            rhs_dilation=None,
            dimension_numbers=('NCHW', 'OIHW', 'NCHW')  # Dimension ordering
        )
        # return convolved.squeeze(0)
        return convolved
    
    def dip_distance_map(self, pos):
        Q = jnp.arange(self.N_x)  # Shape: (N_x) [0, 1, ..., N_x-1]
        R = jnp.arange(self.N_y)  # Shape: (N_y) [0, 1, ..., N_y-1]
        Q = jnp.repeat(jnp.expand_dims(Q, axis=(0)), repeats=self.N_y, axis=(0)) # (N_x) -> (N_y, N_x)
        R = jnp.repeat(jnp.expand_dims(R, axis=(1)), repeats=self.N_x, axis=(1)) # (N_y) -> (N_y, N_x)

        hex_hor = 1
        hex_vert = np.sqrt(3)/2

        Q_ = pos[:,1]
        R_ = pos[:,0]

        x, y = axial_to_cartesian_Pup(Q_, R_, hex_hor, hex_vert)        # Dip-position
        X, Y, = axial_to_cartesian_Pup(Q, R, hex_hor, hex_vert)         # Potential sub-position
        distance_map = jnp.sqrt((Y[None] - y[:, None, None]) ** 2 +
                                (X[None] - x[:, None, None]) ** 2)
        return distance_map

    def mask_logit_to_policy(self, sub_logit: jnp.array, dip_logit: jnp.array, sub_pos_samples, dip_mask):
        dip_logit = jnp.where(dip_mask, dip_logit, -1e10)
        dip_policy = nn.softmax(dip_logit, axis=(-1, -2), where=dip_mask)   # (N, N_y, N_x)

        N = sub_pos_samples.shape[0]
        batch_idx = jnp.arange(N)

        transition_mask_ = jnp.expand_dims(self.transition_mask, 0)                              # (1, D, C, N_y, N_x)
        transition_mask_ = jnp.repeat(transition_mask_, repeats=sub_logit.shape[0], axis=0)   # (N, D, C, N_y, N_x)
        # print(f"transition_mask_: {transition_mask_.shape}  {transition_mask_.dtype}")
        sub_policy = nn.softmax(sub_logit, axis=(2), where=transition_mask_)

        y = sub_pos_samples[:,:,0]
        x = sub_pos_samples[:,:,1]
        dir = sub_pos_samples[:,:,2]
        sampled_sub_policy = sub_policy[batch_idx[:, None], dir, :, y, x]   # (N, D,C,N_y,N_x) --> (N, n, C), where 'n' is the number of sampled 

        return sub_policy, sampled_sub_policy, dip_policy

    ## samples an action for all players. then the step function updates the state with the action of the current player (one action is not used)
    def sample_action(self, key, sv_policy, sub_policy, env_state):
        ## Samples N dip-positions from the dip-policy-distribution
        def SampleDipAction(dip_distribution, key):
            N, Y, X = dip_distribution.shape
            dip_distribution_flat = dip_distribution.reshape(N, Y * X)  # Flatten each (Y, X) distribution
            sampled_indices = jax.random.categorical(key, jnp.log(dip_distribution_flat), axis=-1)  # Sample from flattened distributions
            y_indices, x_indices = divmod(sampled_indices, X)  # Convert flat indices back to (y, x) coordinates
            return jnp.stack([y_indices, x_indices], axis=-1)  # Shape (N, 2)
        
        key, _key = jax.random.split(key)
        action_dip = SampleDipAction(sv_policy, _key)

        ## Samples (N,n) sub-directions, one for each submarine sample (n). returns a (N,n) array of the direction taken
        def SampleActionSub(sub_pos_samples, transition_policy_, key):
            y = sub_pos_samples[:,0]
            x = sub_pos_samples[:,1]
            dir = sub_pos_samples[:,2]
            probs = transition_policy_[dir, :, y, x]   # (D,C,N_y,N_x) --> (n, C), where 'n' is the number of sampled positions in the state
            probs = probs + 1e-15
            action_dir = jax.random.categorical(key, jnp.log(probs), axis=-1) # (n)
            action = jnp.stack([x, y, dir, action_dir], axis=1)


            transition_indices = jnp.array( [[0,0], [1,-1], [1,0], [0,1], [-1,1], [-1,0], [0,-1]], dtype=jnp.int32) # [q,r] Aka: [x,y] 
            d_pos = transition_indices[action_dir, :]          # (n, 2)
            d_pos_x = d_pos[:,0]
            d_pos_y = d_pos[:,1]

            new_pos_x = d_pos_x + x
            new_pos_y = d_pos_y + y

            action = jnp.stack([new_pos_y, new_pos_x, action_dir], axis=1)

            return (action, action_dir)

        key, _key = jax.random.split(key)
        rng_step = jax.random.split(_key, env_state.sub_pos_samples.shape[0]) # (N)
        (sub_samples_action, action_dir) = jax.vmap(SampleActionSub, in_axes=(0,0,0))(env_state.sub_pos_samples, sub_policy, rng_step)

        return (action_dip, sub_samples_action)     ## (N, 2), (N, n, 3)

    def step(self, state: SubState, sub_transition_policy: jnp.ndarray, action_sub_sampled: jnp.ndarray, action_dip_sampled: jnp.ndarray, rng: chex.PRNGKey):
        p_i_prior = state.p_i
        p_prior = state.p

        done_prior = state.done
        done_i_prior = state.done_i
        player = state.player

        detection_map = self.dip_matrix[action_dip_sampled[:,0], action_dip_sampled[:,1]]   # ()

        def ApplyDip(state: SubState, sub_transition_policy: jnp.array, action_dip_sampled: jnp.array, action_sub_sampled: jnp.array, detection_map: jnp.array):

            state.sub_distribution = state.sub_distribution * detection_map # (N, D, N_y, N_x) * ((N, 1, N_y, N_x)-->(N, 1, N_y, N_x))

            # SV_distance2dip = axial_distance(state.sv_pos[1], state.sv_pos[0], action_dip_sampled[1], action_dip_sampled[0])

            # sub_pos_samples: chex.arrray    # (N,n,3) [y,x,d]
            y_i = state.sub_pos_samples[:,:,0]
            x_i = state.sub_pos_samples[:,:,1]
            # dir_i = state.sub_pos_samples[:,:,2]

            batch_idx = jnp.arange(y_i.shape[0])
            # p_not_detected_samples = detection_map[y_i, x_i, dir_i]
            # p_not_detected_samples = detection_map[batch_idx[:,None], dir_i, y_i, x_i]
            p_not_detected_samples = detection_map[batch_idx[:,None], jnp.zeros_like(y_i), y_i, x_i]

            p_not_detected_samples = jnp.where( state.done_i, jnp.ones_like(p_not_detected_samples), p_not_detected_samples)
            # state.p_i = state.p_i * p_not_detected_samples
            # state.p *= jnp.sum(state.sub_distribution, axis=(-1,-2,-3), keepdims=False)

            def dip_distance2time(distance):
                return self.dip_time + (distance*self.tile_size/self.sv_speed)*(distance*self.tile_size/self.sv_speed) / (2.0 + (distance*self.tile_size/self.sv_speed))

            # state.time_to_next_dip = state.time_to_next_dip + dip_distance2time(SV_distance2dip)   ## If the SV moves waits'time_to_next_dip'h to dip after making the decision/action
            # state.dip_destination = action_dip_sampled
            state.time_to_next_dip = state.time_to_next_dip + self.dip_time
            state.sv_pos = action_dip_sampled
            dip_distance_map = self.dip_distance_map(action_dip_sampled)
            state.dip_mask = jnp.logical_and(self.board_mask[None,:,:], dip_distance_map<self.r)

            return state, p_not_detected_samples
        
        def MoveSub(state: SubState, sub_transition_policy: jnp.array, action_dip_sampled: jnp.array, action_sub_sampled: jnp.array, detection_map: jnp.array):
            ## Update state_distributions from action_sub --> transition_policy
            state_distribution_expanded = jnp.expand_dims(state.sub_distribution, axis=2)       # (N, D, N_y, N_x) --> (N, D, 1, N_y, N_x)
            transition_weight = sub_transition_policy * state_distribution_expanded # (N, D, C, N_y, N_x)*(N, D, 1, N_y, N_x) --> (N, D, C, N_y, N_x)
            transition_weight = jnp.sum(transition_weight, axis=(1))            # since the new direction is independent of the previous direction, only which tile it came from
            new_sub_distribution = self.apply_fixed_conv(transition_weight=transition_weight, transition_kernel=self.transition_kernel)  # Shape: (N, D, N_y, N_x) # Perform the convolution to get the new state distribution
            state.sub_distribution = new_sub_distribution
            state.sub_pos_samples = jnp.where( state.done_i[:,:,None], state.sub_pos_samples, action_sub_sampled)

            state.t += self.dt
            state.time_to_next_dip -= self.dt
            return state, jnp.ones_like(state.done_i, dtype=jnp.float32)


        state.player = jnp.array(state.time_to_next_dip<=0.0, dtype=jnp.int32)
        new_state, p_not_detected_samples = lax.cond(
                    (state.time_to_next_dip<=0.0)[0], 
                    # state.player==1,
                    ApplyDip, MoveSub, 
                    state, sub_transition_policy, action_dip_sampled, action_sub_sampled, detection_map
        )

        new_state.player = jnp.array(new_state.time_to_next_dip<=0.0, dtype=jnp.int32)


        # mask of target hexagons/positions
        terminal_reward_mask = jnp.zeros_like(new_state.sub_distribution, dtype=jnp.float32)   # (N, D, N_y, N_x)
        terminal_reward_mask1 = terminal_reward_mask.at[:,:,1,16].set(1.0)  ## Left objective
        terminal_reward_mask2 = terminal_reward_mask.at[:,:,1,45].set(1.0)  ## Right objective
        terminal_reward_mask = terminal_reward_mask1 + terminal_reward_mask2
        terminal_reward_mask_ = terminal_reward_mask[0]     # (N, D, N_y, N_x) --> # (D, N_y, N_x)

        ps = jnp.sum(new_state.sub_distribution* terminal_reward_mask, axis=(-1,-2,-3), keepdims=False)
        # ps_i =  terminal_reward_mask_[new_state.sub_pos_samples[:,:,2], new_state.sub_pos_samples[:,:,0], new_state.sub_pos_samples[:,:,1]] # (dir, y, x)
        ps_i =  terminal_reward_mask_[jnp.zeros_like(new_state.sub_pos_samples[:,:,2]), new_state.sub_pos_samples[:,:,0], new_state.sub_pos_samples[:,:,1]] # (dir, y, x) --> (N, n). The objective is independent of the sub direction/ (which direction it came from)

        pd = jnp.where(new_state.done, jnp.zeros_like(new_state.done), 1.0-jnp.sum(new_state.sub_distribution, axis=(-1, -2, -3), keepdims=False) )
        # pd_i = 1.0 - p_not_detected_samples
        pd_i = (1.0 - p_not_detected_samples) * (1.0-jnp.array((ps_i>0), dtype=jnp.float32))  # if the sub got detected at the same time that it succeded with the mission then remove the detection reward. This is not necessary since actions are sequential and not simultanious 


        ## Extra metrics for logging. This logs how much probability of subs is reaching each of the two target objectives in this scenario. 
        ps_vec = jnp.stack([jnp.sum(new_state.sub_distribution* terminal_reward_mask1, axis=(-1,-2,-3), keepdims=False), jnp.sum(new_state.sub_distribution* terminal_reward_mask2, axis=(-1,-2,-3), keepdims=False)], axis=1)

        new_state.sub_distribution = new_state.sub_distribution * jnp.array((terminal_reward_mask==0), dtype=jnp.float32)   # Remove the probability of the heatmat that is on the target-hex/hexes
        new_state.p = p_prior * jnp.sum(new_state.sub_distribution+1e-16, axis=(-1, -2, -3), keepdims=False)
        new_state.p_i = p_i_prior * (1.0-pd_i) * (1.0 - jnp.array((ps_i>0), dtype=jnp.float32) )


        r = ps*self.a - pd*self.b
        r_i = ps_i*self.a - pd_i*self.b

        ## Exploration-correlated loss. Increase incentive for the sub to explore more spatial-positions (Spread out loss)
        y_i = new_state.sub_pos_samples[:,:,0]
        x_i = new_state.sub_pos_samples[:,:,1]

        batch_idx = jnp.arange(new_state.sub_pos_samples.shape[0])        # 1:N
        flat_sub_distribution = jnp.sum(new_state.sub_distribution, axis=1)   # (N,D,N_y, N_x) --> (N,N_y, N_x)
        r_p_i = flat_sub_distribution[batch_idx[:,None],y_i, x_i]    # (N,N_y, N_x) --> (N,n)

        r_p_i_mean = jnp.sum(flat_sub_distribution*flat_sub_distribution, axis=(-1,-2))     # Compute the mean probability of a sampled submarine being in its current spatial-position. Analytically computed from the belief distribution so it does not depend on the size of 'n'. This is used to make the transformed reward unbiased.
        r_p_i_std = jnp.sqrt( jnp.sum(flat_sub_distribution*flat_sub_distribution*flat_sub_distribution, axis=(-1,-2)) - r_p_i_mean*r_p_i_mean )    # Compute the variance of the probability of a sampled submarine being in its current spatial-position. Analytically computed from the belief distribution so it does not depend on the size of 'n'. This is used to normalize the update to stabilize update (the variance is highly dependent on how spread out the belief-distribution and can be very low if the initial position is one or a few positions).

        r_p_i_mean = jnp.expand_dims(r_p_i_mean, axis=(1))
        r_p_i_std = jnp.expand_dims(r_p_i_std, axis=(1))

        r_p_i = (r_p_i-r_p_i_mean)/(r_p_i_std + 1e-10 )                       # Standardize exploration to make the reward unbiased
        r_p_i = jnp.where(jnp.logical_and(jnp.logical_not(new_state.done_i), player[:,None]==0), r_p_i, jnp.zeros_like(r_p_i))
        r_i += -r_p_i * self.c                                                                                    # Standardize exploration (based on belief distribution).

        # done = new_state.turn>self.max_turn
        done = new_state.t >= self.T
        done = jnp.logical_or(done, done_prior)

        # done_i = new_state.sampled_sub_detected
        done_i = jnp.logical_or(new_state.p_i<=self.p_is_done, ps_i>0.0)
        done_i = jnp.logical_or(done_i, jnp.expand_dims(done, axis=1))
        done_i = jnp.logical_or(done_i, done_i_prior)

        new_state.done = done
        new_state.done_i = done_i

        done2 = jnp.logical_and(done_prior, new_state.done)
        done_i2 = jnp.logical_and(done_i_prior, new_state.done_i)

        new_state.sub_distribution = new_state.sub_distribution/jnp.sum(new_state.sub_distribution+1e-16, axis=(-1, -2, -3), keepdims=True)

        # return new_state, r, r_i, done, done_i, done2, done_i2, p_prior, p_i_prior, new_state.p, new_state.p_i, pd, ps, ps_i, pd_i
        return new_state, r, r_i, done, done_i, done2, done_i2, p_prior, p_i_prior, new_state.p, new_state.p_i, pd, ps, ps_i, pd_i, ps_vec
    



def create_env(config, board, verbose=True):
    if 'SL_loss' not in config:
        config['SL_loss'] = 0
    if 'RL_loss' not in config:
        config['RL_loss'] = 0

    board_mask = jnp.logical_and(board<0.5, board>0).astype(jnp.int32)
    dip_matrix = CreateDipMatrix_SonEkv_Topography(config['N_y'], config['N_x'], config['p_min'], config['tile_size'], config['SL'], config['TS'], config['DT'], config['NL'], config['DI'], config['sig'], board, config['SL_loss'], config['RL_loss'])
   
    transition_mask, transition_mask_B = get_transition_mask_MaskedHexesHexCord(config, board_mask)
    transition_kernel = create_transition_kernel_HexCord()

    env = SubEnv(N_y=int(config["N_y"]), N_x=int(config["N_x"]), C=int(config["C"]), D=int(config['D']), 
                tile_size=config['tile_size'], sv_speed=config["sv_speed"], dip_time=config["dip_time"], time_to_first_dip=config["time_to_first_dip"], dt=config["dt"], T=config["T"], 
                a=config["a"], b=config["b"], c=config["c"], r=config["r"], 
                transition_mask=transition_mask_B, board_mask=board_mask,
                dip_matrix=dip_matrix, transition_kernel=transition_kernel, p_is_done=0.001)

    return env