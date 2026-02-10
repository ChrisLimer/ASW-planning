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



from ActorCriticNetworks import ActorCritic
from asw_env import Transition, Target, SubState, SubEnv



def make_train(config_, env, network, optimizer):


    def train(rng, model_params_, params_target_, optimizer_state_, config, model_reg_params_t1, model_reg_params_t2):
        
        alpha_kl = config['alpha_kl']
        max_reg_step = config['max_reg_step']
        gamma_averaging = config['gamma_averaging']
        
        
        def _update_step(runner_state_update_step, unused):

            (model_params, params_target, optimizer_state, rng, update_step) = runner_state_update_step

            alpha = jnp.array((update_step), float)/max_reg_step

            ## function to step environment and store trajectory-transitions
            def _env_step(runner_state, unused):
                params, env_state, last_obs, rng = runner_state     # carries over each step. 'params' is constant each jax.lax.scan iteration and can be placed outside the function
                
                player = jnp.array(env_state.time_to_next_dip<=0.0,  dtype=jnp.int32)   # Retrieve the active player-array of who is making the action in this step.

                sub_pos_samples = jnp.array(env_state.sub_pos_samples)  # (N, n, 3): array with double-batch (N,n) and position (y, x, dir) of the sampled subs.
                prior_sub_distribution = env_state.sub_distribution     # (N, D, N_y, N_x): Public belief distribution of the submarines position for each batch (N).

                rng, _rng = jax.random.split(rng)
                (sub_logit, sv_logit, v, v_map) = network.apply(params, last_obs)   # Forward pass of the neural-network

                sub_policy, sampled_sub_policy, sv_policy = env.mask_logit_to_policy(sub_logit, sv_logit, env_state.sub_pos_samples, env_state.dip_mask)    # convert logits to policies. (N, D, C, N_y, N_x), (N, n, C), (N, N_y, N_x)

                rng, _rng = jax.random.split(rng)
                (action_dip, action_sub_samples) = env.sample_action( _rng, sv_policy, sub_policy, env_state)   # Sample actions from policy

                rng, _rng = jax.random.split(rng)
                (new_state, r, r_i, done, done_i, done2, done_i2, p_prior, p_i_prior, p, p_i, pd, ps, ps_i, pd_i, ps_vec) = env.step(env_state, sub_policy, action_sub_samples, action_dip, _rng)   # step the environment with the sampled actions and update the sub-belief-distribution with the 'sub_policy'

                obs = env.get_obs(new_state)                    # get the observation of the new state
                runner_state = (params, new_state, obs, rng)    # create the carry for the next iteration

                transition = Transition(done, done_i, done2, done_i2, r, r_i, v, v_map, prior_sub_distribution, p_prior, p_i_prior, p, p_i, sub_pos_samples, env_state.dip_mask, last_obs, sub_logit, sv_logit, sub_policy, sv_policy, new_state.sv_pos, action_sub_samples, action_dip, pd, ps, ps_i, pd_i, player, ps_vec)

                return runner_state, transition
            ## _env_step
            

            rng, _rng = jax.random.split(rng)
            obsv, env_state_0 = env.reset(config_["NUM_ENVS"], config_["NUM_SUB_ENVS"], _rng)   # get a batch of initial states and their observations

            runner_state = (model_params, env_state_0, obsv, rng)   # construct the initial carry for the '_env_step'
            runner_state, transition = jax.lax.scan(
                _env_step, runner_state, None, config_["NUM_ENV_STEPS"]
            )   # Run the entire trajectory by scanning over '_env_step' 'NUM_ENV_STEPS'-times which is the maximum number of steps of the environment.

            ## Function to backpropagate the trajectory-advantage one step (used with jax.lax.scan for the entire trajectory).
            def compute_advantage(carry, transition):
                (done, done_i, done2, done_i2, r, r_i, p_prior, p_i_prior, ps, ps_i, pd, pd_i, ps_vec) = (transition.done,
                                                    transition.done_i,
                                                    transition.done2,
                                                    transition.done_i2,
                                                    transition.r,
                                                    transition.r_i,
                                                    transition.p_prior,
                                                    transition.p_i_prior,
                                                    transition.ps,
                                                    transition.ps_i,
                                                    transition.pd,
                                                    transition.pd_i,
                                                    transition.ps_vec)
                
                (R_t_next, R_ti_next, Pd_next, Pd_i_next, Ps_next, Ps_i_next, Ps_vec_next) = carry

                ## only used to log cumulative detection probability
                Pd_tT_ = Pd_next + pd*p_prior*(1-done2)
                Pd_tTi_ = Pd_i_next + pd_i*p_i_prior*(1-done_i2)

                ## only used to log cumulative success probability
                Ps_tT_ = Ps_next + ps*p_prior*(1-done2)
                Ps_tTi_ = Ps_i_next + ps_i*p_i_prior*(1-done_i2)

                ## only used to log cumulative success per objective (two objectives)
                Ps_vec_tT_ = Ps_vec_next + ps_vec*jnp.expand_dims(p_prior*(1-done2), axis=-1)

                # Note, 'r' and 'r_i' contain the probability to reach that reward from the current_state-->next_state r=r_*p(a)
                R_tT_ = R_t_next + r*p_prior*(1-done2)                  # expected future reward * P(state, h)
                R_tTi_ = R_ti_next + r_i*p_i_prior*(1-done_i2)          # expected future reward * P(state_i, h)


                R_tT = R_tT_/(p_prior+1e-10)      # expected future reward from the current state
                R_tTi = R_tTi_/(p_i_prior+1e-10)  # expected future reward from the current state

                R_tTi = jnp.where(done_i2, 0.0, R_tTi)      # mask out for states that are done. Aka: The max time is reached or the cumulative detection probability is less than 
                R_tT = jnp.where(done2, 0.0, R_tT)          # mask out for states that are done. Aka: --||-- or the sampled-sub have been detected by the 

                return (R_tT_, R_tTi_, Pd_tT_, Pd_tTi_, Ps_tT_, Ps_tTi_, Ps_vec_tT_), (R_tT, R_tTi, r, r_i)            

            

            init_carry = (jnp.zeros_like(transition.r[0]), jnp.zeros_like(transition.r_i[0]), 
                          jnp.zeros_like(transition.pd[0]), jnp.zeros_like(transition.pd_i[0]),
                          jnp.zeros_like(transition.ps[0]), jnp.zeros_like(transition.ps_i[0]),
                          jnp.zeros_like(transition.ps_vec[0]))
            final_carry, R = jax.lax.scan(compute_advantage, init_carry, transition, reverse=True, unroll=16)
            (R_tT, R_tTi, r, r_i) = R
            (R_tT_, R_tTi_, Pd_tT_, Pd_tTi_, Ps_tT_, Ps_tTi_, Ps_vec_tT_) = final_carry  ## using 'Pd_tT_' and 'Pd_tTi_' for logging cumulative detection probability

            targets = Target(R_tT, R_tTi, r, r_i)   # Compute the target with the backpropagated advantage. Used to update/fit the neural network weights ('params')

            
            #### UPDATE NETWORK
            def _update_epoch(update_state, unused):
                (model_params, params_target, optimizer_state, transition, targets, rng) = update_state


                def _update_minibatch(model_opt_sate, batch_info):
                    (transition, targets) = batch_info
                    (model_params, params_target, optimizer_state) = model_opt_sate

                    def _loss_fn_r_nad(params, transition, targets):
                        sub_pos_samples = transition.sub_pos_samples
                        dip_mask = transition.dip_mask

                        player0_i_mask = jnp.logical_and(jnp.logical_not(transition.done_i2), (transition.player==0)[:,None])
                        player0_mask = jnp.logical_and(jnp.logical_not(transition.done2), (transition.player==0))
                        player1_mask = jnp.logical_and(jnp.logical_not(transition.done2), (transition.player==1))

                        # (logits, v, v_map) = network.apply(params, transition.obs)
                        (sub_logit, sv_logit, v, v_map) = network.apply(params, transition.obs)
                        v = v[:,0]  # (N, 1) --> (N, )

                        ## Get masked policy from logits
                        pi_sub_policy, pi_sampled_sub_policy, pi_sv_policy = env.mask_logit_to_policy(sub_logit, sv_logit, transition.sub_pos_samples, dip_mask)
                        

                        ##
                        #### Regularization for regularized objective loss
                        (sub_logit_reg1, sv_logit_reg1, v_reg1, v_map_reg1) = network.apply(model_reg_params_t1, transition.obs)
                        (sub_logit_reg2, sv_logit_reg2, v_reg2, v_map_reg2) = network.apply(model_reg_params_t2, transition.obs)

                        pi_sub_policy_reg1, pi_sampled_sub_policy_reg1, pi_sv_policy_reg1 = env.mask_logit_to_policy(sub_logit_reg1, sv_logit_reg1, transition.sub_pos_samples, dip_mask)
                        pi_sub_policy_reg2, pi_sampled_sub_policy_reg2, pi_sv_policy_reg2 = env.mask_logit_to_policy(sub_logit_reg2, sv_logit_reg2, transition.sub_pos_samples, dip_mask)


                        ## Create policy masks
                        policy_mask_sub = jnp.logical_and(jnp.logical_and(jnp.logical_not(transition.done2), transition.player==0)[:,None,None,None,None], pi_sub_policy>0 )             # not-done and player==0 and action-prob>0 (valid action)
                        policy_mask_sub_i = jnp.logical_and(jnp.logical_and(jnp.logical_not(transition.done_i2), (transition.player==0)[:,None])[:,:,None], pi_sampled_sub_policy>0 )  # not-done and player==0 and action-prob>0 (valid action)
                        policy_mask_sv = jnp.logical_and(jnp.logical_and(jnp.logical_not(transition.done2), transition.player==1)[:,None,None], pi_sv_policy>0 )                # not-done and player==0 and action-prob>0 (valid action)

                        ## mask out actions of the opponent player (not current) and games that are done (terminal)
                        pi_sub_policy = jnp.where(policy_mask_sub, pi_sub_policy, 0.0)
                        pi_sampled_sub_policy = jnp.where(policy_mask_sub_i, pi_sampled_sub_policy, 0.0)
                        pi_sv_policy = jnp.where(policy_mask_sv, pi_sv_policy, 0.0)


                        ## Compute regularized policies
                        pi_sub_policy_reg = jnp.where(policy_mask_sub, pi_sub_policy_reg1*(1.0-alpha) + alpha*pi_sub_policy_reg2, 1.0)
                        pi_sampled_sub_policy_reg = jnp.where(policy_mask_sub_i, pi_sampled_sub_policy_reg1*(1.0-alpha) + alpha*pi_sampled_sub_policy_reg2, 1.0)
                        pi_sv_policy_reg = jnp.where(policy_mask_sv, pi_sv_policy_reg1*(1.0-alpha) + alpha*pi_sv_policy_reg2, 1.0)

                        # ## Make sure that the regularized policies have epsilon probability on each. Not that necessary if the Ent-Loss is non-zero
                        # epsilon_pl1 = 0.1  # 10% of sampling the action from a uniform policy. adding it to the regularized policy
                        # epsilon_pl2 = 0.1  # 10% of sampling the action from a uniform policy. adding it to the regularized policy
                        # # policy_mask_sv_F = policy_mask_sv.astype(pi_sv_policy_reg.dtype)
                        # flat_sub_distribution = jnp.sum(transition.sub_distribution, axis=(1), keepdims=False)
                        # policy_mask_sv_F = (flat_sub_distribution>0).astype(pi_sv_policy_reg.dtype)
                        # counts_sv = policy_mask_sv_F.sum(axis=(-1, -2), keepdims=True)
                        # uniform_policy_sv = jnp.where((counts_sv>0), policy_mask_sv_F/counts_sv, 0.0)
                        # pi_sv_policy_reg = pi_sv_policy_reg * (1.0-epsilon_pl2) + uniform_policy_sv*epsilon_pl2                   # (N,N_y,N_x)

                        # policy_mask_sub_F = policy_mask_sub.astype(pi_sub_policy_reg.dtype)
                        # counts_sub = policy_mask_sub_F.sum(axis=(-1), keepdims=True)
                        # uniform_policy_sub = jnp.where((counts_sub>0), policy_mask_sub_F/counts_sub, 0.0)
                        # pi_sub_policy_reg = pi_sub_policy_reg * (1.0-epsilon_pl1) + uniform_policy_sub*epsilon_pl1                       # (N,N_y,N_x,D)

                        # policy_mask_sub_i_F = policy_mask_sub_i.astype(pi_sampled_sub_policy_reg.dtype)
                        # counts_sub_i = policy_mask_sub_i_F.sum(axis=(-1), keepdims=True)
                        # uniform_policy_sub_i = jnp.where((counts_sub_i>0), policy_mask_sub_i_F/counts_sub_i, 0.0)
                        # # print(f"uniform_policy_sub_i: {uniform_policy_sub_i.shape}")
                        # # print(f"pi_sampled_sub_policy_reg: {pi_sampled_sub_policy_reg.shape}")
                        # pi_sampled_sub_policy_reg = pi_sampled_sub_policy_reg * (1.0-epsilon_pl1) + uniform_policy_sub_i*epsilon_pl1    # (N,n,D)




                        ## Compute KL distance between the current policy and the regularized policy
                        KL_distance_sub_i = jnp.sum(pi_sampled_sub_policy * (jnp.log(pi_sampled_sub_policy+1e-15) - jnp.log(pi_sampled_sub_policy_reg+1e-15)),  axis=(-1), where=policy_mask_sub_i)    # (N, n, D)         Sum over D -> (-1)
                        KL_distance_sv = jnp.sum(pi_sv_policy *         (jnp.log(pi_sv_policy+1e-15)        - jnp.log(pi_sv_policy_reg+1e-15)),         axis=(-1, -2), where=policy_mask_sv)                       # (N, N_y, N_x) -> (N)     Sum over (N_y, N_x) -> (-1, -2)
                        # KL_distance_sub = jnp.sum(pi_sub_policy *           (jnp.log(pi_sub_policy+1e-15)         - jnp.log(pi_sub_policy_reg+1e-15)),          axis=(-3), where=policy_mask_sub)                                # (N, D, N_y, N_x)  Sum over (D) -> (-3)
                        # KL_distance_sub = jnp.sum(KL_distance_sub * transition.sub_distribution, axis=(-1, -2, -3), keepdims=False)

                        ## Compute the KL-loss
                        # loss_kl_sub = jnp.mean(KL_distance_sub, where=player0_mask, keepdims=False)   # If using on analytic sub-disitrbution
                        loss_kl_sub_i = jnp.mean(KL_distance_sub_i*transition.p_i_prior, where=player0_i_mask, keepdims=False)  # If using on sampled sub-disitrbution. Use this since the policy update is based on the same sampes.
                        loss_kl_sv = jnp.mean(KL_distance_sv*transition.p_prior, where=player1_mask, keepdims=False)



                        ## State distribution entropy loss
                        # def entropy_NDXY(state_distribution):   # (N, D, N_x, N_y)
                        #     state_distribution = jnp.sum(state_distribution, axis=(1))  # (N, D, N_y, N_x) --> (N, N_y, N_x)
                        #     p = state_distribution + 1e-10  # Avoid log(0) by adding a small epsilon
                        #     entropy = -jnp.sum(p * jnp.log(p), axis=tuple(range(1, p.ndim)))  # Compute entropy over space (N_x,N_y) and directions (D)
                        #     return entropy
                        # transition_weight = pi_sub_policy * transition.sub_distribution[:,:,None] # (N, D, C, N_y, N_x)*(N, D, 1, N_y, N_x) --> (N, D, C, N_y, N_x)
                        # transition_weight = jnp.sum(transition_weight, axis=(1))
                        # new_mean_state_distribution = env.apply_fixed_conv(transition_weight=transition_weight, transition_kernel=env.transition_kernel)
                        # state_entropy = jnp.mean(entropy_NDXY(new_mean_state_distribution), where=jnp.logical_not(transition.done2))
                        # state_entropy_loss = -state_entropy

                        ## Sub policy entropy loss  
                        sub_entropy_loss = jnp.sum(jnp.log(pi_sub_policy+1e-14)*pi_sub_policy, axis=(2), where=policy_mask_sub)     # (N, D, C, N_y, N_x) --> (N, D, N_y, N_x)
                        sub_entropy_loss = jnp.sum(sub_entropy_loss* transition.sub_distribution, axis=(-1,-2,-3))                  # (N, D, N_y, N_x) --> (N)
                        sub_entropy_loss = jnp.mean(sub_entropy_loss*transition.p_prior, where=player0_mask)       # (N) --> (,). Also multiply by the prior-probability of the state

                        sub_i_entropy = -jnp.sum(pi_sampled_sub_policy * jnp.log(pi_sampled_sub_policy+1e-15),  axis=(-1), where=policy_mask_sub_i)    # (N, n, D) -> (N, n)        Sum over D -> (-1)
                        sub_i_entropy_loss_mean = -jnp.mean(sub_i_entropy*transition.p_i_prior, where=player0_i_mask )        # (N, n) -> (,)

                        #### SV KL policy loss
                        # ## KL-distance to uniform valid UV locations
                        # flat_sub_distribution = jnp.sum(transition.sub_distribution, axis=(1), keepdims=False)
                        # sv_ref_policy = (flat_sub_distribution>0).astype(jnp.float32)
                        # sv_ref_policy = sv_ref_policy / (jnp.sum(sv_ref_policy, axis=(-1,-2), keepdims=True) +1e-14)
                        # sv_entropy_loss = jnp.sum( (jnp.log(pi_sv_policy+1e-14) - jnp.log(sv_ref_policy+1e-14))*pi_sv_policy, axis=(-1, -2), where=policy_mask_sv )   # KL-distance to uniform valid 
                        # sv_entropy_loss = jnp.mean(sv_entropy_loss*transition.p_prior, where=player1_mask)

                        ## normal entropy loss
                        sv_entropy_loss = jnp.sum( (jnp.log(pi_sv_policy+1e-14)*pi_sv_policy), where=policy_mask_sv, axis=(-1,-2))      # normal entropy loss: (N, N_y, N_x) --> (N)
                        sv_entropy = -sv_entropy_loss
                        sv_entropy_loss_mean = jnp.mean(sv_entropy_loss*transition.p_prior, where=player1_mask)                         # (N) --> (,)


                        #### Reward Advantage and policy gradients
                        (R_tT, R_tTi, r, r_i) = targets
                        batch_idx = jnp.arange(sub_pos_samples.shape[0])        # 1:N
                        sub_batch_idx = jnp.arange(sub_pos_samples.shape[1])    # 1:n

                        ## Retrieve and compute sampled policy mu (should be equal to the current policy if there is no replay buffer or minibatches or multiple epoch-updates)
                        mu_sub_logit = transition.sub_logit
                        mu_dip_logit = transition.dip_logit
                        # mu_sub_policy, mu_sampled_sub_policy, mu_sv_policy = env.mask_logit_to_policy(mu_sub_logit, mu_dip_logit, transition.sub_pos_samples, dip_mask)
                        mu_sub_policy, mu_sampled_sub_policy, mu_sv_policy = env.mask_logit_to_policy(mu_sub_logit, mu_dip_logit, transition.sub_pos_samples, dip_mask)
                        mu_sub_policy = jnp.where(policy_mask_sub, mu_sub_policy, 1.0)
                        mu_sampled_sub_policy = jnp.where(policy_mask_sub_i, mu_sampled_sub_policy, 1.0)
                        mu_sv_policy = jnp.where(policy_mask_sv, mu_sv_policy, 1.0)

                        ## Get pi(a_i) and mu(a_i) that is neccessary to compute the policy loss/gradient 
                        pi_sampled_sub_p = pi_sampled_sub_policy[batch_idx[:, None], sub_batch_idx[None, :], transition.action_sub_samples[:,:,2]]
                        mu_sampled_sub_p = mu_sampled_sub_policy[batch_idx[:, None], sub_batch_idx[None, :], transition.action_sub_samples[:,:,2]]
                        pi_sampled_sub_p_reg = pi_sampled_sub_policy_reg[batch_idx[:, None], sub_batch_idx[None, :], transition.action_sub_samples[:,:,2]]
                        pi_sampled_sub_p = jnp.where(player0_i_mask, pi_sampled_sub_p, 0.0)
                        mu_sampled_sub_p = jnp.where(player0_i_mask, mu_sampled_sub_p, 1.0)
                        pi_sampled_sub_p_reg = jnp.where(player0_i_mask, pi_sampled_sub_p_reg, 1.0)

                        ## Get the V[sub_pos_samples] that is neccessary to compute the advantage that is used in the policy update
                        v_i = v_map[batch_idx[:,None], sub_pos_samples[:,:,0], sub_pos_samples[:,:,1]]
                        # v_i_traj_batch = transition.v_map[batch_idx[:,None], sub_pos_samples[:,:,0], sub_pos_samples[:,:,1]]



                        ## Simple policy gradient loss
                        ## loss = log(pi(a^)) * Advantage,  Advantage = R-v
                        pi_sv_p = pi_sv_policy[batch_idx[:], transition.action_dip[:,0], transition.action_dip[:,1]]
                        mu_sv_p = mu_sv_policy[batch_idx[:], transition.action_dip[:,0], transition.action_dip[:,1]]
                        pi_sv_p_reg = pi_sv_policy_reg[batch_idx[:], transition.action_dip[:,0], transition.action_dip[:,1]]
                        pi_sv_p = jnp.where(player1_mask, pi_sv_p, 0.0)
                        mu_sv_p = jnp.where(player1_mask, mu_sv_p, 1.0)
                        pi_sv_p_reg = jnp.where(player1_mask, pi_sv_p_reg, 1.0)

                        ## Compute Advantage
                        Adv_sub_samples = jnp.where(player0_i_mask, R_tTi - v_i, 0.0)
                        Adv_sv = jnp.where(player1_mask ,-(R_tT - v), 0.0 )

                        Adv_sub_samples = jax.lax.stop_gradient(Adv_sub_samples)
                        Adv_sv = jax.lax.stop_gradient(Adv_sv)


                        #### PPO-similar policy loss
                        #### Submarine policy loss
                        sub_ratio_samples = jnp.where(player0_i_mask, pi_sampled_sub_p / (mu_sampled_sub_p+1e-10), 0.0)         # policy ration from mu
                        # sub_ratio_samples = jnp.where(player0_i_mask, pi_sampled_sub_p / pi_sampled_sub_p_reg, 0.0)   # policy ration from regularization policy. This slows down training significantly
                        ratio_sub_clip_rate = jnp.logical_or(sub_ratio_samples>(1+config["CLIP_EPS"]), sub_ratio_samples<(1-config["CLIP_EPS"]))
                        ratio_sub_clip_rate = jnp.sum(ratio_sub_clip_rate) / jnp.maximum(jnp.sum(player0_i_mask), 1)
                        
                        Adv_sub_samples_clip_1 = Adv_sub_samples * sub_ratio_samples
                        Adv_sub_samples_clip_2 = ( jnp.clip(
                                sub_ratio_samples,
                                1.0 - config["CLIP_EPS"],
                                1.0 + config["CLIP_EPS"],
                            ) * Adv_sub_samples)

                        loss_actor_sub = -jnp.minimum(Adv_sub_samples_clip_1, Adv_sub_samples_clip_2)   # (N, n)
                        loss_actor_sub = jnp.where(player0_i_mask, loss_actor_sub, 0.0)


                        #### Surface-vehicle policy loss
                        sv_ratio = jnp.where(player1_mask, pi_sv_p / (mu_sv_p+1e-10), 0.0)  # (N)

                        Adv_sv_clip_1 = Adv_sv * sv_ratio
                        Adv_sv_clip_2 = ( jnp.clip(
                                sv_ratio,
                                1.0 - config["CLIP_EPS"],
                                1.0 + config["CLIP_EPS"],
                            ) * Adv_sv)

                        loss_actor_sv = -jnp.minimum(Adv_sv_clip_1, Adv_sv_clip_2)      # (N)
                        loss_actor_sv = jnp.where(player1_mask, loss_actor_sv, 0.0)     # (N)


                        ## Value loss
                        loss_v = optax.losses.huber_loss( jnp.where( jnp.logical_not(transition.done2), v-R_tT, 0.0) )          # Loss for v    (N, )
                        loss_vi = optax.losses.huber_loss( jnp.where( jnp.logical_not(transition.done_i2),v_i-R_tTi, 0.0) )     # Loss for v_map    (N, n)

                        loss_v_ = jnp.mean(loss_v*transition.p_prior, where=jnp.logical_not(transition.done2))
                        loss_vi_ = jnp.mean(loss_vi*transition.p_i_prior, where=jnp.logical_not(transition.done_i2))
                        loss_actor_sub_ = jnp.mean(loss_actor_sub*transition.p_i_prior, where=player0_i_mask)
                        loss_actor_sv_ = jnp.mean(loss_actor_sv*transition.p_prior, where=player1_mask)


                        loss = loss_v_ + loss_vi_ + loss_actor_sub_ + loss_actor_sv_

                        # loss += state_entropy_loss      * config['STATE_ENT_LOSS']
                        loss += sv_entropy_loss_mean * config['SV_ENT_LOSS']
                        # loss += sub_entropy_loss        * config['SUB_ENT_LOSS']
                        loss += sub_i_entropy_loss_mean * config['SUB_ENT_LOSS']

                        loss += (loss_kl_sub_i + loss_kl_sv) * alpha_kl

                        # return loss, (loss_v, loss_vi, loss_actor_sub, loss_actor_sv, loss_kl_sub, loss_kl_sub_i, loss_kl_sv, jnp.max(KL_distance_sub), jnp.max(KL_distance_sub_i), jnp.max(KL_distance_sv), pi_sampled_sub_policy, mu_sampled_sub_policy, pi_sampled_sub_p, mu_sampled_sub_p)
                        return loss, (loss_v_, loss_vi_, loss_actor_sub_, loss_actor_sv_, loss_kl_sub_i, loss_kl_sv, sv_entropy_loss_mean, sub_i_entropy_loss_mean,
                                        sv_ratio, sub_ratio_samples, player0_i_mask, player0_mask, player1_mask, 
                                        Adv_sv, Adv_sub_samples, KL_distance_sub_i, KL_distance_sv, sv_entropy, sub_i_entropy)
                                    #   jnp.max(KL_distance_sub_i, initial=0.0, where=player0_i_mask), jnp.max(KL_distance_sv, initial=0.0, where=player1_mask), jnp.mean(KL_distance_sub_i, where=player0_i_mask), jnp.mean(KL_distance_sv, where=player1_mask),
                                    #   sub_i_entropy_loss_mean, sub_i_entropy_loss_max, sv_entropy_loss_mean, sv_entropy_loss_max,
                                    #   ratio_sv_clip_rate, ratio_sub_clip_rate, sv_ratio_max, sv_ratio_min, sub_ratio_samples_max, sub_ratio_samples_min,
                                    #   sv_ratio_high, sv_ratio_low, sub_ratio_samples_high, sub_ratio_samples_low )

                        # sv_ratio_high, sv_ratio_low, sub_ratio_samples_high, sub_ratio_samples_low
                    ## _loss_fn_r_nad


                    grad_rnad_fn = jax.value_and_grad(_loss_fn_r_nad, has_aux=True)

                    losses, grads = grad_rnad_fn(model_params, transition, targets)

                    # max_grad_element = tree_util.tree_reduce(
                    #     lambda x, y: jnp.maximum(x, y),
                    #     tree_util.tree_map(lambda g: jnp.max(jnp.abs(g)), grads)
                    # )
                    ## Log gradient-Norm. 
                    global_grad_norm = jnp.sqrt(
                        tree_util.tree_reduce(
                            lambda x, y: x + y,
                            tree_util.tree_map(lambda g: jnp.sum(g ** 2), grads),
                            initializer=0.0,
                        )
                    )

                    updates, optimizer_state = optimizer.update(grads, optimizer_state, model_params)   # Manual Alternative
                    model_params = optax.apply_updates(model_params, updates)                           # Manual Alternative

                    params_target = jax.tree_util.tree_map(lambda p2, p1: (1.0 - gamma_averaging) * p2 + gamma_averaging * p1, params_target, model_params)

                    return (model_params, params_target, optimizer_state), (losses, global_grad_norm)
                ## _update_minibatch

                rng, _rng = jax.random.split(rng) 

                batch_size = config_["MINIBATCH_SIZE"] * config_["NUM_MINIBATCHES"]
                assert (
                    batch_size == config_["NUM_ENV_STEPS"] * config_["NUM_ENVS"]
                ), f"batch size must be equal to number of steps * number of envs. MB_Size: {config_['MINIBATCH_SIZE']}  N_MB: {config_['NUM_MINIBATCHES']}  NS: {config_['NUM_ENV_STEPS']}  NE: {config_['NUM_ENVS']}"
                permutation = jax.random.permutation(_rng, batch_size)


                batch = (transition, targets)
                batch = jax.tree_util.tree_map(
                    lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
                )

                # Split into minibatches and shuffle
                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=0), batch
                )
                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(
                        x, [config_["NUM_MINIBATCHES"], -1] + list(x.shape[1:])
                    ),
                    shuffled_batch, # batch (no shuffle)
                )


                model_opt_sate = (model_params, params_target, optimizer_state)
                model_opt_sate, losses_and_grad = jax.lax.scan(_update_minibatch, model_opt_sate,  minibatches)
                # (losses, grad) = losses_and_grad
                (losses, global_grad_norm) = losses_and_grad
                (model_params, params_target, optimizer_state) = model_opt_sate

                update_state = (model_params, params_target, optimizer_state, transition, targets, rng)
                # losses_ = (losses[0], *losses[1], grad)
                losses_ = (losses[0], losses[1], global_grad_norm)

                return update_state, losses_
            ## _update_epoch
            #### UPDATE NETWORK


            (model_params, params_target, optimizer_state, rng, update_step) = runner_state_update_step
            update_state = (model_params, params_target, optimizer_state, transition, targets, rng)
            update_state, losses = jax.lax.scan(_update_epoch, update_state, None, config_["UPDATE_EPOCHS"])  # loss_info: (total_loss, (value_loss, loss_actor, entropy, Info))

            model_params = update_state[0]
            params_target = update_state[1]
            optimizer_state = update_state[2]
            rng = update_state[5]
            
            R_tT_std = jnp.std(R_tT_)       # std of final carry of advantage
            Ps_tT_std = jnp.std(Ps_tT_)     # std of final carry of advantage
            Pd_tT_std = jnp.std(Pd_tT_)     # std of final carry of advantage

            ## log-losses
            training_loss = losses[0]
            metrics = losses[1]
            global_grad_norm = losses[2]
            # (loss_v, loss_vi, loss_actor_sub, loss_actor_sv, loss_kl_sub_i, loss_kl_sv, KL_dist_sub_i_max, KL_dist_sv_max, KL_dist_sub_i_mean, KL_dist_sv_mean, 
            #  sub_i_entropy_loss_mean, sub_i_entropy_loss_max, sv_entropy_loss_mean, sv_entropy_loss_max, ratio_sv_clip_rate, ratio_sub_clip_rate, 
            #  sv_ratio_max, sv_ratio_min, sv_ratio_high, sv_ratio_low, sub_ratio_samples_max, sub_ratio_samples_min, sub_ratio_samples_high, sub_ratio_samples_low) = metrics
            (   loss_v, loss_vi, loss_actor_sub, loss_actor_sv, loss_kl_sub_i, loss_kl_sv, sv_entropy_loss, sub_entropy_loss,
                sv_ratio, sub_ratio, player0_i_mask, player0_mask, player1_mask, 
                Adv_sv, Adv_sub, KL_distance_sub_i, KL_distance_sv, sv_entropy, sub_entropy) = metrics
            

            ## Compute metrics for logging. This can add ~30% overhead to the training
            # print(f"R_tT_: {R_tT_.shape}  R_tTi_: {R_tTi_.shape}")        # (N,)  (N, n)
            # print(f"Pd_tT_: {Pd_tT_.shape}  Pd_tTi_: {Pd_tTi_.shape}")    # (N,)  (N, n)
            # print(f"Ps_tT_: {Ps_tT_.shape}  Ps_tTi_: {Ps_tTi_.shape}")    # (N,)  (N, n)
            # print(f"Ps_vec_tT_: {Ps_vec_tT_.shape}")                      # (N,)  (N, 2)
            # print(f"loss_kl_sub_i: {loss_kl_sub_i.shape}")                # (NE, Nmb)
            # print(f"loss_kl_sv: {loss_kl_sv.shape}")                      # (NE, Nmb)
            # print(f"sv_ratio: {sv_ratio.shape}  sub_ratio: {sub_ratio.shape}")                                # (NE, Nmb, mb_size) (NE, Nmb, mb_size, n)
            # print(f"player0_i_mask: {player0_i_mask.shape}  player0_mask: {player0_mask.shape}  player1_mask: {player1_mask.shape}")    # (NE, Nmb, mb_size, n) (NE, Nmb, mb_size)
            # print(f"Adv_sv: {Adv_sv.shape}  Adv_sub: {Adv_sub.shape}")                                        # (NE, Nmb, mb_size, n), (NE, Nmb, mb_size)
            # print(f"KL_distance_sub_i: {KL_distance_sub_i.shape}  KL_distance_sv: {KL_distance_sv.shape}")    # (NE, Nmb, mb_size, n), (NE, Nmb, mb_size)
            # print(f"sub_entropy: {sub_entropy.shape}  sv_entropy: {sv_entropy.shape}")                        # (NE, Nmb, mb_size, n), (NE, Nmb, mb_size)
            # print(f"player0_i_mask: {player0_i_mask.shape}  player1_mask: {player1_mask.shape}\n\n\n\n")


            sv_ratio_clip_rate = jnp.logical_and(jnp.logical_or(sv_ratio>(1+config["CLIP_EPS"]), sv_ratio<(1-config["CLIP_EPS"])), player1_mask).astype(jnp.float32)
            sv_ratio_clip_rate = jnp.sum(sv_ratio_clip_rate) / jnp.maximum(jnp.sum(player1_mask), 1)
            sv_ratio_dist = jnp.array([jnp.min(sv_ratio, initial=1e+10, where=player1_mask),
                                       jnp.mean(sv_ratio, where=jnp.logical_and(player1_mask, sv_ratio<1.0)),
                                       jnp.mean(sv_ratio, where=player1_mask),
                                       jnp.mean(sv_ratio, where=jnp.logical_and(player1_mask, sv_ratio>=1.0)),
                                       jnp.max(sv_ratio, initial=0, where=player1_mask)])

            sub_ratio_clip_rate = jnp.logical_and(jnp.logical_or(sub_ratio>(1+config["CLIP_EPS"]), sub_ratio<(1-config["CLIP_EPS"])), player0_i_mask).astype(jnp.float32)
            sub_ratio_clip_rate = jnp.sum(sub_ratio_clip_rate) / jnp.maximum(jnp.sum(player0_i_mask), 1)

            sub_ratio_mean = jnp.mean(sub_ratio, where=player0_i_mask)
            sub_ratio_dist = jnp.array([jnp.min(sub_ratio, initial=2.0, where=player0_i_mask),
                                       jnp.mean(sub_ratio, where=jnp.logical_and(player0_i_mask, sub_ratio<1.0)),
                                       sub_ratio_mean,
                                       jnp.mean(sub_ratio, where=jnp.logical_and(player0_i_mask, sub_ratio>=1.0)),
                                       jnp.max(sub_ratio, initial=-1.0, where=player0_i_mask)])
            
            sv_entropy_mean = jnp.mean(sv_entropy, where=player1_mask)
            sv_entropy_vec = jnp.array([    jnp.min(sv_entropy, initial=1e+10, where=player1_mask),
                                            jnp.mean(sv_entropy, where=jnp.logical_and(player1_mask, sv_entropy<sv_entropy_mean)),
                                            sv_entropy_mean,
                                            jnp.mean(sv_entropy, where=jnp.logical_and(player1_mask, sv_entropy>=sv_entropy_mean)),
                                            jnp.max(sv_entropy, initial=-1e+10, where=player1_mask)])
            sub_entropy_mean = jnp.mean(sub_entropy, where=player0_i_mask)
            sub_entropy_vec = jnp.array([   jnp.min(sub_entropy, initial=1e+10, where=player0_i_mask),
                                            jnp.mean(sub_entropy, where=jnp.logical_and(player0_i_mask, sub_entropy<sub_entropy_mean)),
                                            sub_entropy_mean,
                                            jnp.mean(sub_entropy, where=jnp.logical_and(player0_i_mask, sub_entropy>=sub_entropy_mean)),
                                            jnp.max(sub_entropy, initial=-1e+10, where=player0_i_mask)])
            
            sv_KL_distance_mean = jnp.mean(KL_distance_sv, where=player1_mask)
            sv_KL_distance_vec = jnp.array([   jnp.min(KL_distance_sv, initial=1e+10, where=player1_mask),
                                            jnp.mean(KL_distance_sv, where=jnp.logical_and(player1_mask, KL_distance_sv<sv_KL_distance_mean)),
                                            sv_KL_distance_mean,
                                            jnp.mean(KL_distance_sv, where=jnp.logical_and(player1_mask, KL_distance_sv>=sv_KL_distance_mean)),
                                            jnp.max(KL_distance_sv, initial=-1.0, where=player1_mask)])
            sub_KL_distance_mean = jnp.mean(KL_distance_sub_i, where=player0_i_mask)
            sub_KL_distance_vec = jnp.array([   jnp.min(KL_distance_sub_i, initial=1e+10, where=player0_i_mask),
                                            jnp.mean(KL_distance_sub_i, where=jnp.logical_and(player0_i_mask, KL_distance_sub_i<sub_KL_distance_mean)),
                                            jnp.mean(KL_distance_sub_i, where=player0_i_mask),
                                            jnp.mean(KL_distance_sub_i, where=jnp.logical_and(player0_i_mask, KL_distance_sub_i>=sub_KL_distance_mean)),
                                            jnp.max(KL_distance_sub_i, initial=-1.0, where=player0_i_mask)])
            
            sv_Adv_mean = jnp.mean(Adv_sv, where=player1_mask)
            sv_Adv_vec = jnp.array([   jnp.min(Adv_sv, initial=1e+10, where=player1_mask),
                                            jnp.mean(Adv_sv, where=jnp.logical_and(player1_mask, Adv_sv<sv_Adv_mean)),
                                            sv_Adv_mean,
                                            jnp.mean(Adv_sv, where=jnp.logical_and(player1_mask, Adv_sv>=sv_Adv_mean)),
                                            jnp.max(Adv_sv, initial=-1e+10, where=player1_mask)])
            sub_Adv_mean = jnp.mean(Adv_sub, where=player0_i_mask)
            sub_Adv_vec = jnp.array([   jnp.min(Adv_sub, initial=1e+10, where=player0_i_mask),
                                            jnp.mean(Adv_sub, where=jnp.logical_and(player0_i_mask, Adv_sub<sub_Adv_mean)),
                                            sub_Adv_mean,
                                            jnp.mean(Adv_sub, where=jnp.logical_and(player0_i_mask, Adv_sub>=sub_Adv_mean)),
                                            jnp.max(Adv_sub, initial=-1e+10, where=player0_i_mask)])
            
            Pd_mean = jnp.mean(Pd_tT_)
            Pd_vec = jnp.array([ jnp.min(Pd_tT_, initial=2.0),
                                jnp.mean(Pd_tT_, where=Pd_tT_<Pd_mean),
                                Pd_mean,
                                jnp.mean(Pd_tT_, where=Pd_tT_>=Pd_mean),
                                jnp.max(Pd_tT_, initial=-2.0)])
            Ps_mean = jnp.mean(Ps_tT_)
            Ps_vec = jnp.array([ jnp.min(Ps_tT_, initial=2.0),
                                jnp.mean(Ps_tT_, where=Ps_tT_<Ps_mean),
                                Ps_mean,
                                jnp.mean(Ps_tT_, where=Ps_tT_>=Ps_mean),
                                jnp.max(Ps_tT_, initial=-2.0)])
            R_mean = jnp.mean(R_tT_)
            R_vec = jnp.array([ jnp.min(R_tT_, initial=2.0),
                                jnp.mean(R_tT_, where=R_tT_<R_mean),
                                R_mean,
                                jnp.mean(R_tT_, where=R_tT_>=R_mean),
                                jnp.max(R_tT_, initial=-2.0)])
            R_i_mean = jnp.mean(R_tTi_)
            R_i_vec = jnp.array([ jnp.min(R_tTi_, initial=2.0),
                                    jnp.mean(R_tTi_, where=R_tTi_<R_i_mean),
                                    R_i_mean,
                                    jnp.mean(R_tTi_, where=R_tTi_>=R_i_mean),
                                    jnp.max(R_tTi_, initial=-2.0)])
            
            Ps1_mean = jnp.mean(Ps_vec_tT_[:,0])
            Ps1_vec = jnp.array([ jnp.min(Ps_vec_tT_[:,0], initial=2.0),
                                    jnp.mean(Ps_vec_tT_[:,0], where=Ps_vec_tT_[:,0]<Ps1_mean),
                                    Ps1_mean,
                                    jnp.mean(Ps_vec_tT_[:,0], where=Ps_vec_tT_[:,0]>=Ps1_mean),
                                    jnp.max(Ps_vec_tT_[:,0], initial=-2.0)])
            Ps2_mean = jnp.mean(Ps_vec_tT_[:,1])
            Ps2_vec = jnp.array([ jnp.min(Ps_vec_tT_[:,1], initial=2.0),
                                    jnp.mean(Ps_vec_tT_[:,1], where=Ps_vec_tT_[:,1]<Ps2_mean),
                                    Ps2_mean,
                                    jnp.mean(Ps_vec_tT_[:,1], where=Ps_vec_tT_[:,1]>=Ps2_mean),
                                    jnp.max(Ps_vec_tT_[:,1], initial=-2.0)])

            ## log-metrics
            metric_outcome = (R_vec, R_i_vec, Ps_vec, Pd_vec, Ps1_vec, Ps2_vec, R_tT_std, Ps_tT_std, Pd_tT_std)
            metric_policy = (sv_KL_distance_vec, sub_KL_distance_vec, sub_entropy_vec, sv_entropy_vec)
            metric_losses = (training_loss, loss_v, loss_vi, loss_actor_sub, loss_actor_sv, loss_kl_sub_i, loss_kl_sv, sv_entropy_loss, sub_entropy_loss )
            metric_clip_norm_ratio = (global_grad_norm, sub_ratio_dist, sv_ratio_dist, sv_ratio_clip_rate, sub_ratio_clip_rate, sv_Adv_vec, sub_Adv_vec)


            ## NE: Number of epochs (usually =1)
            ## MbN: number of minibatches (usually 1 or 4)
            # print(f"Ps_tT_: {Ps_tT_.shape}")                # (N)
            # print(f"Pd_tT_: {Pd_tT_.shape}")                # (N)
            # print(f"R_tT_: {R_tT_.shape}")                  # (N)
            # print(f"R_tTi_: {R_tTi_.shape}")                # (N, n)

            # print(f"\nloss_kl_sub_i: {loss_kl_sub_i.shape}")                    # (NE, MbN)
            # print(f"loss_kl_sv: {loss_kl_sv.shape}")                            # (NE, MbN)
            # print(f"KL_dist_sub_i_max: {KL_dist_sub_i_max.shape}")              # (NE, MbN)
            # print(f"KL_dist_sv_max: {KL_dist_sv_max.shape}")                    # (NE, MbN)
            # print(f"KL_dist_sub_i_mean: {KL_dist_sub_i_mean.shape}")            # (NE, MbN)
            # print(f"KL_dist_sv_mean: {KL_dist_sv_mean.shape}")                  # (NE, MbN)

            # print(f"\nsub_i_entropy_loss_mean: {sub_i_entropy_loss_mean.shape}")# (NE, MbN)
            # print(f"sub_i_entropy_loss_max: {sub_i_entropy_loss_max.shape}")    # (NE, MbN)
            # print(f"sv_entropy_loss_mean: {sv_entropy_loss_mean.shape}")        # (NE, MbN)
            # print(f"sv_entropy_loss_max: {sv_entropy_loss_max.shape}")          # (NE, MbN)

            # print(f"training_loss: {training_loss.shape}")                      # (NE, MbN)
            # print(f"loss_v: {loss_v.shape}")                                    # (NE, MbN)
            # print(f"loss_vi: {loss_vi.shape}")                                  # (NE, MbN)
            # print(f"loss_actor_sub: {loss_actor_sub.shape}")                    # (NE, MbN)
            # print(f"loss_actor_sv: {loss_actor_sv.shape}\n")                    # (NE, MbN)

            
            runner_state_update_step = (model_params, params_target, optimizer_state, rng, update_step+1)
            # return runner_state_update_step, (metric_trajectory, metric_losses, metric_kl, metric_entropy, metric_clip_norm_ratio)
            return runner_state_update_step, (metric_outcome, metric_policy, metric_losses, metric_clip_norm_ratio)
        
        rng, _rng = jax.random.split(rng)
        runner_state = (model_params_, params_target_, optimizer_state_, _rng, config['current_reg_step'])
        runner_state, (metric_outcome, metric_policy, metric_losses, metric_clip_norm_ratio) = jax.lax.scan( _update_step, runner_state, None, config_["N_JIT_STEPS"] )

        return {"runner_state": runner_state, "metric_outcome": metric_outcome, "metric_policy": metric_policy, "metric_losses": metric_losses, "metric_clip_norm_ratio": metric_clip_norm_ratio}

    return train