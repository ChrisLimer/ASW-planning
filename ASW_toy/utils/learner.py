import os
# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


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


from ActorCriticNetwork import MLP
from asw_toy_env import Transition, Target, SubState, SubEnv


def make_train(config_, env, network, optimizer):
    
    def train(rng, model_params_, params_target_, optimizer_state_, config, model_reg_params_t1, model_reg_params_t2):

        alpha_kl = config['alpha_kl']
        max_reg_step = config['max_reg_step']
        gamma_averaging = config['gamma_averaging']

        # TRAIN LOOP
        def _update_step(runner_state_update_step, unused):
            
            (model_params, params_target, optimizer_state, rng, update_step) = runner_state_update_step

            alpha = jnp.array((update_step), float)/max_reg_step

            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):

                params, opt_state, env_state, rng = runner_state

                last_obs = env.get_obs(env_state)
                player = jnp.where( (env_state.turn%2==0), 1, 0)
                player = jnp.where( (env_state.turn==0), -1, player)
                sub_pos_samples = jnp.array(env_state.sub_pos_samples)

                prior_sub_distribution = env_state.sub_distribution.copy()

                (sub_logit, dip_logit, v, v_map) = network.apply(params, last_obs)

                dip_mask = env.available_sv_dips[jnp.maximum(env_state.turn//2-1, 0)]

                sub_policy, sampled_sub_policy, sv_policy = env.mask_logit_to_policy(sub_logit, dip_logit, env_state.sub_pos_samples, dip_mask)
                
                rng, _rng = jax.random.split(rng)
                action_sub_samples, action_dip = env.sample_action(_rng, sv_policy, sampled_sub_policy)
                (new_state, r, r_i, done, done_i, done2, done_i2, p_prior, p_i_prior, p, p_i, pd, ps, ps_i, pd_i) = env.step(env_state, sub_policy, action_dip, action_sub_samples)

                
                runner_state = (params, opt_state, new_state, rng)

                transition = Transition(done, done_i, done2, done_i2, r, r_i, v, v_map, prior_sub_distribution, p_prior, p_i_prior, p, p_i, sub_pos_samples, dip_mask, last_obs, sub_logit, dip_logit, sub_policy, sv_policy, action_sub_samples, action_dip, pd, ps, ps_i, pd_i, player)

                return runner_state, transition



            # (model_params, optimizer_state, env_state_0, obsv, rng, update_step) = runner_state_update_step
            rng, _rng = jax.random.split(rng)
            obsv, env_state_0 = env.reset(config_["NUM_ENVS"], config_["NUM_SUB_ENVS"], _rng)  # Create multiple environments
            runner_state = (model_params, optimizer_state, env_state_0, rng)


            runner_state, transition = jax.lax.scan(
                _env_step, runner_state, None, config_["NUM_STEPS"]
            )

            def compute_advantage(carry, transition):
                (done, done_i, done2, done_i2, r, r_i, p_prior, p_i_prior) = (transition.done,
                                                  transition.done_i,
                                                  transition.done2,
                                                  transition.done_i2,
                                                  transition.r,
                                                  transition.r_i,
                                                  transition.p_prior,
                                                  transition.p_i_prior)
                (R_t_next, R_ti_next) = carry

                # Note, 'r' and 'r_i' contain the probability to reach that reward from the current_state-->next_state r=r_*p(a)
                R_tT_ = R_t_next + r*p_prior*(1-done2)                  # expected future reward * P(state, h)
                R_tTi_ = R_ti_next + r_i*p_i_prior*(1-done_i2)          # expected future reward * P(state_i, h)

                R_tT = R_tT_/p_prior      # expected future reward from the current state
                R_tTi = R_tTi_/p_i_prior  # expected future reward from the current state

                R_tTi = jnp.where(done_i2, 0.0, R_tTi)
                R_tT = jnp.where(done2, 0.0, R_tT)


                return (R_tT_, R_tTi_), (R_tT, R_tTi, r, r_i)

            init_carry = (jnp.zeros_like(transition.r[0]), jnp.zeros_like(transition.r_i[0]))
            final_carry, R = jax.lax.scan(compute_advantage, init_carry, transition, reverse=True, unroll=16)
            (R_tT, R_tTi, r, r_i) = R
            (R_tT_, R_tTi_) = final_carry

            targets = Target(R_tT, R_tTi, r, r_i)

            #### UPDATE NETWORK
            def _update_epoch(update_state, unused):
                (model_params, params_target, optimizer_state, transition, targets, rng) = update_state


                def _update_minbatch(model_opt_sate, batch_info):
                    (transition, targets) = batch_info

                    (model_params, params_target, optimizer_state) = model_opt_sate

                    def _loss_fn_r_nad(params, transition, targets):
                        sub_pos_samples = transition.sub_pos_samples
                        dip_mask = transition.dip_mask

                        player0_i_mask = jnp.logical_and(jnp.logical_not(transition.done_i2), (transition.player==0)[:,None])
                        player0_mask = jnp.logical_and(jnp.logical_not(transition.done2), (transition.player==0))
                        player1_mask = jnp.logical_and(jnp.logical_not(transition.done2), (transition.player==1))

                        (sub_logit, dip_logit, v, v_map) = network.apply(params, transition.obs)

                        pi_sub_policy, pi_sampled_sub_policy, pi_sv_policy = env.mask_logit_to_policy(sub_logit, dip_logit, transition.sub_pos_samples, dip_mask)

                        ##
                        #### Regularization for regularized objective loss
                        (sub_logit_reg1, dip_logit_reg1, v, v_map) = network.apply(model_reg_params_t1, transition.obs)
                        (sub_logit_reg2, dip_logit_reg2, v, v_map) = network.apply(model_reg_params_t2, transition.obs)

                        ## Regularize against a uniform policy
                        # dip_logit_reg1 = jnp.ones_like(dip_logit_reg1)
                        # dip_logit_reg2 = jnp.ones_like(dip_logit_reg2)
                        # sub_logit_reg1 = jnp.ones_like(sub_logit_reg1)
                        # sub_logit_reg2 = jnp.ones_like(sub_logit_reg2)

                        pi_sub_policy_reg1, pi_sampled_sub_policy_reg1, pi_sv_policy_reg1 = env.mask_logit_to_policy(sub_logit_reg1, dip_logit_reg1, transition.sub_pos_samples, dip_mask)
                        pi_sub_policy_reg2, pi_sampled_sub_policy_reg2, pi_sv_policy_reg2 = env.mask_logit_to_policy(sub_logit_reg2, dip_logit_reg2, transition.sub_pos_samples, dip_mask)
                        #### Regularization for regularized objective loss
                        ##

                        ## Create policy masks
                        policy_mask_sub = jnp.logical_and(jnp.logical_and(jnp.logical_not(transition.done2), transition.player==0)[:,None,None], pi_sub_policy_reg1>0 )             # not-done and player==0 and action-prob>0 (valid action)
                        policy_mask_sub_i = jnp.logical_and(jnp.logical_and(jnp.logical_not(transition.done_i2), (transition.player==0)[:,None])[:,:,None], pi_sampled_sub_policy_reg1>0 )  # not-done and player==0 and action-prob>0 (valid action)
                        policy_mask_sv = jnp.logical_and(jnp.logical_and(jnp.logical_not(transition.done2), transition.player==1)[:,None], pi_sv_policy_reg1>0 )                # not-done and player==1 and action-prob>0 (valid action)
                        
                        pi_sub_policy = jnp.where(policy_mask_sub, pi_sub_policy, 0.0)
                        pi_sampled_sub_policy = jnp.where(policy_mask_sub_i, pi_sampled_sub_policy, 0.0)
                        pi_sv_policy = jnp.where(policy_mask_sv, pi_sv_policy, 0.0)

                        ## Compute regularized policies
                        pi_sub_policy_reg = jnp.where(policy_mask_sub, pi_sub_policy_reg1*(1.0-alpha) + alpha*pi_sub_policy_reg2, 1.0)
                        pi_sampled_sub_policy_reg = jnp.where(policy_mask_sub_i, pi_sampled_sub_policy_reg1*(1.0-alpha) + alpha*pi_sampled_sub_policy_reg2, 1.0)
                        pi_sv_policy_reg = jnp.where(policy_mask_sv, pi_sv_policy_reg1*(1.0-alpha) + alpha*pi_sv_policy_reg2, 1.0)

                        ## Retrieve and compute sampled policy (should be equal to the current policy if there is no replay buffer or minibatches or multiple epoch-updates)
                        mu_sub_logit = transition.sub_logit
                        mu_dip_logit = transition.dip_logit
                        mu_sub_policy, mu_sampled_sub_policy, mu_sv_policy = env.mask_logit_to_policy(mu_sub_logit, mu_dip_logit, transition.sub_pos_samples, dip_mask)
                        mu_sub_policy = jnp.where(policy_mask_sub, mu_sub_policy, 1.0)
                        mu_sampled_sub_policy = jnp.where(policy_mask_sub_i, mu_sampled_sub_policy, 1.0)
                        mu_sv_policy = jnp.where(policy_mask_sv, mu_sv_policy, 1.0)

                        ## Compute KL distance between the current policy and the regularized policy
                        KL_distance_sub_i = jnp.sum(pi_sampled_sub_policy * jnp.log(pi_sampled_sub_policy/ (pi_sampled_sub_policy_reg+1e-15)+1e-15), axis=(-1), where=policy_mask_sub_i)
                        KL_distance_sv = jnp.sum(pi_sv_policy * jnp.log(pi_sv_policy/ (pi_sv_policy_reg+1e-15)+1e-15), axis=(-1), where=policy_mask_sv)
                        KL_distance_sub = jnp.sum(pi_sub_policy * jnp.log(pi_sub_policy/ (pi_sub_policy_reg+1e-15)+1e-15), axis=(-1), where=policy_mask_sub)
                        KL_distance_sub = jnp.sum(KL_distance_sub * transition.sub_distribution, axis=(-1), keepdims=False)

                        loss_kl_sub = jnp.mean(KL_distance_sub, where=player0_mask, keepdims=False)
                        loss_kl_sub_i = jnp.mean(KL_distance_sub_i, where=player0_i_mask, keepdims=False)
                        loss_kl_sv = jnp.mean(KL_distance_sv, where=player1_mask, keepdims=False)


                        #### Reward Advantage and policy gradients
                        (R_tT, R_tTi, r, r_i) = targets
                        batch_idx = jnp.arange(dip_logit.shape[0])
                        sub_batch_idx = jnp.arange(sub_pos_samples.shape[1])

                        ## get pi(a_i) and mu(a_i)
                        pi_sampled_sub_p = pi_sampled_sub_policy[batch_idx[:, None], sub_batch_idx[None, :], transition.action_sub_samples]
                        mu_sampled_sub_p = mu_sampled_sub_policy[batch_idx[:, None], sub_batch_idx[None, :], transition.action_sub_samples]
                        pi_sampled_sub_p_reg = pi_sampled_sub_policy_reg[batch_idx[:, None], sub_batch_idx[None, :], transition.action_sub_samples]
                        pi_sampled_sub_p = jnp.where(player0_i_mask, pi_sampled_sub_p, 0.0)
                        mu_sampled_sub_p = jnp.where(player0_i_mask, mu_sampled_sub_p, 1.0)
                        pi_sampled_sub_p_reg = jnp.where(player0_i_mask, pi_sampled_sub_p_reg, 1.0)

                        v_i = v_map[batch_idx[:,None], sub_pos_samples]
                        v_i_traj_batch = transition.v_map[batch_idx[:,None], sub_pos_samples]

                        ## Simple policy gradient loss
                        ## loss = log(pi(a^)) * Advantage,  Advantage = R-v
                        pi_sv_policy = jnp.where(policy_mask_sv, pi_sv_policy, 0.0)
                        mu_sv_policy = jnp.where(policy_mask_sv, mu_sv_policy, 0.0)
                        pi_sv_policy_reg = jnp.where(policy_mask_sv, pi_sv_policy_reg, 0.0)
                        pi_sv_p = pi_sv_policy[batch_idx[:], transition.action_dip]
                        mu_sv_p = mu_sv_policy[batch_idx[:], transition.action_dip]
                        pi_sv_p_reg = pi_sv_policy_reg[batch_idx[:], transition.action_dip]
                        pi_sv_p = jnp.where(player1_mask, pi_sv_p, 0.0)
                        mu_sv_p = jnp.where(player1_mask, mu_sv_p, 1.0)
                        pi_sv_p_reg = jnp.where(player1_mask, pi_sv_p_reg, 1.0)


                        Adv_sub_samples = jnp.where(player0_i_mask, R_tTi - v_i, 0.0)
                        Adv_sv = jnp.where(player1_mask ,-(R_tT - v), 0.0 )

                        Adv_sub_samples = jax.lax.stop_gradient(Adv_sub_samples)
                        Adv_sv = jax.lax.stop_gradient(Adv_sv)

                        #### PPO-similar policy loss
                        #### Submarine policy loss
                        sub_ratio_samples = jnp.where(player0_i_mask, pi_sampled_sub_p / (mu_sampled_sub_p+1e-14), 0.0)
                        # sub_ratio_samples = jnp.where(player0_i_mask, pi_sampled_sub_p / pi_sampled_sub_p_reg, 0.0)

                        Adv_sub_samples_clip_1 = Adv_sub_samples * sub_ratio_samples
                        Adv_sub_samples_clip_2 = ( jnp.clip(
                                sub_ratio_samples,
                                1.0 - config["CLIP_EPS"],
                                1.0 + config["CLIP_EPS"],
                            ) * Adv_sub_samples)

                        loss_actor_sub = -jnp.minimum(Adv_sub_samples_clip_1, Adv_sub_samples_clip_2)   # (N, n)
                        loss_actor_sub = jnp.where(player0_i_mask, loss_actor_sub, 0.0)


                        #### Surface Vehicle policy loss
                        sv_ratio = jnp.where(player1_mask, pi_sv_p / (mu_sv_p+1e-14), 0.0)
                        # sv_ratio = jnp.where(player1_mask, pi_sv_p / pi_sv_p_reg, 0.0)

                        Adv_sv_clip_1 = Adv_sv * sv_ratio
                        Adv_sv_clip_2 = ( jnp.clip(
                                sv_ratio,
                                1.0 - config["CLIP_EPS"],
                                1.0 + config["CLIP_EPS"],
                            ) * Adv_sv)

                        loss_actor_sv = -jnp.minimum(Adv_sv_clip_1, Adv_sv_clip_2)   # (N, n)
                        loss_actor_sv = jnp.where(player1_mask, loss_actor_sv, 0.0)


                        ## Entropy loss
                        sub_i_entropy = -jnp.sum(pi_sampled_sub_policy * jnp.log(pi_sampled_sub_policy+1e-15),  axis=(-1), where=policy_mask_sub_i)    # (N, n, N_D) -> (N, n)        Sum over D -> (-1)
                        sub_i_entropy_loss_mean = -jnp.mean(sub_i_entropy*transition.p_i_prior, where=player0_i_mask )        # (N, n) -> (,)

                        sv_entropy = -jnp.sum( (jnp.log(pi_sv_policy+1e-14)*pi_sv_policy), where=policy_mask_sv, axis=(-1))      # normal entropy loss: (N, N_D) --> (N)
                        sv_entropy_loss_mean = -jnp.mean(sv_entropy*transition.p_prior, where=player1_mask)                         # (N) --> (,)

                        loss_v = optax.losses.huber_loss( jnp.where( jnp.logical_not(transition.done2), v-R_tT, 0.0) )
                        loss_vi = optax.losses.huber_loss( jnp.where( jnp.logical_not(transition.done_i2),v_i-R_tTi, 0.0) )

                        
                        loss = jnp.mean(loss_v*transition.p_prior, where=jnp.logical_not(transition.done2))
                        loss += jnp.mean(loss_vi*transition.p_i_prior, where=jnp.logical_not(transition.done_i2))
                        loss += jnp.mean(loss_actor_sub*transition.p_i_prior, where=jnp.logical_not(transition.done_i2))
                        loss += jnp.mean(loss_actor_sv*transition.p_prior, where=jnp.logical_not(transition.done2))

                        loss += sub_i_entropy_loss_mean * config['SUB_ENT_LOSS']
                        loss += sv_entropy_loss_mean * config['SV_ENT_LOSS']

                        loss += (loss_kl_sub_i + loss_kl_sv) * alpha_kl

                        return loss, (loss_v, loss_vi, loss_actor_sub, loss_actor_sv, loss_kl_sub, loss_kl_sub_i, loss_kl_sv, sub_i_entropy_loss_mean, sv_entropy_loss_mean)
                    


                    grad_fn = jax.value_and_grad(_loss_fn_r_nad, has_aux=True)
                    losses, grads = grad_fn(model_params, transition, targets)

                    max_grad_magnitude = tree_util.tree_reduce(
                        lambda x, y: jnp.maximum(x, y),
                        tree_util.tree_map(lambda g: jnp.max(jnp.abs(g)), grads)
                    )


                    updates, optimizer_state = optimizer.update(grads, optimizer_state, model_params)   # Manual Alternative
                    model_params = optax.apply_updates(model_params, updates)                           # Manual Alternative

                    # linear_fit_params(model_params)
                    params_target = jax.tree_util.tree_map(lambda p2, p1: (1.0 - gamma_averaging) * p2 + gamma_averaging * p1, params_target, model_params)

                    return (model_params, params_target, optimizer_state), (losses, max_grad_magnitude)
                ## _update_minbatch


                rng, _rng = jax.random.split(rng)  

                batch_size = config_["MINIBATCH_SIZE"] * config_["NUM_MINIBATCHES"]
                assert (
                    batch_size == config_["NUM_STEPS"] * config_["NUM_ENVS"]
                ), f"batch size must be equal to number of steps * number of envs. MB_Size: {config_['MINIBATCH_SIZE']}  N_MB: {config_['NUM_MINIBATCHES']}  NS: {config_['NUM_STEPS']}  NE: {config_['NUM_ENVS']}"
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
                    shuffled_batch,
                )


                # # Split into minibatches (no shuffle)
                # minibatches = jax.tree_util.tree_map(
                #     lambda x: jnp.reshape(
                #         x, [config_["NUM_MINIBATCHES"], -1] + list(x.shape[1:])
                #     ),
                #     batch,
                # )


                model_opt_sate = (model_params, params_target, optimizer_state)
                model_opt_sate, losses_and_grad = jax.lax.scan(_update_minbatch, model_opt_sate,  minibatches)
                (losses, grad) = losses_and_grad
                (model_params, params_target, optimizer_state) = model_opt_sate

                update_state = (model_params, params_target, optimizer_state, transition, targets, rng)

                # losses_ = (losses[0], *losses[1], grad)
                losses_ = (losses[0], *losses[1])
                
                return update_state, losses_
            ## _update_epoch
            #### UPDATE NETWORK

            (model_params, params_target, optimizer_state, rng, update_step) = runner_state_update_step
            update_state = (model_params, params_target, optimizer_state, transition, targets, rng)
            update_state, losses = jax.lax.scan(_update_epoch, update_state, None, config_["NUM_EPOCHS"])  # loss_info: (total_loss, (value_loss, loss_actor, entropy, Info))

            model_params = update_state[0]
            params_target = update_state[1]
            optimizer_state = update_state[2]
            rng = update_state[5]



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
            
            metric_outcome = (R_vec, R_i_vec)
            metric_losses = losses

            runner_state_update_step = (model_params, params_target, optimizer_state, rng, update_step+1)
            return runner_state_update_step, (metric_losses, metric_outcome)
        ## _update_step
        

        rng, _rng = jax.random.split(rng)
        runner_state = (model_params_, params_target_, optimizer_state_, _rng, config['current_reg_step'])

        runner_state, (metric_losses, metric_outcome) = jax.lax.scan( _update_step, runner_state, None, config_["N_JIT_STEPS"] )
        

        return {"runner_state": runner_state, "metric_losses": metric_losses, "metric_outcome": metric_outcome}

    return train