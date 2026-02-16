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

                # (logits, v, v_map) = network.apply(params, last_obs)
                # sub_logit = logits[:, :13, :]
                # dip_logit = logits[:, 13, :]
                (sub_logit, dip_logit, v, v_map) = network.apply(params, last_obs)

                # sub_policy, sampled_sub_policy, sv_policy = env.mask_logit_to_policy(env_state, logits)
                dip_mask = env.available_sv_dips[jnp.maximum(env_state.turn//2-1, 0)] # (N, 3,13) --> (N, 13)

                # print(f"logits: {logits.shape}")
                print(f"sub_logit: {sub_logit.shape}")
                print(f"dip_logit: {dip_logit.shape}")
                # sub_policy, sampled_sub_policy, sv_policy = env.mask_logit_to_policy(logits, env_state.sub_pos_samples, dip_mask)
                sub_policy, sampled_sub_policy, sv_policy = env.mask_logit_to_policy(sub_logit, dip_logit, env_state.sub_pos_samples, dip_mask)
                
                rng, _rng = jax.random.split(rng)
                action_sub_samples, action_dip = env.sample_action(_rng, sv_policy, sampled_sub_policy)
                (new_state, r, r_i, done, done_i, done2, done_i2, p_prior, p_i_prior, p, p_i, pd, ps, ps_i, pd_i) = env.step(env_state, sub_policy, action_dip, action_sub_samples)

                
                runner_state = (params, opt_state, new_state, rng)

                # transition = Transition(done, done_i, done2, done_i2, r, r_i, v, v_map, new_state.sub_distribution, p_prior, p_i_prior, p, p_i, sub_pos_samples, dip_mask, last_obs, sub_logit, dip_logit, sub_policy, sv_policy, action_sub_samples, action_dip, pd, ps, ps_i, pd_i, player)
                transition = Transition(done, done_i, done2, done_i2, r, r_i, v, v_map, prior_sub_distribution, p_prior, p_i_prior, p, p_i, sub_pos_samples, dip_mask, last_obs, sub_logit, dip_logit, sub_policy, sv_policy, action_sub_samples, action_dip, pd, ps, ps_i, pd_i, player)

                print(f"return _env_step")
                return runner_state, transition



            # (model_params, optimizer_state, env_state_0, obsv, rng, update_step) = runner_state_update_step
            rng, _rng = jax.random.split(rng)
            obsv, env_state_0 = env.reset(config_["NUM_ENVS"], config_["NUM_SUB_ENVS"], _rng)  # Create multiple environments
            runner_state = (model_params, optimizer_state, env_state_0, rng)


            runner_state, transition = jax.lax.scan(
                _env_step, runner_state, None, config_["NUM_STEPS"]
            )

            def compute_advantage(carry, transition):
                print(f"    in compute_advantage")
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

            print(f"\ncompute_advantage...")
            print(f"transition.r: {transition.r.shape}")
            print(f"transition.r_i: {transition.r_i.shape}")
            init_carry = (jnp.zeros_like(transition.r[0]), jnp.zeros_like(transition.r_i[0]))
            final_carry, R = jax.lax.scan(compute_advantage, init_carry, transition, reverse=True, unroll=16)
            (R_tT, R_tTi, r, r_i) = R
            (R_tT_, R_tTi_) = final_carry

            print(f"R_tT: {R_tT.shape}")
            print(f"R_tTi: {R_tTi.shape}")
            print(f"Done with compute_advantage scan")

            # return Target(R_tT, R_tT_n, Pd_tT, Pd_tT_n)
            targets = Target(R_tT, R_tTi, r, r_i)

            #### UPDATE NETWORK
            def _update_epoch(update_state, unused):
                print(f"_update_epoch...")
                (model_params, params_target, optimizer_state, transition, targets, rng) = update_state


                def _update_minbatch(model_opt_sate, batch_info):
                    print(f"_update_minbatch...")
                    (transition, targets) = batch_info
                    # print(f"    transition: {transition.shape}")
                    # print(f"    targets: {targets.shape}")
                    (model_params, params_target, optimizer_state) = model_opt_sate
                    print(f"    Done!")

                    # params_target

                    def _loss_fn_r_nad(params, transition, targets):
                        print(f"\n_loss_fn_custom...")
                        # (pi, v, v_yx), (sub_policy, dip_policy) = network.apply(params, transition.obs, env.transition_mask)
                        sub_pos_samples = transition.sub_pos_samples
                        dip_mask = transition.dip_mask

                        player0_i_mask = jnp.logical_and(jnp.logical_not(transition.done_i2), (transition.player==0)[:,None])
                        player0_mask = jnp.logical_and(jnp.logical_not(transition.done2), (transition.player==0))
                        player1_mask = jnp.logical_and(jnp.logical_not(transition.done2), (transition.player==1))
                        
                        print(f"Apply params")
                        # (logits, v, v_map) = network.apply(params, transition.obs)
                        # sub_logit = logits[:, :13, :]
                        # dip_logit = logits[:, 13, :]
                        (sub_logit, dip_logit, v, v_map) = network.apply(params, transition.obs)

                        print(f"mask_logit_to_policy")
                        pi_sub_policy, pi_sampled_sub_policy, pi_sv_policy = env.mask_logit_to_policy(sub_logit, dip_logit, transition.sub_pos_samples, dip_mask)

                        ##
                        #### Regularization for regularized objective loss
                        # (logits_reg1, v_reg1, v_map_reg1) = network.apply(model_reg_params_t1, transition.obs)
                        # (logits_reg2, v_reg2, v_map_reg2) = network.apply(model_reg_params_t2, transition.obs)
                        (sub_logit_reg1, dip_logit_reg1, v, v_map) = network.apply(params, transition.obs)
                        (sub_logit_reg2, dip_logit_reg2, v, v_map) = network.apply(params, transition.obs)

                        # sub_logit_reg1 = logits_reg1[:, :13, :]
                        # dip_logit_reg1 = logits_reg1[:, 13, :]
                        # sub_logit_reg2 = logits_reg2[:, :13, :]
                        # dip_logit_reg2 = logits_reg2[:, 13, :]
                        print(f"sub_logit_reg1: {sub_logit_reg1.shape}")
                        print(f"dip_logit_reg1: {dip_logit_reg1.shape}")


                        print(f"mask_logit_to_policy uniform regularized policy")
                        pi_sub_policy_reg1, pi_sampled_sub_policy_reg1, pi_sv_policy_reg1 = env.mask_logit_to_policy(sub_logit_reg1, dip_logit_reg1, transition.sub_pos_samples, dip_mask)
                        pi_sub_policy_reg2, pi_sampled_sub_policy_reg2, pi_sv_policy_reg2 = env.mask_logit_to_policy(sub_logit_reg2, dip_logit_reg2, transition.sub_pos_samples, dip_mask)
                        # pi_sub_policy_reg1, pi_sampled_sub_policy_reg1, pi_sv_policy_reg1 = env.mask_logit_to_policy(jnp.zeros_like(sub_logit_reg1), jnp.zeros_like(dip_logit_reg1), transition.sub_pos_samples, dip_mask)
                        # pi_sub_policy_reg2, pi_sampled_sub_policy_reg2, pi_sv_policy_reg2 = env.mask_logit_to_policy(jnp.zeros_like(sub_logit_reg2), jnp.zeros_like(dip_logit_reg2), transition.sub_pos_samples, dip_mask)
                        #### Regularization for regularized objective loss
                        ##

                        ## Create policy masks
                        policy_mask_sub = jnp.logical_and(jnp.logical_and(jnp.logical_not(transition.done2), transition.player==0)[:,None,None], pi_sub_policy_reg1>0 )             # not-done and player==0 and action-prob>0 (valid action)
                        policy_mask_sub_i = jnp.logical_and(jnp.logical_and(jnp.logical_not(transition.done_i2), (transition.player==0)[:,None])[:,:,None], pi_sampled_sub_policy_reg1>0 )  # not-done and player==0 and action-prob>0 (valid action)
                        policy_mask_sv = jnp.logical_and(jnp.logical_and(jnp.logical_not(transition.done2), transition.player==1)[:,None], pi_sv_policy_reg1>0 )                # not-done and player==1 and action-prob>0 (valid action)
                        
                        pi_sub_policy = jnp.where(policy_mask_sub, pi_sub_policy, 0.0)
                        pi_sampled_sub_policy = jnp.where(policy_mask_sub_i, pi_sampled_sub_policy, 0.0)
                        pi_sv_policy = jnp.where(policy_mask_sv, pi_sv_policy, 0.0)


                        # ## Entropy loss
                        # entropy_sub = jnp.sum(-pi_sub_policy * jnp.log(pi_sub_policy+1e-15), where=policy_mask_sub, axis=-1, keepdims=False)
                        # entropy_sv = jnp.sum(-pi_sv_policy * jnp.log(pi_sv_policy+1e-15), where=policy_mask_sv, axis=-1, keepdims=False)

                        # entropy_loss_sub = jnp.sum(-entropy_sub * transition.sub_distribution, axis=-1, keepdims=False)
                        # entropy_loss_sv = jnp.mean(-entropy_sv, axis=-1, keepdims=False)
                        # entropy_loss_sub = jnp.mean(entropy_loss_sub, axis=-1, keepdims=False)

                        # entropy_loss_sub = jnp.where(player0_mask, entropy_loss_sub, 0.0)
                        # entropy_loss_sv = jnp.where(player1_mask, entropy_loss_sv, 0.0)
                        # entropy_loss_sub = jnp.mean(entropy_loss_sub, keepdims=False)
                        # entropy_loss_sv = jnp.mean(entropy_loss_sv, keepdims=False)


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
                        print(f"Compute KL distance between the current policy and the regularized policy")
                        KL_distance_sub_i = jnp.sum(pi_sampled_sub_policy * jnp.log(pi_sampled_sub_policy/ (pi_sampled_sub_policy_reg+1e-15)+1e-15), axis=(-1), where=policy_mask_sub_i)
                        KL_distance_sv = jnp.sum(pi_sv_policy * jnp.log(pi_sv_policy/ (pi_sv_policy_reg+1e-15)+1e-15), axis=(-1), where=policy_mask_sv)
                        KL_distance_sub = jnp.sum(pi_sub_policy * jnp.log(pi_sub_policy/ (pi_sub_policy_reg+1e-15)+1e-15), axis=(-1), where=policy_mask_sub)
                        KL_distance_sub = jnp.sum(KL_distance_sub * transition.sub_distribution, axis=(-1), keepdims=False)

                        # print(f"Mean KL-distance")
                        # KL_distance_sv = jnp.mean(KL_distance_sv, where=player1_mask)
                        # KL_distance_sub_i = jnp.mean(KL_distance_sub_i, where=player0_i_mask)
                        # KL_distance_sub = jnp.mean(KL_distance_sub, where=player0_mask)

                        print(f"KL loss")
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
                        print(f"\n\nmu_sampled_sub_p: {mu_sampled_sub_p.shape}")
                        print(f"pi_sampled_sub_p: {pi_sampled_sub_p.shape}\n\n")

                        v_i = v_map[batch_idx[:,None], sub_pos_samples]
                        print(f"v_i: {v_i.shape}")
                        v_i_traj_batch = transition.v_map[batch_idx[:,None], sub_pos_samples]
                        print(f"v_i_traj_batch: {v_i_traj_batch.shape}")

                        
                        # jax.debug.print("sub_pos_samples: \n{}\n  action_sub_samples: \n{}\n  pi_policy: {}  mu_policy: {}  pi_p: {}  mu_p: {}\n  prior_pi_sampled_sub_policy: \n{}\n  prior_mu_sampled_sub_policy: \n{}\n  player: \n{}", 
                        #                     transition.sub_pos_samples, transition.action_sub_samples, pi_sampled_sub_policy, mu_sampled_sub_policy, pi_sampled_sub_p, mu_sampled_sub_p,
                        #                     prior_pi_sampled_sub_policy, prior_mu_sampled_sub_policy, transition.player)

                        ## Simple policy gradient loss
                        ## loss = log(pi(a^)) * Advantage,  Advantage = R-v
                        print(f"Simple policy gradient loss")
                        pi_sv_policy = jnp.where(policy_mask_sv, pi_sv_policy, 0.0)
                        mu_sv_policy = jnp.where(policy_mask_sv, mu_sv_policy, 0.0)
                        pi_sv_policy_reg = jnp.where(policy_mask_sv, pi_sv_policy_reg, 0.0)
                        pi_sv_p = pi_sv_policy[batch_idx[:], transition.action_dip]
                        mu_sv_p = mu_sv_policy[batch_idx[:], transition.action_dip]
                        pi_sv_p_reg = pi_sv_policy_reg[batch_idx[:], transition.action_dip]
                        pi_sv_p = jnp.where(player1_mask, pi_sv_p, 0.0)
                        mu_sv_p = jnp.where(player1_mask, mu_sv_p, 1.0)
                        pi_sv_p_reg = jnp.where(player1_mask, pi_sv_p_reg, 1.0)


                        print(f"Advantages")
                        Adv_sub_samples = jnp.where(player0_i_mask, R_tTi - v_i, 0.0)
                        Adv_sv = jnp.where(player1_mask ,-(R_tT - v), 0.0 )

                        Adv_sub_samples = jax.lax.stop_gradient(Adv_sub_samples)
                        Adv_sv = jax.lax.stop_gradient(Adv_sv)

                        # #### Simple vanilla policy gradient loss
                        # print(f"Simple vanilla policy gradient loss")
                        # print(f"    player1_mask: {player1_mask.shape}")
                        # print(f"    pi_sv_p: {pi_sv_p.shape}")
                        # print(f"    Adv_sv: {Adv_sv.shape}")
                        # print(f"    player0_i_mask: {player0_i_mask.shape}")
                        # print(f"    pi_sampled_sub_p: {pi_sampled_sub_p.shape}")
                        # print(f"    Adv_sub_samples: {Adv_sub_samples.shape}")
                        # loss_actor_sv = jnp.where( player1_mask ,Adv_sv * jnp.log(pi_sv_p), 0.0 )
                        # loss_actor_sub = jnp.where( player0_i_mask ,Adv_sub_samples * jnp.log(pi_sampled_sub_p), 0.0 )

                        #### PPO-similar policy loss
                        #### Submarine policy loss
                        print(f"PPO-similar policy loss")
                        sub_ratio_samples = jnp.where(player0_i_mask, pi_sampled_sub_p / (mu_sampled_sub_p+1e-14), 0.0)
                        # sub_ratio_samples = jnp.where(player0_i_mask, pi_sampled_sub_p / pi_sampled_sub_p_reg, 0.0)

                        Adv_sub_samples_clip_1 = Adv_sub_samples * sub_ratio_samples
                        print(f"Adv_sub_samples_clip_1: {Adv_sub_samples_clip_1.shape}")
                        Adv_sub_samples_clip_2 = ( jnp.clip(
                                sub_ratio_samples,
                                1.0 - config["CLIP_EPS"],
                                1.0 + config["CLIP_EPS"],
                            ) * Adv_sub_samples)

                        print(f"Adv_sub_samples_clip_2: {Adv_sub_samples_clip_2.shape}")    
                        loss_actor_sub = -jnp.minimum(Adv_sub_samples_clip_1, Adv_sub_samples_clip_2)   # (N, n)
                        loss_actor_sub = jnp.where(player0_i_mask, loss_actor_sub, 0.0)
                        print(f"loss_actor_sub: {loss_actor_sub.shape}")


                        #### Surface Vehicle policy loss
                        print(f"    player1_mask: {player1_mask.shape}")
                        print(f"    pi_sv_p: {pi_sv_p.shape}")
                        print(f"    pi_sv_p_reg: {pi_sv_p_reg.shape}")
                        print(f"    sub_ratio_samples: {sub_ratio_samples.shape}")
                        print(f"    Adv_sv: {Adv_sv.shape}")
                        sv_ratio = jnp.where(player1_mask, pi_sv_p / (mu_sv_p+1e-14), 0.0)
                        # sv_ratio = jnp.where(player1_mask, pi_sv_p / pi_sv_p_reg, 0.0)
                        print(f"    sv_ratio: {sv_ratio.shape}")

                        Adv_sv_clip_1 = Adv_sv * sv_ratio
                        print(f"Adv_sv_clip_1: {Adv_sv_clip_1.shape}")
                        Adv_sv_clip_2 = ( jnp.clip(
                                sv_ratio,
                                1.0 - config["CLIP_EPS"],
                                1.0 + config["CLIP_EPS"],
                            ) * Adv_sv)

                        print(f"Adv_sv_clip_2: {Adv_sv_clip_2.shape}")    
                        loss_actor_sv = -jnp.minimum(Adv_sv_clip_1, Adv_sv_clip_2)   # (N, n)
                        loss_actor_sv = jnp.where(player1_mask, loss_actor_sv, 0.0)
                        print(f"loss_actor_sv: {loss_actor_sv.shape}")


                        ## Entropy loss
                        print(f"Learning\n\n\n pi_sampled_sub_policy: {pi_sampled_sub_policy.shape}")
                        print(f"pi_sv_policy: {pi_sv_policy.shape}")

                        sub_i_entropy = -jnp.sum(pi_sampled_sub_policy * jnp.log(pi_sampled_sub_policy+1e-15),  axis=(-1), where=policy_mask_sub_i)    # (N, n, N_D) -> (N, n)        Sum over D -> (-1)
                        sub_i_entropy_loss_mean = -jnp.mean(sub_i_entropy*transition.p_i_prior, where=player0_i_mask )        # (N, n) -> (,)

                        sv_entropy = -jnp.sum( (jnp.log(pi_sv_policy+1e-14)*pi_sv_policy), where=policy_mask_sv, axis=(-1))      # normal entropy loss: (N, N_D) --> (N)
                        print(f"Learning\n\n\n sv_entropy: {sv_entropy.shape}\n\n\n")
                        sv_entropy_loss_mean = -jnp.mean(sv_entropy*transition.p_prior, where=player1_mask)                         # (N) --> (,)


                        print(f"v: {v.shape}")
                        print(f"v_i: {v_i.shape}")
                        print(f"R_tT: {R_tT.shape}")
                        print(f"R_tTi: {R_tTi.shape}")
                        loss_v = optax.losses.huber_loss( jnp.where( jnp.logical_not(transition.done2), v-R_tT, 0.0) )
                        loss_vi = optax.losses.huber_loss( jnp.where( jnp.logical_not(transition.done_i2),v_i-R_tTi, 0.0) )

                        
                        loss = jnp.mean(loss_v*transition.p_prior, where=jnp.logical_not(transition.done2))
                        loss += jnp.mean(loss_vi*transition.p_i_prior, where=jnp.logical_not(transition.done_i2))
                        loss += jnp.mean(loss_actor_sub*transition.p_i_prior, where=jnp.logical_not(transition.done_i2))
                        loss += jnp.mean(loss_actor_sv*transition.p_prior, where=jnp.logical_not(transition.done2))

                        loss += sub_i_entropy_loss_mean * config['SUB_ENT_LOSS']
                        loss += sv_entropy_loss_mean * config['SV_ENT_LOSS']

                        loss += (loss_kl_sub_i + loss_kl_sv) * alpha_kl

                        print(f"Return losses")
                        # return loss, (loss_v, loss_vi, loss_actor_sub, loss_actor_sv, loss_kl_divergence, pi_sampled_sub_policy, mu_sampled_sub_policy, pi_sampled_sub_p, mu_sampled_sub_p)
                        # return loss, (loss_v, loss_vi, loss_actor_sub, loss_actor_sv, loss_kl_sub, loss_kl_sub_i, loss_kl_sv, pi_sampled_sub_policy, mu_sampled_sub_policy, pi_sampled_sub_p, mu_sampled_sub_p)
                        # return loss, (loss_v, loss_vi, loss_actor_sub, loss_actor_sv, loss_kl_sub, loss_kl_sub_i, loss_kl_sv, jnp.max(KL_distance_sub), jnp.max(KL_distance_sub_i), jnp.max(KL_distance_sv), pi_sampled_sub_policy, mu_sampled_sub_policy, pi_sampled_sub_p, mu_sampled_sub_p)
                        return loss, (loss_v, loss_vi, loss_actor_sub, loss_actor_sv, loss_kl_sub, loss_kl_sub_i, loss_kl_sv, sub_i_entropy_loss_mean, sv_entropy_loss_mean)
                    


                    grad_fn = jax.value_and_grad(_loss_fn_r_nad, has_aux=True)
                    losses, grads = grad_fn(model_params, transition, targets)

                    max_grad_magnitude = tree_util.tree_reduce(
                        lambda x, y: jnp.maximum(x, y),
                        tree_util.tree_map(lambda g: jnp.max(jnp.abs(g)), grads)
                    )
                    print(f"max_grad_magnitude Done")

                    # updates, optimizer_state = optimizer.update(grads, optimizer_state, model_params)

                    updates, optimizer_state = optimizer.update(grads, optimizer_state, model_params)   # Manual Alternative
                    print(f"Got grad updates")
                    model_params = optax.apply_updates(model_params, updates)                           # Manual Alternative
                    print(f"Upgraded params from updates")

                    # linear_fit_params(model_params)
                    params_target = jax.tree_util.tree_map(lambda p2, p1: (1.0 - gamma_averaging) * p2 + gamma_averaging * p1, params_target, model_params)

                    print(f"    _update_minbatch losses: {len(losses)}")
                    return (model_params, params_target, optimizer_state), (losses, max_grad_magnitude)
                ## _update_minbatch


                print(f"(Ps,Pd,Pr, advantages) = targets...")
                rng, _rng = jax.random.split(rng)  
                print(f"(Ps,Pd,Pr, advantages) = targets Done!") 

                batch_size = config_["MINIBATCH_SIZE"] * config_["NUM_MINIBATCHES"]
                assert (
                    batch_size == config_["NUM_STEPS"] * config_["NUM_ENVS"]
                ), f"batch size must be equal to number of steps * number of envs. MB_Size: {config_['MINIBATCH_SIZE']}  N_MB: {config_['NUM_MINIBATCHES']}  NS: {config_['NUM_STEPS']}  NE: {config_['NUM_ENVS']}"
                permutation = jax.random.permutation(_rng, batch_size)
                # batch = (transition, advantages, targets)
                batch = (transition, targets)
                print(f"batch_size: {batch_size}")
                print(f"MINIBATCH_SIZE: {config_['MINIBATCH_SIZE']}")
                print(f"NUM_MINIBATCHES: {config_['NUM_MINIBATCHES']}")
                print(f"NUM_STEPS: {config_['NUM_STEPS']}")
                print(f"NUM_ENVS: {config_['NUM_ENVS']}")
                # print(f"NUM_STEPS: {config_["NUM_STEPS"]}")
                print(f"transition.sub_pos_samples: {transition.sub_pos_samples.shape}")    # Transitions
                print(f"transition.sub_policy: {transition.sub_policy.shape}")

                # print(f"batch: \n{batch}\n\n")
                print(f"transition.player: {transition.player.shape}")
                # print(f"transition.turn: {transition.turn.shape}")


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


                print(f"_update_minbatch...")
                model_opt_sate = (model_params, params_target, optimizer_state)
                model_opt_sate, losses_and_grad = jax.lax.scan(_update_minbatch, model_opt_sate,  minibatches)
                (losses, grad) = losses_and_grad
                print(f"jax.lax.scan(_update_minbatch Done!")
                (model_params, params_target, optimizer_state) = model_opt_sate

                update_state = (model_params, params_target, optimizer_state, transition, targets, rng)

                print(f"    _update_epoch losses: {len(losses)}")
                print(f"    _update_epoch losses: {type(losses[1])}")
                print(f"    _update_epoch losses: {len(losses[1])}")

                # losses_ = (losses[0], *losses[1], grad)
                losses_ = (losses[0], *losses[1])
                

                print(f"_update_epoch Done!")
                return update_state, losses_
            ## _update_epoch
            #### UPDATE NETWORK

            print(f"scan _update_epoch...")
            # (model_params, optimizer_state, env_state_0, obsv, rng) = runner_state
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
            # metric = (losses, transition, targets)

            print(f"    _update_step losses: {len(losses)}")
            metric_losses = losses

            # runner_state = (model_params, optimizer_state, env_state, last_obs, rng)
            runner_state_update_step = (model_params, params_target, optimizer_state, rng, update_step+1)
            return runner_state_update_step, (metric_losses, metric_outcome)
        ## _update_step
        

        # rng, _rng = jax.random.split(rng)
        # obsv, env_state = env.reset(config_["NUM_ENVS"], config_["NUM_SUB_ENVS"], _rng)  # Create multiple environments
        rng, _rng = jax.random.split(rng)

        # current_reg_step = config['current_reg_step']
        # max_reg_step = config['max_reg_step']
        runner_state = (model_params_, params_target_, optimizer_state_, _rng, config['current_reg_step'])
        # (model_params, optimizer_state, env_state_0, obsv, rng, update_step) = runner_state_update_step

        print(f"\n_update_step...")
        # runner_state, metric = jax.lax.scan( _update_step, runner_state, None, config_["NUM_UPDATES"] )
        runner_state, (metric_losses, metric_outcome) = jax.lax.scan( _update_step, runner_state, None, config_["N_JIT_STEPS"] )
        


        return {"runner_state": runner_state, "metric_losses": metric_losses, "metric_outcome": metric_outcome}

    return train