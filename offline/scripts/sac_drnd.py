import os
import uuid
import pyrallis
from pprint import pprint
import time
import json

import jax
import numpy as np
import jax.numpy as jnp
import optax

from functools import partial
from dataclasses import dataclass, asdict
from flax.core import FrozenDict
from typing import Dict, Tuple, Any, Optional, Callable
from tqdm.auto import trange

from flax.training.train_state import TrainState
from flax.training import checkpoints
from networks import DRND, Actor, EnsembleCritic, Alpha
from utils.buffer import ReplayBuffer
from utils.common import Metrics, make_env, evaluate
from utils.running_moments import RunningMeanStd

@dataclass
class Config:
    # model params
    actor_learning_rate: float = 1e-4
    critic_learning_rate: float = 1e-4
    alpha_learning_rate: float = 1e-4
    hidden_dim: int = 256
    gamma: float = 0.99
    tau: float = 5e-3
    actor_lambda: float = 1.0
    critic_lambda: float = 1.0
    num_critics: int = 2
    critic_layernorm: bool = True
    # drnd params
    drnd_learning_rate: float = 1e-6
    drnd_hidden_dim: int = 256
    drnd_embedding_dim: int = 32
    drnd_mlp_type: str = "concat_first"
    drnd_target_mlp_type: Optional[str] = None
    drnd_switch_features: bool = True
    drnd_update_epochs: int = 100
    num_target: int = 10
    drnd_alpha: float = 0.9
    use_norm: bool = False
    # training params
    dataset_name: str = "halfcheetah-medium-v2"
    batch_size: int = 256
    num_epochs: int = 3000
    num_updates_on_epoch: int = 1000
    normalize_reward: bool = False
    # evaluation params
    eval_episodes: int = 10
    eval_every: int = 100
    # general params
    train_seed: int = 10
    eval_seed: int = 42
    random_eval_seed: bool = True
    
    save_path: str = f"data/{dataset_name}"


class DRNDTrainState(TrainState):
    rms1: RunningMeanStd
    rms2: RunningMeanStd
    alpha: float


class CriticTrainState(TrainState):
    target_params: FrozenDict

def restore_checkpoint(update_carry, save_path):
    _update_carry = {
        "actor": update_carry["actor"],
        "critic": update_carry["critic"],
        "alpha": update_carry["alpha"],
    }
    _update_carry = checkpoints.restore_checkpoint(ckpt_dir=os.path.join(save_path, 'actor_critic_alpha'), target=_update_carry)
    
    # load drnd json
    with open(os.path.join(save_path, 'drnd_json.json'), 'r') as f:
        drnd_json = json.load(f)
    tmp_drnd = checkpoints.restore_checkpoint(ckpt_dir=os.path.join(save_path, 'drnd'), target=TrainState.create(
        apply_fn=update_carry["drnd"].apply_fn,
        params=update_carry["drnd"].params,
        tx=update_carry["drnd"].tx,
    ))
    drnd = DRNDTrainState.create(
        apply_fn=tmp_drnd.apply_fn,
        params=tmp_drnd.params,
        tx=tmp_drnd.tx,
        rms1=RunningMeanStd.create(mean=drnd_json['rms1']['mean'][0], var=drnd_json['rms1']['var'][0], count=drnd_json['rms1']['count'][0]),
        rms2=RunningMeanStd.create(mean=drnd_json['rms2']['mean'][0], var=drnd_json['rms2']['var'][0], count=drnd_json['rms2']['count'][0]),
        alpha=jnp.array(drnd_json['alpha']),
    )
    update_carry.update({
        "actor": _update_carry["actor"],
        "critic": _update_carry["critic"],
        "alpha": _update_carry["alpha"],
        "drnd": drnd
    })
    return update_carry



def save_checkpoint(update_carry, save_path):
    _update_carry = {
        "actor": update_carry["actor"],
        "critic": update_carry["critic"],
        "alpha": update_carry["alpha"],
    }
    _save_checkpoint(_update_carry, os.path.join(save_path, 'actor_critic_alpha'))
    # create a new trainstate tmp_drnd for save, use params from drnd without rms
    tmp_drnd = TrainState.create(
        apply_fn=update_carry["drnd"].apply_fn,
        params=update_carry["drnd"].params,
        tx=update_carry["drnd"].tx,
    )
    _save_checkpoint(tmp_drnd, os.path.join(save_path, 'drnd'))
    # generate a json to save rms and alpha in drnd
    drnd_json = {
        "rms1": jax.tree_map(lambda x: jax.device_get(x).tolist(), update_carry["drnd"].rms1.state),
        "rms2": jax.tree_map(lambda x: jax.device_get(x).tolist(), update_carry["drnd"].rms2.state),
        "alpha": jax.device_get(update_carry["drnd"].alpha).tolist(),
    }
    with open(os.path.join(save_path, 'drnd_json.json'), 'w') as f:
        json.dump(drnd_json, f, indent=4)


def _save_checkpoint(ckpt, save_path):
    checkpoints.save_checkpoint(ckpt_dir=save_path,
                            target=ckpt,
                            step=0,
                            overwrite=True,
                            keep=2)


# DRND functions
def drnd_bonus(
        drnd: DRNDTrainState,
        state: jax.Array,
        action: jax.Array,
) -> jax.Array:
    preds, targets = drnd.apply_fn(drnd.params, state, action)
    targets = jnp.transpose(targets, (1, 0, 2))
    mu = jnp.mean(targets, axis=1) 
    bonus1 = jnp.sum((preds - mu) ** 2, axis=1) / drnd.rms1.std
    if Config.use_norm:
        std = jnp.std(targets, axis=1)
        bonus2 = jnp.sqrt(jnp.mean(((preds - mu)/std) ** 2, axis=1))
    else:
        B2 = jnp.mean(targets ** 2, axis=1)
        bonus2 = jnp.mean(jnp.sqrt(jnp.clip(abs(preds ** 2 - mu ** 2) / (B2 - mu ** 2),1e-3,1)), axis=1)
    return drnd.alpha * bonus1 + (1 - drnd.alpha) * bonus2


def update_drnd(
        key: jax.random.PRNGKey,
        drnd: DRNDTrainState,
        batch: Dict[str, jax.Array],
        metrics: Metrics
) -> Tuple[jax.random.PRNGKey, DRNDTrainState, Metrics]:
    def drnd_loss_fn(params, idx):
        pred, targets = drnd.apply_fn(params, batch["states"], batch["actions"])
        targets = jnp.transpose(targets, (1, 0, 2))
        target = targets[jnp.arange(targets.shape[0]), idx, :]
        raw_loss = ((pred - target) ** 2).sum(axis=1)  # in rnd, loss = bonus. In DRND, loss≠bonus
        mu = jnp.mean(targets, axis=1) 
        # bonus1
        bonus1 = jnp.sum((pred - mu) ** 2, axis=1) 
        new_rms1 = drnd.rms1.update(bonus1)
        # bonus2
        if Config.use_norm:
            std = jnp.std(targets, axis=1)
            bonus2 = jnp.mean(((pred - mu)/std) ** 2, axis=1)  #equal to the following expression
        else:
            B2 = jnp.mean(targets ** 2, axis=1)
            bonus2 = jnp.sqrt(jnp.mean(jnp.clip(abs(pred ** 2 - mu ** 2) / (B2 - mu ** 2),1e-3,1), axis=1))
        new_rms2 = drnd.rms2.update(bonus2)
        loss = raw_loss.mean(axis=0)
        return loss, (new_rms1, new_rms2,bonus1.mean(axis=0),bonus2.mean(axis=0))
    key, actions_key, idx_key = jax.random.split(key, 3)
    idx = jax.random.randint(idx_key, shape=(batch["states"].shape[0],), minval=0, maxval=Config.num_target)
    (loss, (new_rms1, new_rms2, bonus1, bonus2)), grads = jax.value_and_grad(drnd_loss_fn, has_aux=True)(drnd.params, idx)
    new_drnd = drnd.apply_gradients(grads=grads).replace(rms1=new_rms1, rms2=new_rms2)

    # log drnd bonus for random actions
    random_actions = jax.random.uniform(actions_key, shape=batch["actions"].shape, minval=-1.0, maxval=1.0)
    new_metrics = metrics.update({
        "drnd_loss": loss,
        "drnd_bonus_total":bonus1/ drnd.rms1.std*drnd.alpha+bonus2/ drnd.rms2.std*(1-drnd.alpha),
        "drnd_bonus1": bonus1/ drnd.rms1.std*drnd.alpha,
        "drnd_bonus2": bonus2/ drnd.rms2.std*(1-drnd.alpha),
        "drnd_std1": new_drnd.rms1.std,
        "drnd_std2": new_drnd.rms2.std,
    })
    return key, new_drnd, new_metrics


def update_actor(
        key: jax.random.PRNGKey,
        actor: TrainState,
        drnd: DRNDTrainState,
        critic: TrainState,
        alpha: TrainState,
        batch: Dict[str, jax.Array],
        actor_lambda: float,
        metrics: Metrics
) -> Tuple[jax.random.PRNGKey, TrainState, jax.Array, Metrics]:
    key, actions_key, random_action_key = jax.random.split(key, 3)

    def actor_loss_fn(params):
        actions_dist = actor.apply_fn(params, batch["states"])
        actions, actions_logp = actions_dist.sample_and_log_prob(seed=actions_key)

        drnd_penalty = drnd_bonus(drnd, batch["states"], actions)
        q_values = critic.apply_fn(critic.params, batch["states"], actions).min(0)
        loss = (alpha.apply_fn(alpha.params) * actions_logp.sum(-1) + actor_lambda * drnd_penalty - q_values).mean()

        # logging stuff
        actor_entropy = -actions_logp.sum(-1).mean()
        random_actions = jax.random.uniform(random_action_key, shape=batch["actions"].shape, minval=-1.0, maxval=1.0)
        new_metrics = metrics.update({
            "batch_entropy": actor_entropy,
            "actor_loss": loss,
            "drnd_policy": drnd_penalty.mean(),
            "drnd_random": drnd_bonus(drnd, batch["states"], random_actions).mean(),
            "action_mse": ((actions - batch["actions"]) ** 2).mean()
        })
        return loss, (actor_entropy, new_metrics)

    grads, (actor_entropy, new_metrics) = jax.grad(actor_loss_fn, has_aux=True)(actor.params)
    new_actor = actor.apply_gradients(grads=grads)

    return key, new_actor, actor_entropy, new_metrics


def update_alpha(
        alpha: TrainState,
        entropy: jax.Array,
        target_entropy: float,
        metrics: Metrics
) -> Tuple[TrainState, Metrics]:
    def alpha_loss_fn(params):
        alpha_value = alpha.apply_fn(params)
        loss = (alpha_value * (entropy - target_entropy)).mean()

        new_metrics = metrics.update({
            "alpha": alpha_value,
            "alpha_loss": loss
        })
        return loss, new_metrics

    grads, new_metrics = jax.grad(alpha_loss_fn, has_aux=True)(alpha.params)
    new_alpha = alpha.apply_gradients(grads=grads)

    return new_alpha, new_metrics


def update_critic(
        key: jax.random.PRNGKey,
        actor: TrainState,
        drnd: DRNDTrainState,
        critic: CriticTrainState,
        alpha: TrainState,
        batch: Dict[str, jax.Array],
        gamma: float,
        critic_lambda: float,
        tau: float,
        metrics: Metrics
) -> Tuple[jax.random.PRNGKey, TrainState, Metrics]:
    key, actions_key = jax.random.split(key, 2)

    next_actions_dist = actor.apply_fn(actor.params, batch["next_states"])
    next_actions, next_actions_logp = next_actions_dist.sample_and_log_prob(seed=actions_key)
    drnd_penalty = drnd_bonus(drnd, batch["next_states"], next_actions)

    next_q = critic.apply_fn(critic.target_params, batch["next_states"], next_actions).min(0)
    next_q = next_q - alpha.apply_fn(alpha.params) * next_actions_logp.sum(-1) - critic_lambda * drnd_penalty

    target_q = batch["rewards"] + (1 - batch["dones"]) * gamma * next_q

    def critic_loss_fn(critic_params):
        # [N, batch_size] - [1, batch_size]
        q = critic.apply_fn(critic_params, batch["states"], batch["actions"])
        q_min = q.min(0).mean()
        loss = ((q - target_q[None, ...]) ** 2).mean(1).sum(0)
        return loss, q_min

    (loss, q_min), grads = jax.value_and_grad(critic_loss_fn, has_aux=True)(critic.params)
    new_critic = critic.apply_gradients(grads=grads)
    new_critic = new_critic.replace(
        target_params=optax.incremental_update(new_critic.params, new_critic.target_params, tau)
    )
    new_metrics = metrics.update({
        "critic_loss": loss,
        "q_min": q_min,
    })
    return key, new_critic, new_metrics


def update_sac(
        key: jax.random.PRNGKey,
        drnd: DRNDTrainState,
        actor: TrainState,
        critic: CriticTrainState,
        alpha: TrainState,
        batch: Dict[str, Any],
        target_entropy: float,
        gamma: float,
        actor_lambda: float,
        critic_lambda: float,
        tau: float,
        metrics: Metrics,
):
    key, new_actor, actor_entropy, new_metrics = update_actor(key, actor, drnd, critic, alpha, batch, actor_lambda, metrics)
    new_alpha, new_metrics = update_alpha(alpha, actor_entropy, target_entropy, new_metrics)
    key, new_critic, new_metrics = update_critic(
        key, new_actor, drnd, critic, alpha, batch, gamma, critic_lambda, tau, new_metrics
    )
    return key, new_actor, new_critic, new_alpha, new_metrics


def action_fn(actor: TrainState) -> Callable:
    @jax.jit
    def _action_fn(obs: jax.Array) -> jax.Array:
        dist = actor.apply_fn(actor.params, obs)
        action = dist.mean()
        return action
    return _action_fn


@pyrallis.wrap()
def main(config: Config):
    save_path = config.save_path+'/'+str(time.time())+'/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # save config with json
    with open(save_path + 'config.json', 'w') as f:
        json.dump(asdict(config), f, indent=4)
    buffer = ReplayBuffer.create_from_d4rl(config.dataset_name, config.normalize_reward)
    state_mean, state_std = buffer.get_moments("states")
    action_mean, action_std = buffer.get_moments("actions")

    key = jax.random.PRNGKey(seed=config.train_seed)
    key, drnd_key, actor_key, critic_key, alpha_key, idx_key = jax.random.split(key, 6)

    init_state = buffer.data["states"][0][None, ...]
    init_action = buffer.data["actions"][0][None, ...]
    target_entropy = -init_action.shape[-1]

    drnd_module = DRND(
        hidden_dim=config.drnd_hidden_dim,
        embedding_dim=config.drnd_embedding_dim,
        state_mean=state_mean,
        state_std=state_std,
        action_mean=action_mean,
        action_std=action_std,
        mlp_type=config.drnd_mlp_type,
        target_mlp_type=config.drnd_target_mlp_type,
        switch_features=config.drnd_switch_features,
        num_target=config.num_target,
    )
    drnd = DRNDTrainState.create(
        apply_fn=drnd_module.apply,
        params=drnd_module.init(drnd_key, init_state, init_action),
        tx=optax.adam(learning_rate=config.drnd_learning_rate),
        rms1=RunningMeanStd.create(),
        rms2=RunningMeanStd.create(),
        alpha=config.drnd_alpha,
    )
    actor_module = Actor(action_dim=init_action.shape[-1], hidden_dim=config.hidden_dim)
    actor = TrainState.create(
        apply_fn=actor_module.apply,
        params=actor_module.init(actor_key, init_state),
        tx=optax.adam(learning_rate=config.actor_learning_rate),
    )
    alpha_module = Alpha()
    alpha = TrainState.create(
        apply_fn=alpha_module.apply,
        params=alpha_module.init(alpha_key),
        tx=optax.adam(learning_rate=config.alpha_learning_rate)
    )
    critic_module = EnsembleCritic(
        hidden_dim=config.hidden_dim, num_critics=config.num_critics, layernorm=config.critic_layernorm
    )
    critic = CriticTrainState.create(
        apply_fn=critic_module.apply,
        params=critic_module.init(critic_key, init_state, init_action),
        target_params=critic_module.init(critic_key, init_state, init_action),
        tx=optax.adam(learning_rate=config.critic_learning_rate),
    )

    update_sac_partial = partial(
        update_sac, target_entropy=target_entropy, gamma=config.gamma, actor_lambda=config.actor_lambda,critic_lambda=config.critic_lambda, tau=config.tau
    )

    def drnd_loop_update_step(i, carry):
        key, batch_key = jax.random.split(carry["key"])
        batch = carry["buffer"].sample_batch(batch_key, batch_size=config.batch_size)

        key, new_drnd, new_metrics = update_drnd(key, carry["drnd"], batch, carry["metrics"])
        carry.update(
            key=key, drnd=new_drnd, metrics=new_metrics
        )
        return carry

    def sac_loop_update_step(i, carry):
        key, batch_key = jax.random.split(carry["key"])
        batch = carry["buffer"].sample_batch(batch_key, batch_size=config.batch_size)

        key, new_actor, new_critic, new_alpha, new_metrics = update_sac_partial(
            key=key,
            drnd=carry["drnd"],
            actor=carry["actor"],
            critic=carry["critic"],
            alpha=carry["alpha"],
            batch=batch,
            metrics=carry["metrics"]
        )
        carry.update(
            key=key, actor=new_actor, critic=new_critic, alpha=new_alpha, metrics=new_metrics
        )
        return carry

    # metrics
    drnd_metrics_to_log = [
        "drnd_loss", "drnd_bonus_total",  "drnd_bonus1","drnd_bonus2", "drnd_std1", "drnd_std2"
    ]
    bc_metrics_to_log = [
        "critic_loss", "q_min", "actor_loss", "batch_entropy",
        "drnd_policy", "drnd_random", "action_mse", "alpha_loss", "alpha"
    ]
    # shared carry for update loops
    update_carry = {
        "key": key,
        "actor": actor,
        "drnd": drnd,
        "critic": critic,
        "alpha": alpha,
        "buffer": buffer,
    }
    # PRETRAIN DRND
    for epoch in range(config.drnd_update_epochs):
        print("drnd epoch: ", epoch)
        with open(save_path+'log.json', "a") as f:
            f.write(f"drnd epoch: {epoch}" + "\n")
        update_carry["metrics"] = Metrics.create(drnd_metrics_to_log)
        update_carry = jax.lax.fori_loop(
            lower=0,
            upper=config.num_updates_on_epoch,
            body_fun=drnd_loop_update_step,
            init_val=update_carry
        )
        # log mean over epoch for each metric
        mean_metrics = update_carry["metrics"].compute()
        pprint(
            {f"DRND/{k}": v for k, v in mean_metrics.items()}
        )
        output_str = json.dumps({f"DRND/{k}": v for k, v in mean_metrics.items()}, indent=4)
        with open(save_path+'log.json', "a") as f:
            f.write(output_str + "\n")
    eval_max = -1000.0
    eval_last = -1000.0
    # TRAIN
    for epoch in range(config.num_epochs):
        print("epoch: ", epoch)
        with open(save_path+'log.json', "a") as f:
            f.write(f"epoch: {epoch}" + "\n")
        update_carry["metrics"] = Metrics.create(bc_metrics_to_log)
        update_carry = jax.lax.fori_loop(
            lower=0,
            upper=config.num_updates_on_epoch,
            body_fun=sac_loop_update_step,
            init_val=update_carry
        )
        # log mean over epoch for each metric
        mean_metrics = update_carry["metrics"].compute()
        if epoch % config.eval_every == 0 or epoch == config.num_epochs - 1:
            s = config.eval_seed if not config.random_eval_seed else np.random.randint(0, 100000)
            eval_env = make_env(config.dataset_name, seed=s)
            actor_action_fn = action_fn(actor=update_carry["actor"])

            eval_returns = evaluate(eval_env, actor_action_fn, config.eval_episodes, seed=s)
            normalized_score = eval_env.get_normalized_score(eval_returns) * 100.0
            pprint({
                    "eval/return_mean": np.mean(eval_returns),
                    "eval/return_std": np.std(eval_returns),
                    "eval/normalized_score_mean": np.mean(normalized_score),
                    "eval/normalized_score_std": np.std(normalized_score)
                })
            output_str = json.dumps({
                    "eval/return_mean": np.mean(eval_returns),
                    "eval/return_std": np.std(eval_returns),
                    "eval/normalized_score_mean": np.mean(normalized_score),
                    "eval/normalized_score_std": np.std(normalized_score)
                }, indent=4)
            with open(save_path+'log.json', "a") as f:
                f.write(output_str + "\n")
            if np.mean(normalized_score) > eval_max:
                eval_max = np.mean(normalized_score)
                save_checkpoint(update_carry, os.path.join(save_path, 'max_checkpoint'))
            eval_last = np.mean(normalized_score)
    os.mkdir(os.path.join(save_path, str(eval_max)+"_eval_max"))
    os.mkdir(os.path.join(save_path, str(eval_last)+"_eval_last"))
    save_checkpoint(update_carry, os.path.join(save_path, 'last_checkpoint'))


if __name__ == "__main__":
    main()
