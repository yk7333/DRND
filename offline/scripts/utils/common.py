import gym
import chex
import jax
import pickle
import json
import flax
import numpy as np
import jax.numpy as jnp

from copy import deepcopy
from tqdm.auto import trange
from typing import Sequence, Dict, Callable, Tuple


@chex.dataclass(frozen=True)
class Metrics:
    accumulators: Dict[str, Tuple[jax.Array, jax.Array]]

    @staticmethod
    def create(metrics: Sequence[str]) -> "Metrics":
        init_metrics = {key: (jnp.array([0.0]), jnp.array([0.0])) for key in metrics}
        return Metrics(accumulators=init_metrics)

    def update(self, updates: Dict[str, jax.Array]) -> "Metrics":
        new_accumulators = deepcopy(self.accumulators)
        for key, value in updates.items():
            acc, steps = new_accumulators[key]
            new_accumulators[key] = (acc + value, steps + 1)

        return self.replace(accumulators=new_accumulators)

    def compute(self) -> Dict[str, float]:
        # cumulative_value / total_steps
        return {k: float(v[0] / v[1]) for k, v in self.accumulators.items()}


def normalize(arr: jax.Array, mean: jax.Array, std: jax.Array, eps: float = 1e-8) -> jax.Array:
    return (arr - mean) / (std + eps)


def make_env(env_name: str, seed: int) -> gym.Env:
    env = gym.make(env_name)
    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env


def evaluate(env: gym.Env, action_fn: Callable, num_episodes: int, seed: int) -> np.ndarray:
    env.seed(seed)

    returns = []
    for _ in trange(num_episodes, desc="Eval", leave=False):
    # for _ in range(num_episodes):
        obs, done = env.reset(), False
        total_reward = 0.0
        while not done:
            action = np.asarray(jax.device_get(action_fn(obs)))
            obs, reward, done, _ = env.step(action)
            total_reward += reward
        returns.append(total_reward)

    return np.array(returns)


def save_weights(weights,filename):
    bytes_output=flax.serialization.to_bytes(target=weights)
    pickle.dump(bytes_output,open(filename,"wb"))

def save(nrnd, nrndtrainstate_config, sac_actor, sac_critic, config, path):
    save_weights(nrnd, path + '/nrnd')
    save_weights(sac_actor, path + '/sac_actor')
    save_weights(sac_critic, path + '/sac_critic')
    with open(path + '/nrndtrainstate_config.json', 'w') as f:
        json.dump(nrndtrainstate_config, f, indent=4)

    with open(path + '/config.json', 'w') as f:
        json.dump(config, f, indent=4)



def load(nrnd, sac_actor, sac_critic, config, path):
    nrnd_pkl = pickle.load(open(path + '/nrnd', 'rb'))
    sac_actor_pkl = pickle.load(open(path + '/sac_actor', 'rb'))
    sac_critic_pkl = pickle.load(open(path + '/sac_critic', 'rb'))
    nrndtrainstate_config = json.load(open(path + '/nrndtrainstate_config.json', 'r'))
    config = json.load(open(path + '/config.json', 'r'))
    raw_config = {
        'nrnd': flax.serialization.from_bytes(target=nrnd, bytes=nrnd_pkl),
        'nrndtrainstate_config': nrndtrainstate_config,
        'sac_actor': flax.serialization.from_bytes(target=sac_actor, bytes=sac_actor_pkl),
        'sac_critic': flax.serialization.from_bytes(target=sac_critic, bytes=sac_critic_pkl),
        'config': config
    }
    return raw_config