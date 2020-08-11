import numpy as np
from rllab.misc import tensor_utils
import time


def rollout(env, agent, max_path_length=np.inf, animated=False, speedup=1,
            always_return_paths=False):
    o = env.reset()
    observations = [[] for i in range(len(o))]
    actions = [[] for i in range(len(o))]
    rewards = [[] for i in range(len(o))]
    agent_infos = [[] for i in range(len(o))]
    env_infos = [[] for i in range(len(o))]
    agent.reset()
    path_length = 0
    if animated:
        env.render()

    while path_length < max_path_length:
        for i in range(len(o)):
            a, agent_info = agent.get_action(o[i], i)
            next_o, r, d, env_info = env.step(a, i)
            observations[i].append(env.observation_space[i].flatten(o))
            rewards[i].append(r)
            actions[i].append(env.action_space[i].flatten(a))
            agent_infos[i].append(agent_info)
            env_infos[i].append(env_info)
            if d:
                break
            o = next_o
            if animated:
                env.render()
                timestep = 0.05
                time.sleep(timestep / speedup)
        path_length += 1

    if animated and not always_return_paths:
        return

    return dict(
        observations=tensor_utils.stack_tensor_list(observations),
        actions=tensor_utils.stack_tensor_list(actions),
        rewards=tensor_utils.stack_tensor_list(rewards),
        agent_infos=tensor_utils.stack_tensor_dict_list(agent_infos),
        env_infos=tensor_utils.stack_tensor_dict_list(env_infos),
    )
