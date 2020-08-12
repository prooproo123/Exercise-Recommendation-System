import numpy as np
from rllab.misc import tensor_utils
import time

import global_vars

def rollout(env, agent, max_path_length=np.inf, animated=False, speedup=1,
            always_return_paths=False):
    o = env.reset()
    observations = []
    actions = []
    rewards = []
    agent_infos = []
    env_infos = []
    agent.reset()
    path_length = 0
    if animated:
        env.render()

    while path_length < max_path_length:
        a, agent_info = agent.get_action(o)
        next_o, r, d, env_info = env.step(a, global_vars.get_current_student())
        observations.append(env.observation_space[global_vars.get_current_student()].flatten(o))
        rewards.append(r)
        actions.append(env.action_space[global_vars.get_current_student()].flatten(a))
        agent_infos.append(agent_info)
        env_infos.append(env_info)
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
