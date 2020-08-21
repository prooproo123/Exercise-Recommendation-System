from __future__ import division, division

import copy
import pickle
import sys
import types

import numpy as np
from gym import spaces

from exercise_recommendation.envs import DKVEnv
from exercise_recommendation.tutors import RLTutor


def make_rl_student_env(env):
    """

    Args:
        env:

    Returns:

    """
    env = copy.deepcopy(env)

    env.n_item_feats = int(np.log(2 * env.n_items))

    env.item_feats = np.random.normal(
        np.zeros(2 * env.n_items * env.n_item_feats),
        np.ones(2 * env.n_items * env.n_item_feats)).reshape((2 * env.n_items, env.n_item_feats))

    env.observation_space = spaces.Box(
        np.concatenate((np.ones(env.n_item_feats) * -sys.maxsize, np.zeros(1))),
        np.concatenate((np.ones(env.n_item_feats) * sys.maxsize, np.ones(1)))
    )

    def encode_item(self, item, outcome):
        return self.item_feats[self.n_items * outcome + item, :]

    def vectorize_obs(self, item, outcome):
        return np.concatenate((self.encode_item(item, outcome), np.array([outcome])))
        # return self.encode_item(item, outcome)

    env._obs_orig = env._obs

    def _obs(self):
        item, outcome = env._obs_orig()

        return self.vectorize_obs(item, outcome)

    env.encode_item = types.MethodType(encode_item, env)
    env.vectorize_obs = types.MethodType(vectorize_obs, env)
    env._obs = types.MethodType(_obs, env)

    return env


def run_ep(agent, env):
    """

    Args:
        agent:
        env:

    Returns:

    """
    agent.reset()
    obs = env.reset()
    done = False
    totalr = []
    observations = []
    print('run_ep')
    while not done:
        action = agent.act(obs)
        obs, r, done, _ = env.step(action)
        agent.learn(r)
        totalr.append(r)
        observations.append(obs)
    return np.mean(totalr), observations


def all_reset(agent):
    """
    Reset policy and student model for recommendation (when the agent recommends to a new student)
    """
    agent.raw_policy.env.env.recomreset()
    agent.raw_policy.policy.reset()

    return agent


def simulation(agent, trace, steps):
    """
    Simulate the recommendation given the student history exercise trace
    :param agent: recommendation policy
    :param trace: student history exercise trace
    :param steps: the number of exercises recommended to the student
    :return: recommended exercises and his predicted knowledge status
    """
    recom_trace = []
    a2i = dict(zip(candidate_exercises, range(len(candidate_exercises))))
    trace = [(a2i[i[0]], i[1]) for i in trace]
    for q, a in trace:
        obs = agent.raw_policy.env.env.vectorize_obs(q, a)
        recomq = agent.guide(obs)

    res = []
    for t in range(steps):
        prob = agent.raw_policy.env.env.predict(candidate_exercises[recomq])
        answer = 1 if np.random.random() < prob else 0

        # obs = agent.raw_policy.env.env.vectorize_obs(recomq, answer)
        recom_trace.append((recomq, answer))
        obs = agent.raw_policy.env.env.actualStep(recomq, answer)
        res.append(np.mean(list(map(agent.raw_policy.env.env.predict, candidate_exercises))))
        recomq = agent.guide(obs)

    return recom_trace, res


def evaluation(agent):
    """
    Evaluate the policy when it recommend exercises to different student
    student_traces:[[(923, 1), (175, 0), (1010, 1), (857, 0), (447, 0)], [........], [.........]]
    :param agent:
    :return: different students'predicted knowledge status
    """
    # with open('./好未来数据/student_traces.', 'rb') as f:
    #     student_traces = pickle.load(f)
    allre = [[] for i in range(50)]
    for trace in student_traces:
        agent = all_reset(agent)
        t, res = simulation(agent, trace, 50)
        for j in range(50):
            allre[j].append(res[j])
    result = [np.mean(k) for k in allre]
    return result


def run_eps(agent, env, n_eps=100):
    tot_rew = []
    for i in range(n_eps):
        totalr, _ = run_ep(agent, env)
        tot_rew.append(totalr)
    return tot_rew


student_traces = [[(1, 0), (3, 1)], [(6, 1), (6, 0), (7, 1)]]
# 18
# 1,1,2,2,2,2,1,1,1,1,1,1,1,1,1,3,3,3
# 0,0,0,1,1,1,1,0,0,1,1,1,1,1,1,1,1,1
# 162
# 6,6,7,3,3,3,3,60,33,32,32,32,30,30,5,5,5,5,38,38,38,5,5,30,30,30,33,33,33,33,33,33,40,40,60,60,60,68,68,68,68,40,40,40,5,5,6,6,6,7,7,88,88,88,2,2,2,2,68,68,68,1,1,1,1,1,1,1,1,24,24,24,24,20,20,20,63,32,32,32,30,40,33,33,33,5,5,5,88,88,88,1,40,38,38,38,30,30,30,40,40,40,60,60,60,60,60,60,6,6,6,6,7,7,7,7,7,7,7,68,68,68,2,2,2,2,1,1,1,63,63,63,37,46,46,46,46,46,46,46,46,72,72,72,77,77,77,77,75,75,75,75,1,1,1,1,71,71,71,18,18,18
# 1,0,1,0,1,1,1,0,0,1,1,1,1,0,0,1,1,0,1,1,1,0,1,1,1,1,1,1,0,1,1,1,1,0,1,1,1,1,0,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,0,0,0,0,0,1,1,1,0,1,1,1,1,1,1,0,1,1,1,0,0,1,1,1,1,1,1,1,1,1,0,0,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,0,1,1,1,0,0,1,0,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,0,1,1,0,1,1,1,1,1,1,0,1,1,1,0,1,1,1,0,1,1,1,1,1,1,0,1,0

# candidate_exercises=[]
with open('data/skill_builder/old_cand_ex.pkl', 'rb') as f:
    candidate_exercises = pickle.load(f)

Concepts = 10  # number of concepts
NumQ = 100#number of exercises
n_steps = 5#number of steps of algorithm
n_items = len(candidate_exercises)#number of candidate exercises
# n_items = [len(candidate_exercises[i]) for i in candidate_exercises]
discount = 0.99
n_eps = 1#number of epochs in algorithm

# test = arms
# Concepts = 188
# NumQ = 1982
# n_steps = 30
# n_items = len(arms)
# discount = 0.99
# n_eps = 200

reward_funcs = ['likelihood']
envs = [
    ('DKVMN', DKVEnv)
]

tutor_builders = [
    ('RL', RLTutor)
]

env_kwargs = {
    'n_items': n_items, 'n_steps': n_steps, 'discount': discount
}

env = DKVEnv(**env_kwargs, reward_func='likelihood')
rl_env = make_rl_student_env(env)
agent = RLTutor(n_items)
reward = agent.train(rl_env, n_eps=n_eps)
print(evaluation(agent))
print('ok')
