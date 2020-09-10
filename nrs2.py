# coding: utf-8

from __future__ import division

import copy
import sys
import types

from exercise_recommendation.envs import *
from exercise_recommendation.policies import *
from exercise_recommendation.tutors import *
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.policies.categorical_gru_policy import CategoricalGRUPolicy


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

        print("Preporuceni put: " + str(t))
        logging.info("Preporuceni put: " + str(t))
        for j in range(50):
            allre[j].append(res[j])
    result = [np.mean(k) for k in allre]
    logging.info(result)
    return result


# def run_eps(agent, env, n_eps=100):
#     tot_rew = []
#     for i in range(n_eps):
#         totalr, _ = run_ep(agent, env)
#         tot_rew.append(totalr)
#     return tot_rew

import logging

logging.basicConfig(filename="rs_logs.txt",
                level=logging.DEBUG,
                format='%(levelname)s: %(asctime)s %(message)s',
                datefmt='%m/%d/%Y %I:%M:%S')



# student_traces = [[(1, 0), (3, 1)], [(6, 1), (6, 0), (7, 1)]]
# stu = [[(51424, 0), (51435, 1),(51444, 1)]]
stu = [[(85829, 0), (85838, 1)]]

# the parameters of trained DKVMN-CA model
with open('old/checkpoint/skill_builder0_10batch_2epochs/kt_params', 'rb') as f:
    params = pickle.load(f)

# Knowledge Concepts Corresponding to the exercise
with open('data/skill_builder/chunk_exercise_concepts_mapping.pkl', 'rb') as f:
    e2c = pickle.load(f)

with open('data/skill_builder/chunk_exercises_id_converter.pkl', 'rb') as f:
    exercises_id_converter = pickle.load(f)

# cands=[51424,51435,51444,51395,51481]
cands = [85829, 61089, 85814, 85838]

candidate_exercises = [exercises_id_converter[e] for e in cands]
student_traces = [[(exercises_id_converter[e], a) for e, a in t] for t in stu]


# current problems:
# key error?

Concepts = 9  # number of concepts
NumQ = 2446  # number of exercises
# Concepts = 123  # number of concepts
# NumQ = 17751 # number of exercises
n_steps = 50  # number of steps of algorithm
n_items = len(candidate_exercises)  # number of candidate exercises
# n_items = [len(candidate_exercises[i]) for i in candidate_exercises]
discount = 0.99
n_eps = 5  # number of epochs in algorithm

reward_funcs = ['likelihood']
envs = [
    ('DKVMN', DKVEnv)
]

tutor_builders = [
    ('RL', RLTutor)
]

env_kwargs = {
    'n_items': n_items, 'n_steps': n_steps, 'discount': discount, 'num_questions': NumQ, 'num_concepts': Concepts,
    'candidate_exercises': candidate_exercises
}

logging.info("")
logging.info("Broj vjezbi kandidata: "+str(len(candidate_exercises)))
logging.info("Broj epoha: "+str(n_eps))
logging.info("Broj koraka: "+str(n_steps))


env = DKVEnv(**env_kwargs, reward_func='likelihood')

rl_env = MyGymEnv(make_rl_student_env(env))

policy = CategoricalGRUPolicy(
    env_spec=rl_env.spec, hidden_dim=32,
    state_include_action=False)
raw_policy = LoggedTRPO(
    env=rl_env,
    policy=policy,
    baseline=LinearFeatureBaseline(env_spec=rl_env.spec),
    batch_size=4000,
    max_path_length=rl_env.env.n_steps,
    n_itr=n_eps,
    discount=0.99,
    step_size=0.01,
    verbose=False
)

agent = RLTutor(rl_env=rl_env, raw_policy=raw_policy)

reward = agent.train()
print(evaluation(agent))
