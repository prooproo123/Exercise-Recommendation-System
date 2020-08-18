from __future__ import division

from exercise_recommendation.envs import MyGymEnv
from exercise_recommendation.policies import LoggedTRPO
from new_rs import env, make_rl_student_env, run_eps
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.policies.categorical_gru_policy import CategoricalGRUPolicy


class Tutor(object):

    def __init__(self):
        pass

    def _next_item(self):
        raise NotImplementedError

    def _update(self, item, outcome, timestamp, delay):
        raise NotImplementedError

    def act(self, obs):
        self._update(*list(obs))
        return self._next_item()

    def learn(self, r):
        pass

    def train(self, env, n_eps=10):
        return run_eps(self, env, n_eps=n_eps)

    def reset(self):
        raise NotImplementedError


class DummyTutor(Tutor):

    def __init__(self, policy):
        self.policy = policy

    def act(self, obs):
        return self.policy(obs)

    def reset(self):
        pass


class RLTutor(Tutor):

    def __init__(self, n_items, init_timestamp=0):
        self.raw_policy = None
        self.curr_obs = None
        self.rl_env = MyGymEnv(make_rl_student_env(env))

    def train(self, gym_env, n_eps=10):
        env = MyGymEnv(gym_env)
        policy = CategoricalGRUPolicy(
            env_spec=env.spec, hidden_dim=32,
            state_include_action=False)
        self.raw_policy = LoggedTRPO(
            env=env,
            policy=policy,
            baseline=LinearFeatureBaseline(env_spec=env.spec),
            batch_size=4000,
            max_path_length=env.env.n_steps,
            n_itr=n_eps,
            discount=0.99,
            step_size=0.01,
            verbose=False
        )
        self.raw_policy.train()
        return self.raw_policy.rew_chkpts

    def guide(self, obs):
        agents = DummyTutor(lambda obs: self.raw_policy.policy.get_action(obs)[0])
        action = agents.act(obs)
        return action

    def getObs(self, action, answer):
        obs = self.raw_policy.env.env.actualStep(action, answer)
        return obs

    def reset(self):
        self.curr_obs = None
        self.raw_policy.reset()

    def _next_item(self):
        if self.curr_obs is None:
            raise ValueError
        return self.raw_policy.get_action(self.curr_obs)[0]

    def _update(self, obs):
        self.curr_obs = self.vectorize_obs(obs)

    def act(self, obs):
        self._update(obs)
        return self._next_item()

    def reset(self):
        self.raw_policy.reset()
