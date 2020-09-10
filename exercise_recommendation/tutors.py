from __future__ import division


class Tutor(object):

    def __init__(self):
        pass

    def _next_item(self):
        raise NotImplementedError

    def _update(self, item, outcome, timestamp, delay):
        raise NotImplementedError

    def act(self, obs):
        pass
        # self._update(*list(obs))
        # return self._next_item()

    def learn(self, r):
        pass

    def train(self, env, n_eps=10):
        pass
        # return run_eps(self, env, n_eps=n_eps)

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
    #sta ce mu ,n_items, init_timestamp=0
    def __init__(self, rl_env,raw_policy):
        self.raw_policy = raw_policy
        self.curr_obs = None
        self.rl_env = rl_env

    def train(self):
        #self.rl_env = MyGymEnv(gym_env)

        self.raw_policy.train()
        return self.raw_policy.rew_chkpts

    def guide(self, observations):
        """
        This method creates a DummyTutor agent using input observations that produce an action according
        to RLTutor raw_policy. The DummyTutor then acts according to the observation and an action is returned.
        Args:
            observations: a list of observations

        Returns:
            A proposed action.
        """
        agents = DummyTutor(lambda obs: self.raw_policy.policy.get_action(obs)[0])
        action = agents.act(observations)
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
