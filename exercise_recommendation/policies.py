import numpy as np

from exercise_recommendation.tutors import DummyTutor
from rllab.algos.trpo import *
from rllab.misc.overrides import overrides


class LoggedTRPO(TRPO):

    def __init__(self, *args, **kwargs):
        super(LoggedTRPO, self).__init__(*args, **kwargs)
        self.rew_chkpts = []

    def run_ep(self,agent, env):
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

    @overrides
    def train(self):
        self.start_worker()
        self.init_opt()
        for itr in range(self.current_itr, self.n_itr):
            paths = self.sampler.obtain_samples(itr)
            samples_data = self.sampler.process_samples(itr, paths)
            self.optimize_policy(itr, samples_data)
            my_policy = lambda obs: self.policy.get_action(obs)[0]
            r, _ = self.run_ep(DummyTutor(my_policy), self.env)
            self.rew_chkpts.append(r)
            print(self.rew_chkpts[-1])
        self.shutdown_worker()
