from exercise_recommendation.tutors import DummyTutor
from new_rs import run_ep
from rllab.algos.trpo import *
from rllab.misc.overrides import overrides


class LoggedTRPO(TRPO):

    def __init__(self, *args, **kwargs):
        super(LoggedTRPO, self).__init__(*args, **kwargs)
        self.rew_chkpts = []

    @overrides
    def train(self):
        self.start_worker()
        self.init_opt()
        for itr in range(self.current_itr, self.n_itr):
            paths = self.sampler.obtain_samples(itr)
            samples_data = self.sampler.process_samples(itr, paths)
            self.optimize_policy(itr, samples_data)
            my_policy = lambda obs: self.policy.get_action(obs)[0]
            r, _ = run_ep(DummyTutor(my_policy), self.env)
            self.rew_chkpts.append(r)
            print(self.rew_chkpts[-1])
        self.shutdown_worker()
