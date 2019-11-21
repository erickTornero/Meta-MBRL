from mbrl.network import Dynamics
import torch
import torch.optim as optim
import copy

from IPython.core.debugger import set_trace
class MetaLearner:
    """ 
        Meta Learner class, helps to train the metalearner model 
        
        @dynamics_ml    :   Neural network init weigths for meta-learnerto be specified
        @M              :   The number of datapoints to fine tunning for each task!
        @K              :   The number of future datapoints predicted for test.
        @N              :   Number of sampled tasks
    """

    def __init__(self, dynamics_ml:Dynamics, M, K, N, inner_lr, outer_lr):
        """ Save Dynamics initialization parameters """
        self.state_shape    =   dynamics_ml.state_shape
        self.action_shape   =   dynamics_ml.action_shape
        self.stack_n        =   dynamics_ml.stack_n
        self.sthocastic     =   dynamics_ml.sthocastic
        self.actfn          =   dynamics_ml.actfn
        self.hlayers        =   dynamics_ml.hlayers

        dynamics_class      =   dynamics_ml.__class__

        self.M_points       =   M
        self.K_points       =   K
        self.N_tasks        =   N
        """ Learning rate """
        self.inner_lr       =   inner_lr
        self.outer_lr       =   outer_lr

        self.dynamics_ML    =   dynamics_ml
        """ Create N networks for fine tunning in N tasks"""
        self.ft_networks    =   [dynamics_class((self.state_shape,), (self.action_shape,), self.stack_n, self.sthocastic, self.actfn, self.hlayers) for _ in range(self.N_tasks)]
        """ Optimizers definition """
        self.optimizers     =   [optim.Adam(params=ft_net.parameters(), lr=self.outer_lr) for ft_net in self.ft_networks]
        self.optimizer_ML   =   optim.Adam(params=self.dynamics_ML.parameters(), lr=self.inner_lr)
    
    def _copy_from_ML(self):
        [ft_net.load_state_dict(copy.deepcopy(self.dynamics_ML.state_dict())) for ft_net in self.ft_networks]



if __name__ == "__main__":
    from mbrl.wrapped_env import QuadrotorEnv
    env = QuadrotorEnv(port=28001, reward_type='type1')
    state_shape     =   env.observation_space.shape
    action_shape    =   env.action_space.shape
    stack_n         =   2
    sthocastic      =   False
    actfn           =   torch.tanh
    hlayers         =   [250,250,250]
    lr              =   1e-4

    """ Meta learning parameters """
    M, K, N         =   10, 10, 3
    set_trace()
    dynamics_ml     =   Dynamics(state_shape, action_shape, stack_n, sthocastic, actfn, hlayers)

    mlearnerclass   =   MetaLearner(dynamics_ml, M, K, N, lr)

    mlearnerclass._copy_from_ML()

    