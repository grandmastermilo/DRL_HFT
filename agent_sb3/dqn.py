from configurations import LOGGER
import os
import gym
import gym_trading

from stable_baselines3 import DQN, PPO, A2C



class Agent(object):
    name = 'DQN'

    def __init__(self, 
        is_train = True,
        environment_kwargs:dict = None, 
        model_kwargs:dict = None,
        training_kwargs:dict = None,
        policy_kwargs:dict = None,
        **kwargs):

        """
        Agent constructor
        
        @param is_train 
        @param environment_kwargs: paramters for the environment - checkout the gym environment
        @param model_kwargs: parameters for the agent - dependant on the sb3 agent constructor
        @param training_kwargs: paramters to start training the aget 
        @param policy_kwargs: parameters to define the custom policy network  
    
        """
        # Agent arguments
        self.model_kwargs = model_kwargs
        self.training_kwargs = training_kwargs
        self.policy_kwargs = policy_kwargs
        self.environment_kwargs = environment_kwargs

        self.kwargs = kwargs
        self.train = is_train

        self.rl_algos = {
            'dqn':DQN,
            'ppo':PPO,
            'a2c':A2C
            }


        self.visualize = False

        # Create environment
        self.env = gym.make(**self.environment_kwargs)
        self.env_name = self.kwargs['env_id']
        
        self.cwd = os.path.dirname(os.path.realpath(__file__))

        # create the agent
        if True:
            self.agent = self.rl_algos[self.kwargs['agent']](
                policy="MlpPolicy",
                env=self.env,
                device='cuda:0',
                **self.model_kwargs
            )

        else:
            raise Exception('Not implimented')


    def __str__(self):
        # msg = '\n'
        # return msg.join(['{}={}'.format(k, v) for k, v in self.__dict__.items()])
        return 'Agent = {} | env = {} | number_of_training_steps = {}'.format(
            Agent.name, self.env_name, self.number_of_training_steps)


    def start(self) -> None:
        """
        Entry point for agent training and testing

        :return: (void)
        """
        output_directory = os.path.join(self.cwd, 'dqn_weights')
        if not os.path.exists(output_directory):
            LOGGER.info('{} does not exist. Creating Directory.'.format(output_directory))
            os.mkdir(output_directory)

        weight_name = 'dqn_{}_{}_weights.h5f'.format(
            self.env_name, self.neural_network_type)
        weights_filename = os.path.join(output_directory, weight_name)

        LOGGER.info("weights_filename: {}".format(weights_filename))

        if self.load_weights:
            raise Exception("Not implemented but may be usefull for continual training")
    
        if self.train:
            step_chkpt = '{step}.h5f'
            step_chkpt = 'dqn_{}_weights_{}'.format(self.env_name, step_chkpt)
            checkpoint_weights_filename = os.path.join(self.cwd,
                                                       'dqn_weights',
                                                       step_chkpt)
            LOGGER.info("checkpoint_weights_filename: {}".format(
                checkpoint_weights_filename))
            log_filename = os.path.join(self.cwd, 'dqn_weights',
                                        'dqn_{}_log.json'.format(self.env_name))
            LOGGER.info('log_filename: {}'.format(log_filename))

            #TODO impliment call back for saving checkpoint and sacing the model
            LOGGER.info(f'Starting training...{self.number_of_training_steps}')
            
            self.agent.learn(total_timesteps=self.number_of_training_steps, log_interval=1)

            
            LOGGER.info("training over.")
            LOGGER.info('Saving AGENT weights...')
            # self.agent.save_weights(weights_filename, overwrite=True)
            LOGGER.info("AGENT weights saved.")