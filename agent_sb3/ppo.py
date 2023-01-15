from configurations import LOGGER
import os
import gym
import gym_trading

from stable_baselines3 import PPO

from agent_sb3.network_configurations import crypto_rl_policy

class Agent(object):
    name = 'PPO'

    def __init__(self, 
        is_train = True,
        number_of_training_steps=1e6, 
        load_weights=False,
        visualize=False, 
        nn_type='mlp', 

        **kwargs):

        """
        Agent constructor
        :param window_size: int, number of lags to include in observation
        :param max_position: int, maximum number of positions able to be held in inventory
        :param fitting_file: str, file used for z-score fitting
        :param testing_file: str,file used for dqn experiment
        :param env: environment name
        :param seed: int, random seed number
        :param action_repeats: int, number of steps to take in environment between actions

        :param number_of_training_steps: int, number of steps to train agent for
        :param gamma: float, value between 0 and 1 used to discount future DQN returns

        :param format_3d: boolean, format observation as matrix or tensor
        :param train: boolean, train or test agent
        :param load_weights: boolean, import existing weights
        :param visualize: boolean, visualize environment
   
        """
        # Agent arguments
        self.kwargs = kwargs
        self.train = is_train

        # self.env_name = id
        self.neural_network_type = nn_type
        self.load_weights = load_weights
        self.number_of_training_steps = number_of_training_steps
        self.visualize = visualize

        # Create environment
        self.env = gym.make(**kwargs)
        self.env_name = self.env.env.id

        # Create agent
        # NOTE: 'Keras-RL' uses its own frame-stacker -- this should be happening in the env
        self.memory_frame_stack = 1  # Number of frames to stack e.g., 1.
        
        self.cwd = os.path.dirname(os.path.realpath(__file__))

        # create the agent
        if True:
            self.agent = PPO(
                policy="MlpPolicy",
                policy_kwargs=crypto_rl_policy,
                env=self.env,
                learning_rate = 3e-4,
                gamma=0.99,
                seed=kwargs['seed'],
                n_steps=256,
                batch_size=256,
                gae_lambda=0.97,
                n_epochs=1,
                device='cuda:0',
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
            # LOGGER.info('...loading weights for {} from\n{}'.format(
            #     self.env_name, weights_filename))
            # self.agent.load_weights(weights_filename)

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


        else:
            raise Exception('Not yet implimented ')
            # LOGGER.info('Starting TEST...')
            # self.agent.test(self.env, nb_episodes=2, visualize=self.visualize)
