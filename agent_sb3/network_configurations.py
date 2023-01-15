import torch



crypto_rl_policy = dict(activation_fn=torch.nn.ReLU,
                        net_arch=[256,dict(pi=[64], vf=[128])])

                    