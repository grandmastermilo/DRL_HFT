from gym import spaces
import torch

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class TransformerExtractor(BaseFeaturesExtractor):

    def __init__(self,
            observation_space: spaces.Box, 
            features_dim: int = 256,
            latent_dim:int = 128,
            num_heads:int = 4,
            num_blocks:int = 2,
         
        ):
        """
        TransformerPolicy constructor

        @param observation_space: (gym.Space)
        @param features_dim: (int) Number of features extracted (unit for the last layer)


        - assumption:
            - to be used with multivariant timeseries data

        - method;
            - learnable positional embedding 
            - learnable cls token
            - parse the cls outputput through a features projector to get out features 

        """
        super().__init__(observation_space, features_dim)
        #observation size -> [num time steps, num features]

        #interal parameters
        self._window_size = self._observation_space.shape[0]
        self._latent_dim = latent_dim 
        self._num_blocks = num_blocks
        self._num_heads = num_heads

        #creating usefull internal variables
        self._input_features_dim = self._observation_space.shape[1]
        
        #creating learnable cls token for creating context vectors -> use for the MLP head e.g feature compression
        self.cls = torch.nn.Parameter(data = torch.randn(self._latent_dim))
        #creating learnable positional embeddings
        self.pos_ebd = torch.nn.Parameter(data = torch.randn(self._window_size + 1, self._latent_dim))

        
        #creating the transformer encoder network
        self._input_projector = torch.nn.Linear(in_features=self._input_features_dim, out_features=self._latent_dim )
        self._feature_projector = torch.nn.Linear(in_features=self._latent_dim, out_features=self._features_dim )

        self.encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=self._latent_dim,
            dim_feedforward = self._latent_dim*2,
            nhead=self._num_heads,
            batch_first=True
        )

        self.encoder = torch.nn.TransformerEncoder(
            encoder_layer=self.encoder_layer,
            num_layers=self._num_blocks
        )

        return


    def forward(self, obs:torch.Tensor):
        """
        Method to compute the forward pass of the FE
        """
        #check for batches:
        batch = True

        if obs.dim() > 2:
            print('using batchs--')
        else:
            #forcing the observation into a batch if not already handled
            obs = obs.unsqueeze(0)    
    
        #convert the input feature space into the target latent dimension
        x = self._input_projector(obs)

        #concatonate the cls token to the start of sequence
        cls = self.cls.unsqueeze(0).repeat(x.size(0),1).unsqueeze(1) #duplicate the cls token for the batch size
        x = torch.hstack((cls, x))

        #sum with appropriate positional embeddings
        x += self.pos_ebd

        # apply transoformer encoder 
        x = self.encoder(x)

        #extract cls context vector
        cv = x[:,0,:]

        #apply linear projection from latent dim to feature dims
        x = self._feature_projector(cv)


        #provide the extracted features
        return x


if __name__ == "__main__":

    print("TESTING THE TRANSFORMER FEATURE EXTRACTOR NETWORK")

    from gym import spaces
    import numpy as np

    input_dim = 50
    window_size = 100

    obs_space = spaces.Box(
        low=-np.inf,
        high=np.inf,
        shape=(window_size,input_dim )
    )

    model = TransformerExtractor(
        observation_space=obs_space,
        features_dim=256,
        latent_dim=128
    )


    x = torch.randn(2,window_size, input_dim)

    model.forward(x)

    #TODO test how batches are working with sb3 and account for this 
