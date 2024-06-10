import torch


class Autoencoder(torch.nn.Module):
    def __init__(self, input_dimension, latent_dimension):
        super().__init__()

        self.encoder = torch.nn.Sequential(torch.nn.Linear(input_dimension, 8192),
                                           torch.nn.BatchNorm1d(8192),
                                           torch.nn.ReLU(),
                                           torch.nn.Dropout(0.3),
                                           torch.nn.Linear(8192, 6144),
                                           torch.nn.BatchNorm1d(6144),
                                           torch.nn.ReLU(),
                                           torch.nn.Dropout(0.3),
                                           torch.nn.Linear(6144, 4096),
                                           torch.nn.BatchNorm1d(4096),
                                           torch.nn.ReLU(),
                                           torch.nn.Linear(4096, 2048),
                                           torch.nn.BatchNorm1d(2048),
                                           torch.nn.ReLU(),
                                           torch.nn.Linear(2048, 1024),
                                           torch.nn.BatchNorm1d(1024),
                                           torch.nn.ReLU(),
                                           torch.nn.Linear(1024, latent_dimension))
        

        
        self.decoder = torch.nn.Sequential(torch.nn.Linear(latent_dimension, 1024),
                                           torch.nn.BatchNorm1d(1024),
                                           torch.nn.ReLU(),
                                           torch.nn.Dropout(0.3),
                                           torch.nn.Linear(1024, 2048),
                                           torch.nn.BatchNorm1d(2048),
                                           torch.nn.ReLU(),
                                           torch.nn.Dropout(0.3),
                                           torch.nn.Linear(2048, 4096),
                                           torch.nn.BatchNorm1d(4096),
                                           torch.nn.ReLU(),
                                           torch.nn.Linear(4096, 6144),
                                           torch.nn.BatchNorm1d(6144),
                                           torch.nn.ReLU(),
                                           torch.nn.Linear(6144, 8192),
                                           torch.nn.BatchNorm1d(8192),
                                           torch.nn.ReLU(),
                                           torch.nn.Linear(8192, input_dimension))
        
        
        
    def forward(self, x):
        # print(f"Input shape: {x.shape}")
        latent_variables = self.encoder(x)
        # print(f"Encoded shape: {latent_variables.shape}")
        reconstructed = self.decoder(latent_variables)
        # print(f"Reconstructed shape: {reconstructed.shape}")
        return latent_variables, reconstructed







