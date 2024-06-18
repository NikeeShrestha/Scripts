import torch
from architecture import Autoencoder
from lossfunctions import BarlowTwinsLoss, MSElossFunc
from dataset import Datasetloader
import numpy as np
from tqdm import tqdm
import time

def scalingfactor(trainerLoader):
# Data loading
    # Loader = Datasetloader(data_path=data_path, batch_size=batch_size)
    # trainerLoader, _, _ = Loader.get_dataloaders()
    trainerLoader=trainerLoader

    # Model initialization
    model = Autoencoder(input_dimension=10761, latent_dimension=512)
    barlow_loss = BarlowTwinsLoss()
    mse_loss = torch.nn.L1Loss()

    # Calculate initial scaling factors
    barlow_losses = []
    mse_losses = []

    model.eval()
    with torch.no_grad():
        start = time.time()
        print('Time starts for Scaling Factor Calculation')
        for batch_index, (data,geneID) in enumerate(trainerLoader):
            if batch_index > 10:  # Use first 10 batches for scaling factor calculation
                break
            z1, reconstructed1 = model(data)
            z2, _ = model(data)
            
            barlowLoss_ = barlow_loss(z1, z2)
            mseloss_ = mse_loss(data, reconstructed1)
            
            barlow_losses.append(barlowLoss_.item())
            mse_losses.append(mseloss_.item())

    avg_barlow_loss = np.mean(barlow_losses)
    avg_mse_loss = np.mean(mse_losses)
    initial_scaling_factor = avg_mse_loss/avg_barlow_loss
    print('Scaling Factor Calculated in:', time.time()-start, '(Seconds), Value:', initial_scaling_factor)
    return initial_scaling_factor