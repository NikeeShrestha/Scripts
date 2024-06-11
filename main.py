import torch
from architecture import Autoencoder
from lossfunctions import BarlowTwinsLoss, MSElossFunc, dataaugment
from dataset import Datasetloader
import numpy as np
from tqdm import tqdm
import pandas as pd



##running on GPU
# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

#dataloader implementation
Loader = Datasetloader(data_path='/home/schnablelab/Documents/GenePrediction/All_data/DataWrangled_nanastillthere_nonan.csv', batch_size=128)
#DataWrangled_nanastillthere_nonan.csv Data_subset.csv
trainerLoader, validationLoader, testLoader= Loader.get_dataloaders()
# print('Dataloader Done')
# feature_dimension=pd.read_csv('/home/schnablelab/Documents/GenePrediction/All_data/DataWrangled_nanastillthere_nonan.csv').shape[1]
##model initiation
model = Autoencoder(input_dimension=10761, latent_dimension=512).to(device)
# print('model initialized')

##learning rate and optimizer
optimizer=torch.optim.Adam(model.parameters(), lr=0.0001)
# print('optimizer done')

barlow_loss = BarlowTwinsLoss().to(device)
mse_loss = MSElossFunc().to(device)
# print('losses initialized')

epochs = 1000

patience = 10  # Number of epochs to wait for improvement before stopping
epochs_no_improve = 0  # Counter for epochs with no improvement

total_training_loss=[]
total_val_loss=[]
best_vloss = 1_000_000.

for epoch in range(epochs):
    # print(epoch)
    train_loss_epoch=[]
    train_barlowloss_epoch=[]
    train_mse_loss_epoch=[]

    for batch_index, data in tqdm(enumerate(trainerLoader, 0), unit='batch', total=len(trainerLoader)):
        data = data.to(device)
        data_augmenter = dataaugment(data)
        data1 = data_augmenter.add_noise(noise_level=0.001)
        data2 = data_augmenter.add_noise(noise_level=0.002)

        # print(batch_index)
        z1, reconstructed1 = model(data1)
        z2, _ = model(data2)

        barlowLoss = barlow_loss(z1,z2)
        # barlowLoss_scaled = (barlowLoss - barlowLoss.min())/(barlowLoss.max() - barlowLoss.min()) 
        
        # print('barlowtwinloss')

        mseloss = mse_loss(data, reconstructed1)
        # mseloss_scaled = (mseloss - mseloss.min())/(mseloss.max() - mseloss.min()) 

        total_loss_value = 0.1 * barlowLoss + 1 * mseloss

        optimizer.zero_grad()  # Zero the gradients
        total_loss_value.backward()  # Backpropagate the total loss
        optimizer.step()

        train_loss_epoch.append(total_loss_value.item())
        train_barlowloss_epoch.append(barlowLoss.item())
        train_mse_loss_epoch.append(mseloss.item())


    average_total_loss = np.mean(train_loss_epoch)
    average_barlow_loss = np.mean(train_barlowloss_epoch)
    average_mse_loss = np.mean(train_mse_loss_epoch)
    total_training_loss.append({'total_loss':average_total_loss, 'barlow_loss': average_barlow_loss, 'mse_loss': average_mse_loss})

    print(f'Epoch: {epoch}, Training Barlow Loss: {average_barlow_loss}, Training MSE Loss: {average_mse_loss}, Training Total Loss: {average_total_loss}')

    val_loss_epoch=[]
    val_barlowloss_epoch=[]
    val_mse_loss_epoch=[]
    model.eval()

    with torch.no_grad():
        for val_index, val_data in tqdm(enumerate(validationLoader, 0), unit='batch', total=len(validationLoader)):
            val_data = val_data.to(device)
            val_data_augmenter = dataaugment(val_data)
            val_data1 = val_data_augmenter.add_noise(noise_level=0.002)
            val_data2 = val_data_augmenter.add_noise(noise_level=0.001)

            val1, valreconstructed1 = model(val_data1)
            val2, _ = model(val_data2)

            valbarlowLoss = barlow_loss(val1,val2)
            # valbarlowLoss_scaled = (valbarlowLoss - valbarlowLoss.min())/(valbarlowLoss.max() - valbarlowLoss.min()) 

            valmseloss = mse_loss(val_data, valreconstructed1)
            # valmseloss_scaled = (valmseloss - valmseloss.min())/(valmseloss.max() - valmseloss.min()) 

            total_valloss_value = 0.1 * valbarlowLoss + 1 * valmseloss

            val_loss_epoch.append(total_valloss_value.item())
            val_barlowloss_epoch.append(valbarlowLoss.item())
            val_mse_loss_epoch.append(valmseloss.item())

        average_v_loss = np.mean(val_loss_epoch)
        average_barlow_loss = np.mean(val_barlowloss_epoch)
        average_mse_loss = np.mean(val_mse_loss_epoch)
        total_val_loss.append({'total_loss':average_v_loss, 'barlow_loss': average_barlow_loss, 'mse_loss': average_mse_loss})

        print(f'Epoch: {epoch}, Val Barlow Loss: {average_barlow_loss}, Val MSE Loss: {average_mse_loss}, Val Total Loss: {average_v_loss}')

        if average_v_loss < best_vloss:
            best_vloss = average_v_loss
            torch.save(model.state_dict(), '../Models/best_autoencoder_model.path')
            epochs_no_improve = 0
        else:
            epochs_no_improve +=1

        if epochs_no_improve == patience:
            print(f'Early Stopping at epoch: {epoch}')
            break

    if epoch==50:
        (pd.DataFrame.from_dict(total_training_loss)).to_csv('Training_loss.csv')
        (pd.DataFrame.from_dict(total_val_loss)).to_csv('Validation_loss.csv')


torch.save(model.state_dict(), '../Models/final_autoencoder_model.path')
(pd.DataFrame.from_dict(total_training_loss)).to_csv('Training_loss.csv')
(pd.DataFrame.from_dict(total_val_loss)).to_csv('Validation_loss.csv')











            


















        




