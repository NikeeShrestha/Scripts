import torch
from architecture import Autoencoder
from lossfunctions import BarlowTwinsLoss, MSElossFunc, dataaugment
from dataset import Datasetloader
import numpy as np
from tqdm import tqdm
import pandas as pd
from scalingfactor import scalingfactor
from torch.utils.tensorboard import SummaryWriter
import argparse 
TF_ENABLE_ONEDNN_OPTS=0

# Set up argparse
parser = argparse.ArgumentParser(description="Run autoencoder experiment")
parser.add_argument("--lr", type=float, required=True,help="Learning rate for the experiment")

# Parse arguments
args = parser.parse_args()

# Now you can use args.lr to get the learning rate
lr = args.lr
##Tensorboard logging
writer = SummaryWriter(log_dir=f'runs/autoencoder_experiment_{lr}')

##running on GPU
# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
batch_size=128
#dataloader implementation
Loader = Datasetloader(data_path='/home/schnablelab/Documents/GenePrediction/All_data/DataWrangled_nanastillthere_nonan.csv', batch_size=batch_size)
#DataWrangled_nanastillthere_nonan.csv Data_subset.csv
trainerLoader, validationLoader, testLoader= Loader.get_dataloaders()
# print('Dataloader Done')
# feature_dimension=pd.read_csv('/home/schnablelab/Documents/GenePrediction/All_data/DataWrangled_nanastillthere_nonan.csv').shape[1]
##model initiation
model = Autoencoder(input_dimension=10761, latent_dimension=512).to(device)
# print('model initialized')

##learning rate and optimizer
optimizer=torch.optim.Adam(model.parameters(), lr=lr)
# print('optimizer done')

barlow_loss = BarlowTwinsLoss().to(device)
mse_loss = torch.nn.L1Loss().to(device)
# print('losses initialized')

epochs = 1000

patience = 100  # Number of epochs to wait for improvement before stopping
epochs_no_improve = 0  # Counter for epochs with no improvement

total_training_loss=[]
total_val_loss=[]
best_vloss = best_vbarlowloss = best_vmaeloss =1_000_000.
running_avg_barlow_loss = None
running_avg_barlow_loss_ = None
initialscalingfactor = scalingfactor(trainerLoader=trainerLoader)
perepochlossdf = pd.DataFrame(index=range(len(trainerLoader)))
perepochlosvalsdf =  pd.DataFrame(index=range(len(validationLoader)))

for epoch in range(epochs):
    # print(epoch)
    train_loss_epoch=[]
    train_barlowloss_epoch=[]
    train_mse_loss_epoch=[]
    perepochloss=[]

    for batch_index, (data, geneID) in tqdm(enumerate(trainerLoader, 0), unit='batch', total=len(trainerLoader)):
        # if batch_index==2:
        #     break
        

        optimizer.zero_grad()  # Zero the gradients
        data_augmenter = dataaugment(data)
        data1 = data_augmenter.add_noise(noise_level=0.001)
        data2 = data_augmenter.add_noise(noise_level=0.002)

        # print(batch_index)
        z1, reconstructed1 = model(data1)
        z2, _ = model(data2)

        barlowLoss = barlow_loss(z1,z2)

        if epoch == 0 and batch_index == 0:
            scaled_barlowLoss = initialscalingfactor * barlowLoss
            # print(scaled_barlowLoss)
            running_avg_barlow_loss_ = barlowLoss.item()
            running_avg_barlow_loss = barlowLoss.item()
            # print(running_avg_barlow_loss)
        else:
            # running_avg_barlow_loss = barlowLoss.item()
            scaled_barlowLoss = barlowLoss / running_avg_barlow_loss
            # print(scaled_barlowLoss)

            ##giving priority to both past and current barlow loss.

            running_avg_barlow_loss = 0.9 * running_avg_barlow_loss_ + 0.1 * barlowLoss.item()
            # print(running_avg_barlow_loss)
            running_avg_barlow_loss_ = barlowLoss.item()

        
        # barlowLoss_scaled = (barlowLoss - barlowLoss.min())/(barlowLoss.max() - barlowLoss.min()) 
        
        # print('barlowtwinloss')

        mseloss = mse_loss(data, reconstructed1)
        # mseloss_scaled = (mseloss - mseloss.min())/(mseloss.max() - mseloss.min())
        factor=0.5

        if (factor*scaled_barlowLoss)/mseloss >= 1.2:
            factor=0.25 
        

        total_loss_value = factor*scaled_barlowLoss + mseloss ##next I will try 0.5*scaledbarlowloss +mseloss

        # print(f"Total Loss Requires Grad: {total_loss_value.requires_grad}")
        total_loss_value.backward()  # Backpropagate the total loss
        # print(barlowLoss, mseloss, total_loss_value, geneID)
        optimizer.step()

        train_loss_epoch.append(total_loss_value.item())
        train_barlowloss_epoch.append(barlowLoss.item())
        train_mse_loss_epoch.append(mseloss.item())

        perepochloss.append({f'train_total_{epoch}': total_loss_value.item(), f'train_barlow_{epoch}': barlowLoss.item(), f'train_mse_{epoch}': mseloss.item()})
        
    tempperepochdf=pd.DataFrame.from_dict(perepochloss)
    perepochlossdf = pd.merge(perepochlossdf, tempperepochdf, left_index=True, right_index=True)

    average_total_loss = np.mean(train_loss_epoch)
    average_barlow_loss = np.mean(train_barlowloss_epoch)
    average_mse_loss = np.mean(train_mse_loss_epoch)
    scaled_barlow_loss = average_total_loss - average_mse_loss
    total_training_loss.append({'total_loss':average_total_loss, 'barlow_loss': average_barlow_loss, 'mse_loss': average_mse_loss})

    print(f'Epoch: {epoch}, Training Barlow Loss: {average_barlow_loss}, {scaled_barlow_loss}, Training MSE Loss: {average_mse_loss}, Training Total Loss: {average_total_loss}')

    writer.add_scalar('Loss/train_total', average_total_loss, epoch)
    writer.add_scalar('Loss/train_barlow', average_barlow_loss, epoch)
    writer.add_scalar('Loss/scaled_barlow', scaled_barlow_loss, epoch)
    writer.add_scalar('Loss/train_mse', average_mse_loss, epoch)

    val_loss_epoch=[]
    val_barlowloss_epoch=[]
    val_mse_loss_epoch=[]
    model.eval()
    valperepochloss =[]

    with torch.no_grad():
        for val_index, (val_data, val_geneID) in tqdm(enumerate(validationLoader, 0), unit='batch', total=len(validationLoader)):
            val_data = val_data.to(device)
            val_data_augmenter = dataaugment(val_data)
            val_data1 = val_data_augmenter.add_noise(noise_level=0.001)
            val_data2 = val_data_augmenter.add_noise(noise_level=0.002)

            val1, valreconstructed1 = model(val_data1)
            val2, _ = model(val_data2)

            valbarlowLoss = barlow_loss(val1,val2)
            # valbarlowLoss_scaled = (valbarlowLoss - valbarlowLoss.min())/(valbarlowLoss.max() - valbarlowLoss.min()) 

            valmseloss = mse_loss(val_data, valreconstructed1)
            # valmseloss_scaled = (valmseloss - valmseloss.min())/(valmseloss.max() - valmseloss.min()) 

            if epoch == 0 and batch_index == 0:
                scaled_valbarlowLoss = initialscalingfactor * valbarlowLoss
            else:
                scaled_valbarlowLoss = valbarlowLoss / running_avg_barlow_loss
            
            factor=0.5

            if (factor*scaled_valbarlowLoss)/valmseloss >= 1.2:
                factor=0.25

            total_valloss_value = factor*scaled_valbarlowLoss + valmseloss

            val_loss_epoch.append(total_valloss_value.item())
            val_barlowloss_epoch.append(valbarlowLoss.item())
            val_mse_loss_epoch.append(valmseloss.item())
            valperepochloss.append({f'train_total_{epoch}': total_valloss_value.item(), f'train_barlow_{epoch}': valbarlowLoss.item(), f'train_mse_{epoch}': valmseloss.item()})
        
        valtempperepochdf=pd.DataFrame.from_dict(valperepochloss)
        perepochlosvalsdf = pd.merge(perepochlosvalsdf, valtempperepochdf, left_index=True, right_index=True)

        average_v_loss = np.mean(val_loss_epoch)
        average_v_barlow_loss = np.mean(val_barlowloss_epoch)
        average_v_mse_loss = np.mean(val_mse_loss_epoch)
        scaled_v_barlow_loss = average_v_loss - average_v_mse_loss
        total_val_loss.append({'total_loss':average_v_loss, 'barlow_loss': average_v_barlow_loss, 'mse_loss': average_v_mse_loss})

        print(f'Epoch: {epoch}, Val Barlow Loss: {average_v_barlow_loss}, {scaled_v_barlow_loss}, Val MSE Loss: {average_v_mse_loss}, Val Total Loss: {average_v_loss}')

        writer.add_scalar('Loss/val_total', average_v_loss, epoch)
        writer.add_scalar('Loss/val_barlow', average_v_barlow_loss, epoch)
        writer.add_scalar('Loss/scaled_val_barlow', scaled_v_barlow_loss, epoch)
        writer.add_scalar('Loss/val_mse', average_v_mse_loss, epoch)

        if average_v_loss < best_vloss:
            best_vloss = average_v_loss
            torch.save(model.state_dict(), f'../Models/best_autoencoder_model_total_loss_{lr}.pth')
            epochs_no_improve = 0
        elif average_v_barlow_loss < best_vbarlowloss:
            best_vbarlowloss = average_v_barlow_loss
            torch.save(model.state_dict(), f'../Models/best_autoencoder_model_barlow_loss_{lr}.pth')
            epochs_no_improve = 0
        elif average_v_mse_loss < best_vmaeloss:
            best_vmaeloss = average_v_mse_loss
            torch.save(model.state_dict(), f'../Models/best_autoencoder_model_mae_loss_{lr}.pth')
            epochs_no_improve = 0
        else:
            epochs_no_improve +=1

        if epochs_no_improve == patience:
            print(f'Early Stopping at epoch for {lr}: {epoch}')
            break

    if epoch==50:
        (pd.DataFrame.from_dict(total_training_loss)).to_csv(f'Training_loss_{lr}.csv')
        (pd.DataFrame.from_dict(total_val_loss)).to_csv(f'Validation_loss_{lr}.csv')
        (perepochlossdf.to_csv(f'index_train_{lr}.csv'))
        (valtempperepochdf.to_csv(f'index_val_{lr}.csv'))


torch.save(model.state_dict(), f'../Models/final_autoencoder_model_{lr}.pth')
(pd.DataFrame.from_dict(total_training_loss)).to_csv(f'Training_loss_{lr}.csv')
(pd.DataFrame.from_dict(total_val_loss)).to_csv(f'Validation_loss_{lr}.csv')
(perepochlossdf.to_csv(f'index_train_{lr}.csv'))
valtempperepochdf.to_csv(f'index_val_{lr}.csv')


writer.close()







            


















        




