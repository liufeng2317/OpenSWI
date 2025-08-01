from torch.utils.data import TensorDataset, DataLoader
import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("../../")
from SWIDiffusion.diffusion import *
from SWIDiffusion.dataloader import OpenFWI_Dataset
from tqdm import tqdm
import os
import shutil


# datasets parameter
batch_size = 256
data_base_path = "../../OpenSWI/Datasets/Original/OpenSWI-shallow/CurveVel_A/model"

# model device
device = "cuda:2" if torch.cuda.is_available() else "cpu"

# training parameter
timesteps = 1000
epochs = 5000
sample_interval = 10 # used for saving images
image_size = 64 # used for sampling
last_loss = 1e16 # used for saving the best model

# save directory
save_dir = "./data/diffusion-openfwi-step{}/CurveVel".format(timesteps)


if __name__ == "__main__":
    # create save directory
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # cope the run file to save directory
    current_file = os.path.basename(__file__)
    shutil.copy(__file__, save_dir+'/'+current_file)


    # load datasets
    datasets = OpenFWI_Dataset(data_base_path)
    dataloader = DataLoader(datasets, batch_size=batch_size, shuffle=True)

    # load model
    model = UNetModel(
        in_channels=1,
        model_channels=image_size,
        out_channels=1,
        channel_mult=(1, 2, 2, 2),
        num_res_blocks=2
    ).to(device)

    # diffusion model
    gaussian_diffusion = DenoisingDiffusionProbabilisticModel(timesteps=timesteps)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-6)

    # training
    pbar = tqdm(range(epochs))
    for epoch in pbar:
        total_loss = 0.0
        # training loop
        for step, images in enumerate(dataloader):
            optimizer.zero_grad()

            batch_size = images.shape[0]
            images = images.to(device)

            # sample t uniformally for every example in the batch
            t = torch.randint(0, timesteps, (batch_size,), device=device).long()

            loss = gaussian_diffusion.train_losses(model, images, t)    
            loss.backward()
            optimizer.step()
            total_loss += loss

        # DDPM inference
        if (epoch+1) % sample_interval == 0 or epoch == 0:
            # save the best model
            if total_loss < last_loss:
                torch.save(model.state_dict(), save_dir+'/model.pt')
                # torch.save(optimizer.state_dict(), save_dir+'/optim.pt')
                last_loss = total_loss
            sample = gaussian_diffusion.sample(model, image_size, batch_size=1, channels=1)[-1]
            plt.figure()
            plt.imshow(sample[-1,0,:,:])
            plt.savefig(save_dir+'/ddpm-vp_'+str(epoch)+'.pdf')
            plt.close()

        pbar.set_description(f"Loss: {total_loss:.4f}")