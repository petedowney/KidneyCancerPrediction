import os
import torch
import importlib
import parse_labels
import util
import math
from tqdm import tqdm
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
print(f"Using device: {device}")

base_path = os.getcwd()
label_path = os.path.join(base_path, "raw_data", "labels.txt")
data_path = os.path.join(base_path, "raw_data", "manifest-xxn3N2Qq630907925598003437/TCGA-KIRC")

series_data = parse_labels.get_all_dcm_files(label_path, data_path)
print(f"Found {len(series_data)} scan series total")


save_path = os.path.join(base_path, "preloaded_data_processed_all.pt")
preloaded_data = torch.load(save_path)

# Make sure MONAI is installed: !pip install monai

import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from monai.networks.nets import VarAutoEncoder
import os
import gc
from tqdm import tqdm

# PyTorch Dataset for Autoencoder (Self-supervised - we only use the volumes)
class AutoEncoderDataset(Dataset):
    def __init__(self, data_dict):
        # We need volumes normalized strictly between 0 and 1
        self.volumes = [(vol.float() / 255.0) for vol, lbl, pid in data_dict.values()]
        
    def __len__(self):
        return len(self.volumes)
        
    def __getitem__(self, idx):
        return self.volumes[idx]

# Prepare dataset
ae_dataset = AutoEncoderDataset(preloaded_data)

# 90/10 Train/Val Split
dataset_size = len(ae_dataset)
train_size = int(0.9 * dataset_size)
val_size = dataset_size - train_size

# Set seed for reproducibility
generator = torch.Generator().manual_seed(42)
train_dataset, val_dataset = random_split(ae_dataset, [train_size, val_size], generator=generator)

# Prepare dataloaders - REDUCED BATCH SIZE TO 1
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, pin_memory=True) # Dropped num_workers to not overload memory
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, pin_memory=True)

print(f"Training on {train_size} samples. Validating on {val_size} samples.")

# Initialize MONAI VarAutoEncoder
vae = VarAutoEncoder(
    spatial_dims=3,             # 3D Medical Data
    in_shape=(1, 64, 192, 192), # Input volume shape (C, D, H, W)
    out_channels=1,
    latent_size=512,            # Compress to exactly 512 dimensions for SMOTE later
    channels=(16, 32, 64),      # Downsampling CNN layers
    strides=(2, 2, 2)           # How much to compress at each CNN layer
).to(device)

optimizer = optim.Adam(vae.parameters(), lr=1e-4)

# Training loop
num_epochs = 15
print("Starting MONAI Variational AutoEncoder training...")

for epoch in range(num_epochs):
    vae.train()
    running_loss = 0.0
    
    train_pbar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}] Train")
    for images in train_pbar:
        images = images.to(device)
        
        optimizer.zero_grad()
        
        # VAE returns reconstructed image, latent mean (mu), and log variance
        reconstructed, mu, log_var = vae(images)
        
        # 1. Reconstruction loss (MSE between input and output)
        recon_loss = nn.MSELoss()(reconstructed, images)
        
        # 2. KL Divergence loss (Forces the 512 latent variables to be nicely scattered and smooth)
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        
        # Total loss (Weighting KL divergence is standard in VAEs)
        loss = recon_loss + 0.0001 * kl_loss
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        train_pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
    avg_train_loss = running_loss / len(train_loader)
    
    # Validation step
    vae.eval()
    val_loss = 0.0
    
    val_pbar = tqdm(val_loader, desc=f"Epoch [{epoch+1}/{num_epochs}] Val")
    with torch.no_grad():
        for images in val_pbar:
            images = images.to(device)
            reconstructed, mu, log_var = vae(images)
            recon_loss = nn.MSELoss()(reconstructed, images)
            kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            loss = recon_loss + 0.0001 * kl_loss
            val_loss += loss.item()
            val_pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
    avg_val_loss = val_loss / len(val_loader)
    
    print(f"Epoch [{epoch+1}/{num_epochs}] Summary -> Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}\n")
    
    # Force clear VRAM cache at end of each epoch
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        gc.collect()

# Final overall validation assessment
print("\n--- Final Validation Assessment ---")
vae.eval()
final_val_loss = 0.0
with torch.no_grad():
    for images in tqdm(val_loader, desc="Final Validation"):
        images = images.to(device)
        reconstructed, _, _ = vae(images)
        final_val_loss += nn.MSELoss()(reconstructed, images).item()
        
print(f"Final Validation MSE Loss: {final_val_loss / len(val_loader):.4f}")

# Save the VAE model for later SMOTE operations
vae_model_path = os.path.join(base_path, "kidney_cancer_vae.pth")
torch.save(vae.state_dict(), vae_model_path)
print(f"VarAutoEncoder saved successfully to {vae_model_path}")