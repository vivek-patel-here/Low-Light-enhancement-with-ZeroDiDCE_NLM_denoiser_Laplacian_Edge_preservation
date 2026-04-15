import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import os

import model
import dataloader

# device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)

# model
net = model.enhance_net_nopool().to(device)
optimizer = optim.Adam(net.parameters(), lr=1e-4)

os.makedirs("snapshots", exist_ok=True)

# Dataset
train_dataset = dataloader.lowlight_loader("data/train_data/INPUT/")
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2)

# loss

def L_spa(enhanced, original):
    return torch.mean(torch.abs(
        (enhanced[:, :, :, :-1] - enhanced[:, :, :, 1:]) -
        (original[:, :, :, :-1] - original[:, :, :, 1:])
    )) + torch.mean(torch.abs(
        (enhanced[:, :, :-1, :] - enhanced[:, :, 1:, :]) -
        (original[:, :, :-1, :] - original[:, :, 1:, :]
    )))

def L_col(enhanced):
    mean_rgb = torch.mean(enhanced, dim=[2,3])
    mr, mg, mb = mean_rgb[:,0], mean_rgb[:,1], mean_rgb[:,2]
    return torch.mean((mr - mg)**2 + (mr - mb)**2 + (mg - mb)**2)

def L_tv(A):
    return torch.mean(torch.abs(A[:, :, :, :-1] - A[:, :, :, 1:])) + \
           torch.mean(torch.abs(A[:, :, :-1, :] - A[:, :, 1:, :]))

# patch exposure loss
def L_exp_patch(enhanced, patch_size=16, target=0.6):
    B, C, H, W = enhanced.shape

    patches = enhanced.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    patches = patches.contiguous().view(B, C, -1, patch_size, patch_size)

    patch_means = patches.mean(dim=[1,3,4])
    return torch.mean((patch_means - target) ** 2)

# piecewise_loss
def piecewise_loss(enhanced, patch_size=16, Q1=0.2, Q2=0.8, E=0.6, m=10):
    B, C, H, W = enhanced.shape

    patches = enhanced.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    patches = patches.contiguous().view(B, C, -1, patch_size, patch_size)

    Yi = patches.mean(dim=[1,3,4])

    extreme_mask = (Yi <= Q1) | (Yi >= Q2)
    normal_mask  = (Yi > Q1) & (Yi < Q2)

    L1 = (Yi - E) ** 2
    L2 = ((Yi - E) ** 2) / (1 + m * Yi)

    loss_extreme = L1[extreme_mask].mean() if extreme_mask.any() else 0.0
    loss_normal  = L2[normal_mask].mean() if normal_mask.any() else 0.0

    return loss_extreme + loss_normal



epochs = 100
for epoch in range(epochs):
    net.train()
    epoch_loss = 0

    for img in train_loader:
        img = img.to(device)

        enhanced, A = net(img)

        loss_spa = L_spa(enhanced, img)
        loss_exp = L_exp_patch(enhanced)
        loss_col = L_col(enhanced)
        loss_tv  = L_tv(A)
        loss_piece = piecewise_loss(enhanced)

        loss = loss_spa \
             + 10 * loss_exp \
             + 5 * loss_col \
             + 200 * loss_tv \
             + 5 * loss_piece

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(train_loader)

    if epoch >= 60:
        torch.save(net.state_dict(), f"snapshots/Epoch{epoch+1}.pth")

    print(f"Epoch {epoch} | Avg Loss: {avg_loss:.6f}")