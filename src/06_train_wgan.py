"""
Step 6: Train a Wasserstein GAN (WGAN) on the minority class (NORMAL)
of the Chest X-Ray dataset.

Key differences from standard GAN / DCGAN:
  - Critic instead of Discriminator (no Sigmoid — outputs unbounded score)
  - Wasserstein loss (Earth Mover's distance) instead of BCE
  - Weight clipping to enforce Lipschitz constraint
  - RMSprop optimizer (Adam can cause instability with WGAN)
  - Critic is trained n_critic times per generator update

Reference: Arjovsky et al., "Wasserstein GAN" (2017)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import os

# ── Hyperparameters ──
nc = 1
nz = 100
ngf = 64
ndf = 64
lr = 5e-5           # Lower LR for WGAN stability
clip_value = 0.01    # Weight clipping range [-c, c]
n_critic = 5         # Train critic this many times per generator step
image_size = 64
batch_size = 64
num_epochs = 50

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ── Data Loader ──
def get_data_loader():
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    dataset = datasets.ImageFolder(root='./chest_xray/train', transform=transform)

    target_idx = dataset.class_to_idx['NORMAL']
    indices = [i for i, label in enumerate(dataset.targets)
               if label == target_idx]
    minority_dataset = torch.utils.data.Subset(dataset, indices)

    print(f"Training on minority class (NORMAL): {len(minority_dataset)} images")
    return DataLoader(minority_dataset, batch_size=batch_size, shuffle=True)


# ── Weight Initialization ──
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# ── Generator (same architecture as DCGAN) ──
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)


# ── Critic (no Sigmoid — outputs raw score) ──
class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            # No Sigmoid — WGAN uses raw scores
        )

    def forward(self, x):
        return self.main(x)


# ── Initialization ──
loader = get_data_loader()

netG = Generator().to(device)
netC = Critic().to(device)
netG.apply(weights_init)
netC.apply(weights_init)

# RMSprop is recommended for WGAN (Adam can cause instability)
optimizerC = optim.RMSprop(netC.parameters(), lr=lr)
optimizerG = optim.RMSprop(netG.parameters(), lr=lr)

os.makedirs("./wgan/CheckPoints", exist_ok=True)
log_file = './wgan/wgan_training_log.txt'
with open(log_file, "w") as f:
    f.write("Epoch,Batch,Loss_C,Loss_G\n")

# ── Training Loop ──
print("Starting WGAN training...")

for epoch in range(num_epochs):
    for batch_idx, (real, _) in enumerate(loader):

        real = real.to(device)
        cur_batch = real.shape[0]

        # --- Train Critic: maximize E[C(real)] - E[C(fake)] ---
        netC.zero_grad()

        output_real = netC(real).view(-1)
        loss_real = -torch.mean(output_real)

        noise = torch.randn(cur_batch, nz, 1, 1).to(device)
        fake = netG(noise).detach()
        output_fake = netC(fake).view(-1)
        loss_fake = torch.mean(output_fake)

        loss_C = loss_real + loss_fake
        loss_C.backward()
        optimizerC.step()

        # Weight clipping (enforces Lipschitz constraint)
        for p in netC.parameters():
            p.data.clamp_(-clip_value, clip_value)

        # --- Train Generator every n_critic steps ---
        if batch_idx % n_critic == 0:
            netG.zero_grad()
            noise = torch.randn(cur_batch, nz, 1, 1).to(device)
            fake = netG(noise)
            output = netC(fake).view(-1)

            loss_G = -torch.mean(output)
            loss_G.backward()
            optimizerG.step()

            if batch_idx % 100 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}] Batch {batch_idx}\t"
                      f"Loss C: {loss_C.item():.4f}, Loss G: {loss_G.item():.4f}")

            with open(log_file, "a") as f:
                f.write(f"{epoch+1},{batch_idx},{loss_C.item():.4f},{loss_G.item():.4f}\n")

    if (epoch + 1) % 10 == 0:
        print(f"Saving checkpoint at epoch {epoch+1}...")
        torch.save(netG.state_dict(),
                   f"./wgan/CheckPoints/generator_epoch_{epoch+1}.pth")
        torch.save(netC.state_dict(),
                   f"./wgan/CheckPoints/critic_epoch_{epoch+1}.pth")

print("WGAN training complete.")
torch.save(netG.state_dict(), "./wgan/generator_final.pth")
torch.save(netC.state_dict(), "./wgan/critic_final.pth")
