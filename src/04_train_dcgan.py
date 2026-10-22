"""
Step 4: Train a Deep Convolutional GAN (DCGAN) on the minority class
(NORMAL) of the Chest X-Ray dataset.

Architecture follows the DCGAN guidelines (Radford et al., 2015):
  - Generator:     Transposed convolutions (nz → 512 → 256 → 128 → 64 → 1)
  - Discriminator:  Strided convolutions    (1 → 64 → 128 → 256 → 512 → 1)
  - BatchNorm in both networks
  - LeakyReLU in discriminator, ReLU in generator

Reference: https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import os

# ── Hyperparameters ──
nc = 1          # Number of channels (1 for grayscale)
nz = 100        # Latent vector size
ngf = 64        # Generator feature maps
ndf = 64        # Discriminator feature maps
lr = 0.0002     # Learning rate (tuned for DCGAN)
beta1 = 0.5     # Adam beta1
image_size = 64
batch_size = 128
num_epochs = 50

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ── Data Loader (Minority Class Only) ──
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


# ── Weight Initialization (DCGAN paper) ──
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# ── Generator ──
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


# ── Discriminator ──
class Discriminator(nn.Module):
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
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x)


# ── Initialization ──
loader = get_data_loader()

netG = Generator().to(device)
netD = Discriminator().to(device)
netG.apply(weights_init)
netD.apply(weights_init)

criterion = nn.BCELoss()
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

os.makedirs("./dcgan/CheckPoints", exist_ok=True)
log_file = './dcgan/dcgan_training_log.txt'

with open(log_file, "w") as f:
    f.write("Epoch,Batch,Loss_D,Loss_G\n")

# ── Training Loop ──
print("Starting DCGAN training...")

for epoch in range(num_epochs):
    for batch_idx, (real, _) in enumerate(loader):

        real = real.to(device)
        cur_batch = real.shape[0]

        # --- Train Discriminator ---
        noise = torch.randn(cur_batch, nz, 1, 1).to(device)
        fake = netG(noise)

        disc_real = netD(real).view(-1)
        lossD_real = criterion(disc_real, torch.ones_like(disc_real))

        disc_fake = netD(fake.detach()).view(-1)
        lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake))

        lossD = (lossD_real + lossD_fake) / 2
        netD.zero_grad()
        lossD.backward()
        optimizerD.step()

        # --- Train Generator ---
        output = netD(fake).view(-1)
        lossG = criterion(output, torch.ones_like(output))

        netG.zero_grad()
        lossG.backward()
        optimizerG.step()

        # --- Logging ---
        if batch_idx % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] Batch {batch_idx}\t"
                  f"Loss D: {lossD.item():.4f}, Loss G: {lossG.item():.4f}")

        with open(log_file, "a") as f:
            f.write(f"{epoch+1},{batch_idx},{lossD.item():.4f},{lossG.item():.4f}\n")

    if (epoch + 1) % 10 == 0:
        print(f"Saving checkpoint at epoch {epoch+1}...")
        torch.save(netG.state_dict(),
                   f"./dcgan/CheckPoints/generator_epoch_{epoch+1}.pth")
        torch.save(netD.state_dict(),
                   f"./dcgan/CheckPoints/discriminator_epoch_{epoch+1}.pth")

print("DCGAN training complete.")
torch.save(netG.state_dict(), "./dcgan/generator_final.pth")
torch.save(netD.state_dict(), "./dcgan/discriminator_final.pth")
