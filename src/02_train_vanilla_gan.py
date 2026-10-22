"""
Step 2: Train a Vanilla GAN on the minority class (NORMAL) of the
Chest X-Ray dataset.

Architecture:
  - Generator:     4-layer fully connected network (z → 256 → 512 → 1024 → 4096)
  - Discriminator:  4-layer fully connected network (4096 → 1024 → 512 → 256 → 1)

The generator learns to produce synthetic 64×64 grayscale X-ray images
from a random noise vector, while the discriminator learns to distinguish
real images from fakes.

Checkpoints are saved every 10 epochs to ./vanilla/CheckPoints/.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import os

# ── Device Configuration ──
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ── Hyperparameters ──
lr = 3e-4
z_dim = 64          # Latent vector size
image_size = 64
channels = 1        # 1 = grayscale
img_dim = image_size * image_size * channels
batch_size = 32
num_epochs = 50


# ── Data Preparation ──
def get_data_loader():
    """
    Loads only the minority class (NORMAL) from the chest X-ray dataset.
    Applies resizing, grayscale conversion, and normalization to [-1, 1].
    """
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    data_path = './chest_xray/train'
    dataset = datasets.ImageFolder(root=data_path, transform=transform)

    # Filter for the minority class only
    minority_idx = dataset.class_to_idx['NORMAL']
    indices = [i for i, label in enumerate(dataset.targets)
               if label == minority_idx]
    minority_dataset = torch.utils.data.Subset(dataset, indices)

    print(f"Training on minority class (NORMAL): {len(minority_dataset)} images")
    return DataLoader(minority_dataset, batch_size=batch_size, shuffle=True)


# ── Discriminator Network ──
class Discriminator(nn.Module):
    """Binary classifier that outputs P(real) for a flattened image."""
    def __init__(self, img_dim):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Linear(img_dim, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.disc(x)


# ── Generator Network ──
class Generator(nn.Module):
    """Maps a latent vector z to a flattened image via fully-connected layers."""
    def __init__(self, z_dim, img_dim):
        super().__init__()
        self.gen = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, img_dim),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.gen(x)


# ── Initialization ──
disc = Discriminator(img_dim).to(device)
gen = Generator(z_dim, img_dim).to(device)

opt_disc = optim.Adam(disc.parameters(), lr=lr)
opt_gen = optim.Adam(gen.parameters(), lr=lr)
criterion = nn.BCELoss()

loader = get_data_loader()

os.makedirs("./vanilla/CheckPoints", exist_ok=True)
log_file = './vanilla/vanilla_training_log.txt'

# ── Training Loop ──
print("Starting Vanilla GAN training...")

for epoch in range(num_epochs):
    for batch_idx, (real, _) in enumerate(loader):

        real = real.view(-1, img_dim).to(device)
        cur_batch = real.shape[0]

        # --- Train Discriminator ---
        noise = torch.randn(cur_batch, z_dim).to(device)
        fake = gen(noise)

        disc_real = disc(real).view(-1)
        lossD_real = criterion(disc_real, torch.ones_like(disc_real))

        disc_fake = disc(fake.detach()).view(-1)
        lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake))

        lossD = (lossD_real + lossD_fake) / 2
        disc.zero_grad()
        lossD.backward()
        opt_disc.step()

        # --- Train Generator ---
        output = disc(fake).view(-1)
        lossG = criterion(output, torch.ones_like(output))

        gen.zero_grad()
        lossG.backward()
        opt_gen.step()

        # --- Logging ---
        log_msg = (f"Epoch [{epoch+1}/{num_epochs}] Batch {batch_idx}\t"
                   f"Loss D: {lossD.item():.4f}, Loss G: {lossG.item():.4f}\n")

        with open(log_file, "a") as f:
            f.write(log_msg)

        if batch_idx % 100 == 0:
            print(log_msg.strip())

    # Save checkpoint every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f"Saving checkpoint at epoch {epoch+1}...")
        torch.save(gen.state_dict(),
                   f"./vanilla/CheckPoints/generator_epoch_{epoch+1}.pth")
        torch.save(disc.state_dict(),
                   f"./vanilla/CheckPoints/discriminator_epoch_{epoch+1}.pth")

print("Vanilla GAN training complete.")
torch.save(gen.state_dict(), "./vanilla/generator_final.pth")
torch.save(disc.state_dict(), "./vanilla/discriminator_final.pth")
