"""
Step 7: Generate synthetic NORMAL X-ray images using the trained
WGAN generator.
"""

import torch
import torch.nn as nn
from torchvision.utils import save_image
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── Configuration ──
num_images_to_generate = 2534
output_folder = './wgan/GeneratedData/NORMAL/'

nz = 100
ngf = 64
nc = 1


class Generator(nn.Module):
    """Must match the WGAN generator architecture used during training."""
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


def generate_images():
    print(f"Loading WGAN generator on {device}...")

    netG = Generator().to(device)
    checkpoint = torch.load('./wgan/generator_final.pth', map_location=device)
    netG.load_state_dict(checkpoint)
    netG.eval()

    os.makedirs(output_folder, exist_ok=True)
    print(f"Generating {num_images_to_generate} images → {output_folder}")

    batch_size = 100
    count = 0

    with torch.no_grad():
        while count < num_images_to_generate:
            cur_batch = min(batch_size, num_images_to_generate - count)
            noise = torch.randn(cur_batch, nz, 1, 1).to(device)

            fake_images = netG(noise)
            fake_images = (fake_images * 0.5) + 0.5  # Denormalize

            for i in range(cur_batch):
                save_image(fake_images[i],
                           os.path.join(output_folder, f"wgan_normal_{count}.png"))
                count += 1

            if count % 500 == 0:
                print(f"  Generated {count}/{num_images_to_generate} images...")

    print("WGAN data generation complete.")


if __name__ == "__main__":
    generate_images()
