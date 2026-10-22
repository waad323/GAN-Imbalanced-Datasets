"""
Step 5: Generate synthetic NORMAL X-ray images using the trained
DCGAN generator.

Loads the final DCGAN generator checkpoint and saves generated
images to ./dcgan/GeneratedData/NORMAL/.
"""

import torch
import torch.nn as nn
from torchvision.utils import save_image
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── Configuration ──
num_images_to_generate = 2534
output_folder = './dcgan/GeneratedData/NORMAL/'

nz = 100
ngf = 64
nc = 1


class Generator(nn.Module):
    """Must match the DCGAN architecture used during training."""
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
    print(f"Loading DCGAN generator on {device}...")

    netG = Generator().to(device)
    checkpoint = torch.load('./dcgan/generator_final.pth', map_location=device)
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
                           os.path.join(output_folder, f"dcgan_normal_{count}.png"))
                count += 1

            if count % 500 == 0:
                print(f"  Generated {count}/{num_images_to_generate} images...")

    print("DCGAN data generation complete.")


if __name__ == "__main__":
    generate_images()
