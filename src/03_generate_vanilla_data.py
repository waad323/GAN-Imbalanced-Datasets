"""
Step 3: Generate synthetic NORMAL X-ray images using the trained
Vanilla GAN generator.

Loads the final generator checkpoint, feeds it random noise vectors,
and saves the resulting images to ./vanilla/GeneratedData/NORMAL/.
"""

import torch
import torch.nn as nn
from torchvision.utils import save_image
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── Configuration ──
num_images_to_generate = 2534
output_folder = './vanilla/GeneratedData/NORMAL/'

z_dim = 64
image_size = 64
channels = 1
img_dim = image_size * image_size * channels


class Generator(nn.Module):
    """Must match the architecture used during training."""
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


def generate_images():
    print(f"Loading Vanilla GAN generator on {device}...")

    gen = Generator(z_dim, img_dim).to(device)
    checkpoint = torch.load('vanilla/generator_final.pth', map_location=device)
    gen.load_state_dict(checkpoint)
    gen.eval()

    os.makedirs(output_folder, exist_ok=True)
    print(f"Generating {num_images_to_generate} images → {output_folder}")

    batch_size = 100
    count = 0

    with torch.no_grad():
        while count < num_images_to_generate:
            cur_batch = min(batch_size, num_images_to_generate - count)
            noise = torch.randn(cur_batch, z_dim).to(device)

            fake_vectors = gen(noise)
            fake_images = fake_vectors.view(cur_batch, channels, image_size, image_size)

            # Denormalize from [-1, 1] → [0, 1]
            fake_images = (fake_images * 0.5) + 0.5

            for i in range(cur_batch):
                save_image(fake_images[i],
                           os.path.join(output_folder, f"gen_normal_{count}.png"))
                count += 1

            if count % 500 == 0:
                print(f"  Generated {count}/{num_images_to_generate} images...")

    print("Vanilla GAN data generation complete.")


if __name__ == "__main__":
    generate_images()
