"""
Training Neural network for boundaries detection
"""

import random

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from .utils import final_uint8, load_geotiff


class GeoTiffDataset(Dataset):
    """
    Dataset
    """

    def __init__(self, input_files, output_file, tile_size, transform=None):
        """
        Args:
            input_files (list): List of paths to input GeoTiff files.
            output_file (str): Path to the output GeoTiff file.
            tile_size (int): Size of the tile to be randomly cropped.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.input_files = input_files
        self.output_file = output_file
        self.tile_size = tile_size
        self.transform = transform
        self.output_image = load_geotiff(self.output_file, layout="hwc")["data"]
        print("loaded output")
        self.input_images = [
            load_geotiff(file, layout="hwc")["data"].astype("float32")
            for file in self.input_files
        ]
        print("loaded input")

    def __len__(self):
        return 1000

    def __getitem__(self, idx):
        input_idx = random.randint(0, len(self.input_images) - 1)
        input_image = self.input_images[input_idx]
        h, w, _ = input_image.shape
        top = random.randint(0, h - self.tile_size)
        left = random.randint(0, w - self.tile_size)

        input_tile = input_image[
            top : top + self.tile_size, left : left + self.tile_size, :
        ]

        output_tile = self.output_image[
            top : top + self.tile_size, left : left + self.tile_size, :
        ]

        output_tile = final_uint8(output_tile, "scene").astype("float32")

        sample = {"input": input_tile, "output": output_tile}

        if self.transform:
            sample = self.transform(sample)

        return sample


INPUTS = [
    "/layouts/downscaled/layout_2021-06-15.tif",
    "/layouts/downscaled/layout_2021-08-16.tif",
    "/layouts/downscaled/layout_2021-10-10.tif",
    "/layouts/downscaled/layout_2022-03-17.tif",
]
OUTPUT = "/layouts/downscaled/layout_2021-08-16.tif"


dataset = GeoTiffDataset(INPUTS, OUTPUT, 384)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)

model = smp.Unet(
    encoder_name="resnet18",
    encoder_weights="imagenet",
    in_channels=4,
    classes=1,
)
model = model.cuda()

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

NUM_EPOCHS = 200

for epoch in range(NUM_EPOCHS):
    model.train()
    RUNNING_LOSS = 0.0

    for i, sample in enumerate(dataloader):
        INPUTS = sample["input"].permute(0, 3, 1, 2).float().cuda()
        outputs = sample["output"].permute(0, 3, 1, 2).float().cuda()

        optimizer.zero_grad()

        preds = model(INPUTS)
        loss = criterion(preds, outputs)

        loss.backward()
        optimizer.step()

        RUNNING_LOSS += loss.item()

        if i % 10 == 9:
            print(
                f"Epoch [{epoch + 1}/{NUM_EPOCHS}], "
                f"Step [{i + 1}/{len(dataloader)}], "
                f"Loss: {RUNNING_LOSS / 10:.4f}"
            )
            RUNNING_LOSS = 0.0

torch.save(model.state_dict(), "resnet_model6.pth")
