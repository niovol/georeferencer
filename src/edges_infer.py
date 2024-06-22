"""
Using Neural network for boundaries detection
"""

import numpy as np
import segmentation_models_pytorch as smp
import torch


def pad_image(image, divisor=32):
    """
    Pads the image so that its height and width are divisible by the given divisor.
    """
    h, w, _ = image.shape
    new_h = ((h + divisor - 1) // divisor) * divisor
    new_w = ((w + divisor - 1) // divisor) * divisor
    padded_image = np.zeros((new_h, new_w, image.shape[2]), dtype=image.dtype)
    padded_image[:h, :w, :] = image
    return padded_image, (h, w)


def unpad_image(image, original_shape):
    """
    Removes the padding from the image.
    """
    h, w = original_shape
    return image[:h, :w]


def load_edges_model(backbone_name, model_path):
    """
    Loads a trained model from disk
    """
    model = smp.Unet(
        encoder_name=backbone_name,
        encoder_weights=None,
        in_channels=4,
        classes=1,
    )
    # model.load_state_dict(torch.load(model_path))
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    return model


def infer_edges(model, input_image):
    """
    Start inference using a trained model
    """
    input_image, original_shape = pad_image(input_image)
    input_tensor = torch.tensor(input_image).permute(2, 0, 1).unsqueeze(0).float()

    with torch.no_grad():
        output_tensor = model(input_tensor)

    output_tile = output_tensor.squeeze().cpu().numpy()
    output_tile = unpad_image(output_tile, original_shape)
    output_tile = output_tile.reshape(output_tile.shape[0], output_tile.shape[1], 1)
    output_tile = np.clip(output_tile, 1, 255)

    return output_tile
