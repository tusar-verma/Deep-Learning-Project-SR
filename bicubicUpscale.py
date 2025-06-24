from PIL import Image
import os

def upscale_bicubic(input_path, output_path, scale):
    """
    Upscales an image using bicubic interpolation.

    Args:
        input_path (str): Path to the input low-res image.
        output_path (str): Path to save the upscaled image.
        scale (int): Upscaling factor (e.g., 2, 3, 4).
    """
    img = Image.open(input_path)
    new_size = (img.width * scale, img.height * scale)
    img_upscaled = img.resize(new_size, Image.BICUBIC)
    img_upscaled.save(output_path)

upscale_bicubic("naranjoso_lr.png", "naranjoso_bi.png", scale=3)