from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize
import matplotlib.pyplot as plt
from dataset import ImageDataset
from torch.utils.data import random_split, DataLoader
import torch
from dataDownloader import downloadData


# cuando haya varios datasets, renombrar a carData


def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)

def input_transform(crop_size, upscale_factor):
    return Compose([
        CenterCrop(crop_size),
        Resize(crop_size // upscale_factor),
        ToTensor(),
    ])

def target_transform(crop_size):
    return Compose([
        CenterCrop(crop_size),
        ToTensor(),
    ])


# -----------

def show_images(dataset, num_images=5):
    fig, axes = plt.subplots(2, num_images, figsize=(3*num_images, 6))
    for i in range(num_images):
        in_img, out_img = dataset[i]
        # Convert tensors to PIL Images if needed
        if hasattr(in_img, 'permute'):
            in_img = in_img.permute(1, 2, 0).numpy()
        if hasattr(out_img, 'permute'):
            out_img = out_img.permute(1, 2, 0).numpy()
        axes[0, i].imshow(in_img)
        axes[0, i].axis('off')
        axes[1, i].imshow(out_img)
        axes[1, i].axis('off')
    axes[0, 0].set_ylabel('Input', fontsize=14)
    axes[1, 0].set_ylabel('Output', fontsize=14)
    plt.tight_layout()
    plt.show()


# ----------

def train_test_split(source_path="./out", upscale_factor=2, crop_size=256):
    # Create a training split from the dataset.

    crop_size = calculate_valid_crop_size(crop_size, upscale_factor)
    dataset = ImageDataset(source_path, 
                          transform_in=input_transform(crop_size, upscale_factor), 
                          transform_out=target_transform(crop_size))
    
    # Split the dataset into training and testing sets
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42))
    
    return train_dataset, test_dataset





if __name__ == "__main__":

    # test
    source_path = "./dataSets/cars"
    upscale_factor = 2
    crop_size = 256

    train_dataset, test_dataset = train_test_split(source_path, upscale_factor, crop_size)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    show_images(train_dataset, num_images=2)
