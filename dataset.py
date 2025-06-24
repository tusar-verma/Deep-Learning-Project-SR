import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor


class ImageDataset(Dataset):
    def __init__(self, image_dir, scale_factor=3):
        self.image_dir = image_dir
        self.image_paths = []
        self.coords = []  # List of (image_index, top, left)
        self.scale = scale_factor
        self.patch_size = 17 * self.scale
        self.stride = 14 * self.scale
        self.to_tensor = ToTensor()

        self._index_patches()

    def _index_patches(self):
        # Precompute all patch coordinates for all images
        image_filenames = [f for f in os.listdir(self.image_dir) if f.lower().endswith(('jpg', 'jpeg', 'png'))]
        self.image_paths = [os.path.join(self.image_dir, f) for f in image_filenames]

        for img_idx, path in enumerate(self.image_paths):
            img = Image.open(path).convert('YCbCr')
            y, _, _ = img.split()
            w, h = y.size

            if w < self.patch_size or h < self.patch_size:
                continue

            for top in range(0, h - self.patch_size + 1, self.stride):
                for left in range(0, w - self.patch_size + 1, self.stride):
                    self.coords.append((img_idx, top, left))

    def __len__(self):
        return len(self.coords)
    

    def __getitem__(self, idx):
        img_idx, top, left = self.coords[idx]
        img = Image.open(self.image_paths[img_idx]).convert('YCbCr')
        y, _, _ = img.split()

        hr_patch = y.crop((left, top, left + self.patch_size, top + self.patch_size))
        lr_patch = hr_patch.resize((self.patch_size // self.scale, self.patch_size // self.scale), Image.BICUBIC)

        return self.to_tensor(lr_patch), self.to_tensor(hr_patch)

