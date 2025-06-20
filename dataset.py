import os
from PIL import Image
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    def __init__(self, source_dir, transform_in=None, transform_out=None):
        self.source_dir = source_dir
        self.filenames = sorted(os.listdir(source_dir))  # Assumes all filenames match in both dirs
        self.transform_in = transform_in
        self.transform_out = transform_out

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_name = self.filenames[idx]
        source_img_path = os.path.join(self.source_dir, img_name)

        # quizas otro formato que no sea RGB 
        in_img = Image.open(source_img_path).convert("YCbCr").split()[0]  # Get only the Y channel
        out_img = Image.open(source_img_path).convert("YCbCr").split()[0]  # Get only the Y channel
        # in_img = Image.open(source_img_path).convert("RGB")
        # out_img = Image.open(source_img_path).convert("RGB")

        if self.transform_in:
            in_img = self.transform_in(in_img)
        if self.transform_out:
            out_img = self.transform_out(out_img)

        return in_img, out_img

