import argparse
import torch
import torch.nn as nn
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage
import os
import sys

img_to_tensor = ToTensor()
tensor_to_pil = ToPILImage()

def calculate_psnr(sr, hr):
    criterion = nn.MSELoss()
    mse = criterion(sr, hr)
    if mse == 0:
        return float('inf')
    return 10 * torch.log10(1.0 / mse)

def crop_image_to_multiple(img, scale):
    w, h = img.size
    w_crop = (w // scale) * scale
    h_crop = (h // scale) * scale
    return img.crop((0, 0, w_crop, h_crop))

def generate_lr_image(hr_image, upscale_factor):
    hr_image = crop_image_to_multiple(hr_image, upscale_factor)
    w, h = hr_image.size
    lr_image = hr_image.resize((w // upscale_factor, h // upscale_factor), Image.BICUBIC)
    return lr_image

def generate_all_lr_images_and_save(test_images, test_dir, upscale_factor, lr_save_dir):
    if os.path.isdir(lr_save_dir) and os.listdir(lr_save_dir):
        print(f"LR data already exists in {lr_save_dir}. Skipping generation.")
        return

    os.makedirs(lr_save_dir, exist_ok=True)

    for img_name in test_images:
        img_path = os.path.join(test_dir, img_name)
        img_hr = Image.open(img_path).convert('YCbCr')
        img_hr = crop_image_to_multiple(img_hr, upscale_factor)
        lr_img = generate_lr_image(img_hr, upscale_factor)
        lr_img.save(os.path.join(lr_save_dir, img_name))

def run_testset_through_model(datasetname, model, device, test_images, log_file_name, test_dir, lr_dir, sr_save_dir):
    psnr_total = 0
    count = 0
    string_output = ""

    for img_name in test_images:
        hr_img_path = os.path.join(test_dir, img_name)
        lr_img_path = os.path.join(lr_dir, img_name)

        img_hr = Image.open(hr_img_path).convert('YCbCr')
        img_hr = crop_image_to_multiple(img_hr, opt.upscale_factor)
        img_lr = Image.open(lr_img_path).convert('YCbCr')

        hr_y, hr_cb, hr_cr = img_hr.split()
        lr_y, lr_cb, lr_cr = img_lr.split()

        lr_tensor = img_to_tensor(lr_y).unsqueeze(0).to(device)
        hr_tensor = img_to_tensor(hr_y).unsqueeze(0).to(device)

        with torch.no_grad():
            sr_tensor = model(lr_tensor)

        psnr_val = calculate_psnr(sr_tensor, hr_tensor).item()
        psnr_total += psnr_val
        count += 1
        string_output += f"{img_name}: PSNR = {psnr_val:.4f} dB\n"

        sr_y = sr_tensor.squeeze(0).squeeze(0).cpu().clamp(0, 1)
        sr_y_img = tensor_to_pil(sr_y)

        sr_cb = lr_cb.resize(sr_y_img.size, Image.BICUBIC)
        sr_cr = lr_cr.resize(sr_y_img.size, Image.BICUBIC)

        sr_img = Image.merge('YCbCr', (sr_y_img, sr_cb, sr_cr)).convert('RGB')
        sr_img.save(os.path.join(sr_save_dir, img_name))

    if count > 0:
        avg_psnr = psnr_total / count
        string_output += f"{datasetname}: Average PSNR: {avg_psnr:.4f} dB\n"
        print(f"{datasetname}: Average PSNR: {avg_psnr:.4f} dB")

        if log_file_name:
            with open(log_file_name, 'w') as f:
                f.write(string_output)
    else:
        print("No test images found.")

def main():
    parser = argparse.ArgumentParser(description='Test model and calculate average PSNR (PyTorch-only)')
    parser.add_argument('--datasetname', type=str, required=True, help="dataset name for logging purposes")
    parser.add_argument('--upscale_factor', type=int, required=True, help="Super resolution upscale factor")
    parser.add_argument('--test_dir', type=str, required=True, help='Directory with test images (HR)')
    parser.add_argument('--checkpoint', type=str, required=True, help='Checkpoint file for model')
    parser.add_argument('--cuda', action='store_true', help='Use CUDA')

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    global opt
    opt = parser.parse_args()

    device = torch.device('cuda' if opt.cuda and torch.cuda.is_available() else 'cpu')

    checkpoint = torch.load(opt.checkpoint, weights_only=False)
    model = checkpoint['model'].to(device)
    model.eval()

    test_images = [f for f in os.listdir(opt.test_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    lr_save_dir = os.path.join('experiments', opt.datasetname, 'lr_output')
    sr_save_dir = os.path.join('experiments', opt.datasetname, 'sr_output')
    os.makedirs(lr_save_dir, exist_ok=True)
    os.makedirs(sr_save_dir, exist_ok=True)

    generate_all_lr_images_and_save(test_images, opt.test_dir, opt.upscale_factor, lr_save_dir)

    run_testset_through_model(opt.datasetname, model, device, test_images, os.path.join('experiments', f'{opt.datasetname}_out.txt'), opt.test_dir, lr_save_dir, sr_save_dir)

if __name__ == '__main__':
    main()
